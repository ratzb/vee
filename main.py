# ============================================================
# BOT TRADING V92.0-PAPER – SIMULACIÓN CON SALDO Y COMISIONES
# ============================================================
import os
import time
import io
import json
import hmac
import hashlib
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timezone, timedelta
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xml.etree.ElementTree as ET

# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURACIÓN GENERAL (V92.0-PAPER)
# ============================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
LEVERAGE = 10
MAX_OPEN_TRADES = 3
SLEEP_SECONDS = 300

QTY_BTC = 0.002
TRAILING_OFFSET_ATR = 0.8
SL_MULTIPLIER = 1.8
MIN_RR_RATIO = 1.5
MIN_TRAILING_STEP = 5.0
COMISION_TAKER = 0.0006

GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

# ========== PAPER TRADING CONFIG ==========
PAPER_TRADING = True   # Cambiar a False para modo real
SALDO_INICIAL = 10000.0  # USD
SALDO_SIMULADO = SALDO_INICIAL
POSICIONES_SIMULADAS = []  # Misma estructura que ACTIVE_TRADES pero sin órdenes reales
# ============================================

# Cache de noticias
NEWS_CACHE = {
    "titulo": "No disponible",
    "fuente": "Ninguna",
    "sent_label": "Neutral",
    "sent_score": 0.0,
    "timestamp": None
}
NEWS_CACHE_TTL = 3600

# ============================================================
# CREDENCIALES (solo necesarias para datos, no para órdenes en papel)
# ============================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

sentiment_analyzer = SentimentIntensityAnalyzer()
BASE_URL = "https://api.bybit.com"

# Estadísticas globales (para simulación)
TRADES_TOTALES = 0
TRADES_WIN = 0
TRADES_LOSS = 0
PNL_GLOBAL = 0.0
PNL_GLOBAL_NETO = 0.0
TRADES_DESDE_RESUMEN = 0
MAX_DRAWDOWN = 0.0
BALANCE_MAX = SALDO_INICIAL
TRADE_COUNTER = 0
ACTIVE_TRADES = []  # Se usará para simulación o real

# ============================================================
# TELEGRAM (sin cambios)
# ============================================================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram send error: {e}")

def telegram_grafico(fig):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', facecolor='black')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': TELEGRAM_CHAT_ID}, timeout=15)
        buf.close()
    except Exception as e:
        logger.error(f"Telegram photo error: {e}")

# ============================================================
# FUNCIONES API BYBIT (solo para datos en paper trading)
# ============================================================
def bybit_request(endpoint, method='GET', params=None, payload=None):
    # En paper trading, solo necesitamos GET para datos, pero mantenemos la función.
    time.sleep(0.15)
    timestamp = str(int(datetime.now(timezone.utc).timestamp() * 1000))
    recv_window = '5000'
    if payload is None:
        payload = {}
    if params:
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    else:
        query_string = ''

    if method == 'GET':
        full_url = f"{BASE_URL}{endpoint}?{query_string}" if query_string else f"{BASE_URL}{endpoint}"
        param_str = query_string
    else:
        full_url = f"{BASE_URL}{endpoint}"
        body = json.dumps(payload, sort_keys=True)
        param_str = body

    sign_str = timestamp + BYBIT_API_KEY + recv_window + param_str
    signature = hmac.new(
        bytes(BYBIT_API_SECRET, "utf-8"),
        bytes(sign_str, "utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }

    if method == 'GET':
        response = requests.get(full_url, headers=headers, timeout=15)
    else:
        response = requests.post(full_url, headers=headers, data=body, timeout=15)

    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code} - {response.text}")
    data = response.json()
    if data.get('retCode') != 0:
        raise Exception(f"Bybit API error: {data.get('retMsg')} (code {data.get('retCode')})")
    return data.get('result', {})

def set_leverage(symbol, leverage):
    # En paper no es necesario, pero lo dejamos
    logger.info(f"Simulación: Apalancamiento establecido a {leverage}x para {symbol}")

def obtener_posiciones_abiertas():
    # En paper, devolvemos posiciones simuladas en formato API (si existe)
    # Para mantener compatibilidad, convertimos POSICIONES_SIMULADAS a formato API.
    # Pero como en paper no usamos API, podemos devolver lista vacía.
    # En revisar_posiciones_reales, manejaremos la simulación directamente.
    if PAPER_TRADING:
        # Devolver lista vacía para que no interfiera
        return []
    else:
        endpoint = "/v5/position/list"
        params = {
            "category": "linear",
            "symbol": SYMBOL
        }
        result = bybit_request(endpoint, params=params)
        positions = result.get('list', [])
        abiertas = [p for p in positions if float(p.get('size', 0)) != 0]
        return abiertas

# Funciones de órdenes reales (solo se usarán si PAPER_TRADING=False)
def crear_orden_market(symbol, side, qty, reduce_only=False):
    if PAPER_TRADING:
        # Simular: no hacer nada, solo registrar en log
        logger.info(f"📝 SIMULACIÓN: Orden Market {side} {qty} {symbol}")
        return {"orderId": f"SIM_{int(time.time())}"}
    else:
        endpoint = "/v5/order/create"
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "reduceOnly": reduce_only
        }
        result = bybit_request(endpoint, method='POST', payload=payload)
        return result

def crear_orden_limit(symbol, side, qty, price, reduce_only=False, post_only=False):
    if PAPER_TRADING:
        logger.info(f"📝 SIMULACIÓN: Orden Limit {side} {qty}@{price} {symbol}")
        return {"orderId": f"SIM_LIMIT_{int(time.time())}"}
    else:
        endpoint = "/v5/order/create"
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Limit",
            "qty": str(qty),
            "price": str(price),
            "timeInForce": "GTC",
            "reduceOnly": reduce_only,
            "postOnly": post_only
        }
        result = bybit_request(endpoint, method='POST', payload=payload)
        return result

def crear_orden_stop_market(symbol, side, qty, stop_price, reduce_only=True):
    if PAPER_TRADING:
        logger.info(f"📝 SIMULACIÓN: Orden Stop Market {side} {qty} @ {stop_price} {symbol}")
        return {"orderId": f"SIM_STOP_{int(time.time())}"}
    else:
        endpoint = "/v5/order/create"
        trigger_dir = 2 if side.capitalize() == "Sell" else 1
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC",
            "triggerPrice": str(stop_price),
            "triggerDirection": trigger_dir,
            "reduceOnly": reduce_only
        }
        result = bybit_request(endpoint, method='POST', payload=payload)
        return result

def cancelar_orden(order_id, symbol):
    if PAPER_TRADING:
        logger.info(f"📝 SIMULACIÓN: Cancelar orden {order_id}")
        return {}
    else:
        endpoint = "/v5/order/cancel"
        payload = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id
        }
        result = bybit_request(endpoint, method='POST', payload=payload)
        return result

def modificar_orden_stop(order_id, symbol, stop_price):
    if PAPER_TRADING:
        logger.info(f"📝 SIMULACIÓN: Modificar stop a {stop_price} para orden {order_id}")
        return {}
    else:
        endpoint = "/v5/order/amend"
        payload = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
            "triggerPrice": str(stop_price)
        }
        result = bybit_request(endpoint, method='POST', payload=payload)
        return result

# ============================================================
# OBTENER VELAS, INDICADORES, ESTADO (sin cambios)
# ============================================================
def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    if not r.text:
        raise Exception("Respuesta vacía de Bybit")
    try:
        data_json = r.json()
    except Exception:
        raise Exception(f"Bybit devolvió respuesta no-JSON: {r.text}")
    if not isinstance(data_json, dict):
        raise Exception(f"Bybit devolvió JSON no dict: {type(data_json)}")
    if "retCode" in data_json and data_json["retCode"] != 0:
        raise Exception(f"Bybit Error retCode={data_json.get('retCode')} retMsg={data_json.get('retMsg')}")
    if "result" not in data_json or not isinstance(data_json["result"], dict):
        raise Exception(f"Respuesta inválida Bybit: {data_json}")
    if "list" not in data_json["result"] or not isinstance(data_json["result"]["list"], list):
        raise Exception(f"Bybit result sin 'list' o no es lista: {data_json['result']}")
    data = data_json["result"]["list"][::-1]
    if len(data) == 0:
        raise Exception("Bybit devolvió lista vacía de velas")
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    return df

def extraer_estado_mercado(df, usar_cerrada=True):
    if df.empty:
        return None
    if usar_cerrada and len(df) < 2:
        usar_cerrada = False
    idx = -2 if usar_cerrada and len(df) >= 2 else -1
    if idx < -len(df) or idx >= len(df):
        idx = -1
    fila = df.iloc[idx]
    precio = fila['close']
    ema20 = df['ema20'].iloc[idx]
    atr = df['atr'].iloc[idx]
    if pd.isna(ema20):
        ema_serie = df['ema20'].dropna()
        ema20 = ema_serie.iloc[-1] if not ema_serie.empty else precio
    if pd.isna(atr):
        atr_serie = df['atr'].dropna()
        atr = atr_serie.iloc[-1] if not atr_serie.empty else precio * 0.01

    ema_vals = df['ema20'].dropna().tail(20)
    if len(ema_vals) >= 2:
        x = np.arange(len(ema_vals))
        slope_ema, _, _, _, _ = linregress(x, ema_vals.values)
    else:
        slope_ema = 0.0

    dist_ema = precio - ema20
    dist_ema_atr = dist_ema / atr if atr != 0 else 0.0

    ventana = min(50, len(df))
    if ventana < 2:
        min_50 = df['close'].min()
        max_50 = df['close'].max()
    else:
        min_50 = df['close'].iloc[-ventana:].min()
        max_50 = df['close'].iloc[-ventana:].max()

    zona_soporte = (min_50, min_50 + 0.3 * atr)
    zona_resistencia = (max_50 - 0.3 * atr, max_50)
    soporte = min_50
    resistencia = max_50

    if precio > ema20:
        ema_nivel = 'soporte'
    else:
        ema_nivel = 'resistencia'

    open_actual = fila['open']
    high_actual = fila['high']
    low_actual = fila['low']
    close_actual = fila['close']
    rango = high_actual - low_actual
    cuerpo = abs(close_actual - open_actual)
    cuerpo_relativo = cuerpo / rango if rango > 0 else 0.0
    sombra_superior = high_actual - max(open_actual, close_actual)
    sombra_inferior = min(open_actual, close_actual) - low_actual

    patron = ""
    if cuerpo_relativo > 0.6:
        if close_actual > open_actual:
            patron = "Vela alcista de cuerpo grande"
        else:
            patron = "Vela bajista de cuerpo grande"
    elif sombra_inferior > 2 * cuerpo and sombra_superior < cuerpo:
        patron = "Martillo (posible reversión alcista)"
    elif sombra_superior > 2 * cuerpo and sombra_inferior < cuerpo:
        patron = "Estrella fugaz (posible reversión bajista)"
    elif cuerpo_relativo < 0.2 and sombra_superior > 0 and sombra_inferior > 0:
        patron = "Doji (indecisión)"
    else:
        patron = "Vela normal"

    slope, intercept, tendencia = _detectar_tendencia(df, idx)

    estado = {
        'precio': precio,
        'ema20': ema20,
        'atr': atr,
        'soporte': soporte,
        'resistencia': resistencia,
        'zona_soporte': zona_soporte,
        'zona_resistencia': zona_resistencia,
        'ema_nivel': ema_nivel,
        'cuerpo_relativo': cuerpo_relativo,
        'patron': patron,
        'slope': slope,
        'intercept': intercept,
        'tendencia': tendencia,
        'sentimiento': 1 if slope > 0.02 else (-1 if slope < -0.02 else 0),
        'fecha': df.index[idx],
        'open': open_actual,
        'high': high_actual,
        'low': low_actual,
        'close': close_actual,
        'sombra_superior': sombra_superior,
        'sombra_inferior': sombra_inferior,
        'idx': idx,
        'dist_ema_atr': dist_ema_atr,
        'pendiente_ema': slope_ema
    }
    return estado

def _detectar_tendencia(df, idx, ventana=80):
    if idx < 0:
        idx = len(df) + idx
    if idx < 0:
        idx = 0
    if idx >= len(df):
        idx = len(df) - 1
    inicio = max(0, idx - ventana + 1)
    if inicio > idx:
        inicio = idx
    y = df['close'].values[inicio:idx+1]
    if len(y) < 2:
        return 0, 0, "➡️ LATERAL"
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)
    if slope > 0.02:
        direccion = '📈 ALCISTA'
    elif slope < -0.02:
        direccion = '📉 BAJISTA'
    else:
        direccion = '➡️ LATERAL'
    return slope, intercept, direccion

def extraer_estado_mercado_por_indice(df, idx):
    if df.empty:
        return None
    if idx < 0:
        idx = len(df) + idx
    if idx < 0 or idx >= len(df):
        return None
    df_temp = df.iloc[:idx+1]
    if df_temp.empty:
        return None
    return extraer_estado_mercado(df_temp, usar_cerrada=False)

# ============================================================
# ANÁLISIS DE VELAS Y TENDENCIA (con zonas)
# ============================================================
def analizar_velas_recientes(df, estado_actual, n=5):
    if len(df) < n + 1:
        return {
            'rechazo_inferior': False,
            'rechazo_superior': False,
            'velas_alcistas_consec': 0,
            'velas_bajistas_consec': 0,
            'ruptura_tendencia': None,
            'cerca_soporte': False,
            'cerca_resistencia': False,
            'toques_soporte': 0,
            'toques_resistencia': 0
        }

    idx_actual = estado_actual['idx']
    if idx_actual < 0:
        idx_actual = len(df) + idx_actual

    inicio = max(0, idx_actual - n + 1)
    segmento = df.iloc[inicio:idx_actual+1]
    if len(segmento) < 2:
        return {
            'rechazo_inferior': False,
            'rechazo_superior': False,
            'velas_alcistas_consec': 0,
            'velas_bajistas_consec': 0,
            'ruptura_tendencia': None,
            'cerca_soporte': False,
            'cerca_resistencia': False,
            'toques_soporte': 0,
            'toques_resistencia': 0
        }

    atr = estado_actual['atr']
    soporte = estado_actual['soporte']
    resistencia = estado_actual['resistencia']
    zona_soporte = estado_actual.get('zona_soporte', (soporte, soporte + 0.3*atr))
    zona_resistencia = estado_actual.get('zona_resistencia', (resistencia - 0.3*atr, resistencia))

    rechazo_inferior = False
    rechazo_superior = False
    toques_soporte = 0
    toques_resistencia = 0

    for i in range(len(segmento)):
        vela = segmento.iloc[i]
        low = vela['low']
        high = vela['high']
        close = vela['close']
        open_ = vela['open']
        cuerpo = abs(close - open_)
        rango = high - low
        if rango == 0:
            continue
        sombra_inf = min(open_, close) - low
        sombra_sup = high - max(open_, close)

        if low <= zona_soporte[1] and high >= zona_soporte[0]:
            toques_soporte += 1
            if sombra_inf > 2 * cuerpo and sombra_inf > 0.3 * atr and close > (low + rango * 0.5):
                rechazo_inferior = True
        if high >= zona_resistencia[0] and low <= zona_resistencia[1]:
            toques_resistencia += 1
            if sombra_sup > 2 * cuerpo and sombra_sup > 0.3 * atr and close < (high - rango * 0.5):
                rechazo_superior = True

    alcistas_consec = 0
    bajistas_consec = 0
    for i in range(len(segmento)-1, -1, -1):
        if segmento.iloc[i]['close'] > segmento.iloc[i]['open']:
            if bajistas_consec == 0:
                alcistas_consec += 1
            else:
                break
        else:
            if alcistas_consec == 0:
                bajistas_consec += 1
            else:
                break

    ventana_tend = 80
    ruptura_tendencia = None
    if len(df) >= ventana_tend:
        df_temp = df.iloc[-ventana_tend:]
        x_vals = np.arange(len(df_temp))
        y_vals = df_temp['close'].values
        slope_t, intercept_t, _, _, _ = linregress(x_vals, y_vals)
        if len(df_temp) >= 2:
            linea_actual = slope_t * (len(df_temp)-1) + intercept_t
            linea_anterior = slope_t * (len(df_temp)-2) + intercept_t
            precio_actual = df_temp['close'].iloc[-1]
            precio_anterior = df_temp['close'].iloc[-2]
            if precio_anterior < linea_anterior and precio_actual > linea_actual:
                ruptura_tendencia = 'alcista'
            elif precio_anterior > linea_anterior and precio_actual < linea_actual:
                ruptura_tendencia = 'bajista'

    cerca_soporte = (estado_actual['precio'] >= zona_soporte[0] and estado_actual['precio'] <= zona_soporte[1])
    cerca_resistencia = (estado_actual['precio'] >= zona_resistencia[0] and estado_actual['precio'] <= zona_resistencia[1])

    return {
        'rechazo_inferior': rechazo_inferior,
        'rechazo_superior': rechazo_superior,
        'velas_alcistas_consec': alcistas_consec,
        'velas_bajistas_consec': bajistas_consec,
        'ruptura_tendencia': ruptura_tendencia,
        'cerca_soporte': cerca_soporte,
        'cerca_resistencia': cerca_resistencia,
        'toques_soporte': toques_soporte,
        'toques_resistencia': toques_resistencia
    }

def analizar_rechazo_ema(df, estado_actual, n_velas=5):
    if len(df) < n_velas + 1:
        return False, False, "Sin datos suficientes"

    idx_actual = estado_actual['idx']
    if idx_actual < 0:
        idx_actual = len(df) + idx_actual
    inicio = max(0, idx_actual - n_velas)
    segmento = df.iloc[inicio:idx_actual+1]
    if len(segmento) < 2:
        return False, False, "Sin datos"

    rechazo_alcista = False
    rechazo_bajista = False
    desc = ""

    for i in range(len(segmento)-1):
        vela = segmento.iloc[i]
        low = vela['low']
        high = vela['high']
        ema = vela['ema20']
        close = vela['close']

        if low < ema < close:
            rechazo_alcista = True
            desc = "Rechazo alcista en EMA20"
            break
        if high > ema > close:
            rechazo_bajista = True
            desc = "Rechazo bajista en EMA20"
            break

    return rechazo_alcista, rechazo_bajista, desc

def evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info):
    condiciones = 0
    razones = []
    
    if velas_info['ruptura_tendencia'] is not None:
        condiciones += 1
        razones.append(f"Ruptura de tendencia ({velas_info['ruptura_tendencia']})")
    
    if (velas_info['cerca_soporte'] and velas_info['rechazo_inferior'] and velas_info['toques_soporte'] >= 2) or \
       (velas_info['cerca_resistencia'] and velas_info['rechazo_superior'] and velas_info['toques_resistencia'] >= 2):
        condiciones += 1
        razones.append("Rechazo con mecha en zona de soporte/resistencia (múltiples toques)")
    
    patron = estado_actual['patron']
    if "Martillo" in patron or "Estrella fugaz" in patron or "cuerpo grande" in patron:
        condiciones += 1
        razones.append(f"Patrón de reversión: {patron}")
    
    permitido = condiciones >= 1
    if permitido:
        razones.append(f"✅ Confluencia suficiente ({condiciones}/3) para operar en contra de tendencia")
    else:
        razones.append(f"❌ Confluencia insuficiente ({condiciones}/3) para operar en contra de tendencia")
    
    return permitido, razones

# ============================================================
# MOTOR DE DECISIÓN V92.0 (sin cambios)
# ============================================================
def detectar_patron_multivela(df, n=3):
    if len(df) < n:
        return None, ""
    closes = df['close'].iloc[-n:].values
    opens = df['open'].iloc[-n:].values
    if all(closes[i] > opens[i] and closes[i] > closes[i-1] for i in range(1, n)):
        return "tres_soldados_blancos", "Alcista fuerte (continuación)"
    if all(closes[i] < opens[i] and closes[i] < closes[i-1] for i in range(1, n)):
        return "tres_cuervos_negros", "Bajista fuerte (continuación)"
    return None, ""

def confirmar_patron(estado_ant, estado_act):
    if estado_ant['patron'] == "Martillo (posible reversión alcista)" and estado_act['close'] > estado_ant['close']:
        return True, "Martillo confirmado (alcista)"
    if estado_ant['patron'] == "Estrella fugaz (posible reversión bajista)" and estado_act['close'] < estado_ant['close']:
        return True, "Estrella fugaz confirmada (bajista)"
    return False, ""

def motor_v92(estado_actual, df):
    if estado_actual is None:
        return None, 0, 0, ["Estado nulo"]

    precio = estado_actual['precio']
    soporte = estado_actual['soporte']
    resistencia = estado_actual['resistencia']
    atr = estado_actual['atr']
    tendencia = estado_actual['tendencia']
    ema20 = estado_actual['ema20']
    ema_nivel = estado_actual['ema_nivel']
    patron = estado_actual['patron']
    dist_ema_atr = estado_actual['dist_ema_atr']
    pendiente_ema = estado_actual['pendiente_ema']
    razones = []

    velas_info = analizar_velas_recientes(df, estado_actual)
    rechazo_inf = velas_info['rechazo_inferior']
    rechazo_sup = velas_info['rechazo_superior']
    alcistas_cons = velas_info['velas_alcistas_consec']
    bajistas_cons = velas_info['velas_bajistas_consec']
    ruptura_tend = velas_info['ruptura_tendencia']
    cerca_soporte = velas_info['cerca_soporte']
    cerca_resistencia = velas_info['cerca_resistencia']
    toques_soporte = velas_info['toques_soporte']
    toques_resistencia = velas_info['toques_resistencia']

    # PRIORIDAD 1: RUPTURAS
    if precio > resistencia + 0.3 * atr and patron in ["Vela alcista de cuerpo grande", "Vela normal"]:
        if estado_actual['close'] > resistencia and estado_actual['close'] > estado_actual['open']:
            razones.append("✅ RUPTURA ALCISTA de resistencia confirmada")
            tp1_candidate = precio + 2.0 * atr
            sl_candidate = precio - 1.8 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para ruptura alcista")
                return None, soporte, resistencia, razones

    if precio < soporte - 0.3 * atr and patron in ["Vela bajista de cuerpo grande", "Vela normal"]:
        if estado_actual['close'] < soporte and estado_actual['close'] < estado_actual['open']:
            razones.append("✅ RUPTURA BAJISTA de soporte confirmada")
            tp1_candidate = precio - 2.0 * atr
            sl_candidate = precio + 1.8 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para ruptura bajista")
                return None, soporte, resistencia, razones

    # PRIORIDAD 2: REBOTES EN ZONAS S/R
    if cerca_soporte and (rechazo_inf or patron in ["Martillo (posible reversión alcista)", "Vela alcista de cuerpo grande"]):
        if bajistas_cons < 5:
            razones.append(f"✅ COMPRA en zona de soporte (toques: {toques_soporte})")
            tp1_candidate = min(resistencia, precio + 2.5 * atr)
            sl_candidate = precio - 1.8 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para compra en soporte")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"⚠️ Demasiadas velas bajistas ({bajistas_cons}) → esperar ruptura")
            return None, soporte, resistencia, razones

    if cerca_resistencia and (rechazo_sup or patron in ["Estrella fugaz (posible reversión bajista)", "Vela bajista de cuerpo grande"]):
        if alcistas_cons < 5:
            razones.append(f"✅ VENTA en zona de resistencia (toques: {toques_resistencia})")
            tp1_candidate = max(soporte, precio - 2.5 * atr)
            sl_candidate = precio + 1.8 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para venta en resistencia")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"⚠️ Demasiadas velas alcistas ({alcistas_cons}) → esperar ruptura")
            return None, soporte, resistencia, razones

    # PRIORIDAD 3: EMA
    rechazo_alcista, rechazo_bajista, desc_rechazo = analizar_rechazo_ema(df, estado_actual)

    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and rechazo_alcista:
        razones.append(f"✅ {desc_rechazo} - COMPRA en EMA20")
        tp1_candidate = min(resistencia, precio + 2.5 * atr)
        sl_candidate = precio - 1.8 * atr
        if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
            return 'Buy', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente para compra en EMA")
            return None, soporte, resistencia, razones

    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and rechazo_bajista:
        razones.append(f"✅ {desc_rechazo} - VENTA en EMA20")
        tp1_candidate = max(soporte, precio - 2.5 * atr)
        sl_candidate = precio + 1.8 * atr
        if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
            return 'Sell', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente para venta en EMA")
            return None, soporte, resistencia, razones

    # CONTRA-TENDENCIA
    en_contra_buy = (precio < ema20 and (tendencia == '📉 BAJISTA' or pendiente_ema < -0.01))
    en_contra_sell = (precio > ema20 and (tendencia == '📈 ALCISTA' or pendiente_ema > 0.01))

    if en_contra_buy:
        permitido, razones_contra = evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info)
        razones.extend(razones_contra)
        if not permitido:
            razones.append("❌ No se permite compra en contra de tendencia (sin confluencia suficiente).")
            return None, soporte, resistencia, razones
        else:
            razones.append("⚠️ Compra en contra de tendencia permitida por confluencia.")
            tp1_candidate = min(resistencia, precio + 2.5 * atr)
            sl_candidate = precio - 1.8 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para compra en contra-tendencia")
                return None, soporte, resistencia, razones

    elif en_contra_sell:
        permitido, razones_contra = evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info)
        razones.extend(razones_contra)
        if not permitido:
            razones.append("❌ No se permite venta en contra de tendencia (sin confluencia suficiente).")
            return None, soporte, resistencia, razones
        else:
            razones.append("⚠️ Venta en contra de tendencia permitida por confluencia.")
            tp1_candidate = max(soporte, precio - 2.5 * atr)
            sl_candidate = precio + 1.8 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para venta en contra-tendencia")
                return None, soporte, resistencia, razones

    # FILTROS ADICIONALES
    if abs(dist_ema_atr) > 3.0:
        razones.append(f"❌ Precio demasiado lejos de EMA20 ({dist_ema_atr:.2f} ATR) - NO OPERAR")
        return None, soporte, resistencia, razones

    es_rango = abs(pendiente_ema) < 0.001
    if es_rango and not (precio > resistencia or precio < soporte) and patron not in ["Martillo (posible reversión alcista)", "Estrella fugaz (posible reversión bajista)", "Doji (indecisión)"]:
        razones.append("⚠️ Mercado en rango lateral sin ruptura ni patrón fuerte → NO OPERAR")
        return None, soporte, resistencia, razones

    patron_mult, desc_mult = detectar_patron_multivela(df)
    if patron_mult == "tres_soldados_blancos" and tendencia in ['📈 ALCISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        tp1_candidate = min(resistencia, precio + 2.5 * atr)
        sl_candidate = precio - 1.8 * atr
        if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
            return 'Buy', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente para patrón multivela")
            return None, soporte, resistencia, razones

    if patron_mult == "tres_cuervos_negros" and tendencia in ['📉 BAJISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        tp1_candidate = max(soporte, precio - 2.5 * atr)
        sl_candidate = precio + 1.8 * atr
        if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
            return 'Sell', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente para patrón multivela")
            return None, soporte, resistencia, razones

    confirmado = False
    msg_conf = ""
    if len(df) >= 3:
        if estado_actual['idx'] == -2:
            idx_ant = -3
        else:
            idx_ant = -2
        estado_ant = extraer_estado_mercado_por_indice(df, idx_ant)
        if estado_ant:
            confirmado, msg_conf = confirmar_patron(estado_ant, estado_actual)
            if confirmado:
                razones.append(msg_conf)
                if "Martillo" in estado_ant['patron'] and estado_actual['close'] > estado_ant['close']:
                    return 'Buy', soporte, resistencia, razones
                if "Estrella fugaz" in estado_ant['patron'] and estado_actual['close'] < estado_ant['close']:
                    return 'Sell', soporte, resistencia, razones

    razones.append("Sin confluencia válida")
    return None, soporte, resistencia, razones

# ============================================================
# FILTRO FUNDAMENTAL (sin cambios)
# ============================================================
def actualizar_cache_noticias():
    global NEWS_CACHE
    ahora = datetime.now(timezone.utc)
    if NEWS_CACHE["timestamp"] is not None:
        edad = (ahora - NEWS_CACHE["timestamp"]).total_seconds()
        if edad < NEWS_CACHE_TTL:
            logger.debug(f"Usando caché de noticias (edad: {edad:.0f}s)")
            return
    logger.info("⏳ Actualizando caché de noticias...")
    titulo, fuente, sent_label, sent_score = _obtener_noticias_frescas()
    NEWS_CACHE = {
        "titulo": titulo,
        "fuente": fuente,
        "sent_label": sent_label,
        "sent_score": sent_score,
        "timestamp": ahora
    }
    logger.info(f"✅ Caché actualizada: {titulo} | {sent_label} ({sent_score:.3f})")

def _obtener_noticias_frescas():
    noticias = []
    fuente = "Ninguna"
    if NEWS_API_KEY:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": "BTC OR Bitcoin OR Cryptocurrency",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
                "apiKey": NEWS_API_KEY
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok" and data.get("articles"):
                    noticias = data["articles"]
                    fuente = "NewsAPI"
                    logger.info(f"📰 Obtenidas {len(noticias)} noticias desde NewsAPI")
        except Exception as e:
            logger.error(f"Error en NewsAPI: {e}")
    if not noticias:
        try:
            rss_url = "https://news.google.com/rss/search?q=Bitcoin+OR+Cryptocurrency&hl=en-US&gl=US&ceid=US:en"
            r = requests.get(rss_url, timeout=10)
            if r.status_code == 200:
                root = ET.fromstring(r.content)
                items = root.findall("./channel/item")
                for item in items[:20]:
                    title = item.find("title").text if item.find("title") is not None else ""
                    if title:
                        noticias.append({
                            "title": title,
                            "source": {"name": "Google News"}
                        })
                fuente = "Google News RSS"
                logger.info(f"📰 Obtenidas {len(noticias)} noticias desde Google News RSS")
        except Exception as e:
            logger.error(f"Error en Google News RSS: {e}")
    if noticias:
        titulo = noticias[0].get("title", "No disponible")
        fuente = noticias[0].get("source", {}).get("name", fuente)
        scores = []
        for n in noticias:
            t = n.get("title", "")
            if t:
                scores.append(sentiment_analyzer.polarity_scores(t)["compound"])
        if scores:
            sent_score = sum(scores) / len(scores)
            if sent_score > 0.05:
                sent_label = "Bullish"
            elif sent_score < -0.05:
                sent_label = "Bearish"
            else:
                sent_label = "Neutral"
        else:
            sent_label = "Neutral"
            sent_score = 0.0
        return titulo, fuente, sent_label, sent_score
    return "No disponible", "Ninguna", "Neutral", 0.0

def obtener_noticias_y_sentimiento():
    actualizar_cache_noticias()
    return (
        NEWS_CACHE["titulo"],
        NEWS_CACHE["fuente"],
        NEWS_CACHE["sent_label"],
        NEWS_CACHE["sent_score"]
    )

def filtrar_por_fundamental(decision, sent_label, estado):
    if decision is None:
        return True, "Sin decisión"
    precio = estado['precio']
    soporte = estado['soporte']
    resistencia = estado['resistencia']
    atr = estado['atr']
    dist_soporte = abs(precio - soporte)
    dist_resistencia = abs(precio - resistencia)
    en_soporte = dist_soporte < 0.5 * atr
    en_resistencia = dist_resistencia < 0.5 * atr
    if decision == 'Buy' and sent_label == 'Bearish':
        if en_soporte:
            return True, f"BUY en soporte a pesar de sentimiento bajista"
        else:
            return False, f"Sentimiento bajista bloquea BUY"
    if decision == 'Sell' and sent_label == 'Bullish':
        if en_resistencia:
            return True, f"SELL en resistencia a pesar de sentimiento alcista"
        else:
            return False, f"Sentimiento alcista bloquea SELL"
    return True, f"Sentimiento permitido ({sent_label})"

# ============================================================
# GRÁFICO (sin cambios)
# ============================================================
def generar_grafico_trade(df, decision, soporte, resistencia, razones, estado,
                          precio_entrada, precio_salida=None, tiempo_entrada=None,
                          trade_id=None, motivo_cierre="", noticia_titulo="", sent_label="", sent_score=0.0):
    if df.empty or estado is None:
        return None
    try:
        plt.style.use('dark_background')
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty:
            return None
        times = df_plot.index
        opens = df_plot['open'].values
        highs = df_plot['high'].values
        lows = df_plot['low'].values
        closes = df_plot['close'].values
        x = np.arange(len(df_plot))
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='black')
        ax.set_facecolor('black')

        for i in range(len(df_plot)):
            color = 'lime' if closes[i] >= opens[i] else 'red'
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = max(abs(closes[i] - opens[i]), 0.0001)
            rect = plt.Rectangle((x[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)

        atr = estado['atr']
        zona_soporte = estado.get('zona_soporte', (soporte, soporte + 0.3*atr))
        zona_resistencia = estado.get('zona_resistencia', (resistencia - 0.3*atr, resistencia))
        ax.axhspan(zona_soporte[0], zona_soporte[1], alpha=0.2, color='cyan', label='Zona Soporte')
        ax.axhspan(zona_resistencia[0], zona_resistencia[1], alpha=0.2, color='magenta', label='Zona Resistencia')
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=1, alpha=0.5)

        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')

        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot, r_plot, _, _ = linregress(x_plot, y_plot)
        tendencia_linea = intercept_plot + slope_plot * x_plot
        ax.plot(x_plot, tendencia_linea, color='white', linewidth=1.5, linestyle='-', label="Tendencia")

        entrada_x = 0
        if tiempo_entrada is not None and tiempo_entrada in times:
            entrada_x = np.where(times == tiempo_entrada)[0][0]
        else:
            entrada_x = len(df_plot) - 1 if precio_salida is None else 0

        if decision == 'Buy':
            ax.scatter(entrada_x, precio_entrada, s=250, marker='^', color='lime', edgecolors='black', linewidths=2, label='Entrada BUY', zorder=5)
        else:
            ax.scatter(entrada_x, precio_entrada, s=250, marker='v', color='red', edgecolors='black', linewidths=2, label='Entrada SELL', zorder=5)

        if precio_salida is not None:
            salida_x = len(df_plot) - 1
            pnl_indicador = (precio_salida - precio_entrada) if decision == 'Buy' else (precio_entrada - precio_salida)
            color_salida = 'lime' if pnl_indicador > 0 else ('red' if pnl_indicador < 0 else 'yellow')
            ax.scatter(salida_x, precio_salida, s=250, marker='s', color='yellow', edgecolors='white', linewidths=1.5, label='Salida', zorder=6)
            ax.plot([entrada_x, salida_x], [precio_entrada, precio_salida], color='white', linestyle=':', linewidth=2, alpha=0.7)

        id_text = f" ID: {trade_id}" if trade_id else ""
        razones_text = '\n• '.join(razones) if razones else "Sin razones"
        texto = (
            f"{decision.upper()}{id_text}\n"
            f"Precio entrada: {precio_entrada:.2f}\n"
            f"Hora: {times[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Zona Soporte: {zona_soporte[0]:.2f}-{zona_soporte[1]:.2f}\n"
            f"Zona Resistencia: {zona_resistencia[0]:.2f}-{zona_resistencia[1]:.2f}\n"
            f"EMA20: {estado['ema20']:.2f}  ATR: {estado['atr']:.2f}\n"
            f"Tendencia: {estado['tendencia']}\n"
            f"Patrón: {estado['patron']}\n"
            f"📰 Noticia: {noticia_titulo[:50]}...\n"
            f"📌 Sentimiento: {sent_label} ({sent_score:.3f})\n"
            f"🧠 Razones:\n• {razones_text}"
        )
        if precio_salida is not None:
            texto += f"\nSalida: {precio_salida:.2f} | Motivo: {motivo_cierre}"

        ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=8, verticalalignment='top', color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        ax.set_title(f"{SYMBOL} - Velas {INTERVAL}m - {'Entrada' if precio_salida is None else 'Cierre'}", color='white')
        ax.set_xlabel("Velas", color='white')
        ax.set_ylabel("Precio", color='white')
        ax.grid(True, alpha=0.2, color='gray')
        step = max(1, int(len(df_plot) / 10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in times[::step]], rotation=45, color='white')
        ax.tick_params(colors='white')
        ax.legend(loc='lower left', facecolor='black', edgecolor='white', labelcolor='white')
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error en gráfico: {e}")
        return None

# ============================================================
# GESTIÓN DE POSICIONES – PAPER TRADING Y REAL (unificada)
# ============================================================
def abrir_posicion_real(decision, precio, atr, soporte, resistencia, razones, tiempo, estado, df,
                        noticia_titulo, noticia_fuente, sent_label, sent_score):
    global TRADE_COUNTER, ACTIVE_TRADES, SALDO_SIMULADO, POSICIONES_SIMULADAS

    if len(ACTIVE_TRADES) >= MAX_OPEN_TRADES:
        return None

    qty = QTY_BTC
    side = "Buy" if decision == "Buy" else "Sell"

    # En modo real, ejecutar orden real
    if not PAPER_TRADING:
        logger.info(f"Enviando orden {side} Market por {qty} BTC...")
        try:
            crear_orden_market(SYMBOL, side, qty, reduce_only=False)
            time.sleep(2)
            posiciones = obtener_posiciones_abiertas()
            pos_actual = None
            for p in posiciones:
                if p.get('side') == side.capitalize() and float(p.get('size', 0)) > 0:
                    pos_actual = p
                    break
            if not pos_actual:
                raise Exception("No se encontró la posición recién abierta")
            entry_price = float(pos_actual.get('avgPrice', precio))
            logger.info(f"Entrada ejecutada a {entry_price:.2f}")
        except Exception as e:
            logger.error(f"Error abriendo posición: {e}")
            return None
    else:
        # Simulación: usar el precio actual (cierre de la vela)
        entry_price = precio
        # Comisión de entrada
        comision_entrada = qty * entry_price * COMISION_TAKER
        # En simulación, no descontamos aún del saldo, solo al cierre
        logger.info(f"📝 SIMULACIÓN: Entrada {side} a {entry_price:.2f} (comisión {comision_entrada:.2f} USD)")

    # Determinar si es ruptura o rebote (para SL/TP adaptativos)
    zona_soporte = estado.get('zona_soporte', (soporte, soporte + 0.3*atr))
    zona_resistencia = estado.get('zona_resistencia', (resistencia - 0.3*atr, resistencia))
    es_ruptura = (decision == 'Buy' and entry_price > zona_resistencia[1]) or \
                 (decision == 'Sell' and entry_price < zona_soporte[0])

    if es_ruptura:
        sl_mult = 1.2
        tp_mult = 3.0
    else:
        sl_mult = SL_MULTIPLIER
        tp_mult = 2.5

    if decision == "Buy":
        sl_price = round(entry_price - sl_mult * atr, 2)
        tp1_price = round(min(resistencia, entry_price + tp_mult * atr), 2)
        tp2_price = round(entry_price + 4.0 * atr, 2)
    else:
        sl_price = round(entry_price + sl_mult * atr, 2)
        tp1_price = round(max(soporte, entry_price - tp_mult * atr), 2)
        tp2_price = round(entry_price - 4.0 * atr, 2)

    # Validaciones básicas
    if decision == "Buy" and sl_price >= entry_price:
        if not PAPER_TRADING:
            crear_orden_market(SYMBOL, 'Sell', qty, reduce_only=True)
        return None
    if decision == "Sell" and sl_price <= entry_price:
        if not PAPER_TRADING:
            crear_orden_market(SYMBOL, 'Buy', qty, reduce_only=True)
        return None

    if decision == "Buy" and tp1_price <= entry_price:
        tp1_price = round(entry_price + tp_mult * atr, 2)
    if decision == "Sell" and tp1_price >= entry_price:
        tp1_price = round(entry_price - tp_mult * atr, 2)

    qty_half = qty / 2

    # Crear órdenes (reales o simuladas)
    tp1_order_id = None
    sl_order_id = None

    if not PAPER_TRADING:
        try:
            tp1_order = crear_orden_limit(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty_half, tp1_price, reduce_only=True)
            tp1_order_id = tp1_order.get('orderId')
        except Exception as e:
            logger.error(f"Error creando TP1: {e}")

        try:
            sl_order = crear_orden_stop_market(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty, sl_price, reduce_only=True)
            sl_order_id = sl_order.get('orderId')
        except Exception as e:
            logger.error(f"Error creando SL: {e}")
    else:
        # Simulación: solo generamos IDs ficticios
        tp1_order_id = f"SIM_TP1_{TRADE_COUNTER+1}"
        sl_order_id = f"SIM_SL_{TRADE_COUNTER+1}"

    TRADE_COUNTER += 1
    trade_id = TRADE_COUNTER
    trade_info = {
        'trade_id': trade_id,
        'side': side,
        'entry_price': entry_price,
        'qty_total': qty,
        'qty_remaining': qty,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'status': 'ACTIVE',
        'order_id_sl': sl_order_id,
        'order_id_tp1': tp1_order_id,
        'razones': razones,
        'estado_entrada': estado.copy(),
        'timestamp': tiempo,
        'trailing_active': False,
        'trailing_offset': TRAILING_OFFSET_ATR,
        'es_ruptura': es_ruptura,
        'comision_entrada': qty * entry_price * COMISION_TAKER if PAPER_TRADING else 0.0
    }
    ACTIVE_TRADES.append(trade_info)

    # Mensaje de entrada (con indicación de simulación)
    modo = "📝 SIMULACIÓN" if PAPER_TRADING else "✅ REAL"
    pnl_flotante = (entry_price - precio) * qty if decision == 'Buy' else (precio - entry_price) * qty
    mensaje_entrada = (
        f"{modo} ENTRADA #{trade_id} {decision}\n"
        f"💰 Precio: {entry_price:.2f}\n"
        f"📍 SL: {sl_price:.2f} | TP1 (50%): {tp1_price:.2f} | TP2: {tp2_price:.2f}\n"
        f"📦 Cantidad: {qty:.6f} BTC (nominal ≈ {qty*entry_price:.2f} USD)\n"
        f"💵 Margen aprox: {qty*entry_price/LEVERAGE:.2f} USD\n"
        f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
        f"📰 Noticia: {noticia_titulo} (Fuente: {noticia_fuente}) | Sentimiento: {sent_label} ({sent_score:.3f})\n"
        f"🧠 Razones técnicas:\n• " + "\n• ".join(razones)
    )
    telegram_mensaje(mensaje_entrada)

    fig = generar_grafico_trade(
        df=df, decision=decision, soporte=soporte, resistencia=resistencia,
        razones=razones, estado=estado, precio_entrada=entry_price,
        tiempo_entrada=tiempo, trade_id=trade_id,
        noticia_titulo=noticia_titulo, sent_label=sent_label, sent_score=sent_score
    )
    if fig:
        telegram_grafico(fig)
        plt.close(fig)

    return trade_info

def actualizar_estadisticas(pnl_neto):
    global TRADES_TOTALES, TRADES_WIN, TRADES_LOSS, PNL_GLOBAL, PNL_GLOBAL_NETO, TRADES_DESDE_RESUMEN, SALDO_SIMULADO, BALANCE_MAX
    TRADES_TOTALES += 1
    TRADES_DESDE_RESUMEN += 1
    PNL_GLOBAL += pnl_neto   # PnL bruto sin comisiones? mejor usar neto
    PNL_GLOBAL_NETO += pnl_neto
    SALDO_SIMULADO += pnl_neto
    if SALDO_SIMULADO > BALANCE_MAX:
        BALANCE_MAX = SALDO_SIMULADO
    if pnl_neto > 0:
        TRADES_WIN += 1
    else:
        TRADES_LOSS += 1

def enviar_resumen_balance():
    global PNL_GLOBAL, PNL_GLOBAL_NETO, TRADES_DESDE_RESUMEN, TRADES_TOTALES
    global TRADES_WIN, TRADES_LOSS, SALDO_SIMULADO

    if TRADES_DESDE_RESUMEN >= 10:
        total_trades = TRADES_WIN + TRADES_LOSS
        win_rate = (TRADES_WIN / total_trades * 100) if total_trades > 0 else 0.0
        drawdown = (BALANCE_MAX - SALDO_SIMULADO) if BALANCE_MAX > 0 else 0.0

        mensaje = (
            f"📊 RESUMEN DE BALANCE (últimos {TRADES_DESDE_RESUMEN} trades)\n"
            f"----------------------------------------\n"
            f"💰 PnL Neto: {PNL_GLOBAL_NETO:.4f} USD\n"
            f"🏆 Trades ganados: {TRADES_WIN}\n"
            f"❌ Trades perdidos: {TRADES_LOSS}\n"
            f"🎯 Win Rate: {win_rate:.1f}%\n"
            f"💵 Saldo actual: {SALDO_SIMULADO:.2f} USD\n"
            f"📉 Drawdown máximo: {drawdown:.2f} USD"
        )
        if PAPER_TRADING:
            mensaje += f"\n📌 Modo SIMULACIÓN (saldo inicial: {SALDO_INICIAL:.2f} USD)"
        telegram_mensaje(mensaje)
        logger.info(mensaje)

        # Resetear contadores para el próximo resumen
        TRADES_DESDE_RESUMEN = 0
        PNL_GLOBAL = 0.0
        PNL_GLOBAL_NETO = 0.0
        TRADES_WIN = 0
        TRADES_LOSS = 0

def revisar_posiciones_reales(precio_actual, df_actual, noticia_titulo, noticia_fuente, sent_label, sent_score):
    global ACTIVE_TRADES, TRADES_DESDE_RESUMEN, SALDO_SIMULADO

    # Si es modo real, obtener posiciones de API
    if not PAPER_TRADING:
        posiciones_api = obtener_posiciones_abiertas()
        pos_map = {}
        for p in posiciones_api:
            side = p.get('side', '').lower()
            if float(p.get('size', 0)) != 0:
                pos_map[side] = p
    else:
        pos_map = {}  # No usado en simulación

    trades_a_remover = []

    for idx, trade in enumerate(ACTIVE_TRADES):
        trade_id = trade['trade_id']
        side = trade['side']
        entry = trade['entry_price']
        qty_total = trade['qty_total']
        qty_remaining = trade['qty_remaining']
        sl_price = trade['sl_price']
        tp1_price = trade['tp1_price']
        tp2_price = trade['tp2_price']
        status = trade['status']
        order_id_sl = trade['order_id_sl']
        atr = trade['estado_entrada']['atr']
        trailing_active = trade.get('trailing_active', False)
        trailing_offset = trade.get('trailing_offset', TRAILING_OFFSET_ATR)
        es_ruptura = trade.get('es_ruptura', False)

        # En modo real, verificar tamaño real
        if not PAPER_TRADING:
            pos_api = pos_map.get(side.lower())
            size_actual = float(pos_api.get('size', 0)) if pos_api else 0.0
        else:
            # En simulación, usamos el qty_remaining y actualizamos según condiciones
            size_actual = qty_remaining  # inicialmente igual

        # Si la posición se cerró (real o simulada)
        if size_actual == 0:
            # Determinar motivo de cierre (para simulación, ya se manejó en la lógica de abajo)
            # En real, si se cerró por SL o TP1, ya lo manejamos con las órdenes.
            # Aquí solo limpiamos si ya no hay posición
            trades_a_remover.append(idx)
            continue

        # ========== TRAILING TEMPRANO ==========
        if status == 'ACTIVE' and not trailing_active:
            if side == 'Buy':
                profit_atr = (precio_actual - entry) / atr
            else:
                profit_atr = (entry - precio_actual) / atr

            umbral_activacion = 0.5 if es_ruptura else 1.0
            if profit_atr >= umbral_activacion:
                if side == 'Buy':
                    nuevo_sl = entry + 0.3 * atr if es_ruptura else entry + 0.5 * atr
                else:
                    nuevo_sl = entry - 0.3 * atr if es_ruptura else entry - 0.5 * atr

                # Modificar SL (real o simulado)
                if not PAPER_TRADING:
                    try:
                        modificar_orden_stop(order_id_sl, SYMBOL, nuevo_sl)
                    except Exception as e:
                        logger.error(f"Error modificando SL: {e}")
                else:
                    logger.info(f"📝 SIMULACIÓN: Trailing temprano activado #{trade_id}, SL -> {nuevo_sl:.2f}")
                trade['sl_price'] = nuevo_sl
                trade['trailing_active'] = True
                telegram_mensaje(f"🔄 Trailing temprano activado #{trade_id} - SL movido a {nuevo_sl:.2f}")

        # Trailing continuo
        if trailing_active:
            if side == 'Buy':
                nuevo_sl = precio_actual - trailing_offset * atr
                if nuevo_sl > trade['sl_price'] + MIN_TRAILING_STEP:
                    if not PAPER_TRADING:
                        try:
                            modificar_orden_stop(order_id_sl, SYMBOL, nuevo_sl)
                        except Exception as e:
                            logger.error(f"Error modificando SL: {e}")
                    else:
                        logger.info(f"📝 SIMULACIÓN: Trailing SL #{trade_id} -> {nuevo_sl:.2f}")
                    trade['sl_price'] = nuevo_sl
            else:
                nuevo_sl = precio_actual + trailing_offset * atr
                if nuevo_sl < trade['sl_price'] - MIN_TRAILING_STEP:
                    if not PAPER_TRADING:
                        try:
                            modificar_orden_stop(order_id_sl, SYMBOL, nuevo_sl)
                        except Exception as e:
                            logger.error(f"Error modificando SL: {e}")
                    else:
                        logger.info(f"📝 SIMULACIÓN: Trailing SL #{trade_id} -> {nuevo_sl:.2f}")
                    trade['sl_price'] = nuevo_sl

        # ========== TP1 (50%) ==========
        if status == 'ACTIVE':
            tp1_alcanzado = False
            # En simulación, comprobar si el precio tocó TP1
            if PAPER_TRADING:
                if (side == 'Buy' and precio_actual >= tp1_price) or (side == 'Sell' and precio_actual <= tp1_price):
                    tp1_alcanzado = True
                    # Cerrar la mitad
                    qty_cerrar = qty_total / 2
                    qty_remaining = qty_total - qty_cerrar
                    trade['qty_remaining'] = qty_remaining
                    # Comisión de salida
                    comision_salida = qty_cerrar * tp1_price * COMISION_TAKER
                    pnl_parcial = (tp1_price - entry) * qty_cerrar if side == 'Buy' else (entry - tp1_price) * qty_cerrar
                    pnl_neto_parcial = pnl_parcial - trade.get('comision_entrada', 0) - comision_salida
                    # Actualizar saldo (solo si es simulación, pero lo haremos al final)
                    # Para simplificar, acumulamos en estadísticas al cerrar completamente
                    # Guardamos el PnL parcial en el trade para sumarlo después
                    trade['pnl_parcial'] = pnl_neto_parcial
                    logger.info(f"📝 SIMULACIÓN: TP1 alcanzado #{trade_id}, PnL parcial {pnl_neto_parcial:.2f} USD")
                    # Cancelar SL antiguo y poner SL en BE
                    trade['sl_price'] = entry
                    trade['status'] = 'TP1_HIT'
                    if not trade.get('trailing_active', False):
                        trade['trailing_active'] = True
                    # Mensaje de TP1
                    mensaje = (
                        f"🔓 CIERRE PARCIAL SIMULADO #{trade_id} - TP1 alcanzado\n"
                        f"💰 Precio: {tp1_price:.2f}\n"
                        f"📊 PnL Parcial: {pnl_neto_parcial:.4f} USD\n"
                        f"🔄 SL movido a BE ({entry:.2f})\n"
                        f"🎯 Esperando TP2 ({tp2_price:.2f})..."
                    )
                    telegram_mensaje(mensaje)
                    # Gráfico de cierre parcial
                    fig = generar_grafico_trade(
                        df_actual, side,
                        trade['estado_entrada']['soporte'],
                        trade['estado_entrada']['resistencia'],
                        trade['razones'],
                        trade['estado_entrada'],
                        entry,
                        precio_salida=tp1_price,
                        tiempo_entrada=trade['timestamp'],
                        trade_id=trade_id,
                        motivo_cierre="TP1 Parcial Alcanzado",
                        noticia_titulo=noticia_titulo,
                        sent_label=sent_label,
                        sent_score=sent_score
                    )
                    if fig:
                        telegram_grafico(fig)
                        plt.close(fig)
            else:
                # Modo real: verificar tamaño real o si se ejecutó TP1
                if size_actual <= qty_total * 0.6:
                    tp1_alcanzado = True
                    # ya se ejecutó, solo actualizar estado
                    trade['qty_remaining'] = size_actual
                    trade['status'] = 'TP1_HIT'
                    # Mover SL a BE ya se hizo en la orden real, pero actualizamos
                    trade['sl_price'] = entry
                    if not trade.get('trailing_active', False):
                        trade['trailing_active'] = True

        # ========== TP2 y trailing infinito ==========
        if status == 'TP1_HIT':
            if PAPER_TRADING:
                if (side == 'Buy' and precio_actual >= tp2_price) or (side == 'Sell' and precio_actual <= tp2_price):
                    trade['status'] = 'TRAILING'
                    telegram_mensaje(
                        f"🚀 #TRAILING INFINITO ACTIVADO PARA #{trade_id}\n"
                        f"📍 Precio cruzó TP2 ({tp2_price:.2f})\n"
                        f"📈 Persiguiendo con {trailing_offset} ATR de distancia."
                    )
            else:
                # En real, si el precio cruza TP2, ya se maneja con la orden de TP2? 
                # No tenemos orden de TP2, solo trailing.
                # Podríamos cambiar el status a TRAILING si el precio supera TP2.
                if (side == 'Buy' and precio_actual >= tp2_price) or (side == 'Sell' and precio_actual <= tp2_price):
                    trade['status'] = 'TRAILING'
                    telegram_mensaje(
                        f"🚀 #TRAILING INFINITO ACTIVADO PARA #{trade_id}\n"
                        f"📍 Precio cruzó TP2 ({tp2_price:.2f})\n"
                        f"📈 Persiguiendo con {trailing_offset} ATR de distancia."
                    )

        # ========== COMPROBAR SL (simulación) ==========
        if PAPER_TRADING:
            sl_activado = False
            if (side == 'Buy' and precio_actual <= trade['sl_price']) or (side == 'Sell' and precio_actual >= trade['sl_price']):
                sl_activado = True
                # Cerrar toda la posición restante
                qty_cerrar = trade['qty_remaining']
                exit_price = trade['sl_price']
                pnl = (exit_price - entry) * qty_cerrar if side == 'Buy' else (entry - exit_price) * qty_cerrar
                comision_salida = qty_cerrar * exit_price * COMISION_TAKER
                comision_entrada = trade.get('comision_entrada', 0)
                pnl_neto = pnl - comision_entrada - comision_salida
                # Sumar PnL parcial si existe
                if 'pnl_parcial' in trade:
                    pnl_neto += trade['pnl_parcial']
                # Actualizar estadísticas
                actualizar_estadisticas(pnl_neto)

                motivo = "SL" if status == 'ACTIVE' else "SL después de TP1" if status == 'TP1_HIT' else "SL"
                mensaje = (
                    f"🔴 CIERRE FINAL SIMULADO #{trade_id}\n"
                    f"📝 Motivo: {motivo}\n"
                    f"🎯 Entrada: {entry:.2f}\n"
                    f"🛑 Salida: {exit_price:.2f}\n"
                    f"📊 PnL Neto: {pnl_neto:.4f} USD\n"
                    f"📰 Noticia: {noticia_titulo} | Sentimiento: {sent_label} ({sent_score:.3f})"
                )
                telegram_mensaje(mensaje)
                fig = generar_grafico_trade(
                    df_actual, side,
                    trade['estado_entrada']['soporte'],
                    trade['estado_entrada']['resistencia'],
                    trade['razones'],
                    trade['estado_entrada'],
                    entry,
                    precio_salida=exit_price,
                    tiempo_entrada=trade['timestamp'],
                    trade_id=trade_id,
                    motivo_cierre=motivo,
                    noticia_titulo=noticia_titulo,
                    sent_label=sent_label,
                    sent_score=sent_score
                )
                if fig:
                    telegram_grafico(fig)
                    plt.close(fig)
                trades_a_remover.append(idx)
                continue

        # ========== CIERRE POR TP2 (simulación) o trailing infinito ==========
        if PAPER_TRADING and status == 'TRAILING':
            # Si el precio cruza el SL (ya manejado arriba), se cierra.
            # Si el precio sigue subiendo/bajando, el trailing lo sigue.
            # No hay cierre automático por TP2, solo trailing.
            pass

    # Remover trades cerrados (tanto reales como simulados)
    for idx in sorted(trades_a_remover, reverse=True):
        del ACTIVE_TRADES[idx]

    # En modo real, también sincronizar con API (ya lo hacemos al inicio del loop)
    if not PAPER_TRADING:
        # Opcional: limpiar trades que ya no existen en API
        pass

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_bot():
    global TRADE_COUNTER, ACTIVE_TRADES, SALDO_SIMULADO

    try:
        set_leverage(SYMBOL, LEVERAGE)
    except Exception as e:
        logger.error(f"Error al establecer apalancamiento: {e}")
        telegram_mensaje(f"⚠️ Error apalancamiento: {e}")

    if PAPER_TRADING:
        telegram_mensaje(f"🧪 MODO PAPER TRADING ACTIVADO - Saldo inicial: {SALDO_INICIAL:.2f} USD")
    else:
        telegram_mensaje("⚡ MODO REAL ACTIVADO - ¡Cuidado!")

    telegram_mensaje("🤖 BOT V92.0-PAPER INICIADO (Zonas S/R, Rupturas prioritarias)\n"
                     f"📊 Velas: {INTERVAL}m | Máx. posiciones: {MAX_OPEN_TRADES}\n"
                     f"⚡ Leverage: {LEVERAGE}x | Tamaño: {QTY_BTC} BTC\n"
                     f"🔒 SL adaptativo (ruptura 1.2 ATR, rebote 1.8 ATR)\n"
                     f"📈 Trailing temprano y gestión moderna\n"
                     f"📰 Filtro fundamental activo\n"
                     f"🛑 Filtros: confluencia 1/3, distancia <=3.0 ATR, R/R >= {MIN_RR_RATIO}")

    ultima_fecha = None

    while True:
        try:
            df = obtener_velas()
            df = calcular_indicadores(df)

            if df.empty:
                logger.warning("⚠️ DataFrame vacío. Saltando ciclo...")
                time.sleep(SLEEP_SECONDS)
                continue

            estado = extraer_estado_mercado(df, usar_cerrada=True)
            if estado is None:
                logger.warning("⚠️ Estado nulo. Saltando ciclo...")
                time.sleep(SLEEP_SECONDS)
                continue

            titulo, fuente, sent_label, sent_score = obtener_noticias_y_sentimiento()

            decision, soporte, resistencia, razones = motor_v92(estado, df)

            filtro_ok = True
            motivo_filtro = "Sin filtro"
            if decision:
                filtro_ok, motivo_filtro = filtrar_por_fundamental(decision, sent_label, estado)
                if not filtro_ok:
                    decision = None

            num_abiertas = len(ACTIVE_TRADES)
            logger.info("="*100)
            logger.info(f"🕒 {estado['fecha']} | 💰 BTC: {estado['precio']:.2f}")
            logger.info(f"📐 Tendencia: {estado['tendencia']} | Slope: {estado['slope']:.5f}")
            logger.info(f"🧱 Zona Soporte: {estado.get('zona_soporte', (soporte, soporte))[0]:.2f}-{estado.get('zona_soporte', (soporte, soporte))[1]:.2f}")
            logger.info(f"🧱 Zona Resistencia: {estado.get('zona_resistencia', (resistencia, resistencia))[0]:.2f}-{estado.get('zona_resistencia', (resistencia, resistencia))[1]:.2f}")
            logger.info(f"📊 ATR: {estado['atr']:.2f} | EMA20: {estado['ema20']:.2f}")
            logger.info(f"📏 Patrón: {estado['patron']}")
            logger.info(f"📏 Distancia a EMA20 (ATR): {estado['dist_ema_atr']:.2f}")
            logger.info(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
            logger.info(f"🧠 Razones: {', '.join(razones)}")
            logger.info(f"🔒 Filtro fundamental: {'PERMITIDO' if filtro_ok else 'BLOQUEADO'} - {motivo_filtro}")
            logger.info(f"📊 Posiciones abiertas (memoria): {num_abiertas}/{MAX_OPEN_TRADES}")
            if not PAPER_TRADING:
                try:
                    api_pos = obtener_posiciones_abiertas()
                    logger.info(f"📊 Posiciones abiertas (API): {len(api_pos)}")
                except:
                    pass
            logger.info(f"📰 Noticia: {titulo} (Fuente: {fuente}) | Sentimiento: {sent_label} ({sent_score:.3f})")
            logger.info(f"💵 Saldo simulado: {SALDO_SIMULADO:.2f} USD" if PAPER_TRADING else "")
            logger.info("="*100)

            if decision and num_abiertas < MAX_OPEN_TRADES:
                abrir_posicion_real(
                    decision=decision,
                    precio=estado['precio'],
                    atr=estado['atr'],
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=estado['fecha'],
                    estado=estado,
                    df=df,
                    noticia_titulo=titulo,
                    noticia_fuente=fuente,
                    sent_label=sent_label,
                    sent_score=sent_score
                )

            revisar_posiciones_reales(estado['precio'], df, titulo, fuente, sent_label, sent_score)

            fecha_hoy = datetime.now(timezone.utc).date()
            if ultima_fecha is None:
                ultima_fecha = fecha_hoy
            elif fecha_hoy != ultima_fecha:
                ultima_fecha = fecha_hoy
                logger.info("Nuevo día.")

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            logger.error(f"🚨 ERROR en loop principal: {e}", exc_info=True)
            telegram_mensaje(f"🚨 ERROR BOT: {e}")
            time.sleep(60)

# ============================================================
# START
# ============================================================
if __name__ == '__main__':
    run_bot()
