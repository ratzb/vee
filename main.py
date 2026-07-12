# ============================================================
# BOT PAPER TRADING V92.3 – CORREGIDO (in en lugar de en)
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
# CONFIGURACIÓN GENERAL
# ============================================================
SYMBOL = "BTCUSDT"
INTERVAL = "5"
LEVERAGE = 10
MAX_OPEN_TRADES = 3
SLEEP_SECONDS = 300

QTY_BTC = 0.002
TRAILING_OFFSET_ATR = 0.7
SL_MULTIPLIER = 1.5
MIN_RR_RATIO = 1.2
MIN_TRAILING_STEP = 3.0
COMISION_TAKER = 0.0006

GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

NEWS_CACHE = {
    "titulo": "No disponible",
    "fuente": "Ninguna",
    "sent_label": "Neutral",
    "sent_score": 0.0,
    "timestamp": None
}
NEWS_CACHE_TTL = 3600

# ============================================================
# CREDENCIALES
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

# ============================================================
# ESTADÍSTICAS PAPER
# ============================================================
PAPER_BALANCE = 1000.0
PAPER_POSITIONS = []
PAPER_TOTAL_PNL = 0.0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TOTAL_TRADES = 0
PAPER_COUNTER_RESUMEN = 0

BLOQUEOS = {
    "R/R insuficiente": 0,
    "Distancia a EMA": 0,
    "Contra-tendencia sin confluencia": 0,
    "Rango lateral sin ruptura": 0,
    "Fundamental": 0,
    "Sin confluencia": 0,
    "Otros": 0
}

# ============================================================
# TELEGRAM
# ============================================================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado")
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
# FUNCIONES API (solo lectura)
# ============================================================
def bybit_request(endpoint, method='GET', params=None, payload=None):
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
    logger.info(f"(PAPER) Apalancamiento simulado a {leverage}x")
    pass

def obtener_posiciones_abiertas():
    return PAPER_POSITIONS

def crear_orden_market(symbol, side, qty, reduce_only=False):
    logger.info(f"(PAPER) MARKET {side} {qty}")
    return {"orderId": f"paper_{int(time.time())}"}

def crear_orden_limit(symbol, side, qty, price, reduce_only=False, post_only=False):
    logger.info(f"(PAPER) LIMIT {side} {qty}@{price}")
    return {"orderId": f"paper_limit_{int(time.time())}"}

def crear_orden_stop_market(symbol, side, qty, stop_price, reduce_only=True):
    logger.info(f"(PAPER) STOP MARKET {side} SL={stop_price}")
    return {"orderId": f"paper_stop_{int(time.time())}"}

def cancelar_orden(order_id, symbol):
    logger.info(f"(PAPER) Cancelar {order_id}")
    pass

def modificar_orden_stop(order_id, symbol, stop_price):
    logger.info(f"(PAPER) Modificar SL a {stop_price}")
    pass

# ============================================================
# OBTENER DATOS E INDICADORES
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

# ============================================================
# EVALUAR CONFLUENCIA (1/3)
# ============================================================
def evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info):
    condiciones = 0
    razones = []
    
    if velas_info['ruptura_tendencia'] is not None:
        condiciones += 1
        razones.append(f"Ruptura de tendencia ({velas_info['ruptura_tendencia']})")
    
    if (velas_info['cerca_soporte'] and velas_info['rechazo_inferior'] and velas_info['toques_soporte'] >= 2) or \
       (velas_info['cerca_resistencia'] and velas_info['rechazo_superior'] and velas_info['toques_resistencia'] >= 2):
        condiciones += 1
        razones.append("Rechazo con mecha en zona de S/R (múltiples toques)")
    
    patron = estado_actual['patron']
    if "Martillo" in patron or "Estrella fugaz" in patron or "cuerpo grande" in patron:
        condiciones += 1
        razones.append(f"Patrón de reversión: {patron}")
    
    permitido = condiciones >= 1
    if permitido:
        razones.append(f"✅ Confluencia suficiente ({condiciones}/3)")
    else:
        razones.append(f"❌ Confluencia insuficiente ({condiciones}/3)")
    
    return permitido, razones

# ============================================================
# MOTOR DE DECISIÓN V92.3 (con in corregido)
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

def motor_v92_paper(estado_actual, df):
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

    # ========== PRIORIDAD 1: RUPTURAS ==========
    if precio > resistencia + 0.3 * atr and patron in ["Vela alcista de cuerpo grande", "Vela normal"]:
        if estado_actual['close'] > resistencia and estado_actual['close'] > estado_actual['open']:
            razones.append("✅ RUPTURA ALCISTA")
            tp1_candidate = precio + 2.0 * atr
            sl_candidate = precio - 1.5 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para ruptura")
                return None, soporte, resistencia, razones

    if precio < soporte - 0.3 * atr and patron in ["Vela bajista de cuerpo grande", "Vela normal"]:
        if estado_actual['close'] < soporte and estado_actual['close'] < estado_actual['open']:
            razones.append("✅ RUPTURA BAJISTA")
            tp1_candidate = precio - 2.0 * atr
            sl_candidate = precio + 1.5 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente para ruptura")
                return None, soporte, resistencia, razones

    # ========== PRIORIDAD 2: REBOTES EN ZONAS ==========
    if cerca_soporte and (rechazo_inf or patron in ["Martillo (posible reversión alcista)", "Vela alcista de cuerpo grande"]):
        if bajistas_cons < 6:
            razones.append(f"✅ COMPRA en zona soporte (toques: {toques_soporte})")
            tp1_candidate = min(resistencia, precio + 2.5 * atr)
            sl_candidate = precio - 1.5 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"⚠️ Demasiadas velas bajistas ({bajistas_cons})")
            return None, soporte, resistencia, razones

    if cerca_resistencia and (rechazo_sup or patron in ["Estrella fugaz (posible reversión bajista)", "Vela bajista de cuerpo grande"]):
        if alcistas_cons < 6:
            razones.append(f"✅ VENTA en zona resistencia (toques: {toques_resistencia})")
            tp1_candidate = max(soporte, precio - 2.5 * atr)
            sl_candidate = precio + 1.5 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"⚠️ Demasiadas velas alcistas ({alcistas_cons})")
            return None, soporte, resistencia, razones

    # ========== PRIORIDAD 3: EMA ==========
    rechazo_alcista, rechazo_bajista, desc_rechazo = analizar_rechazo_ema(df, estado_actual)

    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and rechazo_alcista:
        razones.append(f"✅ COMPRA en EMA20")
        tp1_candidate = min(resistencia, precio + 2.5 * atr)
        sl_candidate = precio - 1.5 * atr
        if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
            return 'Buy', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente")
            return None, soporte, resistencia, razones

    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and rechazo_bajista:
        razones.append(f"✅ VENTA en EMA20")
        tp1_candidate = max(soporte, precio - 2.5 * atr)
        sl_candidate = precio + 1.5 * atr
        if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
            return 'Sell', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente")
            return None, soporte, resistencia, razones

    # ========== CONTRA-TENDENCIA (1/3) ==========
    en_contra_buy = (precio < ema20 and (tendencia == '📉 BAJISTA' or pendiente_ema < -0.01))
    en_contra_sell = (precio > ema20 and (tendencia == '📈 ALCISTA' or pendiente_ema > 0.01))

    if en_contra_buy:
        permitido, razones_contra = evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info)
        razones.extend(razones_contra)
        if not permitido:
            razones.append("❌ Contra-tendencia sin confluencia")
            return None, soporte, resistencia, razones
        else:
            razones.append("⚠️ Compra en contra-tendencia permitida")
            tp1_candidate = min(resistencia, precio + 2.5 * atr)
            sl_candidate = precio - 1.5 * atr
            if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente")
                return None, soporte, resistencia, razones

    elif en_contra_sell:
        permitido, razones_contra = evaluar_confluencia_contra_tendencia(estado_actual, df, velas_info)
        razones.extend(razones_contra)
        if not permitido:
            razones.append("❌ Contra-tendencia sin confluencia")
            return None, soporte, resistencia, razones
        else:
            razones.append("⚠️ Venta en contra-tendencia permitida")
            tp1_candidate = max(soporte, precio - 2.5 * atr)
            sl_candidate = precio + 1.5 * atr
            if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("❌ R/R insuficiente")
                return None, soporte, resistencia, razones

    # ========== FILTROS ADICIONALES ==========
    if abs(dist_ema_atr) > 3.5:
        razones.append(f"❌ Distancia a EMA >3.5 ATR")
        return None, soporte, resistencia, razones

    es_rango = abs(pendiente_ema) < 0.0005
    if es_rango and not (precio > resistencia or precio < soporte) and patron not in ["Martillo (posible reversión alcista)", "Estrella fugaz (posible reversión bajista)", "Doji (indecisión)"]:
        razones.append("⚠️ Rango sin ruptura/patrón")
        return None, soporte, resistencia, razones

    patron_mult, desc_mult = detectar_patron_multivela(df)
    if patron_mult == "tres_soldados_blancos" and tendencia in ['📈 ALCISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        tp1_candidate = min(resistencia, precio + 2.5 * atr)
        sl_candidate = precio - 1.5 * atr
        if sl_candidate < precio and (tp1_candidate - precio) / (precio - sl_candidate) >= MIN_RR_RATIO:
            return 'Buy', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente")
            return None, soporte, resistencia, razones

    if patron_mult == "tres_cuervos_negros" and tendencia in ['📉 BAJISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        tp1_candidate = max(soporte, precio - 2.5 * atr)
        sl_candidate = precio + 1.5 * atr
        if sl_candidate > precio and (precio - tp1_candidate) / (sl_candidate - precio) >= MIN_RR_RATIO:
            return 'Sell', soporte, resistencia, razones
        else:
            razones.append("❌ R/R insuficiente")
            return None, soporte, resistencia, razones

    # Confirmación de patrón anterior (corregido: in en lugar de en)
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
# NOTICIAS Y SENTIMIENTO
# ============================================================
def actualizar_cache_noticias():
    global NEWS_CACHE
    ahora = datetime.now(timezone.utc)
    if NEWS_CACHE["timestamp"] is not None:
        edad = (ahora - NEWS_CACHE["timestamp"]).total_seconds()
        if edad < NEWS_CACHE_TTL:
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
# GRÁFICO
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
# SIMULACIÓN PAPER
# ============================================================
def abrir_posicion_paper(decision, precio, atr, soporte, resistencia, razones, tiempo, estado, df,
                         noticia_titulo, noticia_fuente, sent_label, sent_score):
    global PAPER_BALANCE, PAPER_POSITIONS, PAPER_TOTAL_TRADES

    if len(PAPER_POSITIONS) >= MAX_OPEN_TRADES:
        return None

    qty = QTY_BTC
    side = "Buy" if decision == "Buy" else "Sell"

    comision = qty * precio * COMISION_TAKER
    costo_total = qty * precio + comision
    if costo_total > PAPER_BALANCE:
        logger.warning(f"Saldo insuficiente. Balance: {PAPER_BALANCE:.2f}, Costo: {costo_total:.2f}")
        return None

    PAPER_BALANCE -= costo_total

    sl_price = round(precio - 1.5 * atr, 2) if decision == "Buy" else round(precio + 1.5 * atr, 2)
    tp1_price = round(min(resistencia, precio + 2.5 * atr), 2) if decision == "Buy" else round(max(soporte, precio - 2.5 * atr), 2)
    tp2_price = round(precio + 4.0 * atr, 2) if decision == "Buy" else round(precio - 4.0 * atr, 2)

    PAPER_TOTAL_TRADES += 1
    trade_id = PAPER_TOTAL_TRADES

    trade = {
        'trade_id': trade_id,
        'side': side,
        'entry_price': precio,
        'qty': qty,
        'sl_price': sl_price,
        'tp1_price': tp1_price,
        'tp2_price': tp2_price,
        'status': 'ACTIVE',
        'razones': razones,
        'timestamp': tiempo,
        'trailing_active': False,
        'trailing_offset': TRAILING_OFFSET_ATR,
        'pnl_realizado': 0.0,
        'comision_pagada': comision,
        'atr': atr,
        'estado_entrada': estado,
        'decision': decision,
        'soporte': soporte,
        'resistencia': resistencia
    }
    PAPER_POSITIONS.append(trade)

    # --- MENSAJE Y GRÁFICO DE ENTRADA ---
    mensaje = (f"📌 PAPER ENTRADA #{trade_id} {decision}\n"
               f"💰 Precio: {precio:.2f}\n"
               f"📍 SL: {sl_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}\n"
               f"📦 Cantidad: {qty:.6f} BTC\n"
               f"💵 Costo (incl. comisión): {costo_total:.2f} USD\n"
               f"📊 Balance restante: {PAPER_BALANCE:.2f} USD\n"
               f"📰 Noticia: {noticia_titulo} (Fuente: {noticia_fuente}) | Sentimiento: {sent_label} ({sent_score:.3f})\n"
               f"🧠 Razones técnicas:\n• " + "\n• ".join(razones))
    logger.info(mensaje)
    telegram_mensaje(mensaje)

    fig = generar_grafico_trade(
        df=df, decision=decision, soporte=soporte, resistencia=resistencia,
        razones=razones, estado=estado, precio_entrada=precio,
        tiempo_entrada=tiempo, trade_id=trade_id,
        noticia_titulo=noticia_titulo, sent_label=sent_label, sent_score=sent_score
    )
    if fig:
        telegram_grafico(fig)
        plt.close(fig)

    return trade

def cerrar_posicion_paper(trade, precio_salida, motivo, df_actual, noticia_titulo, sent_label, sent_score):
    global PAPER_BALANCE, PAPER_POSITIONS, PAPER_TOTAL_PNL, PAPER_WIN, PAPER_LOSS, PAPER_COUNTER_RESUMEN

    side = trade['side']
    qty = trade['qty']
    entry = trade['entry_price']
    comision_salida = qty * precio_salida * COMISION_TAKER

    if side == 'Buy':
        pnl_bruto = (precio_salida - entry) * qty
    else:
        pnl_bruto = (entry - precio_salida) * qty

    pnl_neto = pnl_bruto - comision_salida - trade['comision_pagada']
    PAPER_BALANCE += qty * precio_salida
    PAPER_BALANCE += pnl_neto

    trade['pnl_realizado'] = pnl_neto
    PAPER_TOTAL_PNL += pnl_neto

    if pnl_neto > 0:
        PAPER_WIN += 1
    else:
        PAPER_LOSS += 1
    PAPER_COUNTER_RESUMEN += 1

    # --- MENSAJE DE CIERRE ---
    mensaje = (f"🔒 PAPER CIERRE #{trade['trade_id']} - {motivo}\n"
               f"💰 Entrada: {entry:.2f} | Salida: {precio_salida:.2f}\n"
               f"📊 PnL Neto: {pnl_neto:.4f} USD\n"
               f"💵 Balance actual: {PAPER_BALANCE:.2f} USD")
    logger.info(mensaje)
    telegram_mensaje(mensaje)

    # --- GRÁFICO DE CIERRE ---
    if df_actual is not None and not df_actual.empty:
        fig = generar_grafico_trade(
            df=df_actual,
            decision=trade['decision'],
            soporte=trade['soporte'],
            resistencia=trade['resistencia'],
            razones=trade['razones'],
            estado=trade['estado_entrada'],
            precio_entrada=entry,
            precio_salida=precio_salida,
            tiempo_entrada=trade['timestamp'],
            trade_id=trade['trade_id'],
            motivo_cierre=motivo,
            noticia_titulo=noticia_titulo,
            sent_label=sent_label,
            sent_score=sent_score
        )
        if fig:
            telegram_grafico(fig)
            plt.close(fig)

    # Eliminar de posiciones
    PAPER_POSITIONS = [p for p in PAPER_POSITIONS if p['trade_id'] != trade['trade_id']]

    # Resumen cada 5 trades
    if PAPER_COUNTER_RESUMEN >= 5:
        total_trades = PAPER_WIN + PAPER_LOSS
        win_rate = (PAPER_WIN / total_trades * 100) if total_trades > 0 else 0
        msg_resumen = (f"📊 RESUMEN PAPER (últimos {PAPER_COUNTER_RESUMEN} trades)\n"
                       f"----------------------------------------\n"
                       f"💰 PnL Total Neto: {PAPER_TOTAL_PNL:.4f} USD\n"
                       f"🏆 Trades ganados: {PAPER_WIN}\n"
                       f"❌ Trades perdidos: {PAPER_LOSS}\n"
                       f"🎯 Win Rate: {win_rate:.1f}%\n"
                       f"💵 Balance actual: {PAPER_BALANCE:.2f} USD")
        telegram_mensaje(msg_resumen)
        logger.info(msg_resumen)
        PAPER_COUNTER_RESUMEN = 0

def revisar_posiciones_paper(precio_actual, df_actual, noticia_titulo, noticia_fuente, sent_label, sent_score):
    global PAPER_POSITIONS

    trades_a_cerrar = []
    for trade in PAPER_POSITIONS:
        side = trade['side']
        entry = trade['entry_price']
        sl = trade['sl_price']
        tp1 = trade['tp1_price']
        tp2 = trade['tp2_price']
        status = trade['status']
        trailing_active = trade.get('trailing_active', False)
        trailing_offset = trade.get('trailing_offset', TRAILING_OFFSET_ATR)
        atr = trade.get('atr', 100)

        # Trailing temprano
        if status == 'ACTIVE' and not trailing_active:
            profit_atr = (precio_actual - entry) / atr if side == 'Buy' else (entry - precio_actual) / atr
            if profit_atr >= 0.8:
                nuevo_sl = entry + 0.4 * atr if side == 'Buy' else entry - 0.4 * atr
                trade['sl_price'] = nuevo_sl
                trade['trailing_active'] = True
                logger.info(f"PAPER Trailing activado #{trade['trade_id']} SL->{nuevo_sl:.2f}")

        # Trailing continuo
        if trailing_active:
            if side == 'Buy':
                nuevo_sl = precio_actual - trailing_offset * atr
                if nuevo_sl > trade['sl_price'] + MIN_TRAILING_STEP:
                    trade['sl_price'] = nuevo_sl
            else:
                nuevo_sl = precio_actual + trailing_offset * atr
                if nuevo_sl < trade['sl_price'] - MIN_TRAILING_STEP:
                    trade['sl_price'] = nuevo_sl

        # Verificar SL
        if side == 'Buy' and precio_actual <= trade['sl_price']:
            trades_a_cerrar.append((trade, trade['sl_price'], "Stop Loss"))
            continue
        if side == 'Sell' and precio_actual >= trade['sl_price']:
            trades_a_cerrar.append((trade, trade['sl_price'], "Stop Loss"))
            continue

        # TP1
        if status == 'ACTIVE':
            if side == 'Buy' and precio_actual >= tp1:
                trades_a_cerrar.append((trade, tp1, "TP1 Alcanzado"))
                continue
            if side == 'Sell' and precio_actual <= tp1:
                trades_a_cerrar.append((trade, tp1, "TP1 Alcanzado"))
                continue

        # TP2
        if status == 'ACTIVE' or status == 'TP1_HIT':
            if side == 'Buy' and precio_actual >= tp2:
                trade['status'] = 'TRAILING'
                logger.info(f"PAPER TP2 alcanzado #{trade['trade_id']}, modo trailing")
            if side == 'Sell' and precio_actual <= tp2:
                trade['status'] = 'TRAILING'
                logger.info(f"PAPER TP2 alcanzado #{trade['trade_id']}, modo trailing")

    # Cerrar trades
    for trade, precio_salida, motivo in trades_a_cerrar:
        cerrar_posicion_paper(trade, precio_salida, motivo, df_actual, noticia_titulo, sent_label, sent_score)

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_paper_bot():
    global PAPER_BALANCE, BLOQUEOS

    msg_inicio = (f"🤖 PAPER BOT V92.3 INICIADO (simulación con Telegram)\n"
                  f"💰 Saldo inicial: {PAPER_BALANCE:.2f} USD\n"
                  f"⚡ Leverage simulado: {LEVERAGE}x\n"
                  f"📊 Intervalo: {INTERVAL}m | Máx posiciones: {MAX_OPEN_TRADES}\n"
                  f"🔒 SL=1.5 ATR, R/R mínimo={MIN_RR_RATIO}, trailing=0.7 ATR\n"
                  f"📈 Estrategia: Zonas S/R, Rupturas prioritarias")
    logger.info(msg_inicio)
    telegram_mensaje(msg_inicio)

    while True:
        try:
            df = obtener_velas()
            df = calcular_indicadores(df)

            if df.empty:
                time.sleep(SLEEP_SECONDS)
                continue

            estado = extraer_estado_mercado(df, usar_cerrada=True)
            if estado is None:
                time.sleep(SLEEP_SECONDS)
                continue

            titulo, fuente, sent_label, sent_score = obtener_noticias_y_sentimiento()

            decision, soporte, resistencia, razones = motor_v92_paper(estado, df)

            filtro_ok = True
            motivo_filtro = "Sin filtro"
            if decision:
                filtro_ok, motivo_filtro = filtrar_por_fundamental(decision, sent_label, estado)
                if not filtro_ok:
                    BLOQUEOS["Fundamental"] += 1
                    decision = None

            if decision is None and razones:
                for r in razones:
                    if "R/R" in r:
                        BLOQUEOS["R/R insuficiente"] += 1
                    elif "Distancia a EMA" in r:
                        BLOQUEOS["Distancia a EMA"] += 1
                    elif "Contra-tendencia sin confluencia" in r:
                        BLOQUEOS["Contra-tendencia sin confluencia"] += 1
                    elif "Rango sin ruptura" in r:
                        BLOQUEOS["Rango lateral sin ruptura"] += 1
                    elif "Sin confluencia" in r:
                        BLOQUEOS["Sin confluencia"] += 1
                    else:
                        BLOQUEOS["Otros"] += 1

            num_abiertas = len(PAPER_POSITIONS)
            logger.info("="*80)
            logger.info(f"🕒 {estado['fecha']} | BTC: {estado['precio']:.2f}")
            logger.info(f"📐 Tendencia: {estado['tendencia']}")
            logger.info(f"🧱 Zona Soporte: {estado.get('zona_soporte', (0,0))[0]:.2f}-{estado.get('zona_soporte', (0,0))[1]:.2f}")
            logger.info(f"🧱 Zona Resistencia: {estado.get('zona_resistencia', (0,0))[0]:.2f}-{estado.get('zona_resistencia', (0,0))[1]:.2f}")
            logger.info(f"📊 ATR: {estado['atr']:.2f} | EMA20: {estado['ema20']:.2f}")
            logger.info(f"📏 Patrón: {estado['patron']} | Dist EMA: {estado['dist_ema_atr']:.2f} ATR")
            logger.info(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
            logger.info(f"🧠 Razones: {', '.join(razones)}")
            logger.info(f"🔒 Filtro fundamental: {'PERMITIDO' if filtro_ok else 'BLOQUEADO'} - {motivo_filtro}")
            logger.info(f"📊 Posiciones abiertas: {num_abiertas}/{MAX_OPEN_TRADES}")
            logger.info(f"💰 Balance actual: {PAPER_BALANCE:.2f} USD")
            logger.info("="*80)

            if decision and num_abiertas < MAX_OPEN_TRADES:
                abrir_posicion_paper(
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

            revisar_posiciones_paper(estado['precio'], df, titulo, fuente, sent_label, sent_score)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            logger.error(f"🚨 ERROR: {e}", exc_info=True)
            telegram_mensaje(f"🚨 ERROR PAPER BOT: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_paper_bot()
