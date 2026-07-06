# ============================================================
# BOT TRADING V90.6 – VERSIÓN REAL (CON TP1, TP2 Y TRAILING)
# ============================================================
# - Velas de 5 minutos, máximo 3 operaciones abiertas
# - Apalancamiento x10, cierre parcial 50% en TP1, SL a BE
# - TP2 dinámico + TRAILING STOP (0.75 ATR) post-TP2
# - Tamaño fijo: 0.002 BTC (para cumplir mínimo en órdenes limit)
# - SL = 2.0 ATR (más amplio para evitar cierres prematuros)
# - Se recalcula SL/TP después de la entrada para evitar ejecución inmediata
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
INTERVAL = "5"                 # 5 minutos
LEVERAGE = 10                  # Apalancamiento 10x
MAX_OPEN_TRADES = 3
SLEEP_SECONDS = 300

# Tamaño de posición (0.002 BTC para que la mitad (0.001) cumpla mínimo)
QTY_BTC = 0.002

# Trailing Stop después de TP2
TRAILING_OFFSET_ATR = 0.75     # múltiplos de ATR

# SL más amplio (2.0 ATR en lugar de 1.5)
SL_MULTIPLIER = 2.0

# Gráficos
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

# Caché de noticias
NEWS_CACHE = {
    "titulo": "No disponible",
    "fuente": "Ninguna",
    "sent_label": "Neutral",
    "sent_score": 0.0,
    "timestamp": None
}
NEWS_CACHE_TTL = 3600

# ============================================================
# CREDENCIALES (VARIABLES DE ENTORNO)
# ============================================================
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

# ============================================================
# INICIALIZAR VADER Y BASE URL
# ============================================================
sentiment_analyzer = SentimentIntensityAnalyzer()
BASE_URL = "https://api.bybit.com"

# ============================================================
# ESTADÍSTICAS GLOBALES
# ============================================================
TRADES_TOTALES = 0
TRADES_WIN = 0
TRADES_LOSS = 0
PNL_GLOBAL = 0.0
MAX_DRAWDOWN = 0.0
BALANCE_MAX = 0.0
TRADE_COUNTER = 0

ACTIVE_TRADES = []

# ============================================================
# TELEGRAM
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
# FUNCIONES API BYBIT (FIRMA CORREGIDA)
# ============================================================
def bybit_request(endpoint, method='GET', params=None, payload=None):
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
    endpoint = "/v5/position/set-leverage"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage)
    }
    bybit_request(endpoint, method='POST', payload=payload)
    logger.info(f"Apalancamiento establecido a {leverage}x para {symbol}")

def obtener_posiciones_abiertas():
    endpoint = "/v5/position/list"
    params = {
        "category": "linear",
        "symbol": SYMBOL
    }
    result = bybit_request(endpoint, params=params)
    positions = result.get('list', [])
    abiertas = [p for p in positions if float(p.get('size', 0)) != 0]
    return abiertas

def crear_orden_market(symbol, side, qty, reduce_only=False):
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
    endpoint = "/v5/order/create"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side.capitalize(),
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "GTC",
        "stopOrderType": "StopLoss",
        "stopPrice": str(stop_price),
        "reduceOnly": reduce_only
    }
    result = bybit_request(endpoint, method='POST', payload=payload)
    return result

def cancelar_orden(order_id, symbol):
    endpoint = "/v5/order/cancel"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id
    }
    result = bybit_request(endpoint, method='POST', payload=payload)
    return result

def modificar_orden_stop(order_id, symbol, stop_price):
    endpoint = "/v5/order/amend"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "orderId": order_id,
        "stopPrice": str(stop_price)
    }
    result = bybit_request(endpoint, method='POST', payload=payload)
    return result

# ============================================================
# OBTENER VELAS, INDICADORES, ESTADO (SIN CAMBIOS)
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

    ventana = min(50, len(df))
    if ventana < 2:
        min_50 = df['close'].min()
        max_50 = df['close'].max()
    else:
        min_50 = df['close'].iloc[-ventana:].min()
        max_50 = df['close'].iloc[-ventana:].max()

    if precio > max_50:
        soporte = max_50
        resistencia = max_50
    else:
        soporte = min_50
        resistencia = max_50
    if soporte == resistencia:
        soporte = precio * 0.99
        resistencia = precio * 1.01

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
        'idx': idx
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
# PATRONES Y MOTOR DE DECISIÓN (IDÉNTICO AL ORIGINAL)
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

def motor_v90(estado_actual, df):
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
    razones = []

    patron_mult, desc_mult = detectar_patron_multivela(df)
    if patron_mult == "tres_soldados_blancos" and tendencia in ['📈 ALCISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        razones.append("Tres soldados blancos en tendencia favorable")
        return 'Buy', soporte, resistencia, razones
    if patron_mult == "tres_cuervos_negros" and tendencia in ['📉 BAJISTA', '➡️ LATERAL']:
        razones.append(desc_mult)
        razones.append("Tres cuervos negros en tendencia favorable")
        return 'Sell', soporte, resistencia, razones

    if "Martillo" in patron:
        if tendencia == '📉 BAJISTA' or abs(precio - soporte) < atr:
            razones.append(f"Patrón de reversión alcista: {patron}")
            razones.append(f"Contexto: {tendencia} | cerca de soporte {soporte:.2f}")
            return 'Buy', soporte, resistencia, razones
    if "Estrella fugaz" in patron:
        if tendencia == '📈 ALCISTA' or abs(precio - resistencia) < atr:
            razones.append(f"Patrón de reversión bajista: {patron}")
            razones.append(f"Contexto: {tendencia} | cerca de resistencia {resistencia:.2f}")
            return 'Sell', soporte, resistencia, razones

    confirmado = False
    msg_conf = ""
    estado_ant = None
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

    dist_soporte = abs(precio - soporte)
    dist_resistencia = abs(precio - resistencia)

    if dist_soporte < 0.5 * atr:
        senal_bajista_fuerte = False
        if patron == "Vela bajista de cuerpo grande" and estado_actual['close'] < estado_actual['open']:
            senal_bajista_fuerte = True
            razones.append("Vela bajista de cuerpo grande en soporte → posible ruptura")
        if "estrella fugaz" in patron.lower() and estado_actual['close'] < estado_actual['open']:
            senal_bajista_fuerte = True
            razones.append("Estrella fugaz en soporte → señal bajista")
        if tendencia == '📉 BAJISTA' and estado_actual['close'] < soporte:
            senal_bajista_fuerte = True
            razones.append("Cierre por debajo del soporte → ruptura bajista")
        if senal_bajista_fuerte:
            razones.append("Señal bajista fuerte en soporte → NO COMPRAMOS")
            if estado_actual['close'] < soporte - 0.3 * atr:
                razones.append("Ruptura confirmada del soporte → SELL")
                return 'Sell', soporte, resistencia, razones
            else:
                razones.append("Esperar confirmación de ruptura")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"Precio cerca de soporte ({soporte:.2f}) sin señales bajistas fuertes → BUY")
            return 'Buy', soporte, resistencia, razones

    if dist_resistencia < 0.5 * atr:
        senal_alcista_fuerte = False
        if patron == "Vela alcista de cuerpo grande" and estado_actual['close'] > estado_actual['open']:
            senal_alcista_fuerte = True
            razones.append("Vela alcista de cuerpo grande en resistencia → posible ruptura")
        if "martillo" in patron.lower() and estado_actual['close'] > estado_actual['open']:
            senal_alcista_fuerte = True
            razones.append("Martillo en resistencia → señal alcista")
        if tendencia == '📈 ALCISTA' and estado_actual['close'] > resistencia:
            senal_alcista_fuerte = True
            razones.append("Cierre por encima de resistencia → ruptura alcista")
        if senal_alcista_fuerte:
            razones.append("Señal alcista fuerte en resistencia → NO VENDEMOS")
            if estado_actual['close'] > resistencia + 0.3 * atr:
                razones.append("Ruptura confirmada de resistencia → BUY")
                return 'Buy', soporte, resistencia, razones
            else:
                razones.append("Esperar confirmación de ruptura")
                return None, soporte, resistencia, razones
        else:
            razones.append(f"Precio cerca de resistencia ({resistencia:.2f}) sin señales alcistas fuertes → SELL")
            return 'Sell', soporte, resistencia, razones

    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and tendencia != '📈 ALCISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde abajo (EMA actúa como resistencia)")
        return 'Sell', soporte, resistencia, razones
    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and tendencia != '📉 BAJISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde arriba (EMA actúa como soporte)")
        return 'Buy', soporte, resistencia, razones
    if estado_actual['close'] > ema20 and estado_actual['open'] < ema20 and estado_actual['close'] > estado_actual['open']:
        razones.append(f"Ruptura alcista de EMA20 ({ema20:.2f})")
        return 'Buy', soporte, resistencia, razones
    if estado_actual['close'] < ema20 and estado_actual['open'] > ema20 and estado_actual['close'] < estado_actual['open']:
        razones.append(f"Ruptura bajista de EMA20 ({ema20:.2f})")
        return 'Sell', soporte, resistencia, razones

    razones.append("Sin confluencia válida")
    return None, soporte, resistencia, razones

# ============================================================
# FILTRO FUNDAMENTAL
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
            return True, f"BUY en soporte a pesar de sentimiento bajista (soporte {soporte:.2f})"
        else:
            return False, f"Sentimiento bajista bloquea BUY (no está en soporte)"
    if decision == 'Sell' and sent_label == 'Bullish':
        if en_resistencia:
            return True, f"SELL en resistencia a pesar de sentimiento alcista (resistencia {resistencia:.2f})"
        else:
            return False, f"Sentimiento alcista bloquea SELL (no está en resistencia)"
    return True, f"Sentimiento permitido ({sent_label})"

# ============================================================
# GRÁFICO DE VELAS
# ============================================================
def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, estado, precio_salida=None, trade_id=None):
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
            cuerpo_h = abs(closes[i] - opens[i])
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
            rect = plt.Rectangle((x[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")
        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')
        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot, r_plot, _, _ = linregress(x_plot, y_plot)
        tendencia_linea = intercept_plot + slope_plot * x_plot
        ax.plot(x_plot, tendencia_linea, color='white', linewidth=1.5, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")
        entrada_x = len(df_plot) - 1
        entrada_precio = closes[-1]
        if decision == 'Buy':
            ax.scatter(entrada_x, entrada_precio, s=250, marker='^', color='lime', edgecolors='black', linewidths=2, label='Entrada BUY', zorder=5)
            ax.annotate('', xy=(entrada_x, entrada_precio), xytext=(entrada_x-2, entrada_precio-0.5*estado['atr']),
                        arrowprops=dict(arrowstyle='->', color='lime', lw=3))
        elif decision == 'Sell':
            ax.scatter(entrada_x, entrada_precio, s=250, marker='v', color='red', edgecolors='black', linewidths=2, label='Entrada SELL', zorder=5)
            ax.annotate('', xy=(entrada_x, entrada_precio), xytext=(entrada_x-2, entrada_precio+0.5*estado['atr']),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
        if precio_salida is not None:
            ax.scatter(entrada_x, precio_salida, s=200, marker='s', color='yellow', edgecolors='black', linewidths=2, label='Salida', zorder=6)
            ax.axhline(precio_salida, color='yellow', linestyle=':', linewidth=1, alpha=0.5)
        id_text = f" ID: {trade_id}" if trade_id else ""
        texto = (
            f"{decision.upper()}{id_text}\n"
            f"Precio entrada: {entrada_precio:.2f}\n"
            f"Hora: {times[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Soporte: {soporte:.2f}  Resistencia: {resistencia:.2f}\n"
            f"EMA20: {estado['ema20']:.2f}  ATR: {estado['atr']:.2f}\n"
            f"Tendencia: {estado['tendencia']}\n"
            f"Patrón: {estado['patron']}\n"
            f"Razones: {', '.join(razones)}"
        )
        if precio_salida is not None:
            texto += f"\nSalida: {precio_salida:.2f}"
        ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=9, verticalalignment='top', color='white',
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
# GESTIÓN DE POSICIONES REALES (CON RECÁLCULO DE SL/TP POST-ENTRADA)
# ============================================================

def abrir_posicion_real(decision, precio, atr, soporte, resistencia, razones, tiempo, estado, df):
    global TRADE_COUNTER, ACTIVE_TRADES

    if len(ACTIVE_TRADES) >= MAX_OPEN_TRADES:
        return None

    qty = QTY_BTC

    # 1. Entrada Market (sin SL/TP todavía)
    side = "Buy" if decision == "Buy" else "Sell"
    logger.info(f"Enviando orden {side} Market por {qty} BTC...")
    try:
        crear_orden_market(SYMBOL, side, qty, reduce_only=False)
        time.sleep(2)  # Esperar a que la posición se refleje
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

    # 2. Recalcular SL, TP1, TP2 basados en el precio de entrada real
    if decision == "Buy":
        sl_price = round(entry_price - SL_MULTIPLIER * atr, 2)
        tp1_price = round(min(resistencia, entry_price + 2.0 * atr), 2)
        tp2_price = round(entry_price + 3.5 * atr, 2)
    else:
        sl_price = round(entry_price + SL_MULTIPLIER * atr, 2)
        tp1_price = round(max(soporte, entry_price - 2.0 * atr), 2)
        tp2_price = round(entry_price - 3.5 * atr, 2)

    logger.info(f"SL: {sl_price:.2f}, TP1: {tp1_price:.2f}, TP2: {tp2_price:.2f}")

    # 3. Crear órdenes TP1 (50%) y SL (total)
    qty_half = qty / 2
    try:
        tp1_order = crear_orden_limit(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty_half, tp1_price, reduce_only=True)
        tp1_order_id = tp1_order.get('orderId')
        logger.info(f"TP1 orden creada: {tp1_order_id}")
    except Exception as e:
        logger.error(f"Error creando TP1: {e}")
        tp1_order_id = None

    try:
        sl_order = crear_orden_stop_market(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty, sl_price, reduce_only=True)
        sl_order_id = sl_order.get('orderId')
        logger.info(f"SL orden creada: {sl_order_id}")
    except Exception as e:
        logger.error(f"Error creando SL: {e}")
        sl_order_id = None

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
        'order_id_tp2': None,
        'razones': razones,
        'estado_entrada': estado.copy(),
        'timestamp': tiempo
    }
    ACTIVE_TRADES.append(trade_info)

    pnl_flotante = (entry_price - precio) * qty if decision == 'Buy' else (precio - entry_price) * qty
    mensaje_entrada = (
        f"📌 ENTRADA REAL #{trade_id} {decision}\n"
        f"💰 Precio: {entry_price:.2f}\n"
        f"📍 SL: {sl_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}\n"
        f"📦 Cantidad: {qty:.6f} BTC (nominal ≈ {qty*entry_price:.2f} USD)\n"
        f"💵 Margen aprox: {qty*entry_price/LEVERAGE:.2f} USD\n"
        f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
        f"🧠 Razones técnicas:\n• " + "\n• ".join(razones)
    )
    telegram_mensaje(mensaje_entrada)

    fig = generar_grafico_entrada(
        df=df,
        decision=decision,
        soporte=soporte,
        resistencia=resistencia,
        slope=estado['slope'],
        intercept=estado['intercept'],
        razones=razones,
        estado=estado,
        trade_id=trade_id
    )
    if fig:
        telegram_grafico(fig)
        plt.close(fig)

    return trade_info

def actualizar_estadisticas(pnl):
    global TRADES_TOTALES, TRADES_WIN, TRADES_LOSS, PNL_GLOBAL
    TRADES_TOTALES += 1
    PNL_GLOBAL += pnl
    if pnl > 0:
        TRADES_WIN += 1
    else:
        TRADES_LOSS += 1

def revisar_posiciones_reales(precio_actual, df_actual, noticia_titulo, noticia_fuente, sent_label, sent_score):
    global ACTIVE_TRADES

    posiciones_api = obtener_posiciones_abiertas()
    pos_map = {}
    for p in posiciones_api:
        side = p.get('side', '').lower()
        size = float(p.get('size', 0))
        if size != 0:
            pos_map[side] = p

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
        order_id_tp1 = trade['order_id_tp1']
        order_id_tp2 = trade['order_id_tp2']
        atr = trade['estado_entrada']['atr']

        pos_api = pos_map.get(side.lower())
        if not pos_api:
            if qty_remaining > 0:
                if side == 'Buy':
                    pnl = (precio_actual - entry) * qty_remaining
                else:
                    pnl = (entry - precio_actual) * qty_remaining
                actualizar_estadisticas(pnl)
                mensaje = f"🔒 POSICIÓN #{trade_id} CERRADA (sin rastro en API) - PnL: {pnl:.4f} USD"
                telegram_mensaje(mensaje)
            trades_a_remover.append(idx)
            continue

        size_actual = float(pos_api.get('size', 0))
        if size_actual == 0:
            if qty_remaining > 0:
                pnl = 0.0
                actualizar_estadisticas(pnl)
            trades_a_remover.append(idx)
            continue

        # -------- SL --------
        if (side == 'Buy' and precio_actual <= sl_price) or (side == 'Sell' and precio_actual >= sl_price):
            telegram_mensaje(f"⚠️ SL alcanzado para #{trade_id} (precio {precio_actual:.2f})")
            # No removemos la posición aquí porque la API la cerrará y la detectaremos en el próximo ciclo.
            continue

        # -------- ACTIVE: TP1 --------
        if status == 'ACTIVE':
            tp1_alcanzado = False
            if side == 'Buy' and precio_actual >= tp1_price:
                tp1_alcanzado = True
            elif side == 'Sell' and precio_actual <= tp1_price:
                tp1_alcanzado = True

            if tp1_alcanzado:
                qty_cerrar = qty_total / 2
                if qty_cerrar > qty_remaining:
                    qty_cerrar = qty_remaining
                if qty_cerrar <= 0:
                    continue

                try:
                    close_side = 'Sell' if side == 'Buy' else 'Buy'
                    crear_orden_market(SYMBOL, close_side, qty_cerrar, reduce_only=True)
                    trade['qty_remaining'] -= qty_cerrar
                    pnl = (tp1_price - entry) * qty_cerrar if side == 'Buy' else (entry - tp1_price) * qty_cerrar
                    actualizar_estadisticas(pnl)

                    nuevo_sl = entry
                    if order_id_sl:
                        try:
                            cancelar_orden(order_id_sl, SYMBOL)
                        except:
                            pass
                    try:
                        new_sl_order = crear_orden_stop_market(
                            SYMBOL,
                            'Sell' if side == 'Buy' else 'Buy',
                            trade['qty_remaining'],
                            nuevo_sl,
                            reduce_only=True
                        )
                        trade['order_id_sl'] = new_sl_order.get('orderId')
                    except Exception as e:
                        logger.error(f"Error creando nuevo SL (BE): {e}")

                    try:
                        tp2_side = 'Sell' if side == 'Buy' else 'Buy'
                        tp2_order = crear_orden_limit(SYMBOL, tp2_side, trade['qty_remaining'], tp2_price, reduce_only=True)
                        trade['order_id_tp2'] = tp2_order.get('orderId')
                    except Exception as e:
                        logger.error(f"Error creando TP2: {e}")

                    trade['status'] = 'TP1_HIT'
                    trade['sl_price'] = nuevo_sl

                    mensaje = (
                        f"🔓 CIERRE PARCIAL #{trade_id} - TP1 alcanzado\n"
                        f"💰 Precio: {tp1_price:.2f}\n"
                        f"📊 PnL Parcial: {pnl:.4f} USD\n"
                        f"🔄 SL movido a BE ({nuevo_sl:.2f})\n"
                        f"🎯 Buscando TP2 ({tp2_price:.2f}) | Restante: {trade['qty_remaining']:.6f} BTC"
                    )
                    telegram_mensaje(mensaje)

                    fig = generar_grafico_entrada(
                        df=df_actual,
                        decision=side,
                        soporte=trade['estado_entrada']['soporte'],
                        resistencia=trade['estado_entrada']['resistencia'],
                        slope=trade['estado_entrada']['slope'],
                        intercept=trade['estado_entrada']['intercept'],
                        razones=trade['razones'],
                        estado=trade['estado_entrada'],
                        precio_salida=tp1_price,
                        trade_id=trade_id
                    )
                    if fig:
                        telegram_grafico(fig)
                        plt.close(fig)

                except Exception as e:
                    logger.error(f"Error en cierre parcial TP1: {e}")
                continue

        # -------- TP1_HIT: TP2 y trailing --------
        if status == 'TP1_HIT':
            tp2_alcanzado = False
            if side == 'Buy' and precio_actual >= tp2_price:
                tp2_alcanzado = True
            elif side == 'Sell' and precio_actual <= tp2_price:
                tp2_alcanzado = True

            if tp2_alcanzado:
                nuevo_sl = tp2_price
                if order_id_sl:
                    try:
                        cancelar_orden(order_id_sl, SYMBOL)
                    except:
                        pass
                try:
                    new_sl_order = crear_orden_stop_market(
                        SYMBOL,
                        'Sell' if side == 'Buy' else 'Buy',
                        trade['qty_remaining'],
                        nuevo_sl,
                        reduce_only=True
                    )
                    trade['order_id_sl'] = new_sl_order.get('orderId')
                except Exception as e:
                    logger.error(f"Error creando SL post-TP2: {e}")
                trade['sl_price'] = nuevo_sl
                trade['status'] = 'TP2_HIT'

                mensaje = (
                    f"🚀 #TP2 ALCANZADO #{trade_id}\n"
                    f"📍 Precio: {tp2_price:.2f}\n"
                    f"🔒 SL movido a {nuevo_sl:.2f}\n"
                    f"📈 TRAILING STOP ACTIVADO (offset {TRAILING_OFFSET_ATR} ATR)\n"
                    f"📦 Restante: {trade['qty_remaining']:.6f} BTC | Dejando correr..."
                )
                telegram_mensaje(mensaje)
                continue

        # -------- TP2_HIT: trailing --------
        if status == 'TP2_HIT':
            if side == 'Buy':
                nuevo_sl = precio_actual - TRAILING_OFFSET_ATR * atr
                if nuevo_sl > trade['sl_price']:
                    try:
                        modificar_orden_stop(trade['order_id_sl'], SYMBOL, nuevo_sl)
                        trade['sl_price'] = nuevo_sl
                        logger.debug(f"#{trade_id} Trailing SL actualizado a {nuevo_sl:.2f}")
                    except Exception as e:
                        logger.error(f"Error modificando SL trailing: {e}")
            else:
                nuevo_sl = precio_actual + TRAILING_OFFSET_ATR * atr
                if nuevo_sl < trade['sl_price']:
                    try:
                        modificar_orden_stop(trade['order_id_sl'], SYMBOL, nuevo_sl)
                        trade['sl_price'] = nuevo_sl
                        logger.debug(f"#{trade_id} Trailing SL actualizado a {nuevo_sl:.2f}")
                    except Exception as e:
                        logger.error(f"Error modificando SL trailing: {e}")

    for idx in sorted(trades_a_remover, reverse=True):
        del ACTIVE_TRADES[idx]

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_bot():
    global TRADE_COUNTER, ACTIVE_TRADES

    try:
        set_leverage(SYMBOL, LEVERAGE)
    except Exception as e:
        logger.error(f"Error al establecer apalancamiento: {e}")
        telegram_mensaje(f"⚠️ Error apalancamiento: {e}")

    telegram_mensaje("🤖 BOT V90.6 REAL INICIADO (CON TRAILING Y RECÁLCULO)\n"
                     f"📊 Velas: {INTERVAL}m | Máx. posiciones: {MAX_OPEN_TRADES}\n"
                     f"⚡ Leverage: {LEVERAGE}x | Tamaño: {QTY_BTC} BTC\n"
                     f"🔒 SL = {SL_MULTIPLIER} ATR | TP1 50% | Trailing post-TP2 ({TRAILING_OFFSET_ATR} ATR)")

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

            decision, soporte, resistencia, razones = motor_v90(estado, df)

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
            logger.info(f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}")
            logger.info(f"📊 ATR: {estado['atr']:.2f} | EMA20: {estado['ema20']:.2f}")
            logger.info(f"📏 Patrón: {estado['patron']}")
            logger.info(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
            logger.info(f"🧠 Razones: {', '.join(razones)}")
            logger.info(f"🔒 Filtro fundamental: {'PERMITIDO' if filtro_ok else 'BLOQUEADO'} - {motivo_filtro}")
            logger.info(f"📊 Posiciones abiertas: {num_abiertas}/{MAX_OPEN_TRADES}")
            logger.info(f"📰 Noticia: {titulo} (Fuente: {fuente}) | Sentimiento: {sent_label} ({sent_score:.3f})")
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
                    df=df
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
