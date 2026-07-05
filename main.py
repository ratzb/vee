# ============================================================
# BOT TRADING V90.5 – GESTIÓN AVANZADA (TP2 + LEVERAGE 10x)
# ============================================================
# - Velas de 5 minutos, máximo 3 operaciones abiertas
# - Fuente principal: NewsAPI (caché 1 hora) + Google RSS
# - Sentimiento con VADER (léxico, sin IA)
# - Gráfico con fondo negro, flecha de entrada y marcador de salida
# - NUEVO: Apalancamiento x10, cierre parcial 50% en TP1, SL a BE, TP2 dinámico
# - NUEVO: Identificador único (#ID) para cada operación
# ============================================================

import os
import time
import io
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
RISK_PER_TRADE = 0.0025        # 0.25% del balance por operación
LEVERAGE = 10                  # <-- NUEVO: Apalancamiento 10x
MAX_OPEN_TRADES = 3            # máximo 3 operaciones abiertas simultáneas
SLEEP_SECONDS = 300            # 5 minutos (revisión)

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
# PAPER TRADING (SIMULACIÓN) – VARIABLES GLOBALES
# ============================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
OPEN_POSITIONS = []
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_MAX_DRAWDOWN = 0.0
PAPER_BALANCE_MAX = PAPER_BALANCE_INICIAL

TRADE_COUNTER = 0              # <-- NUEVO: Contador global de trades

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
# INICIALIZAR VADER
# ============================================================
sentiment_analyzer = SentimentIntensityAnalyzer()
BASE_URL = "https://api.bybit.com"

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
# OBTENER VELAS BYBIT (SIN CAMBIOS)
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

# ============================================================
# CEREBRO DE DATOS (SIN CAMBIOS)
# ============================================================
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
# PATRONES MULTIVELA (SIN CAMBIOS)
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

# ============================================================
# MOTOR DE DECISIÓN V90.4 (SIN CAMBIOS)
# ============================================================
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
# FILTRO FUNDAMENTAL (SIN CAMBIOS)
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
# GRÁFICO DE VELAS (SIN CAMBIOS)
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
# NUEVA GESTIÓN DE POSICIONES CON TP2 Y LEVERAGE
# ============================================================

def paper_abrir_posicion(decision, precio, atr, soporte, resistencia, razones, tiempo, estado):
    global PAPER_BALANCE, PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS, TRADE_COUNTER

    if len(OPEN_POSITIONS) >= MAX_OPEN_TRADES:
        return None

    # Calcular riesgo en USD (0.25% del balance)
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE * LEVERAGE  # <-- Aplicamos apalancamiento

    if decision == "Buy":
        sl = precio - 1.5 * atr
        # TP1: 2.0 * ATR, pero sin sobrepasar la resistencia
        tp1 = min(resistencia, precio + 2.0 * atr)
        # TP2: 3.5 * ATR, dejando correr si rompe resistencia
        tp2 = precio + 3.5 * atr
    else:  # Sell
        sl = precio + 1.5 * atr
        tp1 = max(soporte, precio - 2.0 * atr)
        tp2 = precio - 3.5 * atr

    distancia_sl = abs(precio - sl)
    if distancia_sl == 0:
        return None

    # Tamaño del contrato (con apalancamiento)
    size_btc = riesgo_usd / distancia_sl
    size_usd = size_btc * precio

    # Incrementar contador de trades
    TRADE_COUNTER += 1
    trade_id = TRADE_COUNTER

    posicion = {
        'trade_id': trade_id,
        'decision': decision,
        'entry_price': precio,
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'current_sl': sl,           # SL actual (se moverá)
        'current_tp': tp1,          # TP actual (se moverá a tp2)
        'status': 'ACTIVE',         # ACTIVE, TP1_HIT, TP2_HIT
        'full_size_btc': size_btc,
        'remaining_size_btc': size_btc,
        'size_usd': size_usd,
        'razones': razones,
        'timestamp': tiempo,
        'estado_entrada': estado.copy()
    }
    OPEN_POSITIONS.append(posicion)
    return posicion

def paper_calcular_pnl(posicion, precio_actual):
    size = posicion['remaining_size_btc']
    if posicion['decision'] == "Buy":
        return (precio_actual - posicion['entry_price']) * size
    else:
        return (posicion['entry_price'] - precio_actual) * size

def paper_revisar_posiciones(precio_actual, df_actual, noticia_titulo, noticia_fuente, sent_label, sent_score):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS

    posiciones_a_remover = []

    for idx, pos in enumerate(OPEN_POSITIONS):
        trade_id = pos['trade_id']
        decision = pos['decision']
        entry = pos['entry_price']
        remaining = pos['remaining_size_btc']
        status = pos['status']
        current_sl = pos['current_sl']
        current_tp = pos['current_tp']

        pnl = 0
        cerrar_total = False
        cerrar_parcial = False
        motivo_cierre = ""
        nuevo_status = status
        precio_cierre = None
        cantidad_cerrar = 0

        # -------- Verificar SL (siempre prioritario) --------
        if decision == "Buy":
            if precio_actual <= current_sl:
                cerrar_total = True
                motivo_cierre = "SL"
                precio_cierre = current_sl
        else:  # Sell
            if precio_actual >= current_sl:
                cerrar_total = True
                motivo_cierre = "SL"
                precio_cierre = current_sl

        if cerrar_total:
            pnl = paper_calcular_pnl(pos, precio_cierre)
            pos['remaining_size_btc'] = 0
            cantidad_cerrar = remaining
            posiciones_a_remover.append((idx, pos, pnl, f"CIERRE TOTAL #{trade_id} - {motivo_cierre}"))
            continue

        # -------- Lógica por estado --------
        if status == 'ACTIVE':
            # Verificar TP1
            if decision == "Buy" and precio_actual >= current_tp:
                cerrar_parcial = True
                motivo_cierre = "TP1"
                precio_cierre = current_tp
            elif decision == "Sell" and precio_actual <= current_tp:
                cerrar_parcial = True
                motivo_cierre = "TP1"
                precio_cierre = current_tp

            if cerrar_parcial:
                # Cerrar el 50% del tamaño restante (que es el full)
                cantidad_cerrar = pos['full_size_btc'] * 0.5
                pnl = (precio_cierre - entry) * cantidad_cerrar if decision == "Buy" else (entry - precio_cierre) * cantidad_cerrar
                # Actualizar posición
                pos['remaining_size_btc'] -= cantidad_cerrar
                pos['current_sl'] = entry  # SL a breakeven
                pos['current_tp'] = pos['tp2']  # Apuntar a TP2
                pos['status'] = 'TP1_HIT'
                # Registrar cierre parcial (no se remueve la posición)
                mensaje = (
                    f"🔓 CIERRE PARCIAL #{trade_id} - TP1 alcanzado\n"
                    f"💰 Precio cierre: {precio_cierre:.2f}\n"
                    f"📊 PnL Parcial: {pnl:.4f} USD\n"
                    f"💵 Balance: {PAPER_BALANCE + pnl:.2f} USD\n"
                    f"🔄 SL movido a breakeven ({entry:.2f})\n"
                    f"🎯 Buscando TP2 ({pos['tp2']:.2f}) | Restante: {pos['remaining_size_btc']:.6f} BTC"
                )
                telegram_mensaje(mensaje)
                # Actualizar balance con el PnL parcial
                PAPER_BALANCE += pnl
                PAPER_PNL_GLOBAL += pnl
                if PAPER_BALANCE > PAPER_BALANCE_MAX:
                    PAPER_BALANCE_MAX = PAPER_BALANCE
                drawdown = PAPER_BALANCE_MAX - PAPER_BALANCE
                if drawdown > PAPER_MAX_DRAWDOWN:
                    PAPER_MAX_DRAWDOWN = drawdown
                # Enviar gráfico de cierre parcial
                fig = generar_grafico_entrada(
                    df=df_actual,
                    decision=decision,
                    soporte=pos['estado_entrada']['soporte'],
                    resistencia=pos['estado_entrada']['resistencia'],
                    slope=pos['estado_entrada']['slope'],
                    intercept=pos['estado_entrada']['intercept'],
                    razones=pos['razones'],
                    estado=pos['estado_entrada'],
                    precio_salida=precio_cierre,
                    trade_id=trade_id
                )
                if fig:
                    telegram_grafico(fig)
                    plt.close(fig)
                # Este trade sigue abierto, no lo removemos
                continue

        elif status == 'TP1_HIT':
            # Verificar TP2
            if decision == "Buy" and precio_actual >= current_tp:
                cerrar_parcial = False  # No cerramos, solo movemos SL
                motivo_cierre = "TP2"
                precio_cierre = current_tp
            elif decision == "Sell" and precio_actual <= current_tp:
                cerrar_parcial = False
                motivo_cierre = "TP2"
                precio_cierre = current_tp
            else:
                # Si no se alcanza TP2, seguimos y el SL (en entry) se encargará si retrocede
                continue

            # Alcanzó TP2: mover SL a TP2, eliminar TP, cambiar estado
            pos['current_sl'] = current_tp  # SL se mueve a TP2
            pos['current_tp'] = None        # Sin TP fijo, solo SL trailing
            pos['status'] = 'TP2_HIT'
            mensaje = (
                f"🚀 #TP2 ALCANZADO #{trade_id}\n"
                f"📍 Precio: {precio_cierre:.2f}\n"
                f"🔒 SL movido a {current_tp:.2f} (asegurando ganancias)\n"
                f"📦 Restante: {pos['remaining_size_btc']:.6f} BTC | Dejando correr..."
            )
            telegram_mensaje(mensaje)
            # No se cierra nada, solo se ajusta SL. Seguir en el loop.

        elif status == 'TP2_HIT':
            # Solo nos fijamos en el SL (que es el TP2 anterior)
            # Si el precio retrocede y toca el SL, se cierra el resto.
            # Esto ya lo maneja la verificación de SL al principio del loop.
            # Pero el SL ya está en current_sl, y current_tp es None.
            # La lógica de SL al principio lo cerrará si toca.
            pass

    # -------- Cerrar totalmente las posiciones marcadas --------
    for idx, pos, pnl, mensaje_titulo in reversed(posiciones_a_remover):
        # El PnL ya está calculado para el cierre total
        # Si no se calculó (por alguna razon), calcularlo ahora
        if pnl == 0 and pos['remaining_size_btc'] > 0:
             # Calcular basado en precio_actual si no se definió
            pnl = paper_calcular_pnl(pos, precio_actual)
            # Asegurar que se cierre todo
        if pos['remaining_size_btc'] > 0:
            # Si por alguna razon no se había vaciado, vaciar
            pnl = paper_calcular_pnl(pos, precio_actual) if pnl == 0 else pnl
            pos['remaining_size_btc'] = 0

        PAPER_BALANCE += pnl
        PAPER_PNL_GLOBAL += pnl
        PAPER_TRADES_TOTALES += 1
        if pnl > 0:
            PAPER_WIN += 1
        else:
            PAPER_LOSS += 1

        if PAPER_BALANCE > PAPER_BALANCE_MAX:
            PAPER_BALANCE_MAX = PAPER_BALANCE
        drawdown = PAPER_BALANCE_MAX - PAPER_BALANCE
        if drawdown > PAPER_MAX_DRAWDOWN:
            PAPER_MAX_DRAWDOWN = drawdown

        mensaje_cierre = (
            f"📌 {mensaje_titulo}\n"
            f"📍 Entrada: {pos['entry_price']:.2f}\n"
            f"📍 Salida: {precio_actual:.2f}\n"
            f"💰 PnL Trade: {pnl:.4f} USD\n"
            f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
            f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
            f"🏆 Wins: {PAPER_WIN} | ❌ Loss: {PAPER_LOSS}\n"
            f"📉 Max Drawdown: {PAPER_MAX_DRAWDOWN:.4f} USD\n"
            f"🧠 Razones: {', '.join(pos['razones'])}\n"
            f"📰 Noticia: {noticia_titulo}\n"
            f"📌 Fuente: {noticia_fuente} | Sentimiento: {sent_label} ({sent_score:.3f})"
        )
        telegram_mensaje(mensaje_cierre)

        estado_ent = pos['estado_entrada']
        fig = generar_grafico_entrada(
            df=df_actual,
            decision=pos['decision'],
            soporte=estado_ent['soporte'],
            resistencia=estado_ent['resistencia'],
            slope=estado_ent['slope'],
            intercept=estado_ent['intercept'],
            razones=pos['razones'],
            estado=estado_ent,
            precio_salida=precio_actual,
            trade_id=pos['trade_id']
        )
        if fig:
            telegram_grafico(fig)
            plt.close(fig)

        # Remover la posición (usando pop en orden inverso)
        # Como tenemos la lista de indices, los removemos uno por uno en orden inverso
        # Pero como OPEN_POSITIONS muta, usamos un marker para borrar después
        pass

    # Remover posiciones marcadas (reversed para no alterar indices)
    for idx, pos, pnl, mensaje in reversed(posiciones_a_remover):
        # Encontrar el indice real en OPEN_POSITIONS (por si acaso)
        for i, p in enumerate(OPEN_POSITIONS):
            if p['trade_id'] == pos['trade_id']:
                OPEN_POSITIONS.pop(i)
                break

    return len(posiciones_a_remover) > 0

# ============================================================
# LOOP PRINCIPAL (SIN HEARTBEAT)
# ============================================================
def run_bot():
    telegram_mensaje("🤖 BOT V90.5 INICIADO (MEJORADO)\n"
                     f"📊 Velas: {INTERVAL}m | Máx. posiciones: {MAX_OPEN_TRADES}\n"
                     f"⚡ Leverage: {LEVERAGE}x | TP1 50% | TP2 dinámico")
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

            num_abiertas = len(OPEN_POSITIONS)
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
                pos = paper_abrir_posicion(
                    decision=decision,
                    precio=estado['precio'],
                    atr=estado['atr'],
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=estado['fecha'],
                    estado=estado
                )

                if pos:
                    trade_id = pos['trade_id']
                    pnl_flotante = paper_calcular_pnl(pos, estado['precio'])
                    mensaje_entrada = (
                        f"📌 ENTRADA PAPER #{trade_id} {decision}\n"
                        f"💰 Precio: {estado['precio']:.2f}\n"
                        f"📍 SL: {pos['sl']:.2f} | TP1: {pos['tp1']:.2f} | TP2: {pos['tp2']:.2f}\n"
                        f"📦 Size USD (notional): {pos['size_usd']:.2f} | Size BTC: {pos['full_size_btc']:.6f}\n"
                        f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
                        f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
                        f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
                        f"🧠 Razones técnicas:\n• " + "\n• ".join(razones) + "\n"
                        f"📊 Patrón: {estado['patron']}\n"
                        f"📈 Tendencia: {estado['tendencia']}\n"
                        f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}\n"
                        f"📉 EMA20: {estado['ema20']:.2f} (actúa como {estado['ema_nivel']})\n"
                        f"🔒 Filtro fundamental: {motivo_filtro}\n"
                        f"📊 Posiciones abiertas: {len(OPEN_POSITIONS)}/{MAX_OPEN_TRADES}\n"
                        f"📰 Noticia: {titulo}\n"
                        f"📌 Fuente: {fuente} | Sentimiento: {sent_label} ({sent_score:.3f})"
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

            # Revisar posiciones (TP1, TP2, SL)
            precio_actual = estado['precio']
            paper_revisar_posiciones(precio_actual, df, titulo, fuente, sent_label, sent_score)

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
