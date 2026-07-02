# ============================================================
# BOT TRADING V90.3 – HÍBRIDO DETERMINISTA (SIN IA)
# ============================================================
# - Velas de 5 minutos, máximo 3 operaciones abiertas
# - Heartbeat cada 5 minutos con noticia y sentimiento (desde caché)
# - Fuente principal: NewsAPI (caché 1 hora)
# - Sentimiento con VADER (léxico, sin IA)
# - Gráfico con fondo negro y flecha de entrada
# - MEJORAS: vela cerrada, patrones multivela, confirmación, SL/TP ajustado, gráfico de cierre
# - SEGURIDAD EXTREMA: manejo de DataFrames vacíos o con pocas filas
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
MAX_OPEN_TRADES = 3            # máximo 3 operaciones abiertas simultáneas
SLEEP_SECONDS = 300            # 5 minutos (heartbeat y revisión)

# Gráficos
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

# Caché de noticias (para no exceder límite de NewsAPI)
NEWS_CACHE = {
    "titulo": "No disponible",
    "fuente": "Ninguna",
    "sent_label": "Neutral",
    "sent_score": 0.0,
    "timestamp": None
}
NEWS_CACHE_TTL = 3600          # 1 hora en segundos

# ============================================================
# PAPER TRADING (SIMULACIÓN) – GESTIÓN DE POSICIONES
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

# ============================================================
# BYBIT ENDPOINT
# ============================================================
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
# FIRMA BYBIT
# ============================================================
def sign(params):
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    return hmac.new(BYBIT_API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()

# ============================================================
# OBTENER VELAS BYBIT
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

# ============================================================
# INDICADORES
# ============================================================
def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # No dropna aquí para no perder filas, lo haremos después de verificar
    return df

# ============================================================
# CEREBRO DE DATOS – EXTRAER ESTADO CON SEGURIDAD ABSOLUTA
# ============================================================
def extraer_estado_mercado(df, usar_cerrada=True):
    """
    Extrae el estado del mercado a partir de la última vela cerrada (índice -2)
    si usar_cerrada es True y hay al menos 2 filas; si no, usa la última disponible.
    Maneja todos los casos de DataFrame vacío o con pocas filas.
    """
    if df.empty:
        return None

    # Asegurar que tenemos al menos 2 filas para usar_cerrada, si no, forzamos usar_cerrada=False
    if usar_cerrada and len(df) < 2:
        usar_cerrada = False

    # Elegir índice: -2 (cerrada) o -1 (última)
    idx = -2 if usar_cerrada and len(df) >= 2 else -1

    # Verificar que el índice sea válido
    if idx < -len(df) or idx >= len(df):
        # Si por algún motivo el índice no es válido, usar -1
        idx = -1

    fila = df.iloc[idx]
    precio = fila['close']
    # Si 'ema20' o 'atr' son NaN, tomar el último valor disponible (forward fill)
    ema20 = df['ema20'].iloc[idx]
    atr = df['atr'].iloc[idx]
    if pd.isna(ema20):
        # Buscar el último valor no nulo hacia atrás
        ema_serie = df['ema20'].dropna()
        if not ema_serie.empty:
            ema20 = ema_serie.iloc[-1]
        else:
            ema20 = precio  # fallback
    if pd.isna(atr):
        atr_serie = df['atr'].dropna()
        if not atr_serie.empty:
            atr = atr_serie.iloc[-1]
        else:
            atr = precio * 0.01  # fallback: 1% de precio

    # Calcular soporte/resistencia con ventana de 50, pero solo si hay suficientes datos
    ventana = min(50, len(df))
    if ventana < 2:
        # Si hay muy pocos datos, usar el mínimo y máximo de todo el df
        min_50 = df['close'].min()
        max_50 = df['close'].max()
    else:
        # Usamos rolling, pero debemos asegurar que el resultado no esté vacío
        # Usamos los últimos 'ventana' valores
        min_50 = df['close'].iloc[-ventana:].min()
        max_50 = df['close'].iloc[-ventana:].max()

    if precio > max_50:
        soporte = max_50
        resistencia = max_50  # no hay resistencia definida, usamos el mismo
    else:
        soporte = min_50
        resistencia = max_50

    # Ajustar si soporte y resistencia son iguales (caso extremo)
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
    """
    Detecta tendencia usando los últimos 'ventana' precios hasta el índice idx.
    Maneja casos con pocos datos.
    """
    # Asegurar que idx es válido
    if idx < 0:
        idx = len(df) + idx
    if idx < 0:
        idx = 0
    if idx >= len(df):
        idx = len(df) - 1

    # Determinar el inicio
    inicio = max(0, idx - ventana + 1)
    if inicio > idx:
        inicio = idx
    # Extraer precios
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

# ============================================================
# FUNCIÓN AUXILIAR PARA EXTRAER ESTADO POR ÍNDICE (CON SEGURIDAD)
# ============================================================
def extraer_estado_mercado_por_indice(df, idx):
    if df.empty:
        return None
    if idx < 0:
        idx = len(df) + idx
    if idx < 0 or idx >= len(df):
        return None
    # Si idx es 0, el slice df.iloc[:1] es válido
    df_temp = df.iloc[:idx+1]
    if df_temp.empty:
        return None
    return extraer_estado_mercado(df_temp, usar_cerrada=False)

# ============================================================
# PATRONES MULTIVELA Y CONFIRMACIÓN
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
# MOTOR DE DECISIÓN V90 MEJORADO
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
    if patron_mult:
        razones.append(desc_mult)

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

    # Reversiones confirmadas
    if confirmado and estado_ant is not None:
        if "Martillo" in estado_ant['patron'] and abs(precio - soporte) < atr:
            razones.append("Martillo confirmado en soporte")
            return 'Buy', soporte, resistencia, razones
        if "Estrella" in estado_ant['patron'] and abs(precio - resistencia) < atr:
            razones.append("Estrella fugaz confirmada en resistencia")
            return 'Sell', soporte, resistencia, razones

    if patron_mult == "tres_soldados_blancos" and tendencia == '📈 ALCISTA' and precio > ema20:
        razones.append("Tres soldados blancos en tendencia alcista")
        return 'Buy', soporte, resistencia, razones
    if patron_mult == "tres_cuervos_negros" and tendencia == '📉 BAJISTA' and precio < ema20:
        razones.append("Tres cuervos negros en tendencia bajista")
        return 'Sell', soporte, resistencia, razones

    if (abs(precio - soporte) < atr) and (tendencia == '📈 ALCISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de soporte ({soporte:.2f})")
        razones.append("Tendencia alcista o lateral")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    if (abs(precio - resistencia) < atr) and (tendencia == '📉 BAJISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de resistencia ({resistencia:.2f})")
        razones.append("Tendencia bajista o lateral")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and tendencia != '📈 ALCISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde abajo (EMA actúa como resistencia)")
        razones.append("Sin ruptura al alza")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and tendencia != '📉 BAJISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde arriba (EMA actúa como soporte)")
        razones.append("Sin ruptura a la baja")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    if estado_actual['close'] > ema20 and estado_actual['open'] < ema20 and estado_actual['close'] > estado_actual['open']:
        razones.append(f"Ruptura alcista de EMA20 ({ema20:.2f})")
        razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    if estado_actual['close'] < ema20 and estado_actual['open'] > ema20 and estado_actual['close'] < estado_actual['open']:
        razones.append(f"Ruptura bajista de EMA20 ({ema20:.2f})")
        razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    razones.append("Sin confluencia válida")
    return None, soporte, resistencia, razones

# ============================================================
# FILTRO FUNDAMENTAL CON NEWSAPI + GOOGLE RSS + CACHÉ
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

def filtrar_por_fundamental(decision, sent_label):
    if sent_label == 'Bearish' and decision == 'Buy':
        return False, f"Noticias bajistas → bloqueo LONG"
    if sent_label == 'Bullish' and decision == 'Sell':
        return False, f"Noticias alcistas → bloqueo SHORT"
    return True, f"Sentimiento permitido ({sent_label})"

# ============================================================
# GRÁFICO DE VELAS JAPONESAS CON FONDO NEGRO (ENTRADA)
# ============================================================
def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, estado):
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
        ax.plot(x_plot, tendencia_linea, color='white', linewidth=1.5, linestyle='-',
                label=f"Tendencia slope {slope_plot:.4f}")

        entrada_x = len(df_plot) - 1
        entrada_precio = closes[-1]
        if decision == 'Buy':
            ax.scatter(entrada_x, entrada_precio, s=250, marker='^', color='lime',
                       edgecolors='black', linewidths=2, label='Entrada BUY', zorder=5)
            ax.annotate('', xy=(entrada_x, entrada_precio), xytext=(entrada_x-2, entrada_precio-0.5*estado['atr']),
                        arrowprops=dict(arrowstyle='->', color='lime', lw=3))
        elif decision == 'Sell':
            ax.scatter(entrada_x, entrada_precio, s=250, marker='v', color='red',
                       edgecolors='black', linewidths=2, label='Entrada SELL', zorder=5)
            ax.annotate('', xy=(entrada_x, entrada_precio), xytext=(entrada_x-2, entrada_precio+0.5*estado['atr']),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))

        texto = (
            f"{decision.upper()}\n"
            f"Precio: {entrada_precio:.2f}\n"
            f"Hora: {times[-1].strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Soporte: {soporte:.2f}  Resistencia: {resistencia:.2f}\n"
            f"EMA20: {estado['ema20']:.2f}  ATR: {estado['atr']:.2f}\n"
            f"Tendencia: {estado['tendencia']}\n"
            f"Patrón: {estado['patron']}\n"
            f"Razones: {', '.join(razones)}"
        )
        ax.text(0.02, 0.98, texto, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        ax.set_title(f"{SYMBOL} - Velas {INTERVAL}m - Entrada {decision}", color='white')
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
# GRÁFICO DE CIERRE DE OPERACIÓN
# ============================================================
def generar_grafico_cierre(df, posicion, precio_salida, motivo):
    if df.empty:
        return None
    try:
        entrada_time = posicion['timestamp']
        df_recorte = df[df.index >= entrada_time].copy()
        if len(df_recorte) < 2:
            return None

        plt.style.use('dark_background')
        times = df_recorte.index
        opens = df_recorte['open'].values
        highs = df_recorte['high'].values
        lows = df_recorte['low'].values
        closes = df_recorte['close'].values
        x = np.arange(len(df_recorte))

        fig, ax = plt.subplots(figsize=(14, 7), facecolor='black')
        ax.set_facecolor('black')

        for i in range(len(df_recorte)):
            color = 'lime' if closes[i] >= opens[i] else 'red'
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = abs(closes[i] - opens[i])
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
            rect = plt.Rectangle((x[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)

        ax.axhline(posicion['sl'], color='orange', linestyle='--', linewidth=2, label=f"SL {posicion['sl']:.2f}")
        ax.axhline(posicion['tp'], color='deepskyblue', linestyle='--', linewidth=2, label=f"TP {posicion['tp']:.2f}")

        ax.axvline(x=0, color='white', linestyle=':', linewidth=2, label='Entrada')
        ax.axvline(x=len(df_recorte)-1, color='yellow', linestyle=':', linewidth=2, label='Salida')

        entrada_y = posicion['entry_price']
        ax.scatter(0, entrada_y, s=200, marker='^' if posicion['decision']=='Buy' else 'v',
                   color='lime' if posicion['decision']=='Buy' else 'red', edgecolors='black', zorder=5)
        ax.scatter(len(df_recorte)-1, precio_salida, s=200, marker='s', color='yellow', edgecolors='black', zorder=5)

        texto = (
            f"CIERRE {posicion['decision']} - {motivo}\n"
            f"Entrada: {entrada_y:.2f}  Salida: {precio_salida:.2f}\n"
            f"SL: {posicion['sl']:.2f}  TP: {posicion['tp']:.2f}\n"
            f"PnL: {(precio_salida - entrada_y) * posicion['size_btc']:.4f} USD"
        )
        ax.text(0.02, 0.98, texto, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))

        ax.set_title(f"{SYMBOL} - Cierre de operación", color='white')
        ax.set_xlabel("Velas desde entrada", color='white')
        ax.set_ylabel("Precio", color='white')
        ax.grid(True, alpha=0.2, color='gray')
        ax.tick_params(colors='white')
        ax.legend(loc='lower left', facecolor='black', edgecolor='white', labelcolor='white')
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error en gráfico de cierre: {e}")
        return None

# ============================================================
# PAPER TRADING – ABRIR Y CERRAR POSICIONES
# ============================================================
def paper_abrir_posicion(decision, precio, atr, soporte, resistencia, razones, tiempo, estado):
    global PAPER_BALANCE, PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS

    if len(OPEN_POSITIONS) >= MAX_OPEN_TRADES:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "Buy":
        sl = precio - 1.5 * atr
        tp = min(resistencia, precio + 2.5 * atr)
    else:
        sl = precio + 1.5 * atr
        tp = max(soporte, precio - 2.5 * atr)

    distancia_sl = abs(precio - sl)
    if distancia_sl == 0:
        return False

    size_btc = riesgo_usd / distancia_sl
    size_usd = size_btc * precio

    posicion = {
        'decision': decision,
        'entry_price': precio,
        'sl': sl,
        'tp': tp,
        'size_btc': size_btc,
        'size_usd': size_usd,
        'razones': razones,
        'timestamp': tiempo,
        'estado_entrada': estado.copy()
    }
    OPEN_POSITIONS.append(posicion)
    return True

def paper_calcular_pnl(posicion, precio_actual):
    if posicion['decision'] == "Buy":
        return (precio_actual - posicion['entry_price']) * posicion['size_btc']
    else:
        return (posicion['entry_price'] - precio_actual) * posicion['size_btc']

def paper_revisar_posiciones(precio_actual, df_actual):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS

    posiciones_a_cerrar = []
    for i, pos in enumerate(OPEN_POSITIONS):
        pnl = paper_calcular_pnl(pos, precio_actual)
        cerrar = False
        motivo = ""
        if pos['decision'] == "Buy":
            if precio_actual <= pos['sl']:
                cerrar = True; motivo = "SL"
            elif precio_actual >= pos['tp']:
                cerrar = True; motivo = "TP"
        else:
            if precio_actual >= pos['sl']:
                cerrar = True; motivo = "SL"
            elif precio_actual <= pos['tp']:
                cerrar = True; motivo = "TP"

        if cerrar:
            posiciones_a_cerrar.append((i, pos, pnl, motivo))

    for i, pos, pnl, motivo in reversed(posiciones_a_cerrar):
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
            f"📌 CIERRE PAPER {pos['decision']} ({motivo})\n"
            f"📍 Entrada: {pos['entry_price']:.2f}\n"
            f"📍 Salida: {precio_actual:.2f}\n"
            f"💰 PnL Trade: {pnl:.4f} USD\n"
            f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
            f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
            f"🏆 Wins: {PAPER_WIN} | ❌ Loss: {PAPER_LOSS}\n"
            f"📉 Max Drawdown: {PAPER_MAX_DRAWDOWN:.4f} USD\n"
            f"🧠 Razones: {', '.join(pos['razones'])}"
        )
        telegram_mensaje(mensaje_cierre)

        fig = generar_grafico_cierre(df_actual, pos, precio_actual, motivo)
        if fig:
            telegram_grafico(fig)
            plt.close(fig)

        OPEN_POSITIONS.pop(i)

    return len(posiciones_a_cerrar) > 0

# ============================================================
# HEARTBEAT
# ============================================================
ultimo_heartbeat = 0
ciclo_count = 0

def enviar_heartbeat(precio, titulo, fuente, sent_label, sent_score):
    global ultimo_heartbeat, ciclo_count
    ahora = time.time()
    if ahora - ultimo_heartbeat >= 300:
        ciclo_count += 1
        mensaje = (
            f"🔄 HEARTBEAT #{ciclo_count} - Bot activo\n"
            f"📰 Última noticia: {titulo}\n"
            f"📌 Fuente: {fuente}\n"
            f"🧠 Sentimiento: {sent_label} (score: {sent_score:.3f})\n"
            f"💰 Precio BTC: {precio:.2f}\n"
            f"⏳ Esperando formación de vela...\n"
            f"📊 Posiciones abiertas: {len(OPEN_POSITIONS)}/{MAX_OPEN_TRADES}"
        )
        telegram_mensaje(mensaje)
        ultimo_heartbeat = ahora

# ============================================================
# LOG EN CONSOLA
# ============================================================
def log_estado(estado, decision, razones, filtro_ok, motivo_filtro, num_abiertas):
    logger.info("="*100)
    logger.info(f"🕒 {estado['fecha']} | 💰 BTC: {estado['precio']:.2f}")
    logger.info(f"📐 Tendencia: {estado['tendencia']} | Slope: {estado['slope']:.5f}")
    logger.info(f"🧱 Soporte: {estado['soporte']:.2f} | Resistencia: {estado['resistencia']:.2f}")
    logger.info(f"📊 ATR: {estado['atr']:.2f} | EMA20: {estado['ema20']:.2f}")
    logger.info(f"📏 Patrón: {estado['patron']}")
    logger.info(f"🎯 Decisión: {decision if decision else 'NO TRADE'}")
    logger.info(f"🧠 Razones: {', '.join(razones)}")
    logger.info(f"🔒 Filtro fundamental: {'PERMITIDO' if filtro_ok else 'BLOQUEADO'} - {motivo_filtro}")
    logger.info(f"📊 Posiciones abiertas: {num_abiertas}/{MAX_OPEN_TRADES}")
    logger.info("="*100)

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_bot():
    global ultimo_heartbeat
    telegram_mensaje("🤖 BOT V90.3 INICIADO (MEJORADO)\n"
                     f"📊 Velas: {INTERVAL}m | Heartbeat cada 5min | Máx. posiciones: {MAX_OPEN_TRADES}")
    ultima_fecha = None

    while True:
        try:
            # 1. Obtener velas
            df = obtener_velas()
            df = calcular_indicadores(df)

            # Verificar que el DataFrame tenga al menos 1 fila después de indicadores
            if df.empty:
                logger.warning("⚠️ DataFrame vacío después de indicadores. Saltando ciclo...")
                time.sleep(SLEEP_SECONDS)
                continue

            # 2. Extraer estado usando la última vela cerrada (si es posible)
            estado = extraer_estado_mercado(df, usar_cerrada=True)
            if estado is None:
                logger.warning("⚠️ Estado nulo. Saltando ciclo...")
                time.sleep(SLEEP_SECONDS)
                continue

            # 3. Noticias y sentimiento (desde caché)
            titulo, fuente, sent_label, sent_score = obtener_noticias_y_sentimiento()

            # 4. Heartbeat
            enviar_heartbeat(estado['precio'], titulo, fuente, sent_label, sent_score)

            # 5. Decisión técnica
            decision, soporte, resistencia, razones = motor_v90(estado, df)

            # 6. Filtro fundamental
            filtro_ok = True
            motivo_filtro = "Sin filtro"
            if decision:
                filtro_ok, motivo_filtro = filtrar_por_fundamental(decision, sent_label)
                if not filtro_ok:
                    decision = None

            # 7. Log
            num_abiertas = len(OPEN_POSITIONS)
            log_estado(estado, decision, razones, filtro_ok, motivo_filtro, num_abiertas)

            # 8. Ejecutar operación
            if decision and num_abiertas < MAX_OPEN_TRADES:
                apertura = paper_abrir_posicion(
                    decision=decision,
                    precio=estado['precio'],
                    atr=estado['atr'],
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=estado['fecha'],
                    estado=estado
                )

                if apertura:
                    pnl_flotante = paper_calcular_pnl(OPEN_POSITIONS[-1], estado['precio'])
                    mensaje_entrada = (
                        f"📌 ENTRADA PAPER {decision}\n"
                        f"💰 Precio: {estado['precio']:.2f}\n"
                        f"📍 SL: {OPEN_POSITIONS[-1]['sl']:.2f} | TP: {OPEN_POSITIONS[-1]['tp']:.2f}\n"
                        f"📦 Size USD: {OPEN_POSITIONS[-1]['size_usd']:.2f} | Size BTC: {OPEN_POSITIONS[-1]['size_btc']:.6f}\n"
                        f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
                        f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
                        f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
                        f"🧠 Razones técnicas:\n• " + "\n• ".join(razones) + "\n"
                        f"📊 Patrón: {estado['patron']}\n"
                        f"📈 Tendencia: {estado['tendencia']}\n"
                        f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}\n"
                        f"📉 EMA20: {estado['ema20']:.2f} (actúa como {estado['ema_nivel']})\n"
                        f"🔒 Filtro fundamental: {motivo_filtro}\n"
                        f"📊 Posiciones abiertas: {len(OPEN_POSITIONS)}/{MAX_OPEN_TRADES}"
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
                        estado=estado
                    )
                    if fig:
                        telegram_grafico(fig)
                        plt.close(fig)

            # 9. Revisar SL/TP y cerrar posiciones
            precio_actual = estado['precio']
            paper_revisar_posiciones(precio_actual, df)

            # 10. Reset diario
            fecha_hoy = datetime.now(timezone.utc).date()
            if ultima_fecha is None:
                ultima_fecha = fecha_hoy
            elif fecha_hoy != ultima_fecha:
                ultima_fecha = fecha_hoy
                logger.info("Nuevo día.")

            # 11. Esperar 5 min
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
