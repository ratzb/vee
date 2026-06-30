# ============================================================
# BOT TRADING V90.2 – HÍBRIDO DETERMINISTA (SIN IA)
# ============================================================
# - Velas de 30 minutos, máximo 3 operaciones abiertas
# - Heartbeat cada 5 minutos con noticia y sentimiento (desde caché)
# - Fuente principal: NewsAPI (caché 30 min) + respaldo: Google News RSS
# - Sentimiento con VADER (léxico, sin IA)
# - Gráfico con fondo negro y flecha de entrada
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
INTERVAL = "30"                # 30 minutos
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
NEWS_CACHE_TTL = 1800  # 30 minutos en segundos

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
NEWS_API_KEY = os.getenv("NEWS_API_KEY")   # Opcional pero recomendada

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
# TELEGRAM (SIN PROXY)
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
# FIRMA BYBIT (para futuras órdenes reales)
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
    return df.dropna()

# ============================================================
# CEREBRO DE DATOS: extraer estado del mercado
# ============================================================
def extraer_estado_mercado(df):
    precio = df['close'].iloc[-1]
    ema20 = df['ema20'].iloc[-1]
    atr = df['atr'].iloc[-1]

    ventana = 50
    if len(df) < ventana:
        ventana = len(df)
    min_50 = df['close'].rolling(ventana).min().iloc[-1]
    max_50 = df['close'].rolling(ventana).max().iloc[-1]

    if precio > max_50:
        soporte = max_50
        resistencia = df['close'].rolling(ventana).max().iloc[-1]
    else:
        soporte = min_50
        resistencia = max_50

    if precio > ema20:
        ema_nivel = 'soporte'
    else:
        ema_nivel = 'resistencia'

    open_actual = df['open'].iloc[-1]
    high_actual = df['high'].iloc[-1]
    low_actual = df['low'].iloc[-1]
    close_actual = df['close'].iloc[-1]
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

    slope, intercept, tendencia = _detectar_tendencia(df)
    if slope > 0.02:
        sentimiento = 1
    elif slope < -0.02:
        sentimiento = -1
    else:
        sentimiento = 0

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
        'sentimiento': sentimiento,
        'fecha': df.index[-1],
        'open': open_actual,
        'high': high_actual,
        'low': low_actual,
        'close': close_actual,
        'sombra_superior': sombra_superior,
        'sombra_inferior': sombra_inferior
    }
    return estado

def _detectar_tendencia(df, ventana=80):
    if len(df) < ventana:
        ventana = len(df)
    y = df['close'].values[-ventana:]
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
# MOTOR DE DECISIÓN V90 MEJORADO
# ============================================================
def motor_v90(estado):
    precio = estado['precio']
    soporte = estado['soporte']
    resistencia = estado['resistencia']
    atr = estado['atr']
    tendencia = estado['tendencia']
    ema20 = estado['ema20']
    ema_nivel = estado['ema_nivel']
    patron = estado['patron']

    razones = []

    # Regla 1: Soporte + tendencia alcista
    if (abs(precio - soporte) < atr) and (tendencia == '📈 ALCISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de soporte ({soporte:.2f})")
        razones.append("Tendencia alcista o lateral")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    # Regla 2: Resistencia + tendencia bajista
    if (abs(precio - resistencia) < atr) and (tendencia == '📉 BAJISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de resistencia ({resistencia:.2f})")
        razones.append("Tendencia bajista o lateral")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    # Regla 3: EMA como resistencia
    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and tendencia != '📈 ALCISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde abajo (EMA actúa como resistencia)")
        razones.append("Sin ruptura al alza")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    # Regla 4: EMA como soporte
    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and tendencia != '📉 BAJISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde arriba (EMA actúa como soporte)")
        razones.append("Sin ruptura a la baja")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    # Regla 5: Ruptura de EMA
    if estado['close'] > ema20 and estado['open'] < ema20 and estado['close'] > estado['open']:
        razones.append(f"Ruptura alcista de EMA20 ({ema20:.2f})")
        razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    if estado['close'] < ema20 and estado['open'] > ema20 and estado['close'] < estado['open']:
        razones.append(f"Ruptura bajista de EMA20 ({ema20:.2f})")
        razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    razones.append("Sin confluencia válida")
    return None, soporte, resistencia, razones

# ============================================================
# FILTRO FUNDAMENTAL CON NEWSAPI + GOOGLE RSS + CACHÉ
# ============================================================
def actualizar_cache_noticias():
    """Actualiza la caché de noticias si ha expirado (TTL 30 min)."""
    global NEWS_CACHE
    ahora = datetime.now(timezone.utc)

    # Si la caché no ha expirado, no hacer nada
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
    """Obtiene noticias desde NewsAPI o Google RSS, y calcula sentimiento con VADER."""
    noticias = []
    fuente = "Ninguna"

    # ---- Intento 1: NewsAPI ----
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

    # ---- Intento 2: Google News RSS (respaldo) ----
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

    # ---- Procesar con VADER ----
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

    # Si no hay noticias
    return "No disponible", "Ninguna", "Neutral", 0.0

def obtener_noticias_y_sentimiento():
    """Devuelve la noticia y sentimiento desde la caché (siempre actualizada)."""
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
# GRÁFICO DE VELAS JAPONESAS CON FONDO NEGRO
# ============================================================
def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, estado):
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
# PAPER TRADING – ABRIR Y CERRAR POSICIONES
# ============================================================
def paper_abrir_posicion(decision, precio, atr, soporte, resistencia, razones, tiempo, estado):
    global PAPER_BALANCE, PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS

    if len(OPEN_POSITIONS) >= MAX_OPEN_TRADES:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "Buy":
        sl = precio - atr
        tp = precio + (atr * 2)
    else:
        sl = precio + atr
        tp = precio - (atr * 2)

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

def paper_revisar_posiciones(precio_actual):
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
    telegram_mensaje("🤖 BOT V90.2 INICIADO (SIN IA)\n"
                     f"📊 Velas: {INTERVAL}m | Heartbeat cada 5min | Máx. posiciones: {MAX_OPEN_TRADES}")
    ultima_fecha = None

    while True:
        try:
            # 1. Obtener velas
            df = obtener_velas()
            df = calcular_indicadores(df)
            estado = extraer_estado_mercado(df)

            # 2. Noticias y sentimiento (desde caché)
            titulo, fuente, sent_label, sent_score = obtener_noticias_y_sentimiento()

            # 3. Heartbeat
            enviar_heartbeat(estado['precio'], titulo, fuente, sent_label, sent_score)

            # 4. Decisión técnica
            decision, soporte, resistencia, razones = motor_v90(estado)

            # 5. Filtro fundamental
            filtro_ok = True
            motivo_filtro = "Sin filtro"
            if decision:
                filtro_ok, motivo_filtro = filtrar_por_fundamental(decision, sent_label)
                if not filtro_ok:
                    decision = None

            # 6. Log
            num_abiertas = len(OPEN_POSITIONS)
            log_estado(estado, decision, razones, filtro_ok, motivo_filtro, num_abiertas)

            # 7. Ejecutar operación
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

            # 8. Revisar SL/TP
            precio_actual = estado['precio']
            paper_revisar_posiciones(precio_actual)

            # 9. Reset diario
            fecha_hoy = datetime.now(timezone.utc).date()
            if ultima_fecha is None:
                ultima_fecha = fecha_hoy
            elif fecha_hoy != ultima_fecha:
                ultima_fecha = fecha_hoy
                logger.info("Nuevo día.")

            # 10. Esperar 5 min
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
