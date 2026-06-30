# ============================================================
# BOT TRADING V90.2 – HÍBRIDO DETERMINISTA (SIN IA)
# ============================================================
# - Velas de 30 minutos, máximo 3 operaciones abiertas
# - Reglas: soporte/resistencia, EMA, patrones de velas
# - Filtro fundamental con cryptocurrency.cv (sentimiento gratuito)
# - Heartbeat en Telegram cada 5 minutos con noticia y sentimiento
# - Gráfico con fondo negro, flecha de entrada, todos los niveles
# - Mensajes detallados por Telegram
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
from datetime import datetime, timezone
import logging

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
LEVERAGE = 1
SLEEP_SECONDS = 300            # 5 minutos (heartbeat cada 5 min)

# Gráficos
GRAFICO_VELAS_LIMIT = 120
MOSTRAR_EMA20 = True
MOSTRAR_ATR = False

# ============================================================
# PAPER TRADING (SIMULACIÓN) – GESTIÓN DE POSICIONES
# ============================================================
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
OPEN_POSITIONS = []   # Cada posición es un dict con: decision, entry_price, sl, tp, size_btc, size_usd, razones, timestamp

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

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

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

    # Regla dinámica: si precio > resistencia_anterior, la resistencia se convierte en soporte
    if precio > max_50:
        soporte = max_50
        resistencia = df['close'].rolling(ventana).max().iloc[-1]  # puede ser el mismo precio
    else:
        soporte = min_50
        resistencia = max_50

    # EMA como nivel adicional
    if precio > ema20:
        ema_nivel = 'soporte'
    else:
        ema_nivel = 'resistencia'

    # Análisis de la última vela (patrón)
    open_actual = df['open'].iloc[-1]
    high_actual = df['high'].iloc[-1]
    low_actual = df['low'].iloc[-1]
    close_actual = df['close'].iloc[-1]
    rango = high_actual - low_actual
    cuerpo = abs(close_actual - open_actual)
    cuerpo_relativo = cuerpo / rango if rango > 0 else 0.0
    sombra_superior = high_actual - max(open_actual, close_actual)
    sombra_inferior = min(open_actual, close_actual) - low_actual

    # Identificar patrones básicos
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

    # Tendencia
    slope, intercept, tendencia = _detectar_tendencia(df)

    # Sentimiento técnico
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
# MOTOR DE DECISIÓN V90 MEJORADO (con EMA y patrones)
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

    # ---- Regla 1: Soporte + tendencia alcista ----
    if (abs(precio - soporte) < atr) and (tendencia == '📈 ALCISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de soporte ({soporte:.2f})")
        razones.append("Tendencia alcista o lateral")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    # ---- Regla 2: Resistencia + tendencia bajista ----
    if (abs(precio - resistencia) < atr) and (tendencia == '📉 BAJISTA' or tendencia == '➡️ LATERAL'):
        razones.append(f"Precio cerca de resistencia ({resistencia:.2f})")
        razones.append("Tendencia bajista o lateral")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    # ---- Regla 3: EMA como resistencia (precio toca EMA desde abajo y no la rompe) ----
    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and tendencia != '📈 ALCISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde abajo (EMA actúa como resistencia)")
        razones.append("Sin ruptura al alza")
        if 'bajista' in patron.lower() or 'estrella fugaz' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Sell', soporte, resistencia, razones

    # ---- Regla 4: EMA como soporte (precio toca EMA desde arriba y no la rompe) ----
    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and tendencia != '📉 BAJISTA':
        razones.append(f"Precio tocando EMA20 ({ema20:.2f}) desde arriba (EMA actúa como soporte)")
        razones.append("Sin ruptura a la baja")
        if 'alcista' in patron.lower() or 'martillo' in patron.lower():
            razones.append(f"Patrón de vela: {patron}")
        return 'Buy', soporte, resistencia, razones

    # ---- Regla 5: Ruptura de EMA ----
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
# FILTRO FUNDAMENTAL CON cryptocurrency.cv (sin API key)
# ============================================================
def obtener_noticias_y_sentimiento(ticker="BTC"):
    """
    Obtiene la última noticia y el sentimiento global desde cryptocurrency.cv.
    Retorna: (titulo_noticia, sentimiento_label, sentimiento_score, source)
    """
    try:
        # 1. Obtener última noticia
        url_news = f"https://cryptocurrency.cv/api/news?ticker={ticker}&limit=1"
        r_news = requests.get(url_news, timeout=10)
        data_news = r_news.json()
        if data_news and isinstance(data_news, list) and len(data_news) > 0:
            noticia = data_news[0]
            titulo = noticia.get('title', 'Sin título')
            fuente = noticia.get('source', 'Desconocida')
        else:
            titulo = "No hay noticias recientes"
            fuente = ""

        # 2. Obtener sentimiento
        url_sent = f"https://cryptocurrency.cv/api/ai/sentiment?asset={ticker}"
        r_sent = requests.get(url_sent, timeout=10)
        data_sent = r_sent.json()
        label = data_sent.get('label', 'Neutral')
        score = data_sent.get('score', 0.0)

        logger.info(f"📰 Última noticia: {titulo} | Sentimiento: {label} ({score:.2f})")
        return titulo, label, score, fuente

    except Exception as e:
        logger.error(f"Error al obtener noticias/sentimiento: {e}")
        return "Error al obtener noticias", "Neutral", 0.0, ""

def filtrar_por_fundamental(decision, sentimiento_label):
    """
    Aplica filtro fundamental basado en el sentimiento de noticias.
    Retorna (permitido, motivo).
    """
    if sentimiento_label == 'Bearish' and decision == 'Buy':
        return False, f"Noticias bajistas → bloqueo LONG"
    if sentimiento_label == 'Bullish' and decision == 'Sell':
        return False, f"Noticias alcistas → bloqueo SHORT"
    return True, f"Sentimiento permitido ({sentimiento_label})"

# ============================================================
# GRÁFICO DE VELAS JAPONESAS CON FONDO NEGRO Y FLECHA
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

        # Dibujar velas
        for i in range(len(df_plot)):
            color = 'lime' if closes[i] >= opens[i] else 'red'
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)
            cuerpo_y = min(opens[i], closes[i])
            cuerpo_h = abs(closes[i] - opens[i])
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
            rect = plt.Rectangle((x[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)

        # Niveles
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")

        # EMA20
        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')

        # Línea de tendencia
        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot, r_plot, _, _ = linregress(x_plot, y_plot)
        tendencia_linea = intercept_plot + slope_plot * x_plot
        ax.plot(x_plot, tendencia_linea, color='white', linewidth=1.5, linestyle='-',
                label=f"Tendencia slope {slope_plot:.4f}")

        # Marcar entrada (última vela)
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

        # Texto informativo en el gráfico
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
        logger.info(f"Límite de {MAX_OPEN_TRADES} posiciones abiertas alcanzado.")
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "Buy":
        sl = precio - atr
        tp = precio + (atr * 2)
    elif decision == "Sell":
        sl = precio + atr
        tp = precio - (atr * 2)
    else:
        return False

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
    logger.info(f"📌 Posición abierta: {decision} a {precio:.2f} (SL={sl:.2f}, TP={tp:.2f})")
    return True

def paper_calcular_pnl(posicion, precio_actual):
    if posicion['decision'] == "Buy":
        return (precio_actual - posicion['entry_price']) * posicion['size_btc']
    else:
        return (posicion['entry_price'] - precio_actual) * posicion['size_btc']

def paper_revisar_posiciones(precio_actual, df):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    global PAPER_BALANCE_MAX, PAPER_MAX_DRAWDOWN, OPEN_POSITIONS

    posiciones_a_cerrar = []
    for i, pos in enumerate(OPEN_POSITIONS):
        pnl = paper_calcular_pnl(pos, precio_actual)
        cerrar = False
        motivo = ""
        if pos['decision'] == "Buy":
            if precio_actual <= pos['sl']:
                cerrar = True
                motivo = "SL"
            elif precio_actual >= pos['tp']:
                cerrar = True
                motivo = "TP"
        else:  # Sell
            if precio_actual >= pos['sl']:
                cerrar = True
                motivo = "SL"
            elif precio_actual <= pos['tp']:
                cerrar = True
                motivo = "TP"

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
            f"🧠 Razones entrada: {', '.join(pos['razones'])}"
        )
        telegram_mensaje(mensaje_cierre)
        OPEN_POSITIONS.pop(i)

    return len(posiciones_a_cerrar) > 0

# ============================================================
# HEARTBEAT – mensaje cada 5 minutos por Telegram
# ============================================================
ultimo_heartbeat = 0

def enviar_heartbeat(precio, noticia_titulo, sentimiento_label, sentimiento_score, fuente):
    global ultimo_heartbeat
    ahora = time.time()
    if ahora - ultimo_heartbeat >= 300:  # 5 minutos
        mensaje = (
            f"🔄 HEARTBEAT - Bot activo\n"
            f"📰 Última noticia: {noticia_titulo} (fuente: {fuente})\n"
            f"🧠 Sentimiento: {sentimiento_label} (score: {sentimiento_score:.2f})\n"
            f"💰 Precio BTC: {precio:.2f}\n"
            f"⏳ Esperando formación de vela para análisis técnico...\n"
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
    logger.info(f"📏 Patrón de vela: {estado['patron']}")
    logger.info(f"🧠 Sentimiento técnico: {estado['sentimiento']} ({'Alcista' if estado['sentimiento']>0 else 'Bajista' if estado['sentimiento']<0 else 'Neutral'})")
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
            # 1. Obtener datos de mercado (velas de 30 min)
            df = obtener_velas()
            df = calcular_indicadores(df)

            # 2. Extraer estado técnico
            estado = extraer_estado_mercado(df)

            # 3. Obtener noticias y sentimiento (fundamental)
            titulo, sent_label, sent_score, fuente = obtener_noticias_y_sentimiento("BTC")

            # 4. Enviar heartbeat cada 5 minutos
            enviar_heartbeat(estado['precio'], titulo, sent_label, sent_score, fuente)

            # 5. Decisión del motor V90 (técnico)
            decision, soporte, resistencia, razones = motor_v90(estado)

            # 6. Aplicar filtro fundamental (si hay decisión)
            filtro_ok = True
            motivo_filtro = "Sin filtro"
            if decision:
                filtro_ok, motivo_filtro = filtrar_por_fundamental(decision, sent_label)
                if not filtro_ok:
                    decision = None

            # 7. Log en consola
            num_abiertas = len(OPEN_POSITIONS)
            log_estado(estado, decision, razones, filtro_ok, motivo_filtro, num_abiertas)

            # 8. Ejecutar operación si procede
            if decision and num_abiertas < MAX_OPEN_TRADES:
                precio = estado['precio']
                atr_actual = estado['atr']
                tiempo_actual = estado['fecha']

                apertura = paper_abrir_posicion(
                    decision=decision,
                    precio=precio,
                    atr=atr_actual,
                    soporte=soporte,
                    resistencia=resistencia,
                    razones=razones,
                    tiempo=tiempo_actual,
                    estado=estado
                )

                if apertura:
                    # Mensaje de entrada detallado
                    pnl_flotante = paper_calcular_pnl(OPEN_POSITIONS[-1], precio)
                    mensaje_entrada = (
                        f"📌 ENTRADA PAPER {decision}\n"
                        f"💰 Precio: {precio:.2f}\n"
                        f"📍 SL: {OPEN_POSITIONS[-1]['sl']:.2f} | TP: {OPEN_POSITIONS[-1]['tp']:.2f}\n"
                        f"📦 Size USD: {OPEN_POSITIONS[-1]['size_usd']:.2f} | Size BTC: {OPEN_POSITIONS[-1]['size_btc']:.6f}\n"
                        f"💵 Balance: {PAPER_BALANCE:.2f} USD\n"
                        f"📈 PnL flotante: {pnl_flotante:.4f} USD\n"
                        f"📊 PnL Global: {PAPER_PNL_GLOBAL:.4f} USD\n"
                        f"🧠 Razones técnicas:\n• " + "\n• ".join(razones) + "\n"
                        f"📊 Patrón de vela: {estado['patron']}\n"
                        f"📈 Tendencia: {estado['tendencia']}\n"
                        f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}\n"
                        f"📉 EMA20: {estado['ema20']:.2f} (actúa como {estado['ema_nivel']})\n"
                        f"🔒 Filtro fundamental: {motivo_filtro}\n"
                        f"📊 Posiciones abiertas: {len(OPEN_POSITIONS)}/{MAX_OPEN_TRADES}"
                    )
                    telegram_mensaje(mensaje_entrada)

                    # Gráfico
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
            else:
                if num_abiertas >= MAX_OPEN_TRADES:
                    logger.info(f"Límite de {MAX_OPEN_TRADES} posiciones abiertas alcanzado.")

            # 9. Revisar SL/TP de todas las posiciones abiertas
            precio_actual = estado['precio']
            se_cerro = paper_revisar_posiciones(precio_actual, df)

            # 10. Reset diario (opcional)
            fecha_hoy = datetime.now(timezone.utc).date()
            if ultima_fecha is None:
                ultima_fecha = fecha_hoy
            elif fecha_hoy != ultima_fecha:
                ultima_fecha = fecha_hoy
                logger.info("Nuevo día.")

            # 11. Esperar 5 minutos para el próximo ciclo
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
