# ============================================================
# BOT TRADING V90.7 – VERSIÓN REAL (CON GRÁFICOS DE SALIDA)
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
TRAILING_OFFSET_ATR = 0.75
SL_MULTIPLIER = 3.0

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
# FUNCIONES API BYBIT
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
    params = {"category": "linear", "symbol": SYMBOL}
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
    return bybit_request(endpoint, method='POST', payload=payload)

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
    return bybit_request(endpoint, method='POST', payload=payload)

def crear_orden_stop_market(symbol, side, qty, stop_price, reduce_only=True):
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
    return bybit_request(endpoint, method='POST', payload=payload)

def cancelar_orden(order_id, symbol):
    endpoint = "/v5/order/cancel"
    payload = {"category": "linear", "symbol": symbol, "orderId": order_id}
    return bybit_request(endpoint, method='POST', payload=payload)

def modificar_orden_stop(order_id, symbol, stop_price):
    endpoint = "/v5/order/amend"
    payload = {"category": "linear", "symbol": symbol, "orderId": order_id, "triggerPrice": str(stop_price)}
    return bybit_request(endpoint, method='POST', payload=payload)

# ============================================================
# OBTENER VELAS, INDICADORES, ESTADO
# ============================================================
def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {"category": "linear", "symbol": SYMBOL, "interval": INTERVAL, "limit": limit}
    r = requests.get(url, params=params, timeout=20)
    data_json = r.json()
    data = data_json["result"]["list"][::-1]
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
    fila = df.iloc[idx]
    precio = fila['close']
    ema20 = df['ema20'].iloc[idx]
    atr = df['atr'].iloc[idx]
    if pd.isna(ema20): ema20 = precio
    if pd.isna(atr): atr = precio * 0.01

    ventana = min(50, len(df))
    if ventana < 2:
        min_50, max_50 = df['close'].min(), df['close'].max()
    else:
        min_50, max_50 = df['close'].iloc[-ventana:].min(), df['close'].iloc[-ventana:].max()

    soporte = min_50 if precio <= max_50 else max_50
    resistencia = max_50
    if soporte == resistencia:
        soporte, resistencia = precio * 0.99, precio * 1.01

    ema_nivel = 'soporte' if precio > ema20 else 'resistencia'
    open_actual, high_actual, low_actual, close_actual = fila['open'], fila['high'], fila['low'], fila['close']
    rango = high_actual - low_actual
    cuerpo = abs(close_actual - open_actual)
    cuerpo_relativo = cuerpo / rango if rango > 0 else 0.0
    sombra_superior = high_actual - max(open_actual, close_actual)
    sombra_inferior = min(open_actual, close_actual) - low_actual

    patron = "Vela normal"
    if cuerpo_relativo > 0.6: patron = "Vela alcista de cuerpo grande" if close_actual > open_actual else "Vela bajista de cuerpo grande"
    elif sombra_inferior > 2 * cuerpo and sombra_superior < cuerpo: patron = "Martillo (posible reversión alcista)"
    elif sombra_superior > 2 * cuerpo and sombra_inferior < cuerpo: patron = "Estrella fugaz (posible reversión bajista)"
    elif cuerpo_relativo < 0.2 and sombra_superior > 0 and sombra_inferior > 0: patron = "Doji (indecisión)"

    slope, intercept, tendencia = _detectar_tendencia(df, idx)

    return {
        'precio': precio, 'ema20': ema20, 'atr': atr, 'soporte': soporte, 'resistencia': resistencia,
        'ema_nivel': ema_nivel, 'cuerpo_relativo': cuerpo_relativo, 'patron': patron,
        'slope': slope, 'intercept': intercept, 'tendencia': tendencia,
        'sentimiento': 1 if slope > 0.02 else (-1 if slope < -0.02 else 0),
        'fecha': df.index[idx], 'open': open_actual, 'high': high_actual, 'low': low_actual,
        'close': close_actual, 'sombra_superior': sombra_superior, 'sombra_inferior': sombra_inferior, 'idx': idx
    }

def _detectar_tendencia(df, idx, ventana=80):
    inicio = max(0, idx - ventana + 1)
    y = df['close'].values[inicio:idx+1]
    if len(y) < 2: return 0, 0, "➡️ LATERAL"
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)
    direccion = '📈 ALCISTA' if slope > 0.02 else ('📉 BAJISTA' if slope < -0.02 else '➡️ LATERAL')
    return slope, intercept, direccion

def extraer_estado_mercado_por_indice(df, idx):
    if df.empty or idx < 0 or idx >= len(df): return None
    return extraer_estado_mercado(df.iloc[:idx+1], usar_cerrada=False)

# ============================================================
# PATRONES Y MOTOR DE DECISIÓN
# ============================================================
def detectar_patron_multivela(df, n=3):
    if len(df) < n: return None, ""
    closes, opens = df['close'].iloc[-n:].values, df['open'].iloc[-n:].values
    if all(closes[i] > opens[i] and closes[i] > closes[i-1] for i in range(1, n)): return "tres_soldados_blancos", "Alcista fuerte (continuación)"
    if all(closes[i] < opens[i] and closes[i] < closes[i-1] for i in range(1, n)): return "tres_cuervos_negros", "Bajista fuerte (continuación)"
    return None, ""

def motor_v90(estado_actual, df):
    if estado_actual is None: return None, 0, 0, ["Estado nulo"]
    precio, soporte, resistencia, atr = estado_actual['precio'], estado_actual['soporte'], estado_actual['resistencia'], estado_actual['atr']
    tendencia, ema20, ema_nivel, patron = estado_actual['tendencia'], estado_actual['ema20'], estado_actual['ema_nivel'], estado_actual['patron']
    razones = []

    patron_mult, desc_mult = detectar_patron_multivela(df)
    if patron_mult == "tres_soldados_blancos" and tendencia in ['📈 ALCISTA', '➡️ LATERAL']:
        razones.extend([desc_mult, "Tres soldados blancos en tendencia favorable"])
        return 'Buy', soporte, resistencia, razones
    if patron_mult == "tres_cuervos_negros" and tendencia in ['📉 BAJISTA', '➡️ LATERAL']:
        razones.extend([desc_mult, "Tres cuervos negros en tendencia favorable"])
        return 'Sell', soporte, resistencia, razones

    if "Martillo" in patron and (tendencia == '📉 BAJISTA' or abs(precio - soporte) < atr):
        razones.extend([f"Reversión alcista: {patron}", f"Contexto: {tendencia} | cerca soporte {soporte:.2f}"])
        return 'Buy', soporte, resistencia, razones
    if "Estrella fugaz" in patron and (tendencia == '📈 ALCISTA' or abs(precio - resistencia) < atr):
        razones.extend([f"Reversión bajista: {patron}", f"Contexto: {tendencia} | cerca resistencia {resistencia:.2f}"])
        return 'Sell', soporte, resistencia, razones

    dist_soporte, dist_resistencia = abs(precio - soporte), abs(precio - resistencia)
    if dist_soporte < 0.5 * atr:
        senal_bajista_fuerte = False
        if patron == "Vela bajista de cuerpo grande" and estado_actual['close'] < estado_actual['open']: senal_bajista_fuerte = True
        if "estrella fugaz" in patron.lower() and estado_actual['close'] < estado_actual['open']: senal_bajista_fuerte = True
        if tendencia == '📉 BAJISTA' and estado_actual['close'] < soporte: senal_bajista_fuerte = True
        
        if senal_bajista_fuerte:
            razones.append("Señal bajista fuerte en soporte → NO COMPRAMOS")
            if estado_actual['close'] < soporte - 0.3 * atr:
                razones.append("Ruptura confirmada del soporte → SELL")
                return 'Sell', soporte, resistencia, razones
        else:
            razones.append(f"Precio cerca de soporte ({soporte:.2f}) sin señales bajistas → BUY")
            return 'Buy', soporte, resistencia, razones

    if dist_resistencia < 0.5 * atr:
        senal_alcista_fuerte = False
        if patron == "Vela alcista de cuerpo grande" and estado_actual['close'] > estado_actual['open']: senal_alcista_fuerte = True
        if "martillo" in patron.lower() and estado_actual['close'] > estado_actual['open']: senal_alcista_fuerte = True
        if tendencia == '📈 ALCISTA' and estado_actual['close'] > resistencia: senal_alcista_fuerte = True
        
        if senal_alcista_fuerte:
            razones.append("Señal alcista fuerte en resistencia → NO VENDEMOS")
            if estado_actual['close'] > resistencia + 0.3 * atr:
                razones.append("Ruptura confirmada de resistencia → BUY")
                return 'Buy', soporte, resistencia, razones
        else:
            razones.append(f"Precio cerca de resistencia ({resistencia:.2f}) sin señales alcistas → SELL")
            return 'Sell', soporte, resistencia, razones

    if ema_nivel == 'resistencia' and abs(precio - ema20) < atr * 0.5 and tendencia != '📈 ALCISTA':
        razones.append(f"Rebote bajista en EMA20 ({ema20:.2f})")
        return 'Sell', soporte, resistencia, razones
    if ema_nivel == 'soporte' and abs(precio - ema20) < atr * 0.5 and tendencia != '📉 BAJISTA':
        razones.append(f"Rebote alcista en EMA20 ({ema20:.2f})")
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
    if NEWS_CACHE["timestamp"] is not None and (ahora - NEWS_CACHE["timestamp"]).total_seconds() < NEWS_CACHE_TTL: return
    logger.info("⏳ Actualizando caché de noticias...")
    titulo, fuente, sent_label, sent_score = "No disponible", "Ninguna", "Neutral", 0.0
    NEWS_CACHE = {"titulo": titulo, "fuente": fuente, "sent_label": sent_label, "sent_score": sent_score, "timestamp": ahora}

def obtener_noticias_y_sentimiento():
    actualizar_cache_noticias()
    return NEWS_CACHE["titulo"], NEWS_CACHE["fuente"], NEWS_CACHE["sent_label"], NEWS_CACHE["sent_score"]

def filtrar_por_fundamental(decision, sent_label, estado):
    return True, "Filtro fundamental omitido / Permitido"

# ============================================================
# GRÁFICO MEJORADO (ENTRADAS Y SALIDAS)
# ============================================================
def generar_grafico_trade(df, decision, soporte, resistencia, slope, intercept, razones, estado, precio_entrada, precio_salida=None, tiempo_entrada=None, trade_id=None, motivo_cierre=None):
    if df.empty or estado is None:
        return None
    try:
        plt.style.use('dark_background')
        df_plot = df.copy().tail(GRAFICO_VELAS_LIMIT)
        if df_plot.empty: return None
        
        times = df_plot.index
        opens, highs, lows, closes = df_plot['open'].values, df_plot['high'].values, df_plot['low'].values, df_plot['close'].values
        x = np.arange(len(df_plot))
        
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='black')
        ax.set_facecolor('black')
        
        # Dibujar velas
        for i in range(len(df_plot)):
            color = 'lime' if closes[i] >= opens[i] else 'red'
            ax.vlines(x[i], lows[i], highs[i], color=color, linewidth=1)
            cuerpo_y, cuerpo_h = min(opens[i], closes[i]), max(abs(closes[i] - opens[i]), 0.0001)
            rect = plt.Rectangle((x[i] - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)
            
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")
        if MOSTRAR_EMA20 and 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')

        # Buscar posición X de la entrada
        entrada_x = 0
        if tiempo_entrada is not None and tiempo_entrada in times:
            entrada_x = np.where(times == tiempo_entrada)[0][0]
        else:
            entrada_x = len(df_plot) - 1 if precio_salida is None else 0

        # Dibujar Entrada
        if decision == 'Buy':
            ax.scatter(entrada_x, precio_entrada, s=250, marker='^', color='lime', edgecolors='black', linewidths=2, label='Entrada BUY', zorder=5)
        else:
            ax.scatter(entrada_x, precio_entrada, s=250, marker='v', color='red', edgecolors='black', linewidths=2, label='Entrada SELL', zorder=5)

        # Dibujar Salida y línea de conexión
        if precio_salida is not None:
            salida_x = len(df_plot) - 1
            pnl_indicador = (precio_salida - precio_entrada) if decision == 'Buy' else (precio_entrada - precio_salida)
            color_salida = 'lime' if pnl_indicador > 0 else ('red' if pnl_indicador < 0 else 'yellow')
            
            ax.scatter(salida_x, precio_salida, s=250, marker='X', color=color_salida, edgecolors='white', linewidths=1.5, label='Cierre', zorder=6)
            ax.plot([entrada_x, salida_x], [precio_entrada, precio_salida], color='white', linestyle=':', linewidth=2, alpha=0.7)

        # Textos informativos
        id_text = f" ID: {trade_id}" if trade_id else ""
        texto = (
            f"{decision.upper()}{id_text}\n"
            f"Entrada: {precio_entrada:.2f}\n"
            f"EMA20: {estado['ema20']:.2f}  ATR: {estado['atr']:.2f}\n"
            f"Razones: {', '.join(razones)}"
        )
        if precio_salida is not None:
            texto += f"\nSalida: {precio_salida:.2f}\nMotivo: {motivo_cierre}"
            
        ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top', color='white',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
        
        titulo_graf = f"{SYMBOL} - Velas {INTERVAL}m - {'Entrada' if precio_salida is None else 'Cierre de Posición'}"
        ax.set_title(titulo_graf, color='white')
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
# GESTIÓN DE POSICIONES REALES
# ============================================================
def abrir_posicion_real(decision, precio, atr, soporte, resistencia, razones, tiempo, estado, df):
    global TRADE_COUNTER, ACTIVE_TRADES

    if len(ACTIVE_TRADES) >= MAX_OPEN_TRADES: return None

    qty = QTY_BTC
    side = "Buy" if decision == "Buy" else "Sell"

    try:
        crear_orden_market(SYMBOL, side, qty, reduce_only=False)
        time.sleep(2)
        pos_actual = next((p for p in obtener_posiciones_abiertas() if p.get('side') == side.capitalize()), None)
        if not pos_actual: raise Exception("No se encontró la posición recién abierta")
        entry_price = float(pos_actual.get('avgPrice', precio))
    except Exception as e:
        logger.error(f"Error abriendo posición: {e}")
        return None

    if decision == "Buy":
        sl_price = round(entry_price - SL_MULTIPLIER * atr, 2)
        tp1_price = round(min(resistencia, entry_price + 2.0 * atr), 2)
        tp2_price = round(entry_price + 3.5 * atr, 2)
        if tp1_price <= entry_price: tp1_price = round(entry_price + 2.0 * atr, 2)
    else:
        sl_price = round(entry_price + SL_MULTIPLIER * atr, 2)
        tp1_price = round(max(soporte, entry_price - 2.0 * atr), 2)
        tp2_price = round(entry_price - 3.5 * atr, 2)
        if tp1_price >= entry_price: tp1_price = round(entry_price - 2.0 * atr, 2)

    try: tp1_order_id = crear_orden_limit(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty / 2, tp1_price, reduce_only=True).get('orderId')
    except: tp1_order_id = None

    try: sl_order_id = crear_orden_stop_market(SYMBOL, 'Sell' if decision=='Buy' else 'Buy', qty, sl_price, reduce_only=True).get('orderId')
    except: sl_order_id = None

    TRADE_COUNTER += 1
    trade_info = {
        'trade_id': TRADE_COUNTER, 'side': side, 'entry_price': entry_price, 'qty_total': qty, 'qty_remaining': qty,
        'sl_price': sl_price, 'tp1_price': tp1_price, 'tp2_price': tp2_price, 'status': 'ACTIVE',
        'order_id_sl': sl_order_id, 'order_id_tp1': tp1_order_id, 'order_id_tp2': None,
        'razones': razones, 'estado_entrada': estado.copy(), 'timestamp': tiempo
    }
    ACTIVE_TRADES.append(trade_info)

    telegram_mensaje(f"📌 ENTRADA REAL #{TRADE_COUNTER} {decision}\n💰 Precio: {entry_price:.2f}\n📍 SL: {sl_price:.2f} | TP1: {tp1_price:.2f}\n🧠 " + "\n• ".join(razones))

    fig = generar_grafico_trade(df, decision, soporte, resistencia, estado['slope'], estado['intercept'], razones, estado, precio_entrada=entry_price, tiempo_entrada=tiempo, trade_id=TRADE_COUNTER)
    if fig:
        telegram_grafico(fig)
        plt.close(fig)

    return trade_info

def actualizar_estadisticas(pnl):
    global TRADES_TOTALES, TRADES_WIN, TRADES_LOSS, PNL_GLOBAL
    TRADES_TOTALES += 1; PNL_GLOBAL += pnl
    if pnl > 0: TRADES_WIN += 1
    else: TRADES_LOSS += 1

def revisar_posiciones_reales(precio_actual, df_actual, noticia_titulo, noticia_fuente, sent_label, sent_score):
    global ACTIVE_TRADES
    pos_map = {p.get('side', '').lower(): p for p in obtener_posiciones_abiertas() if float(p.get('size', 0)) != 0}
    trades_a_remover = []

    for idx, trade in enumerate(ACTIVE_TRADES):
        side, entry, qty_remaining, status = trade['side'], trade['entry_price'], trade['qty_remaining'], trade['status']
        trade_id, tp1_price, tp2_price, sl_price = trade['trade_id'], trade['tp1_price'], trade['tp2_price'], trade['sl_price']
        pos_api = pos_map.get(side.lower())

        # ==========================================
        # 1. CIERRE TOTAL DETECTADO (No está en la API)
        # ==========================================
        if not pos_api or float(pos_api.get('size', 0)) == 0:
            if qty_remaining > 0:
                exit_price = sl_price
                if status == 'ACTIVE':
                    motivo, icono = "Stop Loss Original", "🔴"
                elif status == 'TP1_HIT':
                    motivo, icono = "Stop Loss (BreakEven)", "🟡"
                elif status == 'TP2_HIT':
                    motivo, icono = "Trailing Stop Loss", "🟢"
                else:
                    motivo, icono, exit_price = "Desconocido/Manual", "⚪", precio_actual

                pnl = (exit_price - entry) * qty_remaining if side == 'Buy' else (entry - exit_price) * qty_remaining
                actualizar_estadisticas(pnl)

                mensaje = (
                    f"{icono} POSICIÓN #{trade_id} CERRADA TOTALMENTE\n"
                    f"📝 Motivo: {motivo}\n"
                    f"🎯 Precio Entrada: {entry:.2f}\n"
                    f"🛑 Precio Salida: {exit_price:.2f}\n"
                    f"📊 PnL de este tramo: {pnl:.4f} USD"
                )
                telegram_mensaje(mensaje)

                fig = generar_grafico_trade(
                    df_actual, side, trade['estado_entrada']['soporte'], trade['estado_entrada']['resistencia'], 
                    trade['estado_entrada']['slope'], trade['estado_entrada']['intercept'], trade['razones'], 
                    trade['estado_entrada'], precio_entrada=entry, precio_salida=exit_price, 
                    tiempo_entrada=trade['timestamp'], trade_id=trade_id, motivo_cierre=motivo
                )
                if fig:
                    telegram_grafico(fig)
                    plt.close(fig)

            trades_a_remover.append(idx)
            continue

        # ==========================================
        # 2. ACTIVE: Buscando TP1
        # ==========================================
        if status == 'ACTIVE':
            if (side == 'Buy' and precio_actual >= tp1_price) or (side == 'Sell' and precio_actual <= tp1_price):
                qty_cerrar = trade['qty_total'] / 2
                if qty_cerrar > qty_remaining: qty_cerrar = qty_remaining
                if qty_cerrar > 0:
                    try:
                        crear_orden_market(SYMBOL, 'Sell' if side == 'Buy' else 'Buy', qty_cerrar, reduce_only=True)
                        trade['qty_remaining'] -= qty_cerrar
                        pnl = (tp1_price - entry) * qty_cerrar if side == 'Buy' else (entry - tp1_price) * qty_cerrar
                        actualizar_estadisticas(pnl)

                        if trade['order_id_sl']: cancelar_orden(trade['order_id_sl'], SYMBOL)
                        trade['order_id_sl'] = crear_orden_stop_market(SYMBOL, 'Sell' if side == 'Buy' else 'Buy', trade['qty_remaining'], entry, reduce_only=True).get('orderId')
                        trade['order_id_tp2'] = crear_orden_limit(SYMBOL, 'Sell' if side == 'Buy' else 'Buy', trade['qty_remaining'], tp2_price, reduce_only=True).get('orderId')
                        
                        trade['status'] = 'TP1_HIT'
                        trade['sl_price'] = entry

                        telegram_mensaje(f"🔓 CIERRE PARCIAL #{trade_id} (TP1 Alcanzado)\n💰 Salida: {tp1_price:.2f}\n📊 PnL Parcial: {pnl:.4f} USD\n🔄 SL movido a BE ({entry:.2f})")
                        
                        fig = generar_grafico_trade(
                            df_actual, side, trade['estado_entrada']['soporte'], trade['estado_entrada']['resistencia'], 
                            trade['estado_entrada']['slope'], trade['estado_entrada']['intercept'], trade['razones'], 
                            trade['estado_entrada'], precio_entrada=entry, precio_salida=tp1_price, 
                            tiempo_entrada=trade['timestamp'], trade_id=trade_id, motivo_cierre="TP1 Alcanzado (Parcial)"
                        )
                        if fig:
                            telegram_grafico(fig)
                            plt.close(fig)
                    except Exception as e: logger.error(f"Error TP1: {e}")

        # ==========================================
        # 3. TP1_HIT: Buscando TP2
        # ==========================================
        elif status == 'TP1_HIT':
            if (side == 'Buy' and precio_actual >= tp2_price) or (side == 'Sell' and precio_actual <= tp2_price):
                try:
                    if trade['order_id_sl']: cancelar_orden(trade['order_id_sl'], SYMBOL)
                    trade['order_id_sl'] = crear_orden_stop_market(SYMBOL, 'Sell' if side == 'Buy' else 'Buy', trade['qty_remaining'], tp2_price, reduce_only=True).get('orderId')
                    trade['sl_price'] = tp2_price
                    trade['status'] = 'TP2_HIT'
                    telegram_mensaje(f"🚀 TP2 ALCANZADO #{trade_id}\n📍 Precio: {tp2_price:.2f}\n📈 TRAILING STOP ACTIVADO. Dejando correr...")
                except Exception as e: logger.error(f"Error SL post-TP2: {e}")

        # ==========================================
        # 4. TP2_HIT: Trailing Stop
        # ==========================================
        elif status == 'TP2_HIT':
            atr = trade['estado_entrada']['atr']
            nuevo_sl = precio_actual - TRAILING_OFFSET_ATR * atr if side == 'Buy' else precio_actual + TRAILING_OFFSET_ATR * atr
            if (side == 'Buy' and nuevo_sl > trade['sl_price']) or (side == 'Sell' and nuevo_sl < trade['sl_price']):
                try:
                    modificar_orden_stop(trade['order_id_sl'], SYMBOL, nuevo_sl)
                    trade['sl_price'] = nuevo_sl
                except Exception as e: logger.error(f"Error trailing SL: {e}")

    for idx in sorted(trades_a_remover, reverse=True):
        del ACTIVE_TRADES[idx]

# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run_bot():
    try: set_leverage(SYMBOL, LEVERAGE)
    except: pass

    telegram_mensaje("🤖 BOT V90.7 REAL INICIADO (MEJORA DE GRÁFICOS Y SALIDAS)")
    ultima_fecha = None

    while True:
        try:
            df = calcular_indicadores(obtener_velas())
            if df.empty:
                time.sleep(SLEEP_SECONDS)
                continue

            estado = extraer_estado_mercado(df, usar_cerrada=True)
            if not estado:
                time.sleep(SLEEP_SECONDS)
                continue

            titulo, fuente, sent_label, sent_score = obtener_noticias_y_sentimiento()
            decision, soporte, resistencia, razones = motor_v90(estado, df)

            logger.info(f"🕒 {estado['fecha']} | 💰 {estado['precio']:.2f} | 🎯 Decisión: {decision if decision else 'NO TRADE'} | Posiciones: {len(ACTIVE_TRADES)}/{MAX_OPEN_TRADES}")

            if decision and len(ACTIVE_TRADES) < MAX_OPEN_TRADES:
                abrir_posicion_real(decision, estado['precio'], estado['atr'], soporte, resistencia, razones, estado['fecha'], estado, df)

            revisar_posiciones_reales(estado['precio'], df, titulo, fuente, sent_label, sent_score)

            fecha_hoy = datetime.now(timezone.utc).date()
            if ultima_fecha is None or fecha_hoy != ultima_fecha:
                ultima_fecha = fecha_hoy
                logger.info("Nuevo día.")

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"🚨 ERROR: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
