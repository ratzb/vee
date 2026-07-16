#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Escáner de seeds BIP39 con notificaciones por Telegram.
Genera seeds aleatorias, verifica su saldo y reporta cada hora.
Uso educativo. No esperes encontrar nada.
"""

import os
import sys
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from bitcoinlib.mnemonic import Mnemonic
from bitcoinlib.keys import Key
from bitcoinlib.services.services import Service

# ============================================================
# CONFIGURACIÓN (desde variables de entorno o valores por defecto)
# ============================================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Ajustes de la búsqueda
MAX_ATTEMPTS = 0                # 0 = infinito (se ejecuta hasta que se detenga)
DELAY_SECONDS = 36              # 100 intentos/hora = 3600/100 = 36 segundos
DERIVATION_PATH = "m/84'/0'/0'/0/0"   # SegWit nativo (cambiar si se necesita)
NETWORK = 'bitcoin'
PROVIDER = 'blockchair'          # Puede ser 'blockchain.info', 'blockcypher', etc.
TIMEOUT = 20                     # Timeout para peticiones HTTP

# Archivos de persistencia
PROGRESS_FILE = "progress.json"
FOUND_FILE = "found_wallets.txt"
LOG_FILE = "scanner.log"

# Intervalo de reporte (en segundos)
REPORT_INTERVAL = 3600          # 1 hora

# ============================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def enviar_telegram(mensaje):
    """Envía un mensaje por Telegram si están configuradas las variables."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado. No se enviará mensaje.")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": mensaje,
            "parse_mode": "HTML"
        }, timeout=10)
        if resp.status_code == 200:
            logger.info("Mensaje Telegram enviado correctamente.")
            return True
        else:
            logger.error(f"Error al enviar Telegram: {resp.status_code} - {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Excepción al enviar Telegram: {e}")
        return False

def cargar_progreso():
    """Carga el progreso guardado (intentos, encontrados, etc.)"""
    default = {
        "intentos": 0,
        "encontrados": 0,
        "errores": 0,
        "ultimo_reporte": None
    }
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r") as f:
                data = json.load(f)
            logger.info("Progreso cargado exitosamente.")
            return data
        except Exception as e:
            logger.warning(f"No se pudo cargar el progreso: {e}. Se inicia desde cero.")
    return default

def guardar_progreso(data):
    """Guarda el progreso en archivo JSON."""
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error al guardar progreso: {e}")

def guardar_wallet_encontrada(seed, direccion, balance_sat):
    """Guarda la wallet encontrada en un archivo de texto."""
    btc = balance_sat / 100_000_000
    mensaje = (
        f"\n{'='*60}\n"
        f"WALLET ENCONTRADA\n"
        f"Fecha: {datetime.now().isoformat()}\n"
        f"Seed: {seed}\n"
        f"Dirección: {direccion}\n"
        f"Balance: {btc:.8f} BTC ({balance_sat} satoshis)\n"
        f"{'='*60}\n"
    )
    try:
        with open(FOUND_FILE, "a") as f:
            f.write(mensaje)
        logger.info(f"Wallet guardada en {FOUND_FILE}")
    except Exception as e:
        logger.error(f"No se pudo guardar la wallet: {e}")

# ============================================================
# FUNCIÓN PRINCIPAL DE VERIFICACIÓN
# ============================================================

def verificar_seed(seed_phrase):
    """
    Deriva la primera dirección y consulta su saldo.
    Retorna (direccion, balance_en_satoshis) o (None, None) si falla.
    """
    try:
        # Convertir seed a bytes
        mnemonic = Mnemonic()
        seed_bytes = mnemonic.to_seed(seed_phrase)
        # Crear clave maestra
        key = Key(seed=seed_bytes, network=NETWORK)
        # Derivar la primera dirección según el path
        key.derive(DERIVATION_PATH)
        direccion = key.address()
        if not direccion:
            logger.warning("No se pudo generar dirección válida.")
            return None, None
        
        # Consultar saldo con el proveedor
        service = Service(provider=PROVIDER, timeout=TIMEOUT)
        balance = service.getbalance(direccion)
        # Si balance es None o negativo, considerar error
        if balance is None or balance < 0:
            logger.warning(f"Saldo no disponible para {direccion}")
            return direccion, None
        return direccion, balance
    except Exception as e:
        logger.debug(f"Error en verificar_seed: {e}")
        return None, None

# ============================================================
# BUCLE PRINCIPAL
# ============================================================

def main():
    logger.info("=" * 60)
    logger.info("INICIANDO ESCÁNER DE SEEDS BIP39")
    logger.info(f"Frecuencia: 1 intento cada {DELAY_SECONDS} segundos (~{3600/DELAY_SECONDS:.0f} intentos/hora)")
    logger.info(f"Ruta de derivación: {DERIVATION_PATH}")
    logger.info(f"Proveedor: {PROVIDER}")
    logger.info(f"Límite de intentos: {'Infinito' if MAX_ATTEMPTS==0 else MAX_ATTEMPTS}")
    logger.info("=" * 60)

    # Cargar progreso previo
    progreso = cargar_progreso()
    intentos = progreso.get("intentos", 0)
    encontrados = progreso.get("encontrados", 0)
    errores = progreso.get("errores", 0)
    ultimo_reporte = progreso.get("ultimo_reporte")

    # Si hay un ultimo_reporte en formato string, convertirlo a datetime
    if ultimo_reporte and isinstance(ultimo_reporte, str):
        try:
            ultimo_reporte = datetime.fromisoformat(ultimo_reporte)
        except:
            ultimo_reporte = None

    # Inicializar variables para el bucle
    mnemonic = Mnemonic()
    proximo_reporte = datetime.now() + timedelta(seconds=REPORT_INTERVAL)
    if ultimo_reporte:
        # Si el último reporte fue hace más de 1 hora, forzar reporte inmediato
        if (datetime.now() - ultimo_reporte).total_seconds() >= REPORT_INTERVAL:
            proximo_reporte = datetime.now()  # para que envíe al inicio
        else:
            proximo_reporte = ultimo_reporte + timedelta(seconds=REPORT_INTERVAL)

    logger.info(f"Próximo reporte de estado aproximadamente a las {proximo_reporte.strftime('%H:%M:%S')}")

    # Bucle principal
    try:
        while MAX_ATTEMPTS == 0 or intentos < MAX_ATTEMPTS:
            intentos += 1

            # Generar seed (12 palabras)
            seed_phrase = mnemonic.generate(strength=128)
            logger.debug(f"Seed generada: {seed_phrase}")

            # Verificar
            direccion, balance = verificar_seed(seed_phrase)

            # Actualizar contadores
            if balance is None:
                errores += 1
                logger.debug(f"[{intentos}] Error | Balance: N/A")
            else:
                if balance > 0:
                    encontrados += 1
                    btc = balance / 100_000_000
                    logger.info(f"🎉 [{intentos}] ¡ENCONTRADA! {direccion} - {btc:.8f} BTC")
                    # Notificación inmediata
                    mensaje = (
                        f"🎉 <b>¡WALLET ENCONTRADA!</b>\n"
                        f"<b>Intento:</b> {intentos}\n"
                        f"<b>Seed:</b> <code>{seed_phrase}</code>\n"
                        f"<b>Dirección:</b> <code>{direccion}</code>\n"
                        f"<b>Balance:</b> {btc:.8f} BTC ({balance} satoshis)"
                    )
                    enviar_telegram(mensaje)
                    guardar_wallet_encontrada(seed_phrase, direccion, balance)
                else:
                    logger.debug(f"[{intentos}] {direccion[:12]}... | Balance: 0 sat")

            # Guardar progreso cada 10 intentos para no perder el contador
            if intentos % 10 == 0:
                progreso["intentos"] = intentos
                progreso["encontrados"] = encontrados
                progreso["errores"] = errores
                guardar_progreso(progreso)

            # Reporte de estado cada hora
            ahora = datetime.now()
            if ahora >= proximo_reporte:
                # Construir mensaje de estado
                status_msg = (
                    f"📊 <b>Estado del escáner</b>\n"
                    f"<b>Intentos totales:</b> {intentos}\n"
                    f"<b>Errores:</b> {errores} ({errores/intentos*100:.1f}%)\n"
                    f"<b>Wallets encontradas:</b> {encontrados}\n"
                    f"<b>Última dirección:</b> {direccion if direccion else 'N/A'}\n"
                    f"<b>En ejecución desde:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
                enviar_telegram(status_msg)
                logger.info("Reporte de estado enviado por Telegram.")
                # Actualizar progreso con la hora del reporte
                progreso["ultimo_reporte"] = datetime.now().isoformat()
                guardar_progreso(progreso)
                # Programar el próximo reporte
                proximo_reporte = ahora + timedelta(seconds=REPORT_INTERVAL)

            # Esperar antes del siguiente intento
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        logger.info("\n⏹️  Escáner detenido por el usuario.")
    except Exception as e:
        logger.error(f"Error inesperado en el bucle principal: {e}")
    finally:
        # Guardar progreso final
        progreso["intentos"] = intentos
        progreso["encontrados"] = encontrados
        progreso["errores"] = errores
        guardar_progreso(progreso)

        # Enviar mensaje de cierre por Telegram
        mensaje_fin = (
            f"🛑 <b>Escáner detenido</b>\n"
            f"<b>Intentos totales:</b> {intentos}\n"
            f"<b>Wallets encontradas:</b> {encontrados}\n"
            f"<b>Errores:</b> {errores}"
        )
        enviar_telegram(mensaje_fin)
        logger.info("Escáner finalizado correctamente.")

if __name__ == "__main__":
    main()
