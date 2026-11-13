"""
cron_job_worker.py - Worker en segundo plano para ejecucion continua en SAP BTP Cloud Foundry.

Orquesta el ciclo end-to-end de observabilidad y ciberseguridad:
  1. Trigger Programado (Cron / Schedule): Despierta segun el intervalo o cronograma configurado.
  2. Pipeline Matematico (workflow.py): Ingesta, calculo de ventanas temporales,
     deteccion con Isolation Forest, agrupamiento de eventos y clasificacion en SAP HANA.
  3. Capa Forense y Agente MCP: Detecta los incidentes criticos, extrae evidencia granular
     de HANA y genera los reportes Postmortem (persistidos en BD y exportados en PDF).
  4. Gestion de Memoria: Ejecuta recoleccion de basura forzada (gc.collect) para cumplir
     con la cuota de memoria limitada de SAP BTP Cloud Foundry (512MB - 1GB).
"""

import sys
import os
import time
import argparse
import logging
import gc
import schedule
from datetime import datetime
from typing import Optional

# Asegurar resolucion de modulos tanto desde src/ como desde la raiz
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers.conexion import conexion
from helpers.imports import pd

try:
    from mcp_hana_tools import HanaMcpTools
    from agente_forense_postmortem import AgenteForensePostmortem
except ImportError:
    from src.mcp_hana_tools import HanaMcpTools
    from src.agente_forense_postmortem import AgenteForensePostmortem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    force=True
)
log = logging.getLogger('cron_worker')


def ejecutar_ciclo_completo(tabla_fuente: str = 'LOGS_BATCH_1', top_k_postmortems: int = 3) -> bool:
    """
    Ejecuta un ciclo integral: Pipeline de ML en HANA + Capa Agéntica Postmortem con MCP.
    """
    inicio_ciclo = time.perf_counter()
    log.info("[CRON] ========================================================")
    log.info("[CRON] Iniciando ciclo de observabilidad para tabla: %s", tabla_fuente)
    log.info("[CRON] ========================================================")

    conn = None
    try:
        conn = conexion()

        # -------------------------------------------------------------
        # FASE 1: Pipeline de ML y Deteccion de Anomalias
        # -------------------------------------------------------------
        log.info("[CRON][ML] Ejecutando workflow de extraccion, IF y clustering...")
        from workflow import (
            cargar_datos, preparar_semanas, detectar_anomalias,
            generar_eventos, seleccionar_relevantes, normalizar_datasets,
            clasificar, clusterizar, cargar_resultados, resumen_corrida
        )

        df_system, df_llm = cargar_datos(conn, tabla_fuente)
        system_weeks, llm_weeks, semanas = preparar_semanas(conn, df_system, df_llm, tabla_fuente)

        # Deteccion no supervisada con Isolation Forest
        R = detectar_anomalias(conn, semanas, system_weeks, llm_weeks)

        # Agrupamiento de eventos, z-scores, normalizacion y clasificacion
        generar_eventos(R)
        seleccionar_relevantes(R)
        normalizar_datasets(R)
        clasificar(R)
        validacion = clusterizar(conn, R)

        # Cargar resultados analiticos a las tablas RESULT_*_<AAAA_MM>
        carga = cargar_resultados(conn, tabla_fuente, R, semanas, validacion, reset=False)
        log.info("[CRON][ML] Resultados persistidos en HANA exitosamente.")

        # Identificar el mes procesado a partir de las semanas (case-insensitive)
        col_start = [c for c in semanas.columns if 'start' in c.lower()][0]
        mes_procesado = pd.to_datetime(semanas[col_start]).dt.strftime('%Y_%m').iloc[0]
        log.info("[CRON][ML] Mes identificado para postmortems: %s", mes_procesado)

        # -------------------------------------------------------------
        # FASE 2: Capa Agentica Forense (MCP + Postmortems)
        # -------------------------------------------------------------
        log.info("[CRON][AGENTE] Activando Agente Forense con herramientas MCP...")
        mcp = HanaMcpTools(conn=conn)
        agente = AgenteForensePostmortem(mcp_tools=mcp)

        out_pdf_dir = os.getenv('OUTPUT_PDF_DIR', os.path.join(REPO_ROOT, 'outputs', 'postmortems'))
        os.makedirs(out_pdf_dir, exist_ok=True)

        # Obtener candidatos criticos
        candidatos_sys = mcp.get_candidate_incidents(mes=mes_procesado, stream='system', top_k=5)
        candidatos_llm = mcp.get_candidate_incidents(mes=mes_procesado, stream='llm', top_k=5)

        # Seleccionar representativos (ataque de sistema, prompt injection, degradacion)
        seleccionados = []
        if candidatos_sys:
            seleccionados.append(candidatos_sys[0])
        pi_llm = [c for c in candidatos_llm if 'prompt injection' in c['tipo_ataque'].lower()]
        if pi_llm:
            seleccionados.append(pi_llm[0])
        elif candidatos_llm:
            seleccionados.append(candidatos_llm[0])

        log.info("[CRON][AGENTE] Incidentes seleccionados para investigar: %d", len(seleccionados))

        postmortems_generados = 0
        for inc in seleccionados:
            try:
                log.info("[CRON][AGENTE] Investigando Incidente Evento %s (%s)...", inc['evento_id'], inc['tipo_ataque'])
                res = agente.investigar_y_redactar(
                    mes=mes_procesado,
                    incident_data=inc,
                    output_dir=out_pdf_dir
                )
                if res:
                    postmortems_generados += 1
                    log.info("[CRON][OK] Postmortem generado y persistido: %s", res['postmortem_id'])
            except Exception as e_inc:
                log.error("[CRON][ERROR] Error al investigar incidente %s: %s", inc.get('evento_id'), e_inc)

        tiempo_total = time.perf_counter() - inicio_ciclo
        log.info("[CRON] ========================================================")
        log.info("[CRON][SUCCESS] Ciclo completado en %.1f segundos. Postmortems: %d", tiempo_total, postmortems_generados)
        log.info("[CRON] ========================================================")
        return True

    except Exception as e:
        log.error("[CRON][FATAL] Fallo en el ciclo de observabilidad: %s", e, exc_info=True)
        return False

    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        # Liberar memoria de DataFrames grandes para Cloud Foundry
        gc.collect()


def iniciar_daemon(intervalo_minutos: int = 60, tabla: str = 'LOGS_BATCH_1'):
    """
    Inicia el bucle continuo del worker para Cloud Foundry.
    """
    log.info("[DAEMON] Iniciando Worker Daemon en segundo plano.")
    log.info("[DAEMON] Programacion: Cada %d minutos. Tabla objetivo: %s", intervalo_minutos, tabla)

    # 1. Ejecutar una pasada inmediata al arrancar el contenedor
    log.info("[DAEMON] Ejecutando pasada inicial de arranque...")
    ejecutar_ciclo_completo(tabla_fuente=tabla)

    # 2. Configurar el scheduler continuo
    schedule.every(intervalo_minutos).minutes.do(ejecutar_ciclo_completo, tabla_fuente=tabla)

    log.info("[DAEMON] Scheduler activo. Esperando siguiente tick...")
    while True:
        schedule.run_pending()
        time.sleep(10)


def _parse_args():
    p = argparse.ArgumentParser(description='Worker / Cron-Job Daemon para el Pipeline SAP en Cloud Foundry.')
    p.add_argument('--tabla', default=os.getenv('TABLA_DEFAULT', 'LOGS_BATCH_1'),
                   help='Tabla de logs en SAP HANA a procesar.')
    p.add_argument('--run-now', action='store_true',
                   help='Ejecuta un solo ciclo inmediatamente y termina (ideal para pruebas y validaciones).')
    p.add_argument('--intervalo', type=int, default=int(os.getenv('CRON_INTERVAL_MINUTES', '60')),
                   help='Intervalo de ejecucion en minutos para el modo daemon (default: 60 min).')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    if args.run_now:
        log.info("[MODO] Ejecucion unica (--run-now)")
        exito = ejecutar_ciclo_completo(tabla_fuente=args.tabla)
        sys.exit(0 if exito else 1)
    else:
        log.info("[MODO] Ejecucion continua Daemon")
        iniciar_daemon(intervalo_minutos=args.intervalo, tabla=args.tabla)
