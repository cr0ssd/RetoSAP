"""
probar_agente_postmortem.py - Script de prueba en produccion para el Servidor MCP y Agente Forense.

Prueba el flujo completo contra SAP HANA Cloud:
  1. Consulta los eventos relevantes detectados por el pipeline en RESULT_EVENTS_<mes>.
  2. Selecciona incidentes representativos (Ataque System, Prompt Injection LLM, Degradacion LLM).
  3. Ejecuta el Agente Forense Postmortem para cada uno (obteniendo evidencia cruda via MCP).
  4. Persiste los reportes en la tabla RESULT_POSTMORTEMS_<mes> en SAP HANA Cloud.
  5. Guarda copias locales en Markdown y PDF en outputs/postmortems/.
"""

import sys
import os
import time
import json
import argparse
import logging

# Asegurar resolucion de modulos tanto desde src/ como desde la raiz
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from mcp_hana_tools import HanaMcpTools
    from agente_forense_postmortem import AgenteForensePostmortem, construir_postmortem_id
    from generador_pdf_postmortem import generar_pdf_postmortem_mensual
except ImportError:
    from src.mcp_hana_tools import HanaMcpTools
    from src.agente_forense_postmortem import AgenteForensePostmortem, construir_postmortem_id
    from src.generador_pdf_postmortem import generar_pdf_postmortem_mensual

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('probar_agente')


def main(mes='2026_01', reset_postmortems=False, top_k=0):
    log.info("[INFO] Iniciando generacion forense multi-escala (MCP + Gemini) para el mes: %s", mes)
    mcp = HanaMcpTools()
    agente = AgenteForensePostmortem(mcp_tools=mcp)

    out_dir = os.getenv('OUTPUT_PDF_DIR', os.path.join(REPO_ROOT, 'outputs', 'postmortems'))
    os.makedirs(out_dir, exist_ok=True)

    try:
        if reset_postmortems:
            mcp.clear_postmortems_for_month(mes)

        seleccionados = mcp.get_candidate_incidents(
            mes=mes,
            stream=None,
            top_k=top_k if (top_k and top_k > 0) else None,
        )
        existentes = set() if reset_postmortems else mcp.get_existing_postmortem_ids(mes)
        log.info("[INFO] Incidentes unicos deduplicados en %s: %d (ya existentes en HANA: %d)",
                 mes, len(seleccionados), len(existentes))

        reportes_generados = []
        for i, inc in enumerate(seleccionados, start=1):
            pm_id = construir_postmortem_id(mes, inc)
            if pm_id in existentes:
                log.info("[SKIP] (%d/%d) %s ya existe en RESULT_POSTMORTEMS_%s",
                         i, len(seleccionados), pm_id, mes)
                continue
            log.info("--- Procesando Incidente %d de %d (%s | %s | escalas=%s) ---",
                     i, len(seleccionados), pm_id, inc['tipo_ataque'],
                     ",".join(inc.get('escalas_detectadas', [])))

            res = agente.investigar_y_redactar(mes=mes, incident_data=inc, output_dir=None)
            reportes_generados.append(res)
            time.sleep(0.35)

        if reportes_generados:
            pdf_mes_path = os.path.join(out_dir, f"Postmortem_SOC_{mes}.pdf")
            generar_pdf_postmortem_mensual(reportes_generados, mes.replace('_', '-'), pdf_mes_path)
            log.info("[OK] PDF consolidado mensual generado en: %s", pdf_mes_path)

        # Validar en HANA que los registros estan guardados
        tabla_postmortems = f"RESULT_POSTMORTEMS_{mes}"
        cursor = mcp.conn.connection.cursor()
        cursor.execute(f'SELECT POSTMORTEM_ID, STREAM, TIPO_ATAQUE, INICIO, FIN FROM "{tabla_postmortems}" ORDER BY INICIO ASC')
        filas = cursor.fetchall()
        cursor.close()

        print("\n" + "=" * 80)
        print(f"[RESUMEN] POSTMORTEMS PERSISTIDOS EN SAP HANA ({tabla_postmortems}): {len(filas)}")
        print("=" * 80)
        for fila in filas:
            print(f"  - ID: {fila[0]} | Stream: {fila[1]} | Tipo: {fila[2]} | Intervalo: {fila[3]} -> {fila[4]}")
        print("=" * 80 + "\n")

    finally:
        mcp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generador forense multi-escala de Postmortems en SAP HANA.')
    parser.add_argument('--mes', nargs='+', default=['2026_01'], help='Mes(es) AAAA_MM a procesar (default: 2026_01)')
    parser.add_argument('--reset', action='store_true', help='Limpia RESULT_POSTMORTEMS_<mes> antes de regenerar')
    parser.add_argument('--top-k', type=int, default=0, help='Limite opcional de incidentes por mes (0 = todos)')
    args = parser.parse_args()
    for m in args.mes:
        main(mes=m, reset_postmortems=args.reset, top_k=args.top_k)
