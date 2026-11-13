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
import json
import logging

# Asegurar resolucion de modulos tanto desde src/ como desde la raiz
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from mcp_hana_tools import HanaMcpTools
    from agente_forense_postmortem import AgenteForensePostmortem
except ImportError:
    from src.mcp_hana_tools import HanaMcpTools
    from src.agente_forense_postmortem import AgenteForensePostmortem

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('probar_agente')


def main(mes='2026_01'):
    log.info("[INFO] Iniciando prueba del Agente Forense y Herramientas MCP para el mes: %s", mes)
    mcp = HanaMcpTools()
    agente = AgenteForensePostmortem(mcp_tools=mcp)

    out_dir = os.getenv('OUTPUT_PDF_DIR', os.path.join(REPO_ROOT, 'outputs', 'postmortems'))
    os.makedirs(out_dir, exist_ok=True)

    try:
        # 1. Obtener candidatos de System y LLM
        candidatos_sys = mcp.get_candidate_incidents(mes=mes, stream='system', top_k=10)
        candidatos_llm = mcp.get_candidate_incidents(mes=mes, stream='llm', top_k=10)

        log.info("[INFO] Candidatos System encontrados: %d | Candidatos LLM encontrados: %d",
                 len(candidatos_sys), len(candidatos_llm))

        # 2. Seleccionar incidentes representativos
        seleccionados = []

        # Caso 1: Fuerza Bruta / Ataque System
        fb_sys = [c for c in candidatos_sys if 'fuerza bruta' in c['tipo_ataque'].lower() or 'explotación' in c['tipo_ataque'].lower()]
        if fb_sys:
            seleccionados.append(fb_sys[0])
        elif candidatos_sys:
            seleccionados.append(candidatos_sys[0])

        # Caso 2: Prompt Injection en LLM
        pi_llm = [c for c in candidatos_llm if 'prompt injection' in c['tipo_ataque'].lower()]
        if pi_llm:
            seleccionados.append(pi_llm[0])

        # Caso 3: Degradacion / Timeouts en LLM
        deg_llm = [c for c in candidatos_llm if 'degradación' in c['tipo_ataque'].lower() or 'recursos' in c['tipo_ataque'].lower()]
        if deg_llm:
            seleccionados.append(deg_llm[0])

        log.info("[INFO] Incidentes seleccionados para generar Postmortems: %d", len(seleccionados))

        reportes_generados = []
        for i, inc in enumerate(seleccionados, start=1):
            log.info("--- Procesando Incidente %d de %d (Evento %d: %s) ---",
                     i, len(seleccionados), inc['evento_id'], inc['tipo_ataque'])
            
            res = agente.investigar_y_redactar(mes=mes, incident_data=inc, output_dir=out_dir)
            reportes_generados.append(res)

            # Guardar archivo Markdown en disco
            file_name = f"{res['postmortem_id']}.md"
            file_path = os.path.join(out_dir, file_name)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(res['reporte_markdown'])
            log.info("[OK] Reporte exportado a: %s", file_path)

        # 3. Validar en HANA que los registros estan guardados
        tabla_postmortems = f"RESULT_POSTMORTEMS_{mes}"
        cursor = mcp.conn.connection.cursor()
        cursor.execute(f"SELECT POSTMORTEM_ID, TIPO_ATAQUE, SEVERIDAD_SCORE, INICIO, FIN FROM {tabla_postmortems}")
        filas = cursor.fetchall()
        
        print("\n" + "="*80)
        print(f"[RESUMEN] POSTMORTEMS PERSISTIDOS EN SAP HANA ({tabla_postmortems}):")
        print("="*80)
        for fila in filas:
            print(f"  • ID: {fila[0]} | Tipo: {fila[1]} | Severidad: {fila[2]:.4f} | Intervalo: {fila[3]} -> {fila[4]}")
        print("="*80 + "\n")

    finally:
        mcp.close()


if __name__ == '__main__':
    main()
