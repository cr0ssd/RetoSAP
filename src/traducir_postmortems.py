"""
traducir_postmortems.py - Agrega la versión en inglés a los reportes postmortem que ya están en HANA.

Para cada tabla RESULT_POSTMORTEMS_<mes>:
  1. Agrega las columnas RESUMEN_EJECUTIVO_EN y REPORTE_MARKDOWN_EN si no existen.
  2. Para cada reporte sin versión en inglés:
       - con GEMINI_API_KEY: traduce con Gemini el reporte y el resumen (mismo contenido);
       - sin llave: arma el reporte en inglés con la plantilla determinística del agente, a partir de
         los datos guardados del incidente (METADATA_JSON, aplicaciones e IPs).
  3. Guarda la traducción con UPDATE (no cambia la versión en español).

Uso (desde la raíz del repo):
    python src/traducir_postmortems.py                 # todos los meses, solo los que faltan
    python src/traducir_postmortems.py --mes 2026_10   # un mes
    python src/traducir_postmortems.py --forzar        # vuelve a traducir aunque ya exista
    python src/traducir_postmortems.py --sin-gemini    # usa la plantilla aunque haya llave
"""

import argparse
import json
import logging
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dotenv import load_dotenv

from helpers.conexion import conexion
from helpers.idioma import traducir_reporte_en
from mcp_hana_tools import asegurar_columnas_en
from agente_forense_postmortem import AgenteForensePostmortem, MITRE_MAP, _extraer_resumen_de_markdown

log = logging.getLogger("traducir_postmortems")


def _cliente_gemini():
    load_dotenv()
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        return None
    from google import genai
    return genai.Client(api_key=key)


def _lista(val):
    if val is None:
        return []
    txt = str(val).strip()
    if txt.startswith("["):
        try:
            return [str(x) for x in json.loads(txt)]
        except Exception:
            pass
    return [x.strip() for x in txt.split(",") if x.strip()]


def _plantilla_en(agente, fila):
    """Reporte en inglés con la plantilla determinística, usando los datos guardados del incidente."""
    meta = json.loads(fila["METADATA_JSON"] or "{}")
    inc = dict(meta.get("incident_meta") or {})
    inc.setdefault("evento_id", int(fila["EVENTO_ID"] or 0))
    inc.setdefault("stream", str(fila["STREAM"] or "system").lower())
    inc.setdefault("tipo_ataque", str(fila["TIPO_ATAQUE"] or ""))
    inc.setdefault("inicio", str(fila["INICIO"]))
    inc.setdefault("fin", str(fila["FIN"]))
    inc.setdefault("confianza", "N/A")
    inc.setdefault("z_max_feature", "N/A")
    inc["z_max"] = float(inc.get("z_max") or 0.0)
    inc["n_ventanas"] = int(inc.get("n_ventanas") or 1)
    inc.setdefault("escalas_detectadas", meta.get("escalas_detectadas") or [])
    evidencia = {
        "aplicaciones_afectadas": _lista(fila["APLICACIONES_AFECTADAS"]),
        "ips_involucradas": _lista(fila["IPS_INVOLUCRADAS"]),
        "logs_muestra": meta.get("evidence_logs_sample") or [],
    }
    mitre = meta.get("mitre_info") or MITRE_MAP.get(inc["tipo_ataque"],
                                                    {"tactic": str(fila["MITRE_TACTIC"]), "technique": "N/A"})
    return agente._sintetizar_postmortem_deterministico_en(fila["POSTMORTEM_ID"], inc, {}, evidencia, mitre)


def traducir(mes=None, forzar=False, sin_gemini=False):
    conn = conexion()
    cliente = None if sin_gemini else _cliente_gemini()
    agente = AgenteForensePostmortem.__new__(AgenteForensePostmortem)   # solo para usar sus plantillas
    log.info("Modo: %s", "Gemini" if cliente else "plantilla determinística (sin GEMINI_API_KEY)")
    try:
        cur = conn.connection.cursor()
        cur.execute("SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                    "AND TABLE_NAME LIKE 'RESULT_POSTMORTEMS_%' ORDER BY TABLE_NAME")
        tablas = [r[0] for r in cur.fetchall()]
        if mes:
            tablas = [t for t in tablas if t == f"RESULT_POSTMORTEMS_{mes}"]
        total = 0
        for tabla in tablas:
            asegurar_columnas_en(cur, tabla)
            conn.connection.commit()
            filtro = "" if forzar else 'WHERE "REPORTE_MARKDOWN_EN" IS NULL'
            df = conn.sql(f'SELECT * FROM "{tabla}" {filtro}').collect()
            log.info("%s: %d reporte(s) por traducir", tabla, len(df))
            for _, fila in df.iterrows():
                pm_id = fila["POSTMORTEM_ID"]
                md_en = traducir_reporte_en(cliente, fila["REPORTE_MARKDOWN"]) if cliente else None
                if md_en:
                    # El resumen en español se extrajo del reporte con la misma función: se hace igual
                    # con el reporte traducido (traducir el párrafo suelto con Gemini no es confiable).
                    resumen_en = _extraer_resumen_de_markdown(md_en, "")
                else:
                    if cliente:
                        log.warning("%s: Gemini no respondió; se usa la plantilla", pm_id)
                    md_en = _plantilla_en(agente, fila)
                    resumen_en = _extraer_resumen_de_markdown(md_en, "")
                cur.execute(f'UPDATE "{tabla}" SET "RESUMEN_EJECUTIVO_EN" = ?, "REPORTE_MARKDOWN_EN" = ? '
                            'WHERE "POSTMORTEM_ID" = ?', (str(resumen_en)[:2990], md_en, pm_id))
                conn.connection.commit()
                total += 1
                log.info("  [OK] %s", pm_id)
        cur.close()
        log.info("Listo: %d reporte(s) con versión en inglés.", total)
        return total
    finally:
        conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    logging.getLogger("hana_ml").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description="Agrega la versión en inglés a los postmortems de HANA.")
    p.add_argument("--mes", help="solo un mes, p. ej. 2026_10")
    p.add_argument("--forzar", action="store_true", help="vuelve a traducir aunque ya exista")
    p.add_argument("--sin-gemini", action="store_true", help="usa la plantilla aunque haya llave")
    a = p.parse_args()
    traducir(mes=a.mes, forzar=a.forzar, sin_gemini=a.sin_gemini)
