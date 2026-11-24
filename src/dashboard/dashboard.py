import html
import io
import json
import os
import re
import time
import sys
from contextlib import contextmanager
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent
SRC_DIR = DASHBOARD_DIR.parent
REPO_ROOT = SRC_DIR.parent
for p in (REPO_ROOT, SRC_DIR, DASHBOARD_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import hana_ml.dataframe as hana_df

from generador_pdf_postmortem import (
    generar_pdf_postmortem,
    generar_pdf_postmortem_mensual,
    _clasificar_severidad_incidente,
)
from helpers.conexion import _params
from helpers.email_sender import EmailSendError, enviar_postmortem_pdf, estado_servicio_correo
from helpers.explicacion import FAMILIAS_SYSTEM, FAMILIAS_LLM, zscores, perfil_familias
from helpers.clasificacion import FIRMAS, TIPOS_NO_ATAQUE, clasificar_eventos, validar_firmas
from helpers import ventanas as ventanas_mod
from helpers.semanas import calculate_week_range, tabla_semanas, split_into_weeks
from helpers.eventos import agrupar_eventos, resumen_eventos
from helpers.relevantes import marcar_relevantes
from helpers.IsolationForest import (N_ESTIMATORS as IF_N_ESTIMATORS, MAX_SAMPLES as IF_MAX_SAMPLES,
                                     CONTAMINATION as CONTAMINATION_IF, RANDOM_STATE as IF_RANDOM_STATE)
from hana_ml.algorithms.pal.preprocessing import IsolationForest as PalIsolationForest
import workflow as wf   # configuración y pasos del pipeline de producción
from helpers.idioma import TIPOS_EN, CONFIANZA_EN

# Idioma de la interfaz ("es" o "en"). Se elige en la barra lateral; el valor de la corrida
# anterior está en session_state, así que todo el script se dibuja ya en ese idioma.
IDIOMA = st.session_state.get("idioma", "es")


def tr(es, en):
    """Texto en el idioma elegido."""
    return en if IDIOMA == "en" else es


def tipo_txt(t):
    """Tipo de ataque (se guarda en español en HANA) en el idioma elegido."""
    return TIPOS_EN.get(str(t), str(t)) if IDIOMA == "en" else str(t)


def conf_txt(c):
    """Nivel de confianza (alta / media / baja) en el idioma elegido."""
    return CONFIANZA_EN.get(str(c), str(c)) if IDIOMA == "en" else str(c)


# Modularización
from tema import (BRAND, NAVY, GOLD, SERIES, INK, INK2, MUTED, GRID, AXIS, NEUTRO, SURFACE,
                  CARD_BORDER, PAGE, DIV_MID, CRITICO, AZUL_SEQ, CONFIANZA_COLOR)
from datos_hana import _conn, leer, leer_postmortems, _parse_json_safe
from health import estados_hc, inventario_hana, metricas_pipeline, vista_health

def _declarar_grafica_rango():
    """
    Componente con la gráfica de Plotly y su barra de rango original, que avisa a Streamlit cuando
    cambia el periodo visible. Usa el plotly.js incluido en el paquete de Python (misma versión que
    genera las figuras y sin depender de internet).
    """
    import shutil
    import tempfile
    import plotly
    import streamlit.components.v1 as components

    js = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    destino = Path(tempfile.gettempdir()) / f"sap_grafica_rango_{plotly.__version__}"
    destino.mkdir(exist_ok=True)
    shutil.copyfile(DASHBOARD_DIR / "componentes" / "grafica_rango.html", destino / "index.html")
    if not (destino / "plotly.min.js").exists():
        shutil.copyfile(js, destino / "plotly.min.js")
    return components.declare_component("grafica_rango", path=str(destino))


grafica_rango = _declarar_grafica_rango()

st.set_page_config(page_title=tr("Anomalías SAP", "SAP Anomalies"), page_icon=":material/shield:", layout="wide")



COLOR_TIPO = {
    # system
    "Contexto SAP": SERIES[0], "DDoS": SERIES[1], "DoS": SERIES[2], "Fuerza bruta": SERIES[3],
    "Explotación": SERIES[4], "Reconocimiento": SERIES[5], "Falla operativa": "#8d99a8",
    # llm
    "Prompt injection": SERIES[0], "Degradación del servicio": SERIES[1],
    "Abuso de recursos": SERIES[2], "Acceso no autorizado": SERIES[3], "Automatización": SERIES[4],
}
COLOR_TIPO.update({TIPOS_EN[k]: v for k, v in list(COLOR_TIPO.items()) if k in TIPOS_EN})


STREAMS = {"system": tr("Sistema SAP", "SAP System"), "llm": tr("Servicios de IA", "AI Services")}
VOLUMEN = {"system": "TOTAL_REQUESTS", "llm": "TOTAL_LLM_REQUESTS"}
FAMILIAS = {"system": FAMILIAS_SYSTEM, "llm": FAMILIAS_LLM}
PLOT_CFG = {"displayModeBar": False}

# Nombre corto y legible de cada indicador (para gráficas y tablas).
NOMBRE = {
    "TOTAL_REQUESTS": "Volumen de peticiones", "REQUEST_RATE_STD": "Irregularidad del ritmo",
    "INTERVAL_CV": "Variación entre peticiones", "UNIQUE_IPS_COUNT": "IPs distintas",
    "IP_ENTROPY": "Dispersión entre IPs", "IP_TOP_SHARE": "Peso de la IP más activa",
    "ERROR_RATE_400": "Errores del cliente", "ERROR_RATE_500": "Errores del servidor",
    "NON_200_RATE": "Respuestas no exitosas", "RATIO_ERROR": "Registros de error",
    "RATIO_WARN": "Registros de advertencia", "RATIO_SECURITY": "Registros de seguridad",
    "RATIO_INFO": "Registros informativos", "RATIO_AUDIT": "Registros de auditoría",
    "GET_VS_POST_RATIO": "Consultas vs. envíos",
    "JS_HTTP_STATUS_CODE": "Códigos de respuesta", "JS_HEADERS_HTTP_REQUEST_METHOD": "Tipos de petición",
    "JS_SAP_FUNCTION_LOG_TYPE": "Tipos de registro", "JS_HEATHERS_REQUEST_PATH": "Rutas solicitadas",
    "JS_SAP_FUNCTION_APPLICATION": "Aplicaciones SAP", "JS_SAP_SOURCE_TYPE": "Canales de conexión",
    "JS_SAP_APP_ENV": "Entornos", "JS_REGION_CODE": "Regiones", "JS_HEADERS_HTTP_HOST": "Servidores",
    "JS_HEADERS_CONTENT_TYPE": "Formatos de contenido",
    "TOTAL_LLM_REQUESTS": "Volumen de consultas", "AVG_LLM_COST_USD": "Costo promedio",
    "INJECTION_COUNT": "Intentos de manipulación", "SAFETY_POLICY_COUNT": "Bloqueos por política",
    "AUTH_FAILURE_COUNT": "Fallas de autenticación", "ACCESS_DENIED_COUNT": "Accesos denegados",
    "RATE_ABUSE_COUNT": "Límites de uso alcanzados", "DUPLICATE_REQUEST_COUNT": "Peticiones duplicadas",
    "MALFORMED_INPUT_COUNT": "Peticiones mal formadas", "AVG_TOKEN_RATIO": "Largo de respuesta",
    "AVG_RESPONSE_TIME_MS": "Tiempo de respuesta", "LLM_ERROR_RATE": "Tasa de errores",
    "LLM_TIMEOUT_RATE": "Tiempos de espera agotados", "TEMPERATURE_VARIANCE": "Variación de creatividad",
    "PROMPT_CATEGORY_ENTROPY": "Variedad de temas", "STREAM_RATIO": "Respuestas en tiempo real",
    "AVG_PROMPT_LENGTH": "Largo promedio de consulta", "MAX_PROMPT_LENGTH": "Consulta más extensa",
}


# Tipos de ataque posibles y su descripción. El orden y los tipos son los de
# CLASIFICACION_TIPOS_ATAQUE.md ("Tipos posibles en system / LLM"); las descripciones de system
# vienen de su apartado "Cómo leerlas". El md no describe los tipos de LLM, así que para ellos
# solo se muestran los indicadores de su firma (helpers/clasificacion.py).
TIPOS_POSIBLES = {
    "system": ["DoS", "DDoS", "Fuerza bruta", "Reconocimiento", "Explotación", "Contexto SAP",
               "Falla operativa"],
    "llm": ["Prompt injection", "Acceso no autorizado", "Abuso de recursos", "Automatización",
            "Degradación del servicio"],
}
DESCRIPCION_TIPO = {
    "DoS": tr("Sube el volumen de peticiones y lo concentra en pocas IPs.",
              "Raises request volume and concentrates it on a few IPs."),
    "DDoS": tr("Sube el volumen de peticiones repartido entre muchas IPs.",
               "Raises request volume spread across many IPs."),
    "Fuerza bruta": tr("Concentra errores 4xx y eventos de seguridad o auditoría desde una IP "
                       "dominante, con un ritmo regular y más POST que GET.",
                       "Concentrates 4xx errors and security or audit events from a dominant IP, "
                       "with a regular pace and more POST than GET."),
    "Reconocimiento": tr("Produce errores 4xx, más GET y cambios en las rutas pedidas.",
                         "Produces 4xx errors, more GET requests and changes in requested paths."),
    "Explotación": tr("Produce errores 5xx, registros de error y de seguridad, y cambios en "
                      "métodos, rutas y formatos de contenido.",
                      "Produces 5xx errors, error and security logs, and changes in methods, "
                      "paths and content formats."),
    "Contexto SAP": tr("Cambios en la mezcla de aplicaciones, entornos, regiones o tipos de registro "
                       "de SAP.",
                       "Changes in the mix of SAP applications, environments, regions or log types."),
    "Falla operativa": tr("Errores y advertencias con caída del tráfico.",
                          "Errors and warnings with a drop in traffic."),
}
FIRMA = {"system": FIRMAS["F_SYSTEM"], "llm": FIRMAS["F_LLM"]}


NOMBRE_EN = {
    "TOTAL_REQUESTS": "Request volume", "REQUEST_RATE_STD": "Rate irregularity",
    "INTERVAL_CV": "Inter-request variation", "UNIQUE_IPS_COUNT": "Distinct IPs",
    "IP_ENTROPY": "IP dispersion", "IP_TOP_SHARE": "Share of most active IP",
    "ERROR_RATE_400": "Client errors", "ERROR_RATE_500": "Server errors",
    "NON_200_RATE": "Unsuccessful responses", "RATIO_ERROR": "Error logs",
    "RATIO_WARN": "Warning logs", "RATIO_SECURITY": "Security logs",
    "RATIO_INFO": "Informational logs", "RATIO_AUDIT": "Audit logs",
    "GET_VS_POST_RATIO": "Reads vs. writes",
    "JS_HTTP_STATUS_CODE": "Response codes", "JS_HEADERS_HTTP_REQUEST_METHOD": "Request types",
    "JS_SAP_FUNCTION_LOG_TYPE": "Log types", "JS_HEATHERS_REQUEST_PATH": "Requested paths",
    "JS_SAP_FUNCTION_APPLICATION": "SAP applications", "JS_SAP_SOURCE_TYPE": "Connection channels",
    "JS_SAP_APP_ENV": "Environments", "JS_REGION_CODE": "Regions", "JS_HEADERS_HTTP_HOST": "Servers",
    "JS_HEADERS_CONTENT_TYPE": "Content formats",
    "TOTAL_LLM_REQUESTS": "Query volume", "AVG_LLM_COST_USD": "Average cost",
    "INJECTION_COUNT": "Manipulation attempts", "SAFETY_POLICY_COUNT": "Policy blocks",
    "AUTH_FAILURE_COUNT": "Authentication failures", "ACCESS_DENIED_COUNT": "Access denied",
    "RATE_ABUSE_COUNT": "Usage limits reached", "DUPLICATE_REQUEST_COUNT": "Duplicate requests",
    "MALFORMED_INPUT_COUNT": "Malformed requests", "AVG_TOKEN_RATIO": "Response length",
    "AVG_RESPONSE_TIME_MS": "Response time", "LLM_ERROR_RATE": "Error rate",
    "LLM_TIMEOUT_RATE": "Timeouts", "TEMPERATURE_VARIANCE": "Creativity variation",
    "PROMPT_CATEGORY_ENTROPY": "Topic variety", "STREAM_RATIO": "Streamed responses",
    "AVG_PROMPT_LENGTH": "Average query length", "MAX_PROMPT_LENGTH": "Longest query",
}


def nombre(f):
    return (NOMBRE_EN if IDIOMA == "en" else NOMBRE).get(f, f)


def texto_cambios(top3):
    """'IP_TOP_SHARE (+18.9), RATIO_AUDIT (-2.5)' -> 'Peso de la IP más activa ↑, Registros de auditoría ↓'."""
    if not isinstance(top3, str):
        return ""
    return ", ".join(f"{nombre(f)} {'↑' if float(z) > 0 else '↓'}"
                     for f, z in re.findall(r"([A-Z0-9_]+) \(([+-]?[\d.]+)\)", top3))


# Estilos: encabezado con degradado, KPIs y tarjetas con acento de color (SAP Fiori)
SOMBRA = "0 0 2px rgba(34,53,72,.12), 0 2px 8px rgba(34,53,72,.08)"
st.html(f"""
<style>
  [data-testid="stMainBlockContainer"] {{ padding-top: 4.2rem; padding-bottom: 3rem;
                                          max-width: 1480px; }}
  div[class*="st-key-card_"] {{
      background: {SURFACE}; border: 1px solid {CARD_BORDER} !important;
      border-radius: 14px; padding: 1.1rem 1.25rem 0.9rem; box-shadow: {SOMBRA};
  }}
  /* Tarjetas lado a lado: todas ocupan la altura de la fila, sin huecos abajo */
  [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
      > [data-testid="stLayoutWrapper"]:has(> div[class*="st-key-card_"]) {{ flex: 1 1 auto; }}
  [data-testid="stColumn"] [data-testid="stLayoutWrapper"] > div[class*="st-key-card_"] {{ height: 100%; }}
  .card-title {{ font-size: 1.02rem; font-weight: 650; color: {INK}; margin: 0; }}
  .card-sub {{ font-size: 0.84rem; color: {MUTED}; margin: 0.2rem 0 0.2rem; line-height: 1.45; }}

  /* Encabezado de cada sección: claro, sin caja, sobre el fondo de la página */
  .banner {{ padding: .2rem 0 0; margin-bottom: 1.2rem; }}
  .banner .eyebrow {{ font-size: .76rem; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: {BRAND}; }}
  .banner .titulo {{ font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: .2rem 0 0;
      color: {INK}; }}
  .banner .sub {{ font-size: .95rem; color: {MUTED}; margin: .4rem 0 0; max-width: 820px; }}
  .banner .pills {{ display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem; }}
  .banner .pill {{ font-size: .78rem; padding: .22rem .75rem; border-radius: 999px; color: {INK2};
      background: {SURFACE}; border: 1px solid {CARD_BORDER}; }}
  .banner .pill b {{ font-weight: 600; color: {MUTED}; margin-right: .3rem; }}

  /* KPIs */
  .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: .85rem;
      margin-bottom: 1rem; }}
  .kpis.kpis-4 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
  .kpis.kpis-3 {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
  @media (max-width: 1000px) {{ .kpis, .kpis.kpis-4, .kpis.kpis-3 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .kpi {{ background: {SURFACE}; border: 1px solid {CARD_BORDER}; border-top: 4px solid var(--c);
      border-radius: 14px; padding: .85rem 1rem .9rem; box-shadow: {SOMBRA}; overflow: hidden; }}
  .kpi .lab {{ display: flex; align-items: center; gap: .45rem; font-size: .8rem; color: {MUTED};
      font-weight: 600; }}
  .kpi .lab i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); flex-shrink: 0; }}
  .kpi .val {{ font-size: clamp(1.35rem, 2.0vw, 1.85rem); font-weight: 700; color: {INK}; line-height: 1.15;
      margin-top: .35rem; }}
  .kpi .det {{ font-size: .78rem; color: {MUTED}; margin-top: .2rem; }}

  /* Barra superior y barra lateral: fondo de tarjeta y un borde que las separa del contenido */
  [data-testid="stHeader"] {{ background: {SURFACE} !important;
      border-bottom: 1px solid {CARD_BORDER}; box-shadow: 0 1px 3px rgba(34,53,72,.06); }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {CARD_BORDER}; }}
  .side-brand {{ font-size: 1.15rem; font-weight: 700; color: {INK}; }}
  .side-brand span {{ color: {BRAND}; }}
  .side-periodo {{ background: #e5f3ff; border: 1px solid #b5dcff; border-radius: 10px;
      padding: .55rem .7rem; font-size: .84rem; color: {INK}; margin-bottom: .5rem; }}
  .side-periodo b {{ display: block; font-size: .7rem; letter-spacing: .06em; text-transform: uppercase;
      color: #0057d2; margin-bottom: .15rem; }}
  .side-periodo span {{ display: block; font-size: .74rem; color: {MUTED}; margin-top: .2rem; }}
  .side-sub {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: .4rem; }}

  /* Nada se oculta al quitar el mouse: botón para contraer la barra y marcas del deslizador */
  [data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapseButton"] *,
  [data-testid="stSidebarHeader"] button {{ visibility: visible !important; opacity: 1 !important; }}
  [data-testid="stSliderTickBar"] {{ opacity: 1 !important; }}

  /* Pasos */
  .paso {{ border-left: 3px solid {BRAND}; padding: .1rem 0 .1rem .75rem; }}
  .paso b {{ color: {INK}; font-size: .92rem; }}
  .paso p {{ color: {MUTED}; font-size: .82rem; margin: .15rem 0 0; line-height: 1.4; }}

  /* Tarjetas de tipos de ataque: cuadrícula con filas internas alineadas (subgrid) */
  .tipos-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }}
  @media (max-width: 1000px) {{ .tipos-grid {{ grid-template-columns: minmax(0, 1fr); }} }}
  .tipo {{ display: grid; grid-row: span 4; grid-template-rows: subgrid; row-gap: .7rem;
      background: {SURFACE}; border: 1px solid {CARD_BORDER}; border-top: 4px solid var(--c);
      border-radius: 14px; padding: 1rem 1.15rem 1.05rem; box-shadow: {SOMBRA}; }}
  .tipo .nom {{ display: flex; align-items: center; gap: .5rem; font-weight: 650;
      font-size: 1.02rem; color: {INK}; }}
  .tipo .nom i {{ width: 10px; height: 10px; border-radius: 50%; background: var(--c); }}
  .tipo .desc {{ font-size: .84rem; color: {MUTED}; line-height: 1.45; margin: 0; }}
  .tipo .chips {{ display: flex; flex-wrap: wrap; gap: .4rem; align-items: flex-start; }}
  .chip {{ font-size: .75rem; font-weight: 600; padding: .18rem .6rem; border-radius: 999px; }}
  .chip.alerta {{ background: #fff1e3;
      color: #a93e00; }}
  .chip.nada {{ background: {GRID}; color: {MUTED}; }}
  .chip.info {{ background: #e5f3ff;
      color: #0057d2; }}
  .tipo .sen {{ border-top: 1px solid {CARD_BORDER}; padding-top: .65rem; }}
  .tipo .sen p {{ font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
      color: {MUTED}; margin: 0 0 .35rem; }}
  .tipo .sen ul {{ list-style: none; margin: 0; padding: 0; }}
  .tipo .sen li {{ font-size: .83rem; color: {INK2}; padding: .12rem 0; }}
  .tipo .sen li span {{ display: inline-block; width: 1.3rem; color: var(--c); font-weight: 700; }}

  /* MLOps: health checks */
  .hc-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .9rem; }}
  @media (max-width: 1000px) {{ .hc-grid {{ grid-template-columns: minmax(0, 1fr); }} }}
  .hc-group {{ background: {PAGE}; border: 1px solid {CARD_BORDER}; border-radius: 12px; padding: .75rem .9rem; }}
  .hc-gtitle {{ font-size: .72rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase;
      color: {MUTED}; margin: 0 0 .35rem; }}
  .hc-row {{ display: grid; grid-template-columns: 12px minmax(0, 1fr) auto; gap: .7rem; align-items: center;
      padding: .55rem 0; border-top: 1px solid {CARD_BORDER}; }}
  .hc-row:first-of-type {{ border-top: none; }}
  .hc-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
  .hc-dot.ok {{ background: #188918; }} .hc-dot.warn {{ background: #E76500; }}
  .hc-dot.fail {{ background: {CRITICO}; }} .hc-dot.info {{ background: {BRAND}; }}
  .hc-main b {{ display: block; font-size: .88rem; color: {INK}; font-weight: 650; }}
  .hc-main span {{ display: block; font-size: .78rem; color: {MUTED}; line-height: 1.4; margin-top: .1rem; }}
  .hc-val {{ display: flex; flex-direction: column; align-items: flex-end; gap: .25rem; text-align: right; }}
  .hc-val b {{ font-size: .84rem; color: {INK}; font-weight: 650; white-space: pre; line-height: 1.35; }}
  .hc-pill {{ font-size: .7rem; font-weight: 700; padding: .12rem .55rem; border-radius: 999px; white-space: nowrap; }}
  .hc-pill.ok {{ background: #e8f6e8; color: #106b10; }} .hc-pill.warn {{ background: #fff1e3; color: #a93e00; }}
  .hc-pill.fail {{ background: #fdeaea; color: {CRITICO}; }} .hc-pill.info {{ background: #e5f3ff; color: #0057d2; }}
  /* MLOps: pasos del entrenamiento */
  .tr-steps {{ display: flex; flex-direction: column; gap: .45rem; margin-top: .4rem; }}
  .tr-step {{ display: grid; grid-template-columns: 28px minmax(0, 1fr) auto; gap: .7rem; align-items: center;
      padding: .5rem .7rem; border: 1px solid {CARD_BORDER}; border-radius: 10px; background: {SURFACE}; }}
  .tr-step b {{ display: block; font-size: .86rem; color: {INK}; font-weight: 650; }}
  .tr-step > div > span {{ display: block; font-size: .76rem; color: {MUTED}; }}
  .tr-step em {{ font-style: normal; font-size: .78rem; color: {MUTED}; font-variant-numeric: tabular-nums; }}
  .tr-num {{ width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center;
      justify-content: center; line-height: 1; font-size: .78rem; font-variant-numeric: tabular-nums;
      font-weight: 700; background: {GRID}; color: {MUTED}; }}
  .tr-step.corriendo {{ border-color: {BRAND}; background: #f0f7ff; }}
  .tr-step.corriendo .tr-num {{ background: {BRAND}; color: #fff; animation: trpulse 1.2s ease-in-out infinite; }}
  .tr-step.corriendo em {{ color: {BRAND}; font-weight: 600; }}
  .tr-step.ok .tr-num {{ background: #188918; color: #fff; }}
  .tr-step.error {{ border-color: {CRITICO}; background: #fdeaea; }}
  .tr-step.error .tr-num {{ background: {CRITICO}; color: #fff; }}
  @keyframes trpulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: .45; }} }}
  /* Postmortems SOC */
  .pm-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; }}
  .pm-head .id {{ font-size: 1.15rem; font-weight: 700; color: {INK}; margin-right: .2rem; }}
  .pm-sub {{ font-size: .85rem; color: {MUTED}; margin: .35rem 0 0; line-height: 1.45; }}
  .pm-sub b {{ color: {INK2}; font-weight: 600; }}
  .pm-pill {{ font-size: .74rem; font-weight: 700; padding: .18rem .62rem; border-radius: 999px;
      background: {GRID}; color: {INK2}; white-space: nowrap; }}
  .pm-pill.crit {{ background: #fdeaea; color: {CRITICO}; }}
  .pm-pill.warn {{ background: #fff1e3; color: #a93e00; }}
  .pm-pill.brand {{ background: #e5f3ff; color: #0057d2; }}
  .pm-meta {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .75rem; }}
  .pm-meta.dos {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  @media (max-width: 1000px) {{ .pm-meta {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .pm-meta > div {{ background: {PAGE}; border: 1px solid {CARD_BORDER}; border-radius: 10px;
      padding: .7rem .9rem; }}
  .pm-meta .lbl {{ font-size: .72rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
      color: {MUTED}; margin: 0 0 .3rem; }}
  .pm-meta .val {{ font-size: .95rem; font-weight: 650; color: {INK}; word-break: break-word; }}
  .pm-meta .det {{ font-size: .8rem; color: {MUTED}; margin-top: .1rem; }}
  .pm-callout {{ background: #f0f7ff; border: 1px solid #cfe6ff; border-left: 4px solid {BRAND};
      border-radius: 10px; padding: .85rem 1rem; color: {INK}; font-size: .9rem; line-height: 1.55; }}
  .pm-callout b {{ display: block; font-size: .72rem; letter-spacing: .06em; text-transform: uppercase;
      color: #0057d2; margin-bottom: .3rem; }}
  .pm-chips {{ display: flex; flex-wrap: wrap; gap: .4rem; }}
  .pm-chip {{ font-size: .78rem; font-weight: 600; padding: .22rem .65rem; border-radius: 8px;
      background: {SURFACE}; border: 1px solid {CARD_BORDER}; color: {INK2};
      font-variant-numeric: tabular-nums; }}
  .pm-chip.rango {{ background: #e5f3ff; border-color: #b5dcff; color: #0057d2; }}
  .pm-vacio {{ font-size: .84rem; color: {MUTED}; }}
  /* Botones de descarga: neutros, y se pintan de azul el que se señala o se presiona */
  .st-key-pm_descargas button {{ background: {SURFACE}; color: {INK2}; border: 1px solid {CARD_BORDER};
      font-weight: 600; transition: background .15s, color .15s, border-color .15s; }}
  .st-key-pm_descargas button:hover, .st-key-pm_descargas button:focus-visible,
  .st-key-pm_descargas button:active {{ background: {BRAND}; color: #fff; border-color: {BRAND}; }}
  .st-key-pm_descargas button:hover *, .st-key-pm_descargas button:active * {{ color: #fff; }}
  /* Reporte en Markdown: títulos a la escala del resto de la página */
  .st-key-pm_reporte h1 {{ font-size: 1.3rem !important; padding: .6rem 0 .3rem !important; }}
  .st-key-pm_reporte h2 {{ font-size: 1.1rem !important; padding: 1rem 0 .3rem !important;
      border-top: 1px solid {CARD_BORDER}; margin-top: .6rem !important; }}
  .st-key-pm_reporte h3 {{ font-size: .98rem !important; padding: .7rem 0 .2rem !important; }}
  .st-key-pm_reporte p, .st-key-pm_reporte li {{ font-size: .9rem; line-height: 1.6; color: {INK2}; }}
  .st-key-pm_reporte hr {{ display: none; }}
</style>
""")

_n_card = [0]


@contextmanager
def tarjeta(titulo=None, sub=None):
    """Contenedor con borde y fondo de tarjeta, con título y descripción opcionales."""
    _n_card[0] += 1
    with st.container(border=True, key=f"card_{_n_card[0]}"):
        if titulo:
            st.html(f'<p class="card-title">{titulo}</p>'
                    + (f'<p class="card-sub">{sub}</p>' if sub else ""))
        yield


def encabezado(eyebrow, titulo, sub, pills=None, con_periodo=True):
    pills = list(pills or [(tr("Periodo", "Period"), MES.replace("_", "-")),
                           (tr("Origen", "Source"), STREAMS[STREAM]),
                           (tr("Bloques de", "Blocks of"), ETIQUETA[SIZE_S])])
    if con_periodo and TXT_PERIODO:
        pills.append((tr("Seleccionado", "Selected"), TXT_PERIODO))
    st.html(f'<div class="banner"><div class="eyebrow">{eyebrow}</div>'
            f'<p class="titulo">{titulo}</p>'
            + (f'<p class="sub">{sub}</p>' if sub and sub.strip() else "")
            + '<div class="pills">' + "".join(f'<span class="pill"><b>{k}</b>{v}</span>'
                                               for k, v in pills) + '</div></div>')


def estilo(fig, height=340, legend=True, margin=None):
    fig.update_layout(
        height=height, margin=margin or dict(l=56, r=12, t=40 if legend else 8, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        hoverlabel=dict(bgcolor=SURFACE, font_color=INK, bordercolor=AXIS),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None,
                    font=dict(color=INK2), traceorder="normal"),
        bargap=0.35,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, tickfont_color=MUTED, zeroline=False,
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont_color=MUTED, zeroline=False,
                     automargin=True)
    return fig


def mostrar(fig):
    st.plotly_chart(fig, theme=None, config=PLOT_CFG)


def vacio(msg):
    st.info(msg, icon=":material/info:")

# Datos



@st.cache_data(ttl=600, show_spinner=False)
def meses_disponibles():
    t = _conn().sql(
        "SELECT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
        "AND TABLE_NAME LIKE 'RESULT_EVENTS_%'").collect()
    return sorted((n.replace("RESULT_EVENTS_", "") for n in t["TABLE_NAME"]), reverse=True)





def _parse_lista_campo(val):
    """Normaliza listas guardadas como CSV o JSON en columnas NVARCHAR."""
    if isinstance(val, (list, tuple)):
        items = [str(x).strip() for x in val]
    elif val is None or (isinstance(val, float) and pd.isna(val)):
        return []
    else:
        txt = str(val).strip()
        if not txt:
            return []
        if txt.startswith("["):
            try:
                loaded = json.loads(txt)
                if isinstance(loaded, list):
                    return [str(x).strip() for x in loaded if str(x).strip()]
            except Exception:
                pass
        items = [x.strip() for x in txt.split(",")]
    seen = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return seen


def _etiqueta_severidad(score):
    """Clasifica severidad Isolation Forest para badges y estilos."""
    s = float(score or 0.0)
    if s >= 0.70:
        return tr("Crítica", "Critical"), "crit", CRITICO
    if s >= 0.45:
        return tr("Alta", "High"), "warn", "#E76500"
    return tr("Moderada", "Moderate"), "brand", BRAND





def pm_idioma(df):
    """
    Postmortems en el idioma elegido. En inglés usa RESUMEN_EJECUTIVO_EN y REPORTE_MARKDOWN_EN
    (si el reporte aún no tiene traducción se deja en español y se marca en _SIN_EN) y traduce
    el tipo de ataque.
    """
    if df.empty:
        return df.assign(_SIN_EN=pd.Series(dtype=bool))
    d = df.copy()
    d["_SIN_EN"] = False
    if IDIOMA != "en":
        return d
    for col_en, col in (("REPORTE_MARKDOWN_EN", "REPORTE_MARKDOWN"), ("RESUMEN_EJECUTIVO_EN", "RESUMEN_EJECUTIVO")):
        if col_en in d.columns:
            hay = d[col_en].notna() & (d[col_en].astype(str).str.strip() != "")
            d.loc[hay, col] = d.loc[hay, col_en]
            if col == "REPORTE_MARKDOWN":
                d["_SIN_EN"] = ~hay
        elif col == "REPORTE_MARKDOWN":
            d["_SIN_EN"] = True
    d["TIPO_ATAQUE"] = d["TIPO_ATAQUE"].map(tipo_txt)
    return d


def _fmt_ts(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return "N/A"
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


def _fila_a_payload(fila):
    """Normaliza una fila de RESULT_POSTMORTEMS_<MES> al dict esperado por el generador PDF."""
    meta = _parse_json_safe(fila.get("METADATA_JSON"))
    inc = dict(meta.get("incident_meta") or {})
    inc["z_max"] = float(inc.get("z_max") or 0.0)
    inc["z_max_feature"] = str(inc.get("z_max_feature") or "N/A")
    inc["confianza"] = str(inc.get("confianza") or "alta")
    inc["n_ventanas"] = int(inc.get("n_ventanas") or 1)
    if not inc.get("ventana_pico"):
        inc["ventana_pico"] = _fmt_ts(fila.get("INICIO"))
    meta_norm = dict(meta)
    meta_norm["incident_meta"] = inc

    return {
        "postmortem_id": str(fila.get("POSTMORTEM_ID") or "PM-N/A"),
        "evento_id": int(fila.get("EVENTO_ID") or 0),
        "stream": str(fila.get("STREAM") or "system").lower(),
        "tipo_ataque": str(fila.get("TIPO_ATAQUE") or tr("Anomalía", "Anomaly")),
        "mitre_tactic": str(fila.get("MITRE_TACTIC") or tr("Pendiente de mapeo", "Mapping pending")),
        "severidad_score": float(fila.get("SEVERIDAD_SCORE") or 0.0),
        "inicio": _fmt_ts(fila.get("INICIO")),
        "fin": _fmt_ts(fila.get("FIN")),
        "resumen_ejecutivo": str(fila.get("RESUMEN_EJECUTIVO") or tr("Sin dictamen ejecutivo disponible.",
                                                                      "No executive verdict available.")),
        "aplicaciones_afectadas": _parse_lista_campo(fila.get("APLICACIONES_AFECTADAS")),
        "ips_involucradas": _parse_lista_campo(fila.get("IPS_INVOLUCRADAS")),
        "reporte_markdown": str(fila.get("REPORTE_MARKDOWN") or ""),
        "metadata_json": meta_norm,
    }


def _construir_pdf_bytes(fila):
    """Construye en memoria el reporte ejecutivo PDF individual a partir de una fila HANA."""
    buf = io.BytesIO()
    generar_pdf_postmortem(_fila_a_payload(fila), buf, idioma=IDIOMA)
    return buf.getvalue()


def _construir_pdf_mensual_bytes(df_pm, periodo):
    """Construye en memoria el reporte PDF multi-pagina consolidado para todos los incidentes."""
    items = [_fila_a_payload(row) for _, row in df_pm.iterrows()]
    buf = io.BytesIO()
    generar_pdf_postmortem_mensual(items, periodo, buf, idioma=IDIOMA)
    return buf.getvalue()


def _escalas_de_meta(meta):
    """Extrae las escalas de ventana detectadas de METADATA_JSON."""
    inc = (meta or {}).get("incident_meta") or {}
    esc = (
        (meta or {}).get("escalas_detectadas")
        or inc.get("escalas_detectadas")
        or ([str(inc["window_label"])] if inc.get("window_label") else ["5min"])
    )
    return [str(x) for x in esc if str(x).strip()]


def _construir_markdown_mensual(df_pm, periodo):
    """Construye el Dossier Forense Mensual Unificado en Markdown agrupado por campana."""
    orden_esc = {"5min": 1, "15min": 2, "30min": 3, "1h": 4, "3h": 5}
    lineas = [
        tr(f"# DOSSIER FORENSE POSTMORTEM MENSUAL — PERIODO {periodo}",
           f"# MONTHLY FORENSIC POSTMORTEM DOSSIER — PERIOD {periodo}"),
        tr(f"**Total de incidentes unicos multi-escala:** {len(df_pm)}",
           f"**Total unique multi-scale incidents:** {len(df_pm)}"),
        "",
        tr("## 1. Resumen por Campana / Familia de Amenaza", "## 1. Summary by Campaign / Threat Family"),
        "",
        tr("| Campana / Vector | Origen | Tactica MITRE | Incidentes | Escalas Detectadas | Indicador Dominante |",
           "| Campaign / Vector | Source | MITRE Tactic | Incidents | Detected Scales | Dominant Indicator |"),
        "| :--- | :--- | :--- | ---: | :--- | :--- |",
    ]
    grupos = sorted(df_pm.groupby("TIPO_ATAQUE"), key=lambda kv: (-len(kv[1]), str(kv[0])))
    for vec, g in grupos:
        st_vec = ", ".join(sorted(g["STREAM"].dropna().astype(str).str.upper().unique()))
        mitre_v = str(g["MITRE_TACTIC"].iloc[0] or "N/A")
        escs = set()
        best_f, best_z = "N/A", 0.0
        for _, r in g.iterrows():
            m = _parse_json_safe(r.get("METADATA_JSON"))
            for e in _escalas_de_meta(m):
                escs.add(e)
            inc = m.get("incident_meta") or {}
            z = float(inc.get("z_max") or 0.0)
            if z >= best_z:
                best_z, best_f = z, str(inc.get("z_max_feature") or "N/A")
        esc_s = ", ".join(sorted(escs, key=lambda s: orden_esc.get(s, 99)))
        lineas.append(f"| **{vec}** | {st_vec} | {mitre_v} | {len(g)} | {esc_s} | {best_f} (Z={best_z:.1f}) |")

    lineas.extend([
        "",
        tr("## 2. Cronologia Maestra de Incidentes del Periodo", "## 2. Master Timeline of Incidents in the Period"),
        "",
        tr("| Expediente | Origen | Vector / Ataque | Ventana Temporal (UTC) |",
           "| Case | Source | Vector / Attack | Time Window (UTC) |"),
        "| :--- | :--- | :--- | :--- |",
    ])
    for _, r in df_pm.iterrows():
        lineas.append(
            f"| **{r.get('POSTMORTEM_ID', 'N/A')}** | {str(r.get('STREAM', '')).upper()} | "
            f"{r.get('TIPO_ATAQUE', 'N/A')} | {_fmt_ts(r.get('INICIO'))[:16]} -> {_fmt_ts(r.get('FIN'))[:16]} |"
        )

    lineas.extend(["", tr("## 3. Sintesis Forense Unificada por Campana",
                          "## 3. Unified Forensic Summary by Campaign"), ""])
    for vec, g in grupos:
        lineas.append(tr(f"### Campana: {vec} ({len(g)} incidente(s))", f"### Campaign: {vec} ({len(g)} incident(s))"))
        for _, r in g.iterrows():
            m = _parse_json_safe(r.get("METADATA_JSON"))
            inc = m.get("incident_meta") or {}
            zv = float(inc.get("z_max") or 0.0)
            sev_s = float(r.get("SEVERIDAD_SCORE") or 0.0)
            sev_lbl, _, _ = _clasificar_severidad_incidente(zv, sev_s, IDIOMA)
            lineas.append(
                f"- **{r.get('POSTMORTEM_ID', 'N/A')}** (`{_fmt_ts(r.get('INICIO'))[:16]} -> {_fmt_ts(r.get('FIN'))[:16]}`, "
                f"{tr('severidad', 'severity')}: **{sev_lbl}**): {r.get('RESUMEN_EJECUTIVO', '')}"
            )
        lineas.append("")
    return "\n".join(lineas)


def perfil_relevantes(win, rel, familias):
    feats = [f for fs in familias.values() for f in fs]
    z = []
    for w, g in win.groupby("WEEK"):
        idx = g.loc[g["LABEL"] == -1, "WINDOW_INDEX"].tolist()
        if idx:
            z.append(zscores(g, idx, feats).assign(WEEK=w))
    if not z:
        return pd.DataFrame()
    if_df = win.assign(EVENTO_ID=win["EVENTO_ID"].astype("Int64"))
    p = perfil_familias(if_df, pd.concat(z, ignore_index=True), familias)
    if p.empty:
        return p
    p = p[p["EVENTO_ID"].isin(rel["EVENTO_ID"].astype(int))]
    return p.set_index("EVENTO_ID")[list(familias)]


# Barra lateral
try:
    meses = meses_disponibles()
except Exception as e:  # sin conexión no hay nada que mostrar
    st.error(tr("No fue posible conectarse a SAP HANA. Revisa la configuración de conexión.",
                "Could not connect to SAP HANA. Check the connection settings.") + f"\n\n{e}")
    st.stop()
if not meses:
    st.warning(tr("Todavía no hay resultados cargados en SAP HANA.", "There are no results loaded in SAP HANA yet."))
    st.stop()

with st.sidebar:
    st.segmented_control("Idioma / Language", ["es", "en"], default="es", key="idioma",
                         format_func={"es": "Español", "en": "English"}.get, width="stretch",
                         label_visibility="collapsed")
    st.html(tr('<div class="side-brand">Anomalías <span>SAP</span></div>'
               '<div class="side-sub">Monitoreo de comportamiento inusual en los registros de SAP</div>',
               '<div class="side-brand"><span>SAP</span> Anomalies</div>'
               '<div class="side-sub">Monitoring unusual behavior in SAP logs</div>'))
    MES = st.selectbox(tr("Periodo", "Period"), meses, format_func=lambda m: m.replace("_", "-"), key="mes")
    eventos_all = leer("EVENTS", MES)
    resumen_all = leer("SUMMARY", MES)
    valid_all = leer("VALIDATION", MES)
    # Cada periodo se analiza a partir de una sola tabla de logs; si hubiera varias, se usa la que
    # tiene más eventos. No se muestra como filtro para no confundir a quien consulta el dashboard.
    FUENTE = eventos_all["SOURCE_TABLE"].value_counts().index[0]
    STREAM = st.segmented_control(tr("Origen de los registros", "Log source"), list(STREAMS), default="system",
                                  format_func=STREAMS.get, key="stream",
                                  width="stretch") or "system"
    tam = (resumen_all[["WINDOW_SIZE_S", "WINDOW_LABEL"]].drop_duplicates()
           .sort_values("WINDOW_SIZE_S"))
    ETIQUETA = dict(zip(tam["WINDOW_SIZE_S"], tam["WINDOW_LABEL"]))
    SIZE_S = st.select_slider(tr("Intervalo de análisis", "Analysis interval"), list(ETIQUETA), value=list(ETIQUETA)[0],
                              format_func=ETIQUETA.get, key="size")
    aviso_periodo = st.container()   # se llena después de calcular el periodo seleccionado
    st.divider()
    pm_all = pm_idioma(leer_postmortems(MES))
    pm_stream_n = int((pm_all["STREAM"].astype(str).str.lower() == STREAM).sum()) if not pm_all.empty else 0
    st.caption(tr(f"Reportes forenses del periodo: **{len(pm_all)}** ({pm_stream_n} de {STREAMS[STREAM]})",
                  f"Forensic reports in the period: **{len(pm_all)}** ({pm_stream_n} from {STREAMS[STREAM]})"))
    if st.button(tr("Actualizar datos en vivo", "Refresh live data"), icon=":material/refresh:", width="stretch"):
        st.cache_data.clear()
        st.rerun()

win_all = leer("WINDOWS_SYSTEM" if STREAM == "system" else "WINDOWS_LLM", MES)
WIN = win_all[(win_all["SOURCE_TABLE"] == FUENTE) & (win_all["WINDOW_SIZE_S"] == SIZE_S)] \
    .sort_values("WINDOW_START")
EV = eventos_all[(eventos_all["SOURCE_TABLE"] == FUENTE) & (eventos_all["STREAM"] == STREAM)
                 & (eventos_all["WINDOW_SIZE_S"] == SIZE_S)].copy()
EV["CAMBIOS"] = EV["TOP3"].map(texto_cambios)
EV["INDICADOR"] = EV["Z_MAX_FEATURE"].map(nombre)
EV["FIN_REAL"] = EV["FIN"] + pd.Timedelta(seconds=int(SIZE_S))
REL = EV[EV["RELEVANTE"] == 1]
RES_S = resumen_all[(resumen_all["SOURCE_TABLE"] == FUENTE) & (resumen_all["STREAM"] == STREAM)]
RES = RES_S[RES_S["WINDOW_SIZE_S"] == SIZE_S]
Z_UMBRAL = float(RES["Z_UMBRAL"].iloc[0]) if len(RES) else 5.0
VALID = valid_all[(valid_all["SOURCE_TABLE"] == FUENTE) & (valid_all["STREAM"] == STREAM)]

# ---------------------------------------------------------------------------
# Periodo seleccionado (filtro global)
# Se elige en la gráfica de actividad del Resumen (zoom o barra inferior) y se aplica a todas las
# páginas. Se conserva al cambiar de origen o de intervalo y se borra al cambiar de mes.
# ---------------------------------------------------------------------------
def limpiar_periodo():
    st.session_state.pop("periodo_sel", None)
    st.session_state["tl_ver"] = st.session_state.get("tl_ver", 0) + 1   # gráfica nueva, sin zoom


CLAVE_TL = f"tl_{MES}_{STREAM}_{SIZE_S}_{st.session_state.get('tl_ver', 0)}"
_val = st.session_state.get(CLAVE_TL)
if _val is not None and st.session_state.get("tl_procesado") != (CLAVE_TL, _val):
    st.session_state["tl_procesado"] = (CLAVE_TL, _val)       # cada evento de la gráfica, una vez
    if _val.get("todo"):
        st.session_state.pop("periodo_sel", None)
    elif _val.get("x0") and _val.get("x1"):
        st.session_state["periodo_sel"] = (MES, pd.to_datetime(_val["x0"]), pd.to_datetime(_val["x1"]))

# Datos completos del mes (la gráfica de actividad siempre muestra todo el mes)
WIN_MES, EV_MES, REL_MES, PM_MES = WIN, EV, REL, pm_all
T_MIN, T_MAX = WIN_MES["WINDOW_START"].min(), WIN_MES["WINDOW_START"].max()
PERIODO = None
_sel = st.session_state.get("periodo_sel")
if _sel and _sel[0] == MES and (_sel[1] > T_MIN or _sel[2] < T_MAX):
    PERIODO = (max(_sel[1], T_MIN), min(_sel[2], T_MAX))


def en_periodo(df, ini="INICIO", fin="FIN_REAL"):
    """Filas cuyo intervalo [ini, fin] se cruza con el periodo seleccionado."""
    if PERIODO is None or df.empty:
        return df
    return df[(df[ini] <= PERIODO[1]) & (df[fin] >= PERIODO[0])]


if PERIODO:
    WIN = WIN_MES[WIN_MES["WINDOW_START"].between(*PERIODO)]
    EV, REL = en_periodo(EV_MES), en_periodo(REL_MES)
    pm_all = en_periodo(PM_MES, "INICIO", "FIN")
    VALID = VALID.assign(_FIN=VALID["WEEK_START"] + pd.Timedelta(days=7))
    VALID = en_periodo(VALID, "WEEK_START", "_FIN").drop(columns="_FIN")
    TXT_PERIODO = f"{PERIODO[0]:%d/%m %H:%M} – {PERIODO[1]:%d/%m %H:%M}"
    with aviso_periodo:
        st.html(f'<div class="side-periodo"><b>{tr("Periodo seleccionado", "Selected period")}</b>{TXT_PERIODO}'
                f'<span>{tr("Todas las páginas muestran solo este periodo.", "All pages show only this period.")}'
                '</span></div>')
        st.button(tr("Ver todo el mes", "Show whole month"), icon=":material/zoom_out_map:", width="stretch",
                  key="periodo_limpiar_lateral", on_click=limpiar_periodo)
else:
    TXT_PERIODO = None

# Gráficas
def fig_linea_tiempo(height=400, rango=None):
    WIN, REL = WIN_MES, REL_MES   # la gráfica muestra todo el mes; el zoom marca el periodo
    col = VOLUMEN[STREAM]
    fig = go.Figure()
    # ancho mínimo de 30 min para que el evento se vea (igual que graficar_eventos del notebook)
    piso = pd.Timedelta(minutes=30)
    for e in REL.sort_values("INICIO").itertuples():
        ini, fin = e.INICIO, e.FIN + pd.Timedelta(seconds=int(SIZE_S))
        if fin - ini < piso:
            mid = ini + (fin - ini) / 2
            ini, fin = mid - piso / 2, mid + piso / 2
        fig.add_vrect(x0=ini, x1=fin, fillcolor=COLOR_TIPO.get(e.TIPO_ATAQUE, NEUTRO),
                      opacity=0.35, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=WIN["WINDOW_START"], y=WIN[col], mode="lines", name=tr("Registros por bloque", "Logs per block"),
        line=dict(color="#1B90FF", width=1.3),   # azul vivo de SAP Horizon
        hovertemplate="%{x|%d-%b %H:%M}<br>%{y:,.0f} " + tr("registros", "logs") + "<extra></extra>"))
    a = WIN[WIN["LABEL"] == -1]
    fig.add_trace(go.Scatter(
        x=a["WINDOW_START"], y=a[col], mode="markers", name=tr("Bloque atípico", "Atypical block"),
        marker=dict(color=CRITICO, size=8, line=dict(color=SURFACE, width=2)),
        customdata=np.stack([a["EVENTO_ID"].fillna(-1).astype(int),
                             a["TIPO_ATAQUE"].map(lambda t: tipo_txt(t) if pd.notna(t) else
                                                  tr("Evento no relevante", "Non-relevant event"))],
                            axis=-1)
        if len(a) else None,
        hovertemplate="%{x|%d-%b %H:%M}<br>%{y:,.0f} " + tr("registros", "logs") + "<br>"
                      + tr("Evento", "Event") + " %{customdata[0]} · %{customdata[1]}<extra></extra>"))
    for t in sorted(REL["TIPO_ATAQUE"].dropna().unique()):   # leyenda de las franjas
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=tipo_txt(t),
                                 marker=dict(color=COLOR_TIPO.get(t, NEUTRO), size=12,
                                             symbol="square", opacity=0.5)))
    for w, g in WIN.groupby("WEEK"):
        if w > 0:
            fig.add_vline(x=g["WINDOW_START"].min(), line=dict(color=MUTED, dash="dot", width=1))
    fig.update_yaxes(title_text=tr(f"Registros cada {ETIQUETA[SIZE_S]}", f"Logs every {ETIQUETA[SIZE_S]}"),
                     title_font_color=MUTED, rangemode="tozero")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(0,0,0,0)"))
    if rango:
        fig.update_xaxes(range=[rango[0], rango[1]])
    # uirevision: conserva el zoom del usuario entre recálculos; cambia con "Ver todo"
    fig.update_layout(uirevision=st.session_state.get("tl_ver", 0))
    return estilo(fig, height=height)


def fig_tipos(rel=None):
    rel = REL if rel is None else rel
    ct = (rel.groupby(["TIPO_ATAQUE", "CONFIANZA"]).size().unstack(fill_value=0)
          .reindex(columns=["alta", "media", "baja"], fill_value=0))
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    etiquetas = [tipo_txt(t) for t in ct.index]
    fig = go.Figure()
    for conf in ct.columns:
        nom_conf = tr(f"Confianza {conf}", f"{conf_txt(conf).capitalize()} confidence")
        fig.add_trace(go.Bar(
            y=etiquetas, x=ct[conf], orientation="h", name=nom_conf,
            marker=dict(color=CONFIANZA_COLOR[conf], line=dict(color=SURFACE, width=2),
                        cornerradius=4),
            hovertemplate="%{y}<br>" + nom_conf + ": %{x} " + tr("eventos", "events") + "<extra></extra>"))
    tot = ct.sum(axis=1)
    tot.index = etiquetas
    fig.add_trace(go.Scatter(y=tot.index, x=tot.values, mode="text", text=tot.values,
                             textposition="middle right", textfont_color=INK2,
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text=tr("Número de eventos", "Number of events"), title_font_color=MUTED, dtick=1,
                     range=[0, tot.max() * 1.15])
    return estilo(fig, height=max(260, 62 * len(ct) + 120))


FAMILIA_EN = {
    "Volumen y ritmo": "Volume and pace", "Origen (IPs)": "Source (IPs)", "Tipo de log": "Log type",
    "Respuestas HTTP": "HTTP responses", "Peticiones": "Requests", "Contexto SAP": "SAP context",
    "Volumen y recursos": "Volume and resources", "Fallos": "Failures",
    "Contenido del prompt": "Prompt content", "Parámetros de la llamada": "Call parameters",
    "Mensajes de seguridad": "Security messages",
}


def fig_perfil(perfil):
    tipo_ev = REL.set_index("EVENTO_ID")["TIPO_ATAQUE"]
    filas = [f"{tr('Evento', 'Event')} {i} · {tipo_txt(tipo_ev.get(i, '—'))}" for i in perfil.index]
    fig = go.Figure(go.Heatmap(
        z=perfil.values, x=[FAMILIA_EN.get(c, c) if IDIOMA == "en" else c for c in perfil.columns], y=filas,
        colorscale=[[i / (len(AZUL_SEQ) - 1), c] for i, c in enumerate(AZUL_SEQ)],
        xgap=2, ygap=2, text=np.round(perfil.values, 1), texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: " + tr("cambio de", "change of") + " %{z:.1f}<extra></extra>",
        colorbar=dict(title=dict(text=tr("Cambio", "Change"), font_color=MUTED), tickfont_color=MUTED,
                      thickness=10, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return estilo(fig, height=max(240, 36 * len(perfil) + 110), legend=False,
                  margin=dict(l=8, r=8, t=48, b=8))


def tabla_eventos(df, alto=None):
    df = df.assign(TIPO_ATAQUE=df["TIPO_ATAQUE"].map(tipo_txt), CONFIANZA=df["CONFIANZA"].map(conf_txt))
    st.dataframe(
        df.sort_values("Z_MAX", ascending=False)[
            ["EVENTO_ID", "TIPO_ATAQUE", "CONFIANZA", "INICIO", "FIN_REAL", "Z_MAX",
             "INDICADOR", "CAMBIOS"]],
        hide_index=True, height=alto or "auto",
        column_config={
            "EVENTO_ID": st.column_config.NumberColumn(tr("Evento", "Event"), format="%d", width="small"),
            "TIPO_ATAQUE": tr("Tipo", "Type"), "CONFIANZA": tr("Confianza", "Confidence"),
            "INICIO": st.column_config.DatetimeColumn(tr("Inicio", "Start"), format="DD-MMM HH:mm"),
            "FIN_REAL": st.column_config.DatetimeColumn(tr("Fin", "End"), format="DD-MMM HH:mm"),
            "Z_MAX": st.column_config.ProgressColumn(
                tr("Intensidad", "Intensity"), format="%.1f", min_value=0,
                max_value=float(EV["Z_MAX"].max()) if len(EV) else 1.0),
            "INDICADOR": tr("Indicador más afectado", "Most affected indicator"),
            "CAMBIOS": st.column_config.TextColumn(tr("Principales cambios", "Main changes"), width="large"),
        })

# Secciones
def pagina_resumen():
    encabezado(tr("Resumen", "Overview"), tr("Detección de anomalías en SAP", "Anomaly detection in SAP"), " ")
    rango = PERIODO
    rel_r = REL
    if rango:
        ini, fin = rango
        n_win, n_anom = len(WIN), int((WIN["LABEL"] == -1).sum())
        n_ev, n_rel = len(EV), len(REL)
        tipos_ataque = sorted(set(REL["TIPO_ATAQUE"].dropna()) - {"Falla operativa"})
        det_bloques = tr(f"del {ini:%d/%m %H:%M} al {fin:%d/%m %H:%M}", f"from {ini:%d/%m %H:%M} to {fin:%d/%m %H:%M}")
    else:
        # Todo sale de RESULT_SUMMARY (una fila por semana): se suman las semanas del periodo.
        n_win = int(RES["VENTANAS"].sum())
        n_anom = int(RES["VENTANAS_ANOMALAS"].sum())
        n_ev = int(RES["EVENTOS"].sum())
        n_rel = int(RES["EVENTOS_RELEVANTES"].sum())
        # TIPOS_LISTA sin "Falla operativa", que resultados.py (NO_ATAQUE) no cuenta como ataque
        tipos_ataque = sorted({t.strip() for l in RES["TIPOS_LISTA"].dropna() for t in l.split(",")}
                              - {"Falla operativa"})
        det_bloques = tr(f"{len(RES)} semana(s)", f"{len(RES)} week(s)")
    kpis = [
        (tr("Bloques analizados", "Analyzed blocks"), f"{n_win:,}", det_bloques,
         BRAND, tr(f"La actividad se divide en bloques de {ETIQUETA[SIZE_S]} para analizarla.",
                   f"Activity is split into {ETIQUETA[SIZE_S]} blocks for analysis.")),
        (tr("Bloques atípicos", "Atypical blocks"), f"{n_anom:,}",
         tr(f"{n_anom / n_win:.1%} del total", f"{n_anom / n_win:.1%} of total") if n_win else "—",
         "#E76500", tr("Bloques cuyo comportamiento se aleja claramente del habitual.",
                       "Blocks whose behavior clearly departs from the usual.")),
        (tr("Eventos", "Events"), f"{n_ev:,}", tr("detectados", "detected"), "#7858FF",
         tr("Los bloques atípicos cercanos en el tiempo se agrupan en un solo evento.",
            "Atypical blocks close in time are grouped into a single event.")),
        (tr("Eventos relevantes", "Relevant events"), f"{n_rel:,}",
         tr(f"intensidad de {Z_UMBRAL:g} o más", f"intensity of {Z_UMBRAL:g} or more"), CRITICO,
         tr(f"Eventos con una intensidad de {Z_UMBRAL:g} o más.", f"Events with an intensity of {Z_UMBRAL:g} or more.")),
        (tr("Tipos de ataque", "Attack types"), f"{len(tipos_ataque)}",
         ", ".join(tipo_txt(t) for t in tipos_ataque) or tr("ninguno", "none"), "#049F9A",
         tr("Tipos de ataque asignados a los eventos relevantes (sin contar Falla operativa).",
            "Attack types assigned to relevant events (excluding Operational failure).")),
    ]
    st.html('<div class="kpis">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{ayuda}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    with tarjeta(tr("Actividad y eventos en el tiempo", "Activity and events over time"),
                 tr("Volumen de registros a lo largo del periodo. Las franjas de color señalan los "
                    "eventos relevantes según su tipo y los puntos rojos, los bloques atípicos. "
                    "Usa la barra inferior o arrastra sobre la gráfica para acercar un periodo: "
                    "todo el dashboard se recalcula para ese periodo. Doble clic para ver todo.",
                    "Log volume over the period. Colored bands mark relevant events by type and red "
                    "dots mark atypical blocks. Use the bottom bar or drag on the chart to zoom into a "
                    "period: the whole dashboard is recalculated for it. Double-click to see everything.")):
        if rango:
            r1, r2 = st.columns([4, 1], vertical_alignment="center", gap="medium")
            r1.html(f'<div class="pm-chips"><span class="pm-chip rango">{tr("Periodo seleccionado", "Selected period")}: '
                    f'{ini:%d/%m/%Y %H:%M} – {fin:%d/%m/%Y %H:%M}</span></div>')
            r2.button(tr("Ver todo", "Show all"), icon=":material/zoom_out_map:", width="stretch",
                      key="tl_limpiar", on_click=limpiar_periodo)
        grafica_rango(fig=fig_linea_tiempo(rango=rango).to_json(), alto=400, key=CLAVE_TL, default=None)

    sufijo = tr(" en el periodo seleccionado", " in the selected period") if rango else ""
    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1, tarjeta(tr("Eventos relevantes por tipo", "Relevant events by type"),
                     tr("Cuántos eventos se asignaron a cada tipo y con qué nivel de confianza",
                        "How many events were assigned to each type and with what confidence") + sufijo + "."):
        if rel_r.empty:
            vacio(tr("No hay eventos relevantes", "There are no relevant events")
                  + (sufijo or tr(" con este intervalo de análisis", " with this analysis interval")) + ".")
        else:
            mostrar(fig_tipos(rel_r))
    with c2, tarjeta(tr("Eventos de mayor intensidad", "Most intense events"),
                     tr("Los cinco eventos que más se alejaron del comportamiento habitual",
                        "The five events that departed most from usual behavior") + sufijo + ". "
                     + tr("Consulta el detalle completo en la sección Eventos.",
                          "See full details in the Events section.")):
        if rel_r.empty:
            vacio(tr("No hay eventos relevantes", "There are no relevant events") + sufijo + ".")
        else:
            top = rel_r.nlargest(5, "Z_MAX").assign(TIPO_ATAQUE=lambda d: d["TIPO_ATAQUE"].map(tipo_txt))
            st.dataframe(
                top[["EVENTO_ID", "TIPO_ATAQUE", "INICIO", "Z_MAX", "INDICADOR"]],
                hide_index=True,
                column_config={
                    "EVENTO_ID": st.column_config.NumberColumn(tr("Evento", "Event"), format="%d", width="small"),
                    "TIPO_ATAQUE": tr("Tipo", "Type"),
                    "INICIO": st.column_config.DatetimeColumn(tr("Inicio", "Start"), format="DD-MMM HH:mm"),
                    "Z_MAX": st.column_config.ProgressColumn(
                        tr("Intensidad", "Intensity"), format="%.1f", min_value=0,
                        max_value=float(EV["Z_MAX"].max())),
                    "INDICADOR": tr("Indicador más afectado", "Most affected indicator"),
                })

    pm_r = pm_all
    if not PM_MES.empty:
        n_tipos_pm = int(pm_r["TIPO_ATAQUE"].dropna().astype(str).nunique())
        with tarjeta(tr("Reportes forenses", "Forensic reports"),
                     tr("Reportes que generó el agente forense para los eventos relevantes",
                        "Reports generated by the forensic agent for the relevant events")
                     + (sufijo if rango else tr(" del periodo", " in the period")) + "."):
            k1, k2 = st.columns([4, 1], vertical_alignment="center", gap="medium")
            with k1:
                st.html('<div class="pm-chips">'
                        f'<span class="pm-chip">{len(pm_r)} {tr("reporte(s)", "report(s)")}'
                        f'{tr(" en el rango", " in the range") if rango else tr(" en el periodo", " in the period")}</span>'
                        f'<span class="pm-chip">{n_tipos_pm} {tr("tipo(s) de ataque", "attack type(s)")}</span></div>')
            with k2:
                st.page_link(pg_postmortems, label=tr("Ver reportes", "View reports"), icon=":material/arrow_forward:")


def pagina_eventos():
    encabezado(tr("Eventos", "Events"), tr("Eventos detectados", "Detected events"),
               tr("Un evento reúne bloques atípicos cercanos en el tiempo. Aquí se muestra qué "
                  "aspectos de la actividad cambiaron y el detalle de cada uno.",
                  "An event groups atypical blocks that are close in time. This page shows which "
                  "aspects of the activity changed and the details of each event."))
    if EV.empty:
        with tarjeta():
            vacio(tr("No se detectaron eventos con los filtros seleccionados",
                     "No events were detected with the selected filters")
                  + (tr(" en el periodo seleccionado.", " in the selected period.") if PERIODO else "."))
        return

    with tarjeta(tr("Aspectos afectados en cada evento", "Affected aspects in each event"),
                 tr("Qué tanto cambió cada grupo de indicadores durante el evento, comparado con una "
                    "semana normal. Los tonos más oscuros indican un cambio mayor.",
                    "How much each group of indicators changed during the event compared with a normal "
                    "week. Darker shades indicate a larger change.")):
        perfil = perfil_relevantes(WIN_MES, REL, FAMILIAS[STREAM])   # z contra la semana completa
        if perfil.empty:
            vacio(tr("No hay eventos relevantes para mostrar.", "There are no relevant events to show."))
        else:
            mostrar(fig_perfil(perfil))

    with tarjeta(tr("Detalle de eventos", "Event details"),
                 tr("Ordenados por intensidad. La última columna resume los indicadores que más "
                    "cambiaron (↑ aumentó, ↓ disminuyó).",
                    "Sorted by intensity. The last column summarizes the indicators that changed the "
                    "most (↑ increased, ↓ decreased).")):
        t1, t2 = st.tabs([tr(f"Relevantes ({len(REL)})", f"Relevant ({len(REL)})"),
                          tr(f"Todos ({len(EV)})", f"All ({len(EV)})")])
        with t1:
            if REL.empty:
                vacio(tr("No se detectaron eventos relevantes.", "No relevant events were detected."))
            else:
                tabla_eventos(REL)
        with t2:
            tabla_eventos(EV)

def pagina_tipos():
    encabezado(tr("Tipos de ataque", "Attack types"),
               tr("Tipos de ataque que se pueden detectar", "Attack types that can be detected"),
               tr("Cada evento relevante se clasifica en uno de estos tipos según los indicadores que "
                  "más cambiaron. Los tipos posibles dependen del origen de los registros.",
                  "Each relevant event is classified into one of these types based on the indicators "
                  "that changed the most. The possible types depend on the log source."))
    with tarjeta(tr("Cómo se asigna el tipo", "How the type is assigned")):
        pasos = [
            (tr("1 · Indicadores principales", "1 · Main indicators"),
             tr("Se toman los 3 indicadores que más se alejaron de su comportamiento habitual durante "
                "el evento.", "The 3 indicators that departed most from their usual behavior during the "
                "event are taken.")),
            (tr("2 · Votación", "2 · Voting"),
             tr("Cada indicador vota por los tipos de ataque cuyo patrón coincide con la dirección del "
                "cambio (si subió o bajó). Su voto se reparte entre esos tipos.",
                "Each indicator votes for the attack types whose pattern matches the direction of the "
                "change (up or down). Its vote is split among those types.")),
            (tr("3 · Tipo asignado", "3 · Assigned type"),
             tr("Gana el tipo con más votos. Si gana por al menos 1 voto, la confianza es alta; por 0.5 "
                "a 1, media; por menos de 0.5, baja.",
                "The type with the most votes wins. If it wins by at least 1 vote, confidence is high; "
                "by 0.5 to 1, medium; by less than 0.5, low.")),
        ]
        for col, (t, d) in zip(st.columns(len(pasos), gap="medium"), pasos):
            col.html(f'<div class="paso"><b>{t}</b><p>{d}</p></div>')

    conteo = REL["TIPO_ATAQUE"].value_counts()
    firma = FIRMA[STREAM]
    # Orden del md; si las firmas del código tienen algún tipo más, también se muestra.
    tipos = TIPOS_POSIBLES[STREAM] + sorted({t for sg in firma.values() for t in sg}
                                            - set(TIPOS_POSIBLES[STREAM]))
    st.html(f'<p class="card-title" style="margin:.4rem 0 .8rem">'
            f'{tr(f"{len(tipos)} tipos posibles en {STREAMS[STREAM]}", f"{len(tipos)} possible types in {STREAMS[STREAM]}")}</p>')
    flecha = {"+": "↑", "-": "↓", "~": "↕"}
    texto_dir = {"+": tr("aumenta", "increases"), "-": tr("disminuye", "decreases"), "~": tr("cambia", "changes")}
    tarjetas = []
    for t in tipos:
        n = int(conteo.get(t, 0))
        donde = (tr("en el periodo seleccionado", "in the selected period") if PERIODO
                 else tr("en el periodo", "in the period"))
        chips = (f'<span class="chip alerta">{n} {tr("evento", "event")}{"s" if n != 1 else ""} {donde}</span>'
                 if n else f'<span class="chip nada">{tr("Sin eventos", "No events")} {donde}</span>')
        if t in TIPOS_NO_ATAQUE:
            chips += f'<span class="chip info">{tr("No cuenta como ataque", "Not counted as an attack")}</span>'
        senales = "".join(f'<li><span>{flecha[sg[t]]}</span>{nombre(f)} {texto_dir[sg[t]]}</li>'
                          for f, sg in firma.items() if t in sg)
        tarjetas.append(
            f'<div class="tipo" style="--c:{COLOR_TIPO.get(t, NEUTRO)}">'
            f'<div class="nom"><i></i>{tipo_txt(t)}</div>'
            f'<p class="desc">{DESCRIPCION_TIPO.get(t, "")}</p>'
            f'<div class="chips">{chips}</div>'
            f'<div class="sen"><p>{tr("Indicadores que lo señalan", "Indicators that point to it")}</p>'
            f'<ul>{senales}</ul></div></div>')
    st.html('<div class="tipos-grid">' + "".join(tarjetas) + '</div>')


def pagina_modelo():
    encabezado(tr("Metodología", "Methodology"),
               tr("Cómo se detectan y clasifican los eventos", "How events are detected and classified"),
               tr("Los pasos del análisis y dos comprobaciones de calidad: cómo cambian los resultados "
                  "según el intervalo de análisis y qué tan consistente es la clasificación.",
                  "The analysis steps and two quality checks: how results change with the analysis "
                  "interval and how consistent the classification is."))
    with tarjeta(tr("Proceso de análisis", "Analysis process")):
        pasos = [
            (tr("1 · División en bloques", "1 · Splitting into blocks"),
             tr("La actividad se divide en bloques de tiempo de 5 min, 15 min, 30 min, 1 h y 3 h, que se "
                "traslapan a la mitad. En cada bloque se miden 24 indicadores (Sistema SAP) o 18 "
                "(Servicios de IA).",
                "Activity is split into 5 min, 15 min, 30 min, 1 h and 3 h time blocks that overlap by "
                "half. Each block measures 24 indicators (SAP System) or 18 (AI Services).")),
            (tr("2 · Detección de lo atípico", "2 · Detecting the atypical"),
             tr("Un modelo de aprendizaje automático (Isolation Forest) identifica cada semana el 1 % de "
                "bloques con el comportamiento más inusual.",
                "A machine learning model (Isolation Forest) flags, every week, the 1% of blocks with the "
                "most unusual behavior.")),
            (tr("3 · Agrupación en eventos", "3 · Grouping into events"),
             tr("Los bloques atípicos separados por 3 bloques o menos se unen en un evento. Es relevante "
                f"si algún indicador se alejó {Z_UMBRAL:g} desviaciones estándar o más de su promedio "
                "semanal.",
                "Atypical blocks 3 blocks apart or less are merged into one event. It is relevant if any "
                f"indicator moved {Z_UMBRAL:g} standard deviations or more from its weekly mean.")),
            (tr("4 · Clasificación", "4 · Classification"),
             tr("Los 3 indicadores que más cambiaron se comparan con el patrón de cada tipo de ataque "
                "mediante reglas fijas, y se asigna el tipo que mejor coincide.",
                "The 3 indicators that changed the most are compared with each attack type's pattern "
                "using fixed rules, and the best-matching type is assigned.")),
            (tr("5 · Comprobación", "5 · Verification"),
             tr("Un agrupamiento automático (k-means) de los bloques atípicos se compara con la "
                "clasificación para ver si coinciden.",
                "An automatic clustering (k-means) of the atypical blocks is compared with the "
                "classification to see whether they agree.")),
        ]
        for col, (t, d) in zip(st.columns(len(pasos), gap="medium"), pasos):
            col.html(f'<div class="paso"><b>{t}</b><p>{d}</p></div>')

    c3, c4 = st.columns(2, gap="medium")
    with c3, tarjeta(tr("Resultados según el intervalo de análisis", "Results by analysis interval"),
                     tr("Con intervalos cortos se detectan más eventos breves; con intervalos largos, "
                        "solo los que se sostienen en el tiempo.",
                        "Short intervals detect more brief events; long intervals only detect those "
                        "sustained over time.")):
        ev_s = eventos_all[(eventos_all["SOURCE_TABLE"] == FUENTE) & (eventos_all["STREAM"] == STREAM)]
        ev_s = en_periodo(ev_s.assign(FIN_REAL=ev_s["FIN"] + pd.to_timedelta(ev_s["WINDOW_SIZE_S"], unit="s")))
        s = (ev_s.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])
             .agg(EVENTOS=("EVENTO_ID", "size"), EVENTOS_RELEVANTES=("RELEVANTE", "sum"))
             .reindex(pd.MultiIndex.from_frame(RES_S[["WINDOW_SIZE_S", "WINDOW_LABEL"]].drop_duplicates()),
                      fill_value=0)
             .reset_index().sort_values("WINDOW_SIZE_S"))
        fig = go.Figure()
        for colname, nombre, color in (("EVENTOS", tr("Eventos detectados", "Detected events"), SERIES[0]),
                                       ("EVENTOS_RELEVANTES", tr("Eventos relevantes", "Relevant events"), SERIES[2])):
            fig.add_trace(go.Bar(
                x=s["WINDOW_LABEL"], y=s[colname], name=nombre,
                marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                hovertemplate=tr("Intervalo de", "Interval of") + " %{x}<br>" + nombre + ": %{y}<extra></extra>"))
        fig.update_layout(barmode="group", bargroupgap=0.08)
        fig.update_yaxes(title_text=tr("Número de eventos", "Number of events"), title_font_color=MUTED)
        fig.update_xaxes(title_text=tr("Intervalo de análisis", "Analysis interval"), title_font_color=MUTED)
        mostrar(estilo(fig, height=360))

    with c4, tarjeta(tr("Consistencia de la clasificación", "Classification consistency"),
                     tr("Qué tanto coincide la clasificación con un agrupamiento automático (k-means) "
                        "de los bloques atípicos de los eventos relevantes, según dos índices (ARI y "
                        "NMI): 1 es coincidencia total y 0, al azar. No es una validación "
                        "independiente, porque ambos métodos usan los mismos indicadores.",
                        "How well the classification agrees with an automatic clustering (k-means) of "
                        "the atypical blocks of relevant events, by two indices (ARI and NMI): 1 is "
                        "full agreement and 0 is random. It is not an independent validation, since "
                        "both methods use the same indicators.")):
        ve = VALID[VALID["ESTADO"] == "evaluable"]
        if ve.empty:
            vacio(tr("No hay datos suficientes para evaluar la consistencia en este periodo; se "
                     "necesitan al menos dos tipos de evento.",
                     "There is not enough data to assess consistency in this period; at least two "
                     "event types are needed."))
        else:
            g = (ve.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])[["ARI", "NMI"]].mean()
                 .reset_index().sort_values("WINDOW_SIZE_S"))
            n_sem = ve.groupby("WINDOW_SIZE_S").size()
            fig = go.Figure()
            for m, color in (("ARI", SERIES[0]), ("NMI", SERIES[4])):
                indice = tr(f"Índice {m}", f"{m} index")
                fig.add_trace(go.Bar(
                    x=g["WINDOW_LABEL"], y=g[m], name=indice,
                    marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                    customdata=n_sem.reindex(g["WINDOW_SIZE_S"]).values,
                    hovertemplate=tr("Intervalo de", "Interval of") + " %{x}<br>" + indice + ": %{y:.2f}<br>"
                                  + tr("Promedio de %{customdata} semana(s)", "Average of %{customdata} week(s)")
                                  + "<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=AXIS, width=1))
            fig.update_layout(barmode="group", bargroupgap=0.08)
            fig.update_xaxes(title_text=tr("Intervalo de análisis", "Analysis interval"), title_font_color=MUTED)
            fig.update_yaxes(title_text=tr("Coincidencia (0 a 1)", "Agreement (0 to 1)"), title_font_color=MUTED,
                             range=[min(0, g[["ARI", "NMI"]].min().min() - 0.05), 1])
            mostrar(estilo(fig, height=360))
        no_eval = int((VALID["ESTADO"] != "evaluable").sum())
        if no_eval:
            st.caption(tr(f"En {no_eval} de {len(VALID)} combinaciones de semana e intervalo no "
                          "hubo datos suficientes para esta comparación.",
                          f"In {no_eval} of {len(VALID)} week-interval combinations there was not "
                          "enough data for this comparison.")
                       + (tr(" Con un periodo seleccionado se usan las semanas que lo incluyen.",
                             " With a selected period, the weeks that include it are used.") if PERIODO else ""))

def titulo_pm(periodo):
    """Título de Postmortems; "— AAAA-MM" nunca se parte entre dos líneas."""
    return f'{tr("Resumen forense mensual", "Monthly forensic summary")} <span style="white-space: nowrap">— {periodo}</span>'


def _rango(ini, fin):
    """Fecha y horario del incidente en dos líneas cortas."""
    if pd.isna(ini) or pd.isna(fin):
        return _fmt_ts(ini), ""
    if ini.date() == fin.date():
        return f"{ini:%d/%m/%Y}", tr(f"de {ini:%H:%M} a {fin:%H:%M}", f"from {ini:%H:%M} to {fin:%H:%M}")
    return f"{ini:%d/%m/%Y %H:%M}", tr(f"a {fin:%d/%m/%Y %H:%M}", f"to {fin:%d/%m/%Y %H:%M}")


def _sin_titulo(md):
    """Quita el título principal del reporte: ya aparece en el encabezado de la tarjeta."""
    lineas = md.lstrip().splitlines()
    if lineas and lineas[0].startswith("# "):
        lineas = lineas[1:]
    return "\n".join(lineas).lstrip()


def _chips(items, vacio_txt):
    if not items:
        return f'<span class="pm-vacio">{vacio_txt}</span>'
    return '<div class="pm-chips">' + "".join(
        f'<span class="pm-chip">{html.escape(x)}</span>' for x in items) + '</div>'


def pagina_postmortems():
    periodo = MES.replace("_", "-")
    if pm_all.empty and not PM_MES.empty:
        encabezado("Postmortems SOC", titulo_pm(MES.replace('_', '-')),
                   tr("Consolidado ejecutivo de todos los incidentes investigados en el periodo.", "Executive consolidation of all incidents investigated in the period."),
                   pills=[(tr("Periodo", "Period"), MES.replace("_", "-")), (tr("Reportes", "Reports"), 0)])
        vacio(tr(f"No hay reportes en el periodo seleccionado ({TXT_PERIODO}). "
                 f"El mes tiene {len(PM_MES)}; usa Ver todo el mes en la barra lateral para verlos.",
                 f"There are no reports in the selected period ({TXT_PERIODO}). The month has "
                 f"{len(PM_MES)}; use Show whole month in the sidebar to see them."))
        return
    if pm_all.empty:
        encabezado("Postmortems SOC", titulo_pm(periodo),
                   tr("Consolidado ejecutivo de todos los incidentes investigados en el periodo.", "Executive consolidation of all incidents investigated in the period."),
                   pills=[(tr("Periodo", "Period"), periodo), (tr("Reportes", "Reports"), 0)])
        vacio(tr(f"Aún no hay reportes para {periodo}. Aparecerán aquí cuando el agente forense "
                 "(cron_job_worker.py) procese los eventos relevantes del periodo.",
                 f"There are no reports for {periodo} yet. They will appear here when the forensic agent "
                 "(cron_job_worker.py) processes the relevant events of the period."))
        return

    header_slot = st.container()

    # Filtros
    with tarjeta():
        f1, f2, f3 = st.columns([1, 1.4, 1.4], gap="medium")
        origenes = ["todos"] + sorted(pm_all["STREAM"].dropna().astype(str).str.lower().unique())
        sel_origen = f1.selectbox(tr("Origen", "Source"), origenes, key=f"pm_origen_{IDIOMA}",
                                  format_func=lambda s: tr("Todos", "All") if s == "todos" else STREAMS.get(s, s))
        sel_tipos = f2.multiselect(tr("Tipo de ataque", "Attack type"),
                                   sorted(pm_all["TIPO_ATAQUE"].dropna().astype(str).unique()),
                                   placeholder=tr("Todos los tipos", "All types"), key=f"pm_tipos_{IDIOMA}")
        q = f3.text_input(tr("Buscar", "Search"),
                          placeholder=tr("Reporte, táctica MITRE, aplicación o IP", "Report, MITRE tactic, application or IP"),
                          key="pm_buscar").strip().lower()

    pm_f = pm_all.copy()
    if sel_origen != "todos":
        pm_f = pm_f[pm_f["STREAM"].astype(str).str.lower() == sel_origen]
    if sel_tipos:
        pm_f = pm_f[pm_f["TIPO_ATAQUE"].astype(str).isin(sel_tipos)]
    if q:
        cols = ["POSTMORTEM_ID", "TIPO_ATAQUE", "MITRE_TACTIC", "APLICACIONES_AFECTADAS",
                "IPS_INVOLUCRADAS", "RESUMEN_EJECUTIVO"]
        texto = pm_f[[c for c in cols if c in pm_f.columns]].fillna("").astype(str) \
            .agg(" ".join, axis=1).str.lower()
        pm_f = pm_f[texto.str.contains(q, regex=False)]
    if pm_f.empty:
        with header_slot:
            encabezado("Postmortems SOC", titulo_pm(periodo),
                       tr("Consolidado ejecutivo de todos los incidentes investigados en el periodo.", "Executive consolidation of all incidents investigated in the period."),
                       pills=[(tr("Periodo", "Period"), periodo), (tr("Reportes", "Reports"), len(pm_all))])
        vacio(tr("Ningún reporte coincide con los filtros.", "No report matches the filters."))
        return

    pm_f = pm_f.sort_values(["INICIO", "EVENTO_ID"], ascending=[True, True]) \
        .reset_index(drop=True)

    # Cabecera con botones globales de descarga y envio por correo del reporte mensual consolidado
    pdf_mes = None
    with header_slot:
        h_left, h_right = st.columns([1.45, 1.55], vertical_alignment="bottom", gap="medium")
        with h_left:
            encabezado("Postmortems SOC", titulo_pm(periodo),
                       tr("Resumen consolidado de todos los incidentes investigados del mes y "
                          "exportación ejecutiva multipágina.",
                          "Consolidated summary of all incidents investigated in the month and "
                          "multi-page executive export."),
                       pills=[(tr("Periodo", "Period"), periodo), (tr("Incidentes", "Incidents"), len(pm_f))])
        with h_right:
            b1, b2, b3 = st.columns(3, gap="small")
            with b1:
                pdf_mes = None      # el formulario de correo lo usa aunque falle la generación
                try:
                    pdf_mes = _construir_pdf_mensual_bytes(pm_f, periodo)
                    st.download_button(
                        tr("Descargar PDF", "Download PDF"),
                        data=pdf_mes,
                        file_name=f"Postmortem_SOC_{MES}{'_EN' if IDIOMA == 'en' else ''}.pdf",
                        mime="application/pdf",
                        icon=":material/picture_as_pdf:",
                        type="primary",
                        width="stretch",
                        help=tr("Genera el PDF ejecutivo multipágina con el resumen del mes y el detalle de cada incidente.",
                                "Generates the multi-page executive PDF with the monthly summary and each incident's details."),
                    )
                except Exception as err:
                    st.warning(tr("No se pudo generar el PDF consolidado: ", "The consolidated PDF could not be generated: ") + str(err))
            with b2:
                md_mes = _construir_markdown_mensual(pm_f, periodo)
                st.download_button(
                    tr("Descargar Markdown", "Download Markdown"),
                    data=md_mes.encode("utf-8"),
                    file_name=f"Postmortem_SOC_{MES}{'_EN' if IDIOMA == 'en' else ''}.md",
                    mime="text/markdown",
                    icon=":material/description:",
                    width="stretch",
                    help=tr("Descarga el consolidado completo del mes en Markdown.",
                            "Downloads the full monthly consolidation in Markdown."),
                )
            with b3:
                if st.button(
                    tr("Enviar por correo", "Send by email"),
                    icon=":material/mail:",
                    width="stretch",
                    key="btn_toggle_pm_email",
                    help=tr("Abre el formulario para enviar este reporte PDF por correo electrónico.",
                            "Opens the form to send this PDF report by email."),
                ):
                    st.session_state["pm_show_email"] = not st.session_state.get("pm_show_email", False)

        if st.session_state.get("pm_show_email", False):
            estado_mail = estado_servicio_correo()
            pdf_filename = f"Postmortem_SOC_{MES}{'_EN' if IDIOMA == 'en' else ''}.pdf"
            periodo_correo = f"{periodo} ({TXT_PERIODO})" if PERIODO else periodo
            n_tacticas_mail = int(pm_f["MITRE_TACTIC"].dropna().astype(str).nunique())
            apps_mail = sorted({a for v in pm_f["APLICACIONES_AFECTADAS"] for a in _parse_lista_campo(v)})
            size_kb = (len(pdf_mes) / 1024.0) if pdf_mes else 0.0

            with tarjeta(
                tr("Enviar reporte PDF por correo electrónico", "Send PDF report by email"),
                tr(f"Adjunto listo: {pdf_filename} ({size_kb:.1f} KB · {len(pm_f)} incidente(s)) · "
                   f"Proveedor: {estado_mail['provider_label']}.",
                   f"Attachment ready: {pdf_filename} ({size_kb:.1f} KB · {len(pm_f)} incident(s)) · "
                   f"Provider: {estado_mail['provider_label']}."),
            ):
                if not estado_mail["configured"]:
                    st.warning(estado_mail["hint"])
                elif estado_mail.get("sandbox"):
                    st.info(estado_mail["hint"])

                with st.form("form_enviar_pm_correo", clear_on_submit=False):
                    c_to, c_subj = st.columns([1.15, 1.0], gap="medium")
                    destinatario = c_to.text_input(
                        tr("Correo(s) destinatario(s)", "Recipient email(s)"),
                        placeholder="analista@sap.com, soc@empresa.com",
                        help=tr("Puedes ingresar uno o varios correos separados por coma.",
                                "You can enter one or more emails separated by commas."),
                        key="pm_email_to",
                    )
                    asunto_mail = c_subj.text_input(
                        tr("Asunto", "Subject"),
                        value=tr(f"Reporte Forense Postmortem SOC — {periodo_correo} ({len(pm_f)} incidentes)",
                                 f"SOC Forensic Postmortem Report — {periodo_correo} ({len(pm_f)} incidents)"),
                        key=f"pm_email_subject_{MES}_{len(pm_f)}_{bool(PERIODO)}_{IDIOMA}",
                    )
                    nota_mail = st.text_area(
                        tr("Mensaje adicional (opcional)", "Additional message (optional)"),
                        placeholder=tr("Ej. Compartimos el dossier forense mensual revisado para la sesión SOC.",
                                       "E.g. Sharing the reviewed monthly forensic dossier for the SOC session."),
                        height=68,
                        key="pm_email_note",
                    )
                    act_1, act_2, _ = st.columns([1.0, 0.75, 2.25], gap="small")
                    enviar_click = act_1.form_submit_button(
                        tr("Enviar PDF", "Send PDF"),
                        icon=":material/send:",
                        type="primary",
                        width="stretch",
                        disabled=(pdf_mes is None),
                    )
                    cerrar_click = act_2.form_submit_button(
                        tr("Cerrar", "Close"),
                        width="stretch",
                    )

                if cerrar_click:
                    st.session_state["pm_show_email"] = False
                    st.rerun()

                if enviar_click:
                    if not pdf_mes:
                        st.error(tr("No hay un PDF generado para adjuntar.", "There is no generated PDF to attach."))
                    else:
                        with st.spinner(tr("Enviando reporte PDF por correo...", "Sending PDF report by email...")):
                            try:
                                res_mail = enviar_postmortem_pdf(
                                    destinatarios_raw=destinatario,
                                    pdf_bytes=pdf_mes,
                                    filename=pdf_filename,
                                    periodo=periodo_correo,
                                    n_incidentes=len(pm_f),
                                    n_tacticas=n_tacticas_mail,
                                    n_apps=len(apps_mail),
                                    asunto=asunto_mail,
                                    nota_adicional=nota_mail,
                                )
                                recips = ", ".join(res_mail["recipients"])
                                msg_id = f" (ID: {res_mail['message_id']})" if res_mail.get("message_id") else ""
                                st.success(tr(
                                    f"Reporte {pdf_filename} enviado correctamente a {recips} "
                                    f"vía {res_mail['provider']}{msg_id}.",
                                    f"Report {pdf_filename} sent successfully to {recips} "
                                    f"via {res_mail['provider']}{msg_id}."))
                            except EmailSendError as err:
                                st.error(str(err))
                            except Exception as err:
                                st.error(tr("Error inesperado al enviar correo: ", "Unexpected error sending the email: ") + str(err))

    n_sin_en = int(pm_f["_SIN_EN"].sum()) if "_SIN_EN" in pm_f.columns else 0
    if n_sin_en:
        st.info(f"{n_sin_en} of {len(pm_f)} report(s) do not have an English version yet, so they are "
                "shown in Spanish. Run src/traducir_postmortems.py to translate them.",
                icon=":material/translate:")

    # KPIs del periodo (sin severidad)
    n_tacticas = int(pm_f["MITRE_TACTIC"].dropna().astype(str).nunique())
    apps = sorted({a for v in pm_f["APLICACIONES_AFECTADAS"] for a in _parse_lista_campo(v)})
    ips_mes = sorted({ip for v in pm_f["IPS_INVOLUCRADAS"] for ip in _parse_lista_campo(v)})
    kpis = [
        (tr("Incidentes del mes", "Incidents in the month"), f"{len(pm_f)}",
         tr(f"de {len(pm_all)} en el periodo", f"of {len(pm_all)} in the period"), BRAND,
         tr("Reportes forenses incluidos en el consolidado mensual.", "Forensic reports included in the monthly consolidation.")),
        (tr("Tácticas MITRE", "MITRE tactics"), f"{n_tacticas}", tr("tácticas distintas", "distinct tactics"), "#7858FF",
         tr("Tácticas MITRE ATT&CK / ATLAS asignadas en el periodo.", "MITRE ATT&CK / ATLAS tactics assigned in the period.")),
        (tr("Aplicaciones SAP", "SAP applications"), f"{len(apps)}",
         tr(f"{len(ips_mes)} IP(s) involucradas", f"{len(ips_mes)} IP(s) involved"), "#049F9A",
         ", ".join(apps) or tr("Sin aplicaciones registradas.", "No applications recorded.")),
    ]
    st.html('<div class="kpis kpis-3">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{html.escape(ayuda)}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    # Panorama consolidado del mes y resumen por campana
    esc = html.escape
    orden_esc = {"5min": 1, "15min": 2, "30min": 3, "1h": 4, "3h": 5}
    tacticas_mes = sorted(pm_f["MITRE_TACTIC"].dropna().astype(str).unique())
    tipos_mes = sorted(pm_f["TIPO_ATAQUE"].dropna().astype(str).unique())
    ini_global, fin_global = _rango(pm_f["INICIO"].min(), pm_f["FIN"].max())

    grupos_camp = sorted(pm_f.groupby("TIPO_ATAQUE"), key=lambda kv: (-len(kv[1]), str(kv[0])))
    resumen_camp_rows = []
    for vec, g in grupos_camp:
        st_vec = ", ".join(sorted({STREAMS.get(str(s).lower(), str(s)) for s in g["STREAM"].dropna()}))
        mitre_v = str(g["MITRE_TACTIC"].iloc[0] or "N/A")
        escs_v = set()
        best_f, best_z = "N/A", 0.0
        apps_v = set()
        for _, r in g.iterrows():
            m = _parse_json_safe(r.get("METADATA_JSON"))
            for e in _escalas_de_meta(m):
                escs_v.add(e)
            inc = m.get("incident_meta") or {}
            z = float(inc.get("z_max") or 0.0)
            if z >= best_z:
                best_z, best_f = z, str(inc.get("z_max_feature") or "N/A")
            for a in _parse_lista_campo(r.get("APLICACIONES_AFECTADAS")):
                apps_v.add(a)
        resumen_camp_rows.append({
            tr("Campaña / Vector", "Campaign / Vector"): str(vec),
            tr("Origen", "Source"): st_vec,
            tr("Táctica MITRE", "MITRE tactic"): mitre_v,
            tr("Incidentes", "Incidents"): len(g),
            tr("Ventanas detectadas", "Detected windows"): ", ".join(sorted(escs_v, key=lambda s: orden_esc.get(s, 99))),
            tr("Indicador dominante", "Dominant indicator"): f"{nombre(best_f)} (Z = {best_z:.1f})",
            tr("Apps SAP", "SAP apps"): len(apps_v),
        })

    with tarjeta(tr("Síntesis ejecutiva del periodo", "Executive summary of the period"),
                 tr("Panorama unificado por campaña de amenaza, aplicaciones e IPs comprometidas en el mes.",
                    "Unified view by threat campaign, applications and IPs compromised in the month.")):
        familias_txt = esc(", ".join(f"{v} ({len(g)})" for v, g in grupos_camp))
        st.html(
            f'<div class="pm-callout"><b>{tr("Dictamen mensual", "Monthly verdict")} — {esc(periodo)}</b>'
            + tr(f'Se investigaron <strong>{len(pm_f)}</strong> incidente(s) únicos multi-escala '
                 f'en la ventana <strong>{esc(ini_global)} {esc(fin_global)}</strong>, agrupados en '
                 f'<strong>{len(grupos_camp)}</strong> familia(s) de amenaza ({familias_txt}) '
                 f'sobre <strong>{len(apps)}</strong> aplicación(es) SAP y '
                 f'<strong>{len(ips_mes)}</strong> dirección(es) IP.',
                 f'<strong>{len(pm_f)}</strong> unique multi-scale incident(s) were investigated in the '
                 f'window <strong>{esc(ini_global)} {esc(fin_global)}</strong>, grouped into '
                 f'<strong>{len(grupos_camp)}</strong> threat family(ies) ({familias_txt}) across '
                 f'<strong>{len(apps)}</strong> SAP application(s) and <strong>{len(ips_mes)}</strong> '
                 f'IP address(es).')
            + '</div>'
        )
        st.dataframe(pd.DataFrame(resumen_camp_rows), hide_index=True)
        st.html(
            '<div class="pm-meta dos" style="margin-top:.65rem">'
            f'<div><p class="lbl">{tr("Aplicaciones SAP afectadas", "Affected SAP applications")} ({len(apps)})</p>'
            f'{_chips(apps, tr("Sin aplicaciones registradas.", "No applications recorded."))}</div>'
            f'<div><p class="lbl">{tr("Direcciones IP involucradas", "Involved IP addresses")} ({len(ips_mes)})</p>'
            f'{_chips(ips_mes[:18], tr("Sin IPs externas asociadas.", "No external IPs associated."))}</div>'
            '</div>'
        )
        st.html(
            '<div style="margin-top:.55rem">'
            f'<p class="lbl" style="font-size:.72rem;font-weight:700;letter-spacing:.05em;'
            f'text-transform:uppercase;color:{MUTED};margin:0 0 .3rem;">{tr("Tácticas MITRE del mes", "MITRE tactics of the month")}</p>'
            f'{_chips(tacticas_mes, tr("Sin tácticas mapeadas.", "No tactics mapped."))}</div>'
        )

    # Desglose unificado por campana y cronologia completa
    with tarjeta(tr(f"Dossier forense del periodo ({len(pm_f)} incidentes en {len(grupos_camp)} campañas)",
                    f"Forensic dossier of the period ({len(pm_f)} incidents in {len(grupos_camp)} campaigns)"),
                 tr("Consulta el análisis agrupado por familia de amenaza o la cronología completa del mes.",
                    "Browse the analysis grouped by threat family or the full timeline of the month.")):
        tab_camp, tab_cron = st.tabs([
            tr("Por campaña / familia de amenaza", "By campaign / threat family"),
            tr("Cronología completa de incidentes", "Full incident timeline"),
        ])

        with tab_camp:
            for idx_c, (vec, g) in enumerate(grupos_camp):
                st_vec = ", ".join(sorted({STREAMS.get(str(s).lower(), str(s)) for s in g["STREAM"].dropna()}))
                mitre_v = str(g["MITRE_TACTIC"].iloc[0] or "N/A")
                apps_c = sorted({a for v in g["APLICACIONES_AFECTADAS"] for a in _parse_lista_campo(v)})
                ips_c = sorted({ip for v in g["IPS_INVOLUCRADAS"] for ip in _parse_lista_campo(v)})
                escs_c = sorted(
                    {e for _, r in g.iterrows() for e in _escalas_de_meta(_parse_json_safe(r.get("METADATA_JSON")))},
                    key=lambda s: orden_esc.get(s, 99),
                )
                st.html(
                    f'<div class="pm-head"><span class="id">{esc(str(vec))}</span>'
                    f'<span class="pm-pill brand">{len(g)} {tr("incidente(s)", "incident(s)")}</span>'
                    f'<span class="pm-pill">{esc(st_vec)}</span>'
                    f'<span class="pm-pill">{tr("Ventanas", "Windows")}: {esc(", ".join(escs_c))}</span></div>'
                    f'<p class="pm-sub"><b>{esc(mitre_v)}</b> · '
                    f'{len(apps_c)} {tr("app(s) SAP", "SAP app(s)")} · {len(ips_c)} {tr("IP(s) involucradas", "IP(s) involved")}</p>'
                )
                for _, fila in g.iterrows():
                    pm_id = str(fila.get("POSTMORTEM_ID") or "PM-N/A")
                    meta = _parse_json_safe(fila.get("METADATA_JSON"))
                    inc = meta.get("incident_meta") or {}
                    esc_i = ", ".join(_escalas_de_meta(meta))
                    fecha, horas = _rango(fila.get("INICIO"), fila.get("FIN"))
                    z_feat = str(inc.get("z_max_feature") or "N/A")
                    st.html(
                        f'<div class="pm-callout"><b>{esc(pm_id)} · {esc(fecha)} {esc(horas)} · '
                        f'{tr("Ventanas", "Windows")}: {esc(esc_i)} · {esc(nombre(z_feat))} (Z = {float(inc.get("z_max") or 0):.1f})</b>'
                        f'{esc(str(fila.get("RESUMEN_EJECUTIVO") or tr("Sin resumen registrado.", "No summary recorded.")))}</div>'
                    )
                with st.expander(tr(f"Ver reportes individuales y evidencia MCP de {vec} ({len(g)})",
                                    f"View individual reports and MCP evidence for {vec} ({len(g)})")):
                    for _, fila in g.iterrows():
                        pm_id = str(fila.get("POSTMORTEM_ID") or "PM-N/A")
                        meta = _parse_json_safe(fila.get("METADATA_JSON"))
                        md = str(fila.get("REPORTE_MARKDOWN") or tr("Sin reporte registrado.", "No report recorded."))
                        logs = (meta.get("evidence_logs_sample")
                                or (meta.get("raw_evidence") or {}).get("logs_muestra") or [])
                        st.markdown(f"#### {pm_id}")
                        st.markdown(_sin_titulo(md))
                        if logs and isinstance(logs, list):
                            st.caption(tr(f"Muestra de registros MCP ({pm_id})", f"MCP log sample ({pm_id})"))
                            st.dataframe(pd.DataFrame(logs), hide_index=True)
                        st.divider()
                if idx_c < len(grupos_camp) - 1:
                    st.divider()

        with tab_cron:
            for idx, fila in pm_f.iterrows():
                pm_id = str(fila.get("POSTMORTEM_ID") or "PM-N/A")
                origen = STREAMS.get(str(fila.get("STREAM")).lower(), str(fila.get("STREAM")))
                meta = _parse_json_safe(fila.get("METADATA_JSON"))
                inc = meta.get("incident_meta") or {}
                escalas_l = _escalas_de_meta(meta)
                apps_l = _parse_lista_campo(fila.get("APLICACIONES_AFECTADAS"))
                ips_l = _parse_lista_campo(fila.get("IPS_INVOLUCRADAS"))
                fecha, horas = _rango(fila.get("INICIO"), fila.get("FIN"))
                z_feat = str(inc.get("z_max_feature") or "N/A")
                md = str(fila.get("REPORTE_MARKDOWN") or tr("Sin reporte registrado.", "No report recorded."))

                st.html(
                    f'<div class="pm-head"><span class="id">{esc(pm_id)}</span>'
                    f'<span class="pm-pill brand">{esc(origen)}</span>'
                    f'<span class="pm-pill">{tr("Evento", "Event")} {int(fila.get("EVENTO_ID") or 0)}</span>'
                    f'<span class="pm-pill">{tr("Ventanas", "Windows")}: {esc(", ".join(escalas_l))}</span></div>'
                    f'<p class="pm-sub"><b>{esc(str(fila.get("TIPO_ATAQUE") or "N/A"))}</b> · '
                    f'{esc(str(fila.get("MITRE_TACTIC") or tr("Táctica pendiente", "Tactic pending")))} · '
                    f'{esc(fecha)} {esc(horas)} · '
                    f'{tr("Indicador", "Indicator")}: {esc(nombre(z_feat))} (Z = {float(inc.get("z_max") or 0):.1f}) · '
                    f'{len(apps_l)} app(s) · {len(ips_l)} IP(s)</p>'
                )
                st.html(
                    f'<div class="pm-callout"><b>{tr("Resumen ejecutivo", "Executive summary")}</b>'
                    f'{esc(str(fila.get("RESUMEN_EJECUTIVO") or tr("Sin resumen registrado.", "No summary recorded.")))}</div>'
                )

                with st.expander(tr(f"Ver reporte completo y evidencia ({pm_id})", f"View full report and evidence ({pm_id})")):
                    t1, t2 = st.tabs([tr("Reporte completo", "Full report"), tr("Evidencia", "Evidence")])
                    with t1:
                        st.markdown(_sin_titulo(md))
                    with t2:
                        st.html(
                            '<div class="pm-meta dos">'
                            f'<div><p class="lbl">{tr("Aplicaciones SAP afectadas", "Affected SAP applications")}</p>'
                            f'{_chips(apps_l, tr("Sin aplicaciones registradas.", "No applications recorded."))}</div>'
                            f'<div><p class="lbl">{tr("Direcciones IP involucradas", "Involved IP addresses")}</p>'
                            f'{_chips(ips_l, tr("Sin IPs externas asociadas.", "No external IPs associated."))}</div>'
                            '</div>'
                        )
                        logs = (meta.get("evidence_logs_sample")
                                or (meta.get("raw_evidence") or {}).get("logs_muestra") or [])
                        if logs and isinstance(logs, list):
                            st.dataframe(pd.DataFrame(logs), hide_index=True)
                        else:
                            vacio(tr("Este reporte no incluye una muestra de registros.",
                                     "This report does not include a log sample."))
                        st.json(meta or {"nota": tr("Sin metadatos.", "No metadata.")})

                if idx < len(pm_f) - 1:
                    st.divider()




# --- Entrenamiento en vivo -------------------------------------------------
PASOS = [
    (tr("Conexión a SAP HANA", "SAP HANA connection"),
     tr("Abre una sesión propia para las tablas temporales.", "Opens a dedicated session for temporary tables.")),
    (tr("Selección de la semana", "Week selection"),
     tr("Filtra los registros de la tabla a la semana elegida.", "Filters the table's logs to the chosen week.")),
    (tr("Eventos base", "Base events"),
     tr("Convierte timestamps y calcula banderas por registro.", "Parses timestamps and computes per-log flags.")),
    (tr("Construcción de ventanas", "Window construction"),
     tr("Agrega los registros en ventanas con 50 % de traslape.", "Aggregates logs into windows with 50% overlap.")),
    (tr("Entrenamiento de Isolation Forest", "Isolation Forest training"),
     tr("PAL ajusta el bosque de árboles de aislamiento.", "PAL fits the forest of isolation trees.")),
    (tr("Puntuación de ventanas", "Window scoring"),
     tr("Score de anomalía y etiqueta para cada ventana.", "Anomaly score and label for each window.")),
    (tr("Eventos, relevancia y tipo", "Events, relevance and type"),
     tr("Agrupa anomalías, filtra por |z| y asigna el tipo de ataque.",
        "Groups anomalies, filters by |z| and assigns the attack type.")),
    (tr("Limpieza", "Cleanup"),
     tr("Borra las tablas temporales y cierra la sesión.", "Drops the temporary tables and closes the session.")),
]


@st.cache_data(ttl=600, show_spinner=False)
def semanas_de(tabla):
    t0, t_max, n = calculate_week_range(_conn(), tabla, wf.WEEK_DAYS, wf.MIN_LAST_DAYS)
    return tabla_semanas(t0, t_max, n, wf.WEEK_DAYS)


def _html_pasos(estado, dur):
    iconos = {"pendiente": "", "corriendo": "", "ok": "✓", "error": "!"}
    out = '<div class="tr-steps">'
    for i, ((nom, desc), e, d) in enumerate(zip(PASOS, estado, dur), 1):
        t = f"{d:.1f} s" if d is not None else (tr("en curso…", "running…") if e == "corriendo" else "")
        out += (f'<div class="tr-step {e}"><span class="tr-num">{iconos[e] or i}</span>'
                f'<div><b>{nom}</b><span>{desc}</span></div><em>{t}</em></div>')
    return out + '</div>'


def _correr_entrenamiento(tabla, semana, size_s, stream, ui):
    """Ejecuta el pipeline de una semana con el código de producción, mostrando cada paso en vivo."""
    estado, dur, lineas = ["pendiente"] * len(PASOS), [None] * len(PASOS), []
    t_ini = time.perf_counter()
    feats = wf.FEATURES[stream]
    es_sys = stream == "system"

    def log(msg):
        lineas.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        ui["log"].code("\n".join(lineas[-18:]), language=None, wrap_lines=True)

    def pintar():
        hechos = sum(e == "ok" for e in estado)
        ui["pasos"].html(_html_pasos(estado, dur))
        seg = time.perf_counter() - t_ini
        txt = (tr(f"Completado en {seg:.0f} s", f"Completed in {seg:.0f} s") if hechos == len(PASOS) else
               tr(f"Paso {hechos + 1} de {len(PASOS)} · {seg:.0f} s transcurridos",
                  f"Step {hechos + 1} of {len(PASOS)} · {seg:.0f} s elapsed"))
        ui["prog"].progress(hechos / len(PASOS), text=txt)

    @contextmanager
    def paso(i):
        estado[i] = "corriendo"
        pintar()
        log(tr("Inicia: ", "Starting: ") + PASOS[i][0])
        t = time.perf_counter()
        try:
            yield
        except Exception:
            estado[i], dur[i] = "error", time.perf_counter() - t
            pintar()
            raise
        estado[i], dur[i] = "ok", time.perf_counter() - t
        pintar()

    conn, res = None, {"tabla": tabla, "semana": semana, "size_s": size_s, "stream": stream}
    try:
        with paso(0):
            conn = hana_df.ConnectionContext(**_params(), communicationTimeout=600000)
            log(tr("Sesión dedicada abierta: las tablas temporales (#) solo existen en ella.",
                   "Dedicated session opened: temporary tables (#) only exist in it."))
        with paso(1):
            df_sys, df_llm = wf.cargar_datos(conn, tabla)
            sw, lw, sem = split_into_weeks(df_sys, df_llm, conn, tabla, wf.WEEK_DAYS, wf.MIN_LAST_DAYS)
            df_w = (sw if es_sys else lw)[semana]
            res["registros"] = int(df_w.count())
            fila = sem.iloc[semana]
            rango_sem = f"{fila['WEEK_START']:%d/%m} – {fila['WEEK_END']:%d/%m}"
            log(tr(f"{tabla} · semana {semana + 1} ({rango_sem}): {res['registros']:,} registros de {STREAMS[stream]}.",
                   f"{tabla} · week {semana + 1} ({rango_sem}): {res['registros']:,} {STREAMS[stream]} logs."))
        with paso(2):
            (ventanas_mod.prepare_system_base if es_sys else ventanas_mod.prepare_llm_base)(conn, df_w)
            log(tr("Eventos base materializados en ", "Base events materialized in ")
                + ("#BASE_SYSTEM." if es_sys else "#BASE_LLM."))
        with paso(3):
            build = ventanas_mod.build_system_windows if es_sys else ventanas_mod.build_llm_windows
            win = ventanas_mod.materialize(conn, build(conn, size_s), "#WIN_SYSTEM" if es_sys else "#WIN_LLM")
            ventanas_mod.drop_temp(conn, ventanas_mod.SYSTEM_EVENTS_TABLE if es_sys
                                   else ventanas_mod.LLM_EVENTS_TABLE)
            res["ventanas"] = int(win.count())
            log(tr(f"{res['ventanas']:,} ventanas de {ventanas_mod.size_label(size_s)} con {len(feats)} variables.",
                   f"{res['ventanas']:,} {ventanas_mod.size_label(size_s)} windows with {len(feats)} variables."))
        with paso(4):
            n = res["ventanas"]
            model = PalIsolationForest(n_estimators=IF_N_ESTIMATORS, max_samples=min(IF_MAX_SAMPLES, n),
                                       random_state=IF_RANDOM_STATE)
            log(f"{tr('Parámetros', 'Parameters')}: n_estimators={IF_N_ESTIMATORS}, max_samples={min(IF_MAX_SAMPLES, n)}, "
                f"random_state={IF_RANDOM_STATE}.")
            t_fit = time.perf_counter()
            model.fit(data=win, key="WINDOW_INDEX", features=feats)
            res["t_fit"] = time.perf_counter() - t_fit
            log(tr(f"Modelo ajustado en SAP HANA PAL en {res['t_fit']:.1f} s.",
                   f"Model fitted in SAP HANA PAL in {res['t_fit']:.1f} s."))
        with paso(5):
            pred = model.predict(data=win, key="WINDOW_INDEX", features=feats,
                                 contamination=CONTAMINATION_IF, show_explainer=False)
            win_pdf = win.collect()
            r = win_pdf.merge(pred.collect(), on="WINDOW_INDEX")
            idx = r.loc[r["LABEL"] == -1, "WINDOW_INDEX"].tolist()
            z = zscores(win_pdf, idx, feats)
            res["anomalas"] = len(idx)
            log(tr(f"{len(idx)} ventanas anómalas ({len(idx) / len(r):.2%}) con contaminación {CONTAMINATION_IF}.",
                   f"{len(idx)} anomalous windows ({len(idx) / len(r):.2%}) with contamination {CONTAMINATION_IF}."))
        with paso(6):
            r.insert(0, "WEEK", semana)
            z.insert(0, "WEEK", semana)
            r = agrupar_eventos(r, wf.MAX_GAP[size_s][stream])
            ev = resumen_eventos(r)
            ev = marcar_relevantes(r, z, ev, wf.Z_UMBRAL[size_s])
            tipos = clasificar_eventos(r, z, ev, firmas=wf.FIRMAS[stream])
            ev = ev.merge(tipos, on="EVENTO_ID", how="left")
            res.update(eventos=len(ev), relevantes=int(ev["RELEVANTE"].sum()), ev=ev, win=r)
            log(tr(f"{len(ev)} eventos; {res['relevantes']} relevantes (|z| ≥ {wf.Z_UMBRAL[size_s]}).",
                   f"{len(ev)} events; {res['relevantes']} relevant (|z| ≥ {wf.Z_UMBRAL[size_s]})."))
            for _, e in ev[ev["RELEVANTE"]].iterrows():
                log(tr(f"  Evento {int(e['EVENTO_ID'])}: {e['TIPO_ATAQUE']} (confianza {e['CONFIANZA']}, "
                       f"z máx {e['Z_MAX']:.1f} en {e['Z_MAX_FEATURE']}).",
                       f"  Event {int(e['EVENTO_ID'])}: {tipo_txt(e['TIPO_ATAQUE'])} ({conf_txt(e['CONFIANZA'])} "
                       f"confidence, max z {e['Z_MAX']:.1f} in {e['Z_MAX_FEATURE']})."))
        with paso(7):
            ventanas_mod.drop_temp(conn, "#WIN_SYSTEM" if es_sys else "#WIN_LLM")
            ventanas_mod.drop_temp(conn, ventanas_mod.SYSTEM_BASE_TABLE if es_sys
                                   else ventanas_mod.LLM_BASE_TABLE)
            conn.close()
            conn = None
            log(tr("Tablas temporales borradas. No se modificó ninguna tabla RESULT_*.",
                   "Temporary tables dropped. No RESULT_* table was modified."))
        res["total"] = time.perf_counter() - t_ini
        res["durs"] = dur
        log(tr(f"Entrenamiento completo en {res['total']:.0f} s.", f"Training completed in {res['total']:.0f} s."))
        res["log"] = lineas
        return res
    except Exception as e:
        log(f"ERROR: {e}")
        st.error(tr("El entrenamiento se detuvo: ", "Training stopped: ") + str(e))
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _comparar_con_produccion(res):
    """Compara la corrida con lo que el pipeline guardó en RESULT_SUMMARY para la misma semana."""
    inv = inventario_hana()
    meses_proc = sorted({t[len("RESULT_SUMMARY_"):] for t in inv["TABLE_NAME"]
                         if t.startswith("RESULT_SUMMARY_")})
    resumen, _, _ = metricas_pipeline(tuple(meses_proc))
    g = resumen[(resumen["SOURCE_TABLE"] == res["tabla"]) & (resumen["WINDOW_SIZE_S"] == res["size_s"])
                & (resumen["STREAM"] == res["stream"]) & (resumen["WEEK"] == res["semana"])]
    with tarjeta(tr("Comparación con producción", "Comparison with production"),
                 tr("Verifica que el entrenamiento sea reproducible: con la misma semilla (random_state = 42) "
                    "debe obtener lo mismo que guardó el pipeline en RESULT_SUMMARY.",
                    "Checks that training is reproducible: with the same seed (random_state = 42) it must "
                    "produce what the pipeline saved in RESULT_SUMMARY.")):
        if g.empty:
            st.html('<div class="hc-group"><div class="hc-row"><i class="hc-dot info"></i>'
                    f'<div class="hc-main"><b>{tr("Sin referencia", "No reference")}</b><span>'
                    + tr("Esta tabla aún no se procesa en producción, así que no hay resultados guardados para comparar.",
                         "This table has not been processed in production yet, so there are no saved results to compare.")
                    + f'</span></div><div class="hc-val"><span class="hc-pill info">{estados_hc()["info"][0]}</span></div></div></div>')
            return
        g = g.iloc[0]
        filas = [(tr("Ventanas", "Windows"), res["ventanas"], int(g["VENTANAS"])),
                 (tr("Ventanas anómalas", "Anomalous windows"), res["anomalas"], int(g["VENTANAS_ANOMALAS"])),
                 (tr("Eventos", "Events"), res["eventos"], int(g["EVENTOS"])),
                 (tr("Eventos relevantes", "Relevant events"), res["relevantes"], int(g["EVENTOS_RELEVANTES"]))]
        out = '<div class="hc-grid">'
        for i in range(0, len(filas), 2):
            out += '<div class="hc-group">'
            for nom, vivo, prod in filas[i:i + 2]:
                cls = "ok" if vivo == prod else "warn"
                out += (f'<div class="hc-row"><i class="hc-dot {cls}"></i><div class="hc-main"><b>{nom}</b>'
                        f'<span>{tr("Esta corrida", "This run")}: {vivo:,} · '
                        f'{tr("Guardado en producción", "Saved in production")}: {prod:,}</span></div>'
                        f'<div class="hc-val"><span class="hc-pill {cls}">'
                        f'{tr("Coincide", "Matches") if vivo == prod else tr("Difiere", "Differs")}</span></div></div>')
            out += '</div>'
        st.html(out + '</div>')


def _resultado_entrenamiento(res):
    r, ev = res["win"], res["ev"]
    kpis = [
        (tr("Ventanas", "Windows"), f"{res['ventanas']:,}",
         tr(f"de {ventanas_mod.size_label(res['size_s'])}", f"of {ventanas_mod.size_label(res['size_s'])}"), BRAND,
         tr("Ventanas construidas para la semana.", "Windows built for the week.")),
        (tr("Ventanas anómalas", "Anomalous windows"), f"{res['anomalas']}",
         tr(f"{res['anomalas'] / res['ventanas']:.2%} del total", f"{res['anomalas'] / res['ventanas']:.2%} of total"),
         "#E76500", tr("Marcadas por Isolation Forest.", "Flagged by Isolation Forest.")),
        (tr("Eventos relevantes", "Relevant events"), f"{res['relevantes']}",
         tr(f"de {res['eventos']} eventos", f"of {res['eventos']} events"), CRITICO,
         tr("Eventos con |z| ≥ 5 en alguna variable.", "Events with |z| ≥ 5 in some variable.")),
        (tr("Tiempo total", "Total time"), f"{res['total']:.0f} s",
         tr(f"entrenamiento: {res['t_fit']:.1f} s", f"training: {res['t_fit']:.1f} s"), "#049F9A",
         tr("Duración de todo el proceso y del ajuste del modelo.", "Duration of the whole process and of the model fit.")),
    ]
    st.html('<div class="kpis kpis-4">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{ayuda}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    _comparar_con_produccion(res)

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, tarjeta(tr("Puntaje de anomalía en el tiempo", "Anomaly score over time"),
                     tr("Score de cada ventana de la semana. Los puntos rojos son las ventanas anómalas.",
                        "Score of each window in the week. Red dots are the anomalous windows.")):
        a = r[r["LABEL"] == -1]
        fig = go.Figure([
            go.Scatter(x=r["WINDOW_START"], y=r["SCORE"], mode="lines", name="Score",
                       line=dict(color=SERIES[0], width=1.2)),
            go.Scatter(x=a["WINDOW_START"], y=a["SCORE"], mode="markers", name=tr("Anómala", "Anomalous"),
                       marker=dict(color=CRITICO, size=6)),
        ])
        fig.update_yaxes(title_text="Score", title_font_color=MUTED)
        mostrar(estilo(fig, height=300))
    with c2, tarjeta(tr("Distribución del puntaje", "Score distribution"),
                     tr("Histograma de scores. La línea marca el umbral a partir del cual una ventana "
                        "se considera anómala.",
                        "Score histogram. The line marks the threshold above which a window is "
                        "considered anomalous.")):
        umbral = float(r.loc[r["LABEL"] == -1, "SCORE"].min())
        fig = go.Figure(go.Histogram(x=r["SCORE"], nbinsx=40, marker=dict(color=SERIES[0]),
                                     hovertemplate="Score %{x}<br>%{y} " + tr("ventanas", "windows") + "<extra></extra>"))
        fig.add_vline(x=umbral, line=dict(color=CRITICO, dash="dot", width=1.5))
        fig.update_yaxes(title_text=tr("Ventanas", "Windows"), title_font_color=MUTED)
        mostrar(estilo(fig, height=300, legend=False))

    with tarjeta(tr("Eventos detectados en esta corrida", "Events detected in this run"),
                 tr("Eventos agrupados a partir de las ventanas anómalas, ordenados por intensidad.",
                    "Events grouped from the anomalous windows, sorted by intensity.")):
        if ev.empty:
            vacio(tr("No se formaron eventos.", "No events were formed."))
        else:
            st.dataframe(ev.sort_values("Z_MAX", ascending=False)[
                ["EVENTO_ID", "INICIO", "N_VENTANAS", "Z_MAX", "Z_MAX_FEATURE", "RELEVANTE",
                 "TIPO_ATAQUE", "CONFIANZA"]].assign(
                    Z_MAX_FEATURE=lambda d: d["Z_MAX_FEATURE"].map(nombre),
                    TIPO_ATAQUE=lambda d: d["TIPO_ATAQUE"].map(lambda t: tipo_txt(t) if pd.notna(t) else t),
                    CONFIANZA=lambda d: d["CONFIANZA"].map(lambda c: conf_txt(c) if pd.notna(c) else c)),
                hide_index=True, column_config={
                    "EVENTO_ID": st.column_config.NumberColumn(tr("Evento", "Event"), format="%d"),
                    "INICIO": st.column_config.DatetimeColumn(tr("Inicio", "Start"), format="DD/MM HH:mm"),
                    "N_VENTANAS": st.column_config.NumberColumn(tr("Ventanas", "Windows"), format="%d"),
                    "Z_MAX": st.column_config.NumberColumn(tr("Intensidad", "Intensity"), format="%.1f"),
                    "Z_MAX_FEATURE": tr("Indicador más afectado", "Most affected indicator"),
                    "RELEVANTE": st.column_config.CheckboxColumn(tr("Relevante", "Relevant")),
                    "TIPO_ATAQUE": tr("Tipo de ataque", "Attack type"), "CONFIANZA": tr("Confianza", "Confidence"),
                })


def _vista_entrenamiento():
    inv = inventario_hana()
    tablas = sorted(inv.loc[inv["TABLE_NAME"].str.startswith("LOGS"), "TABLE_NAME"])
    with tarjeta(tr("Configuración de la corrida", "Run configuration"),
                 tr("Corre el mismo código del pipeline de producción (src/workflow.py) sobre una semana, "
                    "en tablas temporales de SAP HANA. No modifica los resultados guardados.",
                    "Runs the same production pipeline code (src/workflow.py) on one week, in temporary "
                    "SAP HANA tables. It does not modify the saved results.")):
        f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1], gap="medium")
        tabla = f1.selectbox(tr("Tabla de logs", "Log table"), tablas, key="tr_tabla",
                             index=tablas.index("LOGS_2026_10") if "LOGS_2026_10" in tablas else 0)
        sem = semanas_de(tabla)
        semana = f2.selectbox(tr("Semana", "Week"), list(sem["WEEK"]), key=f"tr_semana_{IDIOMA}",
                              format_func=lambda w: f"{tr('Semana', 'Week')} {w + 1} · {sem.loc[w, 'WEEK_START']:%d/%m} – "
                                                    f"{sem.loc[w, 'WEEK_END']:%d/%m}")
        size_s = f3.selectbox(tr("Intervalo", "Interval"), list(wf.SIZES), index=3, key="tr_size",
                              format_func=ventanas_mod.size_label)
        stream = f4.selectbox(tr("Origen", "Source"), list(STREAMS), format_func=STREAMS.get, key=f"tr_stream_{IDIOMA}")
        correr = st.button(tr("Iniciar entrenamiento", "Start training"), icon=":material/play_arrow:", type="primary",
                           key="tr_correr")
        st.caption(tr("Intervalos cortos generan más ventanas y tardan más: 1 h suele terminar en menos de un minuto; "
                      "5 min puede tardar varios minutos.",
                      "Short intervals create more windows and take longer: 1 h usually finishes in under a minute; "
                      "5 min can take several minutes."))

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1, tarjeta(tr("Progreso", "Progress"), tr("Cada paso se marca al terminar, con su duración.",
                                                     "Each step is checked off when it finishes, with its duration.")):
        ui = {"prog": st.empty(), "pasos": st.empty()}
    with c2, tarjeta(tr("Bitácora", "Log"), tr("Mensajes del proceso en tiempo real.", "Real-time process messages.")):
        ui["log"] = st.empty()

    if correr:
        res = _correr_entrenamiento(tabla, int(semana), int(size_s), stream, ui)
        if res:
            st.session_state["tr_resultado"] = res
    res = st.session_state.get("tr_resultado")
    if res and not correr:
        ui["prog"].progress(1.0, text=f"{tr('Última corrida', 'Last run')}: {res['tabla']} · "
                                      f"{tr('semana', 'week')} {res['semana'] + 1} · "
                                      f"{ventanas_mod.size_label(res['size_s'])} · {STREAMS[res['stream']]}")
        ui["pasos"].html(_html_pasos(["ok"] * len(PASOS), res["durs"]))
        ui["log"].code("\n".join(res["log"][-18:]), language=None, wrap_lines=True)
    elif not correr:
        ui["prog"].progress(0.0, text=tr("Listo para iniciar", "Ready to start"))
        ui["pasos"].html(_html_pasos(["pendiente"] * len(PASOS), [None] * len(PASOS)))
        ui["log"].code(tr("Elige la configuración y presiona Iniciar entrenamiento.",
                          "Choose the configuration and press Start training."), language=None, wrap_lines=True)
    if res:
        _resultado_entrenamiento(res)


def pagina_mlops():
    encabezado("MLOps", tr("Operación del modelo", "Model operations"),
               tr("Estado del sistema en producción y entrenamiento del modelo en tiempo real.",
                  "Production system status and real-time model training."),
               pills=[(tr("Modelo", "Model"), "Isolation Forest (PAL)"),
                      (tr("Contaminación", "Contamination"), f"{CONTAMINATION_IF:.0%}"),
                      (tr("Árboles", "Trees"), IF_N_ESTIMATORS)], con_periodo=False)
    vista = st.segmented_control(tr("Vista", "View"), ["health", "train"], default="health",
                                 format_func={"health": "Health checks",
                                              "train": tr("Entrenamiento en vivo", "Live training")}.get,
                                 key="mlops_vista", label_visibility="collapsed") or "health"
    if vista == "health":
        vista_health(tarjeta, estilo, mostrar)
    else:
        _vista_entrenamiento()


# Navegación
pg_resumen = st.Page(pagina_resumen, title=tr("Resumen", "Overview"), icon=":material/dashboard:",
                     url_path="resumen", default=True)
pg_eventos = st.Page(pagina_eventos, title=tr("Eventos", "Events"), icon=":material/timeline:", url_path="eventos")
pg_tipos = st.Page(pagina_tipos, title=tr("Tipos de ataque", "Attack types"), icon=":material/category:", url_path="tipos")
pg_postmortems = st.Page(pagina_postmortems, title="Postmortems SOC", icon=":material/security:",
                         url_path="postmortems")
pg_modelo = st.Page(pagina_modelo, title=tr("Metodología", "Methodology"), icon=":material/model_training:",
                    url_path="metodologia")

pg_mlops = st.Page(pagina_mlops, title="MLOps", icon=":material/monitor_heart:", url_path="mlops")

pg = st.navigation([pg_resumen, pg_eventos, pg_tipos, pg_postmortems, pg_modelo, pg_mlops], position="top")


# --- Sapy (asistente). Solo la interfaz; la lógica del chatbot va en responder() de sapy.py.
#     Para quitarlo: borra este bloque hasta "fin Sapy". -----------------------------------------
from sapy import mostrar_sapy   # noqa: E402
mostrar_sapy(IDIOMA, tr)
# --- fin Sapy ---------------------------------------------------------------------------------

pg.run()
