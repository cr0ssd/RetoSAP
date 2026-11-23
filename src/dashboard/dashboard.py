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
for p in (REPO_ROOT, SRC_DIR):
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


# Modularización
from tema import (BRAND, NAVY, GOLD, SERIES, INK, INK2, MUTED, GRID, AXIS, NEUTRO, SURFACE,
                  CARD_BORDER, PAGE, DIV_MID, CRITICO, AZUL_SEQ, CONFIANZA_COLOR, STREAMS)
from datos_hana import _conn, leer, leer_postmortems, _parse_json_safe
from health import inventario_hana, metricas_pipeline, vista_health

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

st.set_page_config(page_title="Anomalías SAP", page_icon=":material/shield:", layout="wide")



COLOR_TIPO = {
    # system
    "Contexto SAP": SERIES[0], "DDoS": SERIES[1], "DoS": SERIES[2], "Fuerza bruta": SERIES[3],
    "Explotación": SERIES[4], "Reconocimiento": SERIES[5], "Falla operativa": "#8d99a8",
    # llm
    "Prompt injection": SERIES[0], "Degradación del servicio": SERIES[1],
    "Abuso de recursos": SERIES[2], "Acceso no autorizado": SERIES[3], "Automatización": SERIES[4],
}


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
    "DoS": "Sube el volumen de peticiones y lo concentra en pocas IPs.",
    "DDoS": "Sube el volumen de peticiones repartido entre muchas IPs.",
    "Fuerza bruta": "Concentra errores 4xx y eventos de seguridad o auditoría desde una IP "
                    "dominante, con un ritmo regular y más POST que GET.",
    "Reconocimiento": "Produce errores 4xx, más GET y cambios en las rutas pedidas.",
    "Explotación": "Produce errores 5xx, registros de error y de seguridad, y cambios en "
                   "métodos, rutas y formatos de contenido.",
    "Contexto SAP": "Cambios en la mezcla de aplicaciones, entornos, regiones o tipos de registro "
                    "de SAP.",
    "Falla operativa": "Errores y advertencias con caída del tráfico.",
}
FIRMA = {"system": FIRMAS["F_SYSTEM"], "llm": FIRMAS["F_LLM"]}


def nombre(f):
    return NOMBRE.get(f, f)


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
    pills = list(pills or [("Periodo", MES.replace("_", "-")), ("Origen", STREAMS[STREAM]),
                           ("Bloques de", ETIQUETA[SIZE_S])])
    if con_periodo and TXT_PERIODO:
        pills.append(("Seleccionado", TXT_PERIODO))
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



@st.cache_data(ttl=600, show_spinner="Cargando datos…")
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
        return "Crítica", "crit", CRITICO
    if s >= 0.45:
        return "Alta", "warn", "#E76500"
    return "Moderada", "brand", BRAND





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
        "tipo_ataque": str(fila.get("TIPO_ATAQUE") or "Anomalía"),
        "mitre_tactic": str(fila.get("MITRE_TACTIC") or "Pendiente de mapeo"),
        "severidad_score": float(fila.get("SEVERIDAD_SCORE") or 0.0),
        "inicio": _fmt_ts(fila.get("INICIO")),
        "fin": _fmt_ts(fila.get("FIN")),
        "resumen_ejecutivo": str(fila.get("RESUMEN_EJECUTIVO") or "Sin dictamen ejecutivo disponible."),
        "aplicaciones_afectadas": _parse_lista_campo(fila.get("APLICACIONES_AFECTADAS")),
        "ips_involucradas": _parse_lista_campo(fila.get("IPS_INVOLUCRADAS")),
        "reporte_markdown": str(fila.get("REPORTE_MARKDOWN") or ""),
        "metadata_json": meta_norm,
    }


def _construir_pdf_bytes(fila):
    """Construye en memoria el reporte ejecutivo PDF individual a partir de una fila HANA."""
    buf = io.BytesIO()
    generar_pdf_postmortem(_fila_a_payload(fila), buf)
    return buf.getvalue()


def _construir_pdf_mensual_bytes(df_pm, periodo):
    """Construye en memoria el reporte PDF multi-pagina consolidado para todos los incidentes."""
    items = [_fila_a_payload(row) for _, row in df_pm.iterrows()]
    buf = io.BytesIO()
    generar_pdf_postmortem_mensual(items, periodo, buf)
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
        f"# DOSSIER FORENSE POSTMORTEM MENSUAL — PERIODO {periodo}",
        f"**Total de incidentes unicos multi-escala:** {len(df_pm)}",
        "",
        "## 1. Resumen por Campana / Familia de Amenaza",
        "",
        "| Campana / Vector | Origen | Tactica MITRE | Incidentes | Escalas Detectadas | Indicador Dominante |",
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
        "## 2. Cronologia Maestra de Incidentes del Periodo",
        "",
        "| Expediente | Origen | Vector / Ataque | Ventana Temporal (UTC) |",
        "| :--- | :--- | :--- | :--- |",
    ])
    for _, r in df_pm.iterrows():
        lineas.append(
            f"| **{r.get('POSTMORTEM_ID', 'N/A')}** | {str(r.get('STREAM', '')).upper()} | "
            f"{r.get('TIPO_ATAQUE', 'N/A')} | {_fmt_ts(r.get('INICIO'))[:16]} -> {_fmt_ts(r.get('FIN'))[:16]} |"
        )

    lineas.extend(["", "## 3. Sintesis Forense Unificada por Campana", ""])
    for vec, g in grupos:
        lineas.append(f"### Campana: {vec} ({len(g)} incidente(s))")
        for _, r in g.iterrows():
            m = _parse_json_safe(r.get("METADATA_JSON"))
            inc = m.get("incident_meta") or {}
            zv = float(inc.get("z_max") or 0.0)
            sev_s = float(r.get("SEVERIDAD_SCORE") or 0.0)
            sev_lbl, _, _ = _clasificar_severidad_incidente(zv, sev_s)
            lineas.append(
                f"- **{r.get('POSTMORTEM_ID', 'N/A')}** (`{_fmt_ts(r.get('INICIO'))[:16]} -> {_fmt_ts(r.get('FIN'))[:16]}`, "
                f"severidad: **{sev_lbl}**): {r.get('RESUMEN_EJECUTIVO', '')}"
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
    st.error(f"No fue posible conectarse a SAP HANA. Revisa la configuración de conexión.\n\n{e}")
    st.stop()
if not meses:
    st.warning("Todavía no hay resultados cargados en SAP HANA.")
    st.stop()

with st.sidebar:
    st.html('<div class="side-brand">Anomalías <span>SAP</span></div>'
            '<div class="side-sub">Monitoreo de comportamiento inusual en los registros de SAP</div>')
    MES = st.selectbox("Periodo", meses, format_func=lambda m: m.replace("_", "-"), key="mes")
    eventos_all = leer("EVENTS", MES)
    resumen_all = leer("SUMMARY", MES)
    valid_all = leer("VALIDATION", MES)
    # Cada periodo se analiza a partir de una sola tabla de logs; si hubiera varias, se usa la que
    # tiene más eventos. No se muestra como filtro para no confundir a quien consulta el dashboard.
    FUENTE = eventos_all["SOURCE_TABLE"].value_counts().index[0]
    STREAM = st.segmented_control("Origen de los registros", list(STREAMS), default="system",
                                  format_func=STREAMS.get, key="stream",
                                  width="stretch") or "system"
    tam = (resumen_all[["WINDOW_SIZE_S", "WINDOW_LABEL"]].drop_duplicates()
           .sort_values("WINDOW_SIZE_S"))
    ETIQUETA = dict(zip(tam["WINDOW_SIZE_S"], tam["WINDOW_LABEL"]))
    SIZE_S = st.select_slider("Intervalo de análisis", list(ETIQUETA), value=list(ETIQUETA)[0],
                              format_func=ETIQUETA.get, key="size")
    aviso_periodo = st.container()   # se llena después de calcular el periodo seleccionado
    st.divider()
    pm_all = leer_postmortems(MES)
    pm_stream_n = int((pm_all["STREAM"].astype(str).str.lower() == STREAM).sum()) if not pm_all.empty else 0
    st.caption(f"Reportes forenses del periodo: **{len(pm_all)}** ({pm_stream_n} de {STREAMS[STREAM]})")
    if st.button("Actualizar datos en vivo", icon=":material/refresh:", width="stretch"):
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
        st.html(f'<div class="side-periodo"><b>Periodo seleccionado</b>{TXT_PERIODO}'
                '<span>Todas las páginas muestran solo este periodo.</span></div>')
        st.button("Ver todo el mes", icon=":material/zoom_out_map:", width="stretch",
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
        x=WIN["WINDOW_START"], y=WIN[col], mode="lines", name="Registros por bloque",
        line=dict(color="#1B90FF", width=1.3),   # azul vivo de SAP Horizon
        hovertemplate="%{x|%d-%b %H:%M}<br>%{y:,.0f} registros<extra></extra>"))
    a = WIN[WIN["LABEL"] == -1]
    fig.add_trace(go.Scatter(
        x=a["WINDOW_START"], y=a[col], mode="markers", name="Bloque atípico",
        marker=dict(color=CRITICO, size=8, line=dict(color=SURFACE, width=2)),
        customdata=np.stack([a["EVENTO_ID"].fillna(-1).astype(int),
                             a["TIPO_ATAQUE"].fillna("Evento no relevante")], axis=-1)
        if len(a) else None,
        hovertemplate="%{x|%d-%b %H:%M}<br>%{y:,.0f} registros<br>Evento %{customdata[0]} · "
                      "%{customdata[1]}<extra></extra>"))
    for t in sorted(REL["TIPO_ATAQUE"].dropna().unique()):   # leyenda de las franjas
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=t,
                                 marker=dict(color=COLOR_TIPO.get(t, NEUTRO), size=12,
                                             symbol="square", opacity=0.5)))
    for w, g in WIN.groupby("WEEK"):
        if w > 0:
            fig.add_vline(x=g["WINDOW_START"].min(), line=dict(color=MUTED, dash="dot", width=1))
    fig.update_yaxes(title_text=f"Registros cada {ETIQUETA[SIZE_S]}",
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
    fig = go.Figure()
    for conf in ct.columns:
        fig.add_trace(go.Bar(
            y=ct.index, x=ct[conf], orientation="h", name=f"Confianza {conf}",
            marker=dict(color=CONFIANZA_COLOR[conf], line=dict(color=SURFACE, width=2),
                        cornerradius=4),
            hovertemplate="%{y}<br>Confianza " + conf + ": %{x} eventos<extra></extra>"))
    tot = ct.sum(axis=1)
    fig.add_trace(go.Scatter(y=tot.index, x=tot.values, mode="text", text=tot.values,
                             textposition="middle right", textfont_color=INK2,
                             showlegend=False, hoverinfo="skip"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="Número de eventos", title_font_color=MUTED, dtick=1,
                     range=[0, tot.max() * 1.15])
    return estilo(fig, height=max(260, 62 * len(ct) + 120))


def fig_perfil(perfil):
    tipo_ev = REL.set_index("EVENTO_ID")["TIPO_ATAQUE"]
    filas = [f"Evento {i} · {tipo_ev.get(i, '—')}" for i in perfil.index]
    fig = go.Figure(go.Heatmap(
        z=perfil.values, x=list(perfil.columns), y=filas,
        colorscale=[[i / (len(AZUL_SEQ) - 1), c] for i, c in enumerate(AZUL_SEQ)],
        xgap=2, ygap=2, text=np.round(perfil.values, 1), texttemplate="%{text}",
        hovertemplate="%{y}<br>%{x}: cambio de %{z:.1f}<extra></extra>",
        colorbar=dict(title=dict(text="Cambio", font_color=MUTED), tickfont_color=MUTED,
                      thickness=10, outlinewidth=0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(side="top", showgrid=False)
    return estilo(fig, height=max(240, 36 * len(perfil) + 110), legend=False,
                  margin=dict(l=8, r=8, t=48, b=8))


def tabla_eventos(df, alto=None):
    st.dataframe(
        df.sort_values("Z_MAX", ascending=False)[
            ["EVENTO_ID", "TIPO_ATAQUE", "CONFIANZA", "INICIO", "FIN_REAL", "Z_MAX",
             "INDICADOR", "CAMBIOS"]],
        hide_index=True, height=alto or "auto",
        column_config={
            "EVENTO_ID": st.column_config.NumberColumn("Evento", format="%d", width="small"),
            "TIPO_ATAQUE": "Tipo", "CONFIANZA": "Confianza",
            "INICIO": st.column_config.DatetimeColumn("Inicio", format="DD-MMM HH:mm"),
            "FIN_REAL": st.column_config.DatetimeColumn("Fin", format="DD-MMM HH:mm"),
            "Z_MAX": st.column_config.ProgressColumn(
                "Intensidad", format="%.1f", min_value=0,
                max_value=float(EV["Z_MAX"].max()) if len(EV) else 1.0),
            "INDICADOR": "Indicador más afectado",
            "CAMBIOS": st.column_config.TextColumn("Principales cambios", width="large"),
        })

# Secciones
def pagina_resumen():
    encabezado("Resumen", "Detección de anomalías en SAP", " ")
    rango = PERIODO
    rel_r = REL
    if rango:
        ini, fin = rango
        n_win, n_anom = len(WIN), int((WIN["LABEL"] == -1).sum())
        n_ev, n_rel = len(EV), len(REL)
        tipos_ataque = sorted(set(REL["TIPO_ATAQUE"].dropna()) - {"Falla operativa"})
        det_bloques = f"del {ini:%d/%m %H:%M} al {fin:%d/%m %H:%M}"
    else:
        # Todo sale de RESULT_SUMMARY (una fila por semana): se suman las semanas del periodo.
        n_win = int(RES["VENTANAS"].sum())
        n_anom = int(RES["VENTANAS_ANOMALAS"].sum())
        n_ev = int(RES["EVENTOS"].sum())
        n_rel = int(RES["EVENTOS_RELEVANTES"].sum())
        # TIPOS_LISTA sin "Falla operativa", que resultados.py (NO_ATAQUE) no cuenta como ataque
        tipos_ataque = sorted({t.strip() for l in RES["TIPOS_LISTA"].dropna() for t in l.split(",")}
                              - {"Falla operativa"})
        det_bloques = f"{len(RES)} semana(s)"
    kpis = [
        ("Bloques analizados", f"{n_win:,}", det_bloques,
         BRAND, f"La actividad se divide en bloques de {ETIQUETA[SIZE_S]} para analizarla."),
        ("Bloques atípicos", f"{n_anom:,}", f"{n_anom / n_win:.1%} del total" if n_win else "—",
         "#E76500", "Bloques cuyo comportamiento se aleja claramente del habitual."),
        ("Eventos", f"{n_ev:,}", "detectados", "#7858FF",
         "Los bloques atípicos cercanos en el tiempo se agrupan en un solo evento."),
        ("Eventos relevantes", f"{n_rel:,}", f"intensidad de {Z_UMBRAL:g} o más", CRITICO,
         f"Eventos con una intensidad de {Z_UMBRAL:g} o más."),
        ("Tipos de ataque", f"{len(tipos_ataque)}", ", ".join(tipos_ataque) or "ninguno", "#049F9A",
         "Tipos de ataque asignados a los eventos relevantes (sin contar Falla operativa)."),
    ]
    st.html('<div class="kpis">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{ayuda}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    with tarjeta("Actividad y eventos en el tiempo",
                 "Volumen de registros a lo largo del periodo. Las franjas de color señalan los "
                 "eventos relevantes según su tipo y los puntos rojos, los bloques atípicos. "
                 "Usa la barra inferior o arrastra sobre la gráfica para acercar un periodo: "
                 "todo el dashboard se recalcula para ese periodo. Doble clic para ver todo."):
        if rango:
            r1, r2 = st.columns([4, 1], vertical_alignment="center", gap="medium")
            r1.html(f'<div class="pm-chips"><span class="pm-chip rango">Periodo seleccionado: '
                    f'{ini:%d/%m/%Y %H:%M} – {fin:%d/%m/%Y %H:%M}</span></div>')
            r2.button("Ver todo", icon=":material/zoom_out_map:", width="stretch",
                      key="tl_limpiar", on_click=limpiar_periodo)
        grafica_rango(fig=fig_linea_tiempo(rango=rango).to_json(), alto=400, key=CLAVE_TL, default=None)

    sufijo = " en el periodo seleccionado" if rango else ""
    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1, tarjeta("Eventos relevantes por tipo",
                     "Cuántos eventos se asignaron a cada tipo y con qué nivel de confianza" + sufijo + "."):
        if rel_r.empty:
            vacio("No hay eventos relevantes" + (sufijo or " con este intervalo de análisis") + ".")
        else:
            mostrar(fig_tipos(rel_r))
    with c2, tarjeta("Eventos de mayor intensidad",
                     "Los cinco eventos que más se alejaron del comportamiento habitual" + sufijo + ". "
                     "Consulta el detalle completo en la sección Eventos."):
        if rel_r.empty:
            vacio("No hay eventos relevantes" + sufijo + ".")
        else:
            top = rel_r.nlargest(5, "Z_MAX")
            st.dataframe(
                top[["EVENTO_ID", "TIPO_ATAQUE", "INICIO", "Z_MAX", "INDICADOR"]],
                hide_index=True,
                column_config={
                    "EVENTO_ID": st.column_config.NumberColumn("Evento", format="%d", width="small"),
                    "TIPO_ATAQUE": "Tipo",
                    "INICIO": st.column_config.DatetimeColumn("Inicio", format="DD-MMM HH:mm"),
                    "Z_MAX": st.column_config.ProgressColumn(
                        "Intensidad", format="%.1f", min_value=0,
                        max_value=float(EV["Z_MAX"].max())),
                    "INDICADOR": "Indicador más afectado",
                })

    pm_r = pm_all
    if not PM_MES.empty:
        n_tipos_pm = int(pm_r["TIPO_ATAQUE"].dropna().astype(str).nunique())
        with tarjeta("Reportes forenses",
                     "Reportes que generó el agente forense para los eventos relevantes"
                     + (sufijo if rango else " del periodo") + "."):
            k1, k2 = st.columns([4, 1], vertical_alignment="center", gap="medium")
            with k1:
                st.html('<div class="pm-chips">'
                        f'<span class="pm-chip">{len(pm_r)} reporte(s)'
                        f'{" en el rango" if rango else " en el periodo"}</span>'
                        f'<span class="pm-chip">{n_tipos_pm} tipo(s) de ataque</span></div>')
            with k2:
                st.page_link(pg_postmortems, label="Ver reportes", icon=":material/arrow_forward:")


def pagina_eventos():
    encabezado("Eventos", "Eventos detectados",
               "Un evento reúne bloques atípicos cercanos en el tiempo. Aquí se muestra qué "
               "aspectos de la actividad cambiaron y el detalle de cada uno.")
    if EV.empty:
        with tarjeta():
            vacio("No se detectaron eventos con los filtros seleccionados"
                  + (" en el periodo seleccionado." if PERIODO else "."))
        return

    with tarjeta("Aspectos afectados en cada evento",
                 "Qué tanto cambió cada grupo de indicadores durante el evento, comparado con una "
                 "semana normal. Los tonos más oscuros indican un cambio mayor."):
        perfil = perfil_relevantes(WIN_MES, REL, FAMILIAS[STREAM])   # z contra la semana completa
        if perfil.empty:
            vacio("No hay eventos relevantes para mostrar.")
        else:
            mostrar(fig_perfil(perfil))

    with tarjeta("Detalle de eventos",
                 "Ordenados por intensidad. La última columna resume los indicadores que más "
                 "cambiaron (↑ aumentó, ↓ disminuyó)."):
        t1, t2 = st.tabs([f"Relevantes ({len(REL)})", f"Todos ({len(EV)})"])
        with t1:
            if REL.empty:
                vacio("No se detectaron eventos relevantes.")
            else:
                tabla_eventos(REL)
        with t2:
            tabla_eventos(EV)

def pagina_tipos():
    encabezado("Tipos de ataque", "Tipos de ataque que se pueden detectar",
               "Cada evento relevante se clasifica en uno de estos tipos según los indicadores que "
               "más cambiaron. Los tipos posibles dependen del origen de los registros.")
    with tarjeta("Cómo se asigna el tipo"):
        pasos = [
            ("1 · Indicadores principales", "Se toman los 3 indicadores que más se alejaron de "
                                            "su comportamiento habitual durante el evento."),
            ("2 · Votación", "Cada indicador vota por los tipos de ataque cuyo patrón coincide con "
                             "la dirección del cambio (si subió o bajó). Su voto se reparte entre "
                             "esos tipos."),
            ("3 · Tipo asignado", "Gana el tipo con más votos. Si gana por al menos 1 voto, la "
                                  "confianza es alta; por 0.5 a 1, media; por menos de 0.5, baja."),
        ]
        for col, (t, d) in zip(st.columns(len(pasos), gap="medium"), pasos):
            col.html(f'<div class="paso"><b>{t}</b><p>{d}</p></div>')

    conteo = REL["TIPO_ATAQUE"].value_counts()
    firma = FIRMA[STREAM]
    # Orden del md; si las firmas del código tienen algún tipo más, también se muestra.
    tipos = TIPOS_POSIBLES[STREAM] + sorted({t for sg in firma.values() for t in sg}
                                            - set(TIPOS_POSIBLES[STREAM]))
    st.html(f'<p class="card-title" style="margin:.4rem 0 .8rem">{len(tipos)} tipos posibles '
            f'en {STREAMS[STREAM]}</p>')
    flecha = {"+": "↑", "-": "↓", "~": "↕"}
    texto_dir = {"+": "aumenta", "-": "disminuye", "~": "cambia"}
    tarjetas = []
    for t in tipos:
        n = int(conteo.get(t, 0))
        donde = "en el periodo seleccionado" if PERIODO else "en el periodo"
        chips = (f'<span class="chip alerta">{n} evento{"s" if n != 1 else ""} {donde}</span>'
                 if n else f'<span class="chip nada">Sin eventos {donde}</span>')
        if t in TIPOS_NO_ATAQUE:
            chips += '<span class="chip info">No cuenta como ataque</span>'
        senales = "".join(f'<li><span>{flecha[sg[t]]}</span>{nombre(f)} {texto_dir[sg[t]]}</li>'
                          for f, sg in firma.items() if t in sg)
        tarjetas.append(
            f'<div class="tipo" style="--c:{COLOR_TIPO.get(t, NEUTRO)}">'
            f'<div class="nom"><i></i>{t}</div>'
            f'<p class="desc">{DESCRIPCION_TIPO.get(t, "")}</p>'
            f'<div class="chips">{chips}</div>'
            f'<div class="sen"><p>Indicadores que lo señalan</p><ul>{senales}</ul></div></div>')
    st.html('<div class="tipos-grid">' + "".join(tarjetas) + '</div>')


def pagina_modelo():
    encabezado("Metodología", "Cómo se detectan y clasifican los eventos",
               "Los pasos del análisis y dos comprobaciones de calidad: cómo cambian los resultados "
               "según el intervalo de análisis y qué tan consistente es la clasificación.")
    with tarjeta("Proceso de análisis"):
        pasos = [
            ("1 · División en bloques", "La actividad se divide en bloques de tiempo de 5 min, "
                                        "15 min, 30 min, 1 h y 3 h, que se traslapan a la mitad. "
                                        "En cada bloque se miden 24 indicadores (Sistema SAP) o "
                                        "18 (Servicios de IA)."),
            ("2 · Detección de lo atípico", "Un modelo de aprendizaje automático (Isolation "
                                            "Forest) identifica cada semana el 1 % de bloques "
                                            "con el comportamiento más inusual."),
            ("3 · Agrupación en eventos", "Los bloques atípicos separados por 3 bloques o menos "
                                          "se unen en un evento. Es relevante si algún indicador "
                                          f"se alejó {Z_UMBRAL:g} desviaciones estándar o más de "
                                          "su promedio semanal."),
            ("4 · Clasificación", "Los 3 indicadores que más cambiaron se comparan con el patrón "
                                  "de cada tipo de ataque mediante reglas fijas, y se asigna el "
                                  "tipo que mejor coincide."),
            ("5 · Comprobación", "Un agrupamiento automático (k-means) de los bloques atípicos "
                                 "se compara con la clasificación para ver si coinciden."),
        ]
        for col, (t, d) in zip(st.columns(len(pasos), gap="medium"), pasos):
            col.html(f'<div class="paso"><b>{t}</b><p>{d}</p></div>')

    c3, c4 = st.columns(2, gap="medium")
    with c3, tarjeta("Resultados según el intervalo de análisis",
                     "Con intervalos cortos se detectan más eventos breves; con intervalos largos, "
                     "solo los que se sostienen en el tiempo."):
        ev_s = eventos_all[(eventos_all["SOURCE_TABLE"] == FUENTE) & (eventos_all["STREAM"] == STREAM)]
        ev_s = en_periodo(ev_s.assign(FIN_REAL=ev_s["FIN"] + pd.to_timedelta(ev_s["WINDOW_SIZE_S"], unit="s")))
        s = (ev_s.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])
             .agg(EVENTOS=("EVENTO_ID", "size"), EVENTOS_RELEVANTES=("RELEVANTE", "sum"))
             .reindex(pd.MultiIndex.from_frame(RES_S[["WINDOW_SIZE_S", "WINDOW_LABEL"]].drop_duplicates()),
                      fill_value=0)
             .reset_index().sort_values("WINDOW_SIZE_S"))
        fig = go.Figure()
        for colname, nombre, color in (("EVENTOS", "Eventos detectados", SERIES[0]),
                                       ("EVENTOS_RELEVANTES", "Eventos relevantes", SERIES[2])):
            fig.add_trace(go.Bar(
                x=s["WINDOW_LABEL"], y=s[colname], name=nombre,
                marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                hovertemplate="Intervalo de %{x}<br>" + nombre + ": %{y}<extra></extra>"))
        fig.update_layout(barmode="group", bargroupgap=0.08)
        fig.update_yaxes(title_text="Número de eventos", title_font_color=MUTED)
        fig.update_xaxes(title_text="Intervalo de análisis", title_font_color=MUTED)
        mostrar(estilo(fig, height=360))

    with c4, tarjeta("Consistencia de la clasificación",
                     "Qué tanto coincide la clasificación con un agrupamiento automático (k-means) "
                     "de los bloques atípicos de los eventos relevantes, según dos índices (ARI y "
                     "NMI): 1 es coincidencia total y 0, al azar. No es una validación "
                     "independiente, porque ambos métodos usan los mismos indicadores."):
        ve = VALID[VALID["ESTADO"] == "evaluable"]
        if ve.empty:
            vacio("No hay datos suficientes para evaluar la consistencia en este periodo; se "
                  "necesitan al menos dos tipos de evento.")
        else:
            g = (ve.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])[["ARI", "NMI"]].mean()
                 .reset_index().sort_values("WINDOW_SIZE_S"))
            n_sem = ve.groupby("WINDOW_SIZE_S").size()
            fig = go.Figure()
            for m, color, desc in (("ARI", SERIES[0], "Adjusted Rand Index"),
                                   ("NMI", SERIES[4], "Información mutua normalizada")):
                fig.add_trace(go.Bar(
                    x=g["WINDOW_LABEL"], y=g[m], name=f"Índice {m}",
                    marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                    customdata=n_sem.reindex(g["WINDOW_SIZE_S"]).values,
                    hovertemplate="Intervalo de %{x}<br>Índice " + m + ": %{y:.2f}<br>"
                                  "Promedio de %{customdata} semana(s)<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=AXIS, width=1))
            fig.update_layout(barmode="group", bargroupgap=0.08)
            fig.update_xaxes(title_text="Intervalo de análisis", title_font_color=MUTED)
            fig.update_yaxes(title_text="Coincidencia (0 a 1)", title_font_color=MUTED,
                             range=[min(0, g[["ARI", "NMI"]].min().min() - 0.05), 1])
            mostrar(estilo(fig, height=360))
        no_eval = int((VALID["ESTADO"] != "evaluable").sum())
        if no_eval:
            st.caption(f"En {no_eval} de {len(VALID)} combinaciones de semana e intervalo no "
                       "hubo datos suficientes para esta comparación."
                       + (" Con un periodo seleccionado se usan las semanas que lo incluyen." if PERIODO else ""))

def titulo_pm(periodo):
    """Título de Postmortems; "— AAAA-MM" nunca se parte entre dos líneas."""
    return f'Resumen forense mensual <span style="white-space: nowrap">— {periodo}</span>'


def _rango(ini, fin):
    """Fecha y horario del incidente en dos líneas cortas."""
    if pd.isna(ini) or pd.isna(fin):
        return _fmt_ts(ini), ""
    if ini.date() == fin.date():
        return f"{ini:%d/%m/%Y}", f"de {ini:%H:%M} a {fin:%H:%M}"
    return f"{ini:%d/%m/%Y %H:%M}", f"a {fin:%d/%m/%Y %H:%M}"


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
                   "Consolidado ejecutivo de todos los incidentes investigados en el periodo.",
                   pills=[("Periodo", MES.replace("_", "-")), ("Reportes", 0)])
        vacio(f"No hay reportes en el periodo seleccionado ({TXT_PERIODO}). "
              f"El mes tiene {len(PM_MES)}; usa Ver todo el mes en la barra lateral para verlos.")
        return
    if pm_all.empty:
        encabezado("Postmortems SOC", titulo_pm(periodo),
                   "Consolidado ejecutivo de todos los incidentes investigados en el periodo.",
                   pills=[("Periodo", periodo), ("Reportes", 0)])
        vacio(f"Aún no hay reportes para {periodo}. Aparecerán aquí cuando el agente forense "
              "(cron_job_worker.py) procese los eventos relevantes del periodo.")
        return

    header_slot = st.container()

    # Filtros
    with tarjeta():
        f1, f2, f3 = st.columns([1, 1.4, 1.4], gap="medium")
        origenes = ["todos"] + sorted(pm_all["STREAM"].dropna().astype(str).str.lower().unique())
        sel_origen = f1.selectbox("Origen", origenes, key="pm_origen",
                                  format_func=lambda s: "Todos" if s == "todos" else STREAMS.get(s, s))
        sel_tipos = f2.multiselect("Tipo de ataque",
                                   sorted(pm_all["TIPO_ATAQUE"].dropna().astype(str).unique()),
                                   placeholder="Todos los tipos", key="pm_tipos")
        q = f3.text_input("Buscar", placeholder="Reporte, táctica MITRE, aplicación o IP",
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
                       "Consolidado ejecutivo de todos los incidentes investigados en el periodo.",
                       pills=[("Periodo", periodo), ("Reportes", len(pm_all))])
        vacio("Ningún reporte coincide con los filtros.")
        return

    pm_f = pm_f.sort_values(["INICIO", "EVENTO_ID"], ascending=[True, True]) \
        .reset_index(drop=True)

    # Cabecera con botones globales de descarga del reporte mensual consolidado
    with header_slot:
        h_left, h_right = st.columns([1.75, 1.25], vertical_alignment="bottom", gap="medium")
        with h_left:
            encabezado("Postmortems SOC", titulo_pm(periodo),
                       "Resumen consolidado de todos los incidentes investigados del mes y "
                       "exportación ejecutiva multipágina.",
                       pills=[("Periodo", periodo), ("Incidentes", len(pm_f))])
        with h_right:
            b1, b2 = st.columns(2, gap="small")
            with b1:
                try:
                    pdf_mes = _construir_pdf_mensual_bytes(pm_f, periodo)
                    st.download_button(
                        "Descargar PDF",
                        data=pdf_mes,
                        file_name=f"Postmortem_SOC_{MES}.pdf",
                        mime="application/pdf",
                        icon=":material/picture_as_pdf:",
                        type="primary",
                        width="stretch",
                        help="Genera el PDF ejecutivo multipágina con el resumen del mes y el detalle de cada incidente.",
                    )
                except Exception as err:
                    st.warning(f"No se pudo generar el PDF consolidado: {err}")
            with b2:
                md_mes = _construir_markdown_mensual(pm_f, periodo)
                st.download_button(
                    "Descargar Markdown",
                    data=md_mes.encode("utf-8"),
                    file_name=f"Postmortem_SOC_{MES}.md",
                    mime="text/markdown",
                    icon=":material/description:",
                    width="stretch",
                    help="Descarga el consolidado completo del mes en Markdown.",
                )

    # KPIs del periodo (sin severidad)
    n_tacticas = int(pm_f["MITRE_TACTIC"].dropna().astype(str).nunique())
    apps = sorted({a for v in pm_f["APLICACIONES_AFECTADAS"] for a in _parse_lista_campo(v)})
    ips_mes = sorted({ip for v in pm_f["IPS_INVOLUCRADAS"] for ip in _parse_lista_campo(v)})
    kpis = [
        ("Incidentes del mes", f"{len(pm_f)}", f"de {len(pm_all)} en el periodo", BRAND,
         "Reportes forenses incluidos en el consolidado mensual."),
        ("Tácticas MITRE", f"{n_tacticas}", "tácticas distintas", "#7858FF",
         "Tácticas MITRE ATT&CK / ATLAS asignadas en el periodo."),
        ("Aplicaciones SAP", f"{len(apps)}", f"{len(ips_mes)} IP(s) involucradas", "#049F9A",
         ", ".join(apps) or "Sin aplicaciones registradas."),
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
            "Campaña / Vector": str(vec),
            "Origen": st_vec,
            "Táctica MITRE": mitre_v,
            "Incidentes": len(g),
            "Ventanas detectadas": ", ".join(sorted(escs_v, key=lambda s: orden_esc.get(s, 99))),
            "Indicador dominante": f"{nombre(best_f)} (Z = {best_z:.1f})",
            "Apps SAP": len(apps_v),
        })

    with tarjeta("Síntesis ejecutiva del periodo",
                 "Panorama unificado por campaña de amenaza, aplicaciones e IPs comprometidas en el mes."):
        st.html(
            f'<div class="pm-callout"><b>Dictamen mensual — {esc(periodo)}</b>'
            f'Se investigaron <strong>{len(pm_f)}</strong> incidente(s) únicos multi-escala '
            f'en la ventana <strong>{esc(ini_global)} {esc(fin_global)}</strong>, agrupados en '
            f'<strong>{len(grupos_camp)}</strong> familia(s) de amenaza '
            f'({esc(", ".join(f"{v} ({len(g)})" for v, g in grupos_camp))}) '
            f'sobre <strong>{len(apps)}</strong> aplicación(es) SAP y '
            f'<strong>{len(ips_mes)}</strong> dirección(es) IP.</div>'
        )
        st.dataframe(pd.DataFrame(resumen_camp_rows), hide_index=True)
        st.html(
            '<div class="pm-meta dos" style="margin-top:.65rem">'
            f'<div><p class="lbl">Aplicaciones SAP afectadas ({len(apps)})</p>'
            f'{_chips(apps, "Sin aplicaciones registradas.")}</div>'
            f'<div><p class="lbl">Direcciones IP involucradas ({len(ips_mes)})</p>'
            f'{_chips(ips_mes[:18], "Sin IPs externas asociadas.")}</div>'
            '</div>'
        )
        st.html(
            '<div style="margin-top:.55rem">'
            f'<p class="lbl" style="font-size:.72rem;font-weight:700;letter-spacing:.05em;'
            f'text-transform:uppercase;color:{MUTED};margin:0 0 .3rem;">Tácticas MITRE del mes</p>'
            f'{_chips(tacticas_mes, "Sin tácticas mapeadas.")}</div>'
        )

    # Desglose unificado por campana y cronologia completa
    with tarjeta(f"Dossier forense del periodo ({len(pm_f)} incidentes en {len(grupos_camp)} campañas)",
                 "Consulta el análisis agrupado por familia de amenaza o la cronología completa del mes."):
        tab_camp, tab_cron = st.tabs([
            "Por campaña / familia de amenaza",
            "Cronología completa de incidentes",
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
                    f'<span class="pm-pill brand">{len(g)} incidente(s)</span>'
                    f'<span class="pm-pill">{esc(st_vec)}</span>'
                    f'<span class="pm-pill">Ventanas: {esc(", ".join(escs_c))}</span></div>'
                    f'<p class="pm-sub"><b>{esc(mitre_v)}</b> · '
                    f'{len(apps_c)} app(s) SAP · {len(ips_c)} IP(s) involucradas</p>'
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
                        f'Ventanas: {esc(esc_i)} · {esc(nombre(z_feat))} (Z = {float(inc.get("z_max") or 0):.1f})</b>'
                        f'{esc(str(fila.get("RESUMEN_EJECUTIVO") or "Sin resumen registrado."))}</div>'
                    )
                with st.expander(f"Ver reportes individuales y evidencia MCP de {vec} ({len(g)})"):
                    for _, fila in g.iterrows():
                        pm_id = str(fila.get("POSTMORTEM_ID") or "PM-N/A")
                        meta = _parse_json_safe(fila.get("METADATA_JSON"))
                        md = str(fila.get("REPORTE_MARKDOWN") or "Sin reporte registrado.")
                        logs = (meta.get("evidence_logs_sample")
                                or (meta.get("raw_evidence") or {}).get("logs_muestra") or [])
                        st.markdown(f"#### {pm_id}")
                        st.markdown(_sin_titulo(md))
                        if logs and isinstance(logs, list):
                            st.caption(f"Muestra de registros MCP ({pm_id})")
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
                md = str(fila.get("REPORTE_MARKDOWN") or "Sin reporte registrado.")

                st.html(
                    f'<div class="pm-head"><span class="id">{esc(pm_id)}</span>'
                    f'<span class="pm-pill brand">{esc(origen)}</span>'
                    f'<span class="pm-pill">Evento {int(fila.get("EVENTO_ID") or 0)}</span>'
                    f'<span class="pm-pill">Ventanas: {esc(", ".join(escalas_l))}</span></div>'
                    f'<p class="pm-sub"><b>{esc(str(fila.get("TIPO_ATAQUE") or "N/A"))}</b> · '
                    f'{esc(str(fila.get("MITRE_TACTIC") or "Táctica pendiente"))} · '
                    f'{esc(fecha)} {esc(horas)} · '
                    f'Indicador: {esc(nombre(z_feat))} (Z = {float(inc.get("z_max") or 0):.1f}) · '
                    f'{len(apps_l)} app(s) · {len(ips_l)} IP(s)</p>'
                )
                st.html(
                    f'<div class="pm-callout"><b>Resumen ejecutivo</b>'
                    f'{esc(str(fila.get("RESUMEN_EJECUTIVO") or "Sin resumen registrado."))}</div>'
                )

                with st.expander(f"Ver reporte completo y evidencia ({pm_id})"):
                    t1, t2 = st.tabs(["Reporte completo", "Evidencia"])
                    with t1:
                        st.markdown(_sin_titulo(md))
                    with t2:
                        st.html(
                            '<div class="pm-meta dos">'
                            f'<div><p class="lbl">Aplicaciones SAP afectadas</p>'
                            f'{_chips(apps_l, "Sin aplicaciones registradas.")}</div>'
                            f'<div><p class="lbl">Direcciones IP involucradas</p>'
                            f'{_chips(ips_l, "Sin IPs externas asociadas.")}</div>'
                            '</div>'
                        )
                        logs = (meta.get("evidence_logs_sample")
                                or (meta.get("raw_evidence") or {}).get("logs_muestra") or [])
                        if logs and isinstance(logs, list):
                            st.dataframe(pd.DataFrame(logs), hide_index=True)
                        else:
                            vacio("Este reporte no incluye una muestra de registros.")
                        st.json(meta or {"nota": "Sin metadatos."})

                if idx < len(pm_f) - 1:
                    st.divider()




# --- Entrenamiento en vivo -------------------------------------------------
PASOS = [
    ("Conexión a SAP HANA", "Abre una sesión propia para las tablas temporales."),
    ("Selección de la semana", "Filtra los registros de la tabla a la semana elegida."),
    ("Eventos base", "Convierte timestamps y calcula banderas por registro."),
    ("Construcción de ventanas", "Agrega los registros en ventanas con 50 % de traslape."),
    ("Entrenamiento de Isolation Forest", "PAL ajusta el bosque de árboles de aislamiento."),
    ("Puntuación de ventanas", "Score de anomalía y etiqueta para cada ventana."),
    ("Eventos, relevancia y tipo", "Agrupa anomalías, filtra por |z| y asigna el tipo de ataque."),
    ("Limpieza", "Borra las tablas temporales y cierra la sesión."),
]


@st.cache_data(ttl=600, show_spinner=False)
def semanas_de(tabla):
    t0, t_max, n = calculate_week_range(_conn(), tabla, wf.WEEK_DAYS, wf.MIN_LAST_DAYS)
    return tabla_semanas(t0, t_max, n, wf.WEEK_DAYS)


def _html_pasos(estado, dur):
    iconos = {"pendiente": "", "corriendo": "", "ok": "✓", "error": "!"}
    out = '<div class="tr-steps">'
    for i, ((nom, desc), e, d) in enumerate(zip(PASOS, estado, dur), 1):
        t = f"{d:.1f} s" if d is not None else ("en curso…" if e == "corriendo" else "")
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
        txt = (f"Completado en {time.perf_counter() - t_ini:.0f} s" if hechos == len(PASOS) else
               f"Paso {hechos + 1} de {len(PASOS)} · {time.perf_counter() - t_ini:.0f} s transcurridos")
        ui["prog"].progress(hechos / len(PASOS), text=txt)

    @contextmanager
    def paso(i):
        estado[i] = "corriendo"
        pintar()
        log(f"Inicia: {PASOS[i][0]}")
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
            log("Sesión dedicada abierta: las tablas temporales (#) solo existen en ella.")
        with paso(1):
            df_sys, df_llm = wf.cargar_datos(conn, tabla)
            sw, lw, sem = split_into_weeks(df_sys, df_llm, conn, tabla, wf.WEEK_DAYS, wf.MIN_LAST_DAYS)
            df_w = (sw if es_sys else lw)[semana]
            res["registros"] = int(df_w.count())
            fila = sem.iloc[semana]
            log(f"{tabla} · semana {semana + 1} ({fila['WEEK_START']:%d/%m} – {fila['WEEK_END']:%d/%m}): "
                f"{res['registros']:,} registros de {STREAMS[stream]}.")
        with paso(2):
            (ventanas_mod.prepare_system_base if es_sys else ventanas_mod.prepare_llm_base)(conn, df_w)
            log("Eventos base materializados en " + ("#BASE_SYSTEM." if es_sys else "#BASE_LLM."))
        with paso(3):
            build = ventanas_mod.build_system_windows if es_sys else ventanas_mod.build_llm_windows
            win = ventanas_mod.materialize(conn, build(conn, size_s), "#WIN_SYSTEM" if es_sys else "#WIN_LLM")
            ventanas_mod.drop_temp(conn, ventanas_mod.SYSTEM_EVENTS_TABLE if es_sys
                                   else ventanas_mod.LLM_EVENTS_TABLE)
            res["ventanas"] = int(win.count())
            log(f"{res['ventanas']:,} ventanas de {ventanas_mod.size_label(size_s)} con {len(feats)} variables.")
        with paso(4):
            n = res["ventanas"]
            model = PalIsolationForest(n_estimators=IF_N_ESTIMATORS, max_samples=min(IF_MAX_SAMPLES, n),
                                       random_state=IF_RANDOM_STATE)
            log(f"Parámetros: n_estimators={IF_N_ESTIMATORS}, max_samples={min(IF_MAX_SAMPLES, n)}, "
                f"random_state={IF_RANDOM_STATE}.")
            t_fit = time.perf_counter()
            model.fit(data=win, key="WINDOW_INDEX", features=feats)
            res["t_fit"] = time.perf_counter() - t_fit
            log(f"Modelo ajustado en SAP HANA PAL en {res['t_fit']:.1f} s.")
        with paso(5):
            pred = model.predict(data=win, key="WINDOW_INDEX", features=feats,
                                 contamination=CONTAMINATION_IF, show_explainer=False)
            win_pdf = win.collect()
            r = win_pdf.merge(pred.collect(), on="WINDOW_INDEX")
            idx = r.loc[r["LABEL"] == -1, "WINDOW_INDEX"].tolist()
            z = zscores(win_pdf, idx, feats)
            res["anomalas"] = len(idx)
            log(f"{len(idx)} ventanas anómalas ({len(idx) / len(r):.2%}) con contaminación {CONTAMINATION_IF}.")
        with paso(6):
            r.insert(0, "WEEK", semana)
            z.insert(0, "WEEK", semana)
            r = agrupar_eventos(r, wf.MAX_GAP[size_s][stream])
            ev = resumen_eventos(r)
            ev = marcar_relevantes(r, z, ev, wf.Z_UMBRAL[size_s])
            tipos = clasificar_eventos(r, z, ev, firmas=wf.FIRMAS[stream])
            ev = ev.merge(tipos, on="EVENTO_ID", how="left")
            res.update(eventos=len(ev), relevantes=int(ev["RELEVANTE"].sum()), ev=ev, win=r)
            log(f"{len(ev)} eventos; {res['relevantes']} relevantes (|z| ≥ {wf.Z_UMBRAL[size_s]}).")
            for _, e in ev[ev["RELEVANTE"]].iterrows():
                log(f"  Evento {int(e['EVENTO_ID'])}: {e['TIPO_ATAQUE']} (confianza {e['CONFIANZA']}, "
                    f"z máx {e['Z_MAX']:.1f} en {e['Z_MAX_FEATURE']}).")
        with paso(7):
            ventanas_mod.drop_temp(conn, "#WIN_SYSTEM" if es_sys else "#WIN_LLM")
            ventanas_mod.drop_temp(conn, ventanas_mod.SYSTEM_BASE_TABLE if es_sys
                                   else ventanas_mod.LLM_BASE_TABLE)
            conn.close()
            conn = None
            log("Tablas temporales borradas. No se modificó ninguna tabla RESULT_*.")
        res["total"] = time.perf_counter() - t_ini
        res["durs"] = dur
        log(f"Entrenamiento completo en {res['total']:.0f} s.")
        res["log"] = lineas
        return res
    except Exception as e:
        log(f"ERROR: {e}")
        st.error(f"El entrenamiento se detuvo: {e}")
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
    with tarjeta("Comparación con producción",
                 "Verifica que el entrenamiento sea reproducible: con la misma semilla (random_state = 42) "
                 "debe obtener lo mismo que guardó el pipeline en RESULT_SUMMARY."):
        if g.empty:
            st.html('<div class="hc-group"><div class="hc-row"><i class="hc-dot info"></i>'
                    '<div class="hc-main"><b>Sin referencia</b><span>Esta tabla aún no se procesa en '
                    'producción, así que no hay resultados guardados para comparar.</span></div>'
                    '<div class="hc-val"><span class="hc-pill info">Informativo</span></div></div></div>')
            return
        g = g.iloc[0]
        filas = [("Ventanas", res["ventanas"], int(g["VENTANAS"])),
                 ("Ventanas anómalas", res["anomalas"], int(g["VENTANAS_ANOMALAS"])),
                 ("Eventos", res["eventos"], int(g["EVENTOS"])),
                 ("Eventos relevantes", res["relevantes"], int(g["EVENTOS_RELEVANTES"]))]
        out = '<div class="hc-grid">'
        for i in range(0, len(filas), 2):
            out += '<div class="hc-group">'
            for nom, vivo, prod in filas[i:i + 2]:
                cls = "ok" if vivo == prod else "warn"
                out += (f'<div class="hc-row"><i class="hc-dot {cls}"></i><div class="hc-main"><b>{nom}</b>'
                        f'<span>Esta corrida: {vivo:,} · Guardado en producción: {prod:,}</span></div>'
                        f'<div class="hc-val"><span class="hc-pill {cls}">'
                        f'{"Coincide" if vivo == prod else "Difiere"}</span></div></div>')
            out += '</div>'
        st.html(out + '</div>')


def _resultado_entrenamiento(res):
    r, ev = res["win"], res["ev"]
    kpis = [
        ("Ventanas", f"{res['ventanas']:,}", f"de {ventanas_mod.size_label(res['size_s'])}", BRAND,
         "Ventanas construidas para la semana."),
        ("Ventanas anómalas", f"{res['anomalas']}", f"{res['anomalas'] / res['ventanas']:.2%} del total",
         "#E76500", "Marcadas por Isolation Forest."),
        ("Eventos relevantes", f"{res['relevantes']}", f"de {res['eventos']} eventos", CRITICO,
         "Eventos con |z| ≥ 5 en alguna variable."),
        ("Tiempo total", f"{res['total']:.0f} s", f"entrenamiento: {res['t_fit']:.1f} s", "#049F9A",
         "Duración de todo el proceso y del ajuste del modelo."),
    ]
    st.html('<div class="kpis kpis-4">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{ayuda}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    _comparar_con_produccion(res)

    c1, c2 = st.columns([1.6, 1], gap="medium")
    with c1, tarjeta("Puntaje de anomalía en el tiempo",
                     "Score de cada ventana de la semana. Los puntos rojos son las ventanas anómalas."):
        a = r[r["LABEL"] == -1]
        fig = go.Figure([
            go.Scatter(x=r["WINDOW_START"], y=r["SCORE"], mode="lines", name="Score",
                       line=dict(color=SERIES[0], width=1.2)),
            go.Scatter(x=a["WINDOW_START"], y=a["SCORE"], mode="markers", name="Anómala",
                       marker=dict(color=CRITICO, size=6)),
        ])
        fig.update_yaxes(title_text="Score", title_font_color=MUTED)
        mostrar(estilo(fig, height=300))
    with c2, tarjeta("Distribución del puntaje",
                     "Histograma de scores. La línea marca el umbral a partir del cual una ventana "
                     "se considera anómala."):
        umbral = float(r.loc[r["LABEL"] == -1, "SCORE"].min())
        fig = go.Figure(go.Histogram(x=r["SCORE"], nbinsx=40, marker=dict(color=SERIES[0]),
                                     hovertemplate="Score %{x}<br>%{y} ventanas<extra></extra>"))
        fig.add_vline(x=umbral, line=dict(color=CRITICO, dash="dot", width=1.5))
        fig.update_yaxes(title_text="Ventanas", title_font_color=MUTED)
        mostrar(estilo(fig, height=300, legend=False))

    with tarjeta("Eventos detectados en esta corrida",
                 "Eventos agrupados a partir de las ventanas anómalas, ordenados por intensidad."):
        if ev.empty:
            vacio("No se formaron eventos.")
        else:
            st.dataframe(ev.sort_values("Z_MAX", ascending=False)[
                ["EVENTO_ID", "INICIO", "N_VENTANAS", "Z_MAX", "Z_MAX_FEATURE", "RELEVANTE",
                 "TIPO_ATAQUE", "CONFIANZA"]].assign(Z_MAX_FEATURE=lambda d: d["Z_MAX_FEATURE"].map(nombre)),
                hide_index=True, column_config={
                    "EVENTO_ID": st.column_config.NumberColumn("Evento", format="%d"),
                    "INICIO": st.column_config.DatetimeColumn("Inicio", format="DD/MM HH:mm"),
                    "N_VENTANAS": st.column_config.NumberColumn("Ventanas", format="%d"),
                    "Z_MAX": st.column_config.NumberColumn("Intensidad", format="%.1f"),
                    "Z_MAX_FEATURE": "Indicador más afectado",
                    "RELEVANTE": st.column_config.CheckboxColumn("Relevante"),
                    "TIPO_ATAQUE": "Tipo de ataque", "CONFIANZA": "Confianza",
                })


def _vista_entrenamiento():
    inv = inventario_hana()
    tablas = sorted(inv.loc[inv["TABLE_NAME"].str.startswith("LOGS"), "TABLE_NAME"])
    with tarjeta("Configuración de la corrida",
                 "Corre el mismo código del pipeline de producción (src/workflow.py) sobre una semana, "
                 "en tablas temporales de SAP HANA. No modifica los resultados guardados."):
        f1, f2, f3, f4 = st.columns([1.2, 1.4, 1, 1], gap="medium")
        tabla = f1.selectbox("Tabla de logs", tablas, key="tr_tabla",
                             index=tablas.index("LOGS_2026_10") if "LOGS_2026_10" in tablas else 0)
        sem = semanas_de(tabla)
        semana = f2.selectbox("Semana", list(sem["WEEK"]), key="tr_semana",
                              format_func=lambda w: f"Semana {w + 1} · {sem.loc[w, 'WEEK_START']:%d/%m} – "
                                                    f"{sem.loc[w, 'WEEK_END']:%d/%m}")
        size_s = f3.selectbox("Intervalo", list(wf.SIZES), index=3, key="tr_size",
                              format_func=ventanas_mod.size_label)
        stream = f4.selectbox("Origen", list(STREAMS), format_func=STREAMS.get, key="tr_stream")
        correr = st.button("Iniciar entrenamiento", icon=":material/play_arrow:", type="primary",
                           key="tr_correr")
        st.caption("Intervalos cortos generan más ventanas y tardan más: 1 h suele terminar en menos de un minuto; "
                   "5 min puede tardar varios minutos.")

    c1, c2 = st.columns([1, 1.25], gap="medium")
    with c1, tarjeta("Progreso", "Cada paso se marca al terminar, con su duración."):
        ui = {"prog": st.empty(), "pasos": st.empty()}
    with c2, tarjeta("Bitácora", "Mensajes del proceso en tiempo real."):
        ui["log"] = st.empty()

    if correr:
        res = _correr_entrenamiento(tabla, int(semana), int(size_s), stream, ui)
        if res:
            st.session_state["tr_resultado"] = res
    res = st.session_state.get("tr_resultado")
    if res and not correr:
        ui["prog"].progress(1.0, text=f"Última corrida: {res['tabla']} · semana {res['semana'] + 1} · "
                                      f"{ventanas_mod.size_label(res['size_s'])} · {STREAMS[res['stream']]}")
        ui["pasos"].html(_html_pasos(["ok"] * len(PASOS), res["durs"]))
        ui["log"].code("\n".join(res["log"][-18:]), language=None, wrap_lines=True)
    elif not correr:
        ui["prog"].progress(0.0, text="Listo para iniciar")
        ui["pasos"].html(_html_pasos(["pendiente"] * len(PASOS), [None] * len(PASOS)))
        ui["log"].code("Elige la configuración y presiona Iniciar entrenamiento.", language=None, wrap_lines=True)
    if res:
        _resultado_entrenamiento(res)


def pagina_mlops():
    encabezado("MLOps", "Operación del modelo",
               "Estado del sistema en producción y entrenamiento del modelo en tiempo real.",
               pills=[("Modelo", "Isolation Forest (PAL)"), ("Contaminación", f"{CONTAMINATION_IF:.0%}"),
                      ("Árboles", IF_N_ESTIMATORS)], con_periodo=False)
    vista = st.segmented_control("Vista", ["Health checks", "Entrenamiento en vivo"], default="Health checks",
                                 key="mlops_vista", label_visibility="collapsed") or "Health checks"
    if vista == "Health checks":
        vista_health(tarjeta, estilo, mostrar)
    else:
        _vista_entrenamiento()


# Navegación
pg_resumen = st.Page(pagina_resumen, title="Resumen", icon=":material/dashboard:",
                     url_path="resumen", default=True)
pg_eventos = st.Page(pagina_eventos, title="Eventos", icon=":material/timeline:", url_path="eventos")
pg_tipos = st.Page(pagina_tipos, title="Tipos de ataque", icon=":material/category:", url_path="tipos")
pg_postmortems = st.Page(pagina_postmortems, title="Postmortems SOC", icon=":material/security:",
                         url_path="postmortems")
pg_modelo = st.Page(pagina_modelo, title="Metodología", icon=":material/model_training:",
                    url_path="metodologia")

pg_mlops = st.Page(pagina_mlops, title="MLOps", icon=":material/monitor_heart:", url_path="mlops")

pg = st.navigation([pg_resumen, pg_eventos, pg_tipos, pg_postmortems, pg_modelo, pg_mlops], position="top")
pg.run()
