import html
import io
import json
import re
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

from generador_pdf_postmortem import generar_pdf_postmortem
from helpers.conexion import _params
from helpers.explicacion import FAMILIAS_SYSTEM, FAMILIAS_LLM, zscores, perfil_familias
from helpers.clasificacion import FIRMAS, TIPOS_NO_ATAQUE

st.set_page_config(page_title="Anomalías SAP", page_icon=":material/shield:", layout="wide")

BRAND, NAVY, GOLD = "#0070F2", "#1D2D3E", "#F0AB00"
# Paleta de gráficas de SAP Fiori, en un orden validado para daltonismo.
SERIES = ["#5899DA", "#19A979", "#945ECF", "#ED4A7B", "#13A4B4", "#525DF4"]
# El dashboard usa siempre el tema claro (.streamlit/config.toml no define tema oscuro).
INK, INK2, MUTED, GRID, AXIS = "#1D2D3E", "#3d5166", "#556B82", "#eaedf0", "#d5dadd"
NEUTRO, SURFACE, CARD_BORDER, PAGE = "#c4cbd3", "#ffffff", "#e1e5ea", "#f5f6f7"
DIV_MID = "#f0f2f4"
CRITICO = "#D20A0A"
# Escala de azules de SAP Horizon (de claro a oscuro) para los mapas de calor.
AZUL_SEQ = ["#d1efff", "#a6e0ff", "#7bcfff", "#4db1ff", "#1b90ff", "#0070f2", "#0057d2", "#0040b0"]
CONFIANZA_COLOR = {"alta": "#0040b0", "media": "#0070f2", "baja": "#4db1ff"}

COLOR_TIPO = {
    # system
    "Contexto SAP": SERIES[0], "DDoS": SERIES[1], "DoS": SERIES[2], "Fuerza bruta": SERIES[3],
    "Explotación": SERIES[4], "Reconocimiento": SERIES[5], "Falla operativa": "#8d99a8",
    # llm
    "Prompt injection": SERIES[0], "Degradación del servicio": SERIES[1],
    "Abuso de recursos": SERIES[2], "Acceso no autorizado": SERIES[3], "Automatización": SERIES[4],
}


STREAMS = {"system": "Sistema SAP", "llm": "Servicios de IA"}
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
  @media (max-width: 1000px) {{ .kpis, .kpis.kpis-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
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


def encabezado(eyebrow, titulo, sub, pills=None):
    pills = pills or [("Periodo", MES.replace("_", "-")), ("Origen", STREAMS[STREAM]),
                      ("Bloques de", ETIQUETA[SIZE_S])]
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
@st.cache_resource(show_spinner=False)
def _crear_conexion():
    # Tiempo límite de 2 min por consulta: si la red se cae, falla en lugar de quedarse colgada.
    return hana_df.ConnectionContext(**_params(), communicationTimeout=120000)


def _conn():
    """Conexión guardada entre recargas; si dejó de responder (p. ej. tras horas sin uso), se rehace."""
    c = _crear_conexion()
    try:
        cur = c.connection.cursor()
        cur.execute("SELECT 1 FROM DUMMY")
        cur.close()
    except Exception:
        _crear_conexion.clear()
        c = _crear_conexion()
    return c


@st.cache_data(ttl=600, show_spinner="Cargando datos…")
def meses_disponibles():
    t = _conn().sql(
        "SELECT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
        "AND TABLE_NAME LIKE 'RESULT_EVENTS_%'").collect()
    return sorted((n.replace("RESULT_EVENTS_", "") for n in t["TABLE_NAME"]), reverse=True)


@st.cache_data(ttl=600, show_spinner="Cargando datos…")
def leer(tipo, mes):
    df = _conn().sql(f'SELECT * FROM "RESULT_{tipo}_{mes}"').collect()
    for c in ("WEEK_START", "WEEK_END", "WINDOW_START", "INICIO", "FIN", "VENTANA_PICO"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


_POSTMORTEM_COLS = [
    "POSTMORTEM_ID", "EVENTO_ID", "STREAM", "SOURCE_TABLE", "TIPO_ATAQUE",
    "MITRE_TACTIC", "SEVERIDAD_SCORE", "INICIO", "FIN", "APLICACIONES_AFECTADAS",
    "IPS_INVOLUCRADAS", "RESUMEN_EJECUTIVO", "REPORTE_MARKDOWN", "METADATA_JSON",
    "FECHA_GENERACION",
]


def _parse_json_safe(raw):
    """Convierte METADATA_JSON de HANA a dict de forma segura."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


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


@st.cache_data(ttl=120, show_spinner="Cargando reportes postmortem…")
def leer_postmortems(mes):
    """Lee RESULT_POSTMORTEMS_<mes> si existe en el esquema activo; si no, retorna vacío."""
    tabla = f"RESULT_POSTMORTEMS_{mes}"
    existe = _conn().sql(
        f"SELECT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
        f"AND TABLE_NAME = '{tabla}'"
    ).collect()
    if existe.empty:
        return pd.DataFrame(columns=_POSTMORTEM_COLS)
    df = _conn().sql(
        f'SELECT * FROM "{tabla}" ORDER BY SEVERIDAD_SCORE DESC, FECHA_GENERACION DESC'
    ).collect()
    for c in ("INICIO", "FIN", "FECHA_GENERACION"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def _fmt_ts(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return "N/A"
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d %H:%M:%S")
    return str(val)


def _construir_pdf_bytes(fila):
    """Construye en memoria el reporte ejecutivo PDF a partir de una fila HANA."""
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

    payload = {
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
    buf = io.BytesIO()
    generar_pdf_postmortem(payload, buf)
    return buf.getvalue()


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

# Gráficas
def fig_linea_tiempo(height=400):
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
    return estilo(fig, height=height)


def fig_tipos():
    ct = (REL.groupby(["TIPO_ATAQUE", "CONFIANZA"]).size().unstack(fill_value=0)
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

    # Todo sale de RESULT_SUMMARY (una fila por semana): se suman las semanas del periodo.
    n_win = int(RES["VENTANAS"].sum())
    n_anom = int(RES["VENTANAS_ANOMALAS"].sum())
    n_ev = int(RES["EVENTOS"].sum())
    n_rel = int(RES["EVENTOS_RELEVANTES"].sum())
    # TIPOS_LISTA sin "Falla operativa", que resultados.py (NO_ATAQUE) no cuenta como ataque
    tipos_ataque = sorted({t.strip() for l in RES["TIPOS_LISTA"].dropna() for t in l.split(",")}
                          - {"Falla operativa"})
    kpis = [
        ("Bloques analizados", f"{n_win:,}", f"{len(RES)} semana(s)",
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
                 "Usa la barra inferior para acercar un periodo."):
        mostrar(fig_linea_tiempo())

    c1, c2 = st.columns([1, 1.15], gap="medium")
    with c1, tarjeta("Eventos relevantes por tipo",
                     "Cuántos eventos se asignaron a cada tipo y con qué nivel de confianza."):
        if REL.empty:
            vacio("No se detectaron eventos relevantes con este intervalo de análisis.")
        else:
            mostrar(fig_tipos())
    with c2, tarjeta("Eventos de mayor intensidad",
                     "Los cinco eventos que más se alejaron del comportamiento habitual. "
                     "Consulta el detalle completo en la sección Eventos."):
        if REL.empty:
            vacio("No se detectaron eventos relevantes.")
        else:
            top = REL.nlargest(5, "Z_MAX")
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

    if not pm_all.empty:
        n_crit = int((pm_all["SEVERIDAD_SCORE"] >= 0.70).sum())
        with tarjeta("Reportes forenses",
                     "Reportes que generó el agente forense para los eventos relevantes del periodo."):
            k1, k2 = st.columns([4, 1], vertical_alignment="center", gap="medium")
            with k1:
                st.html('<div class="pm-chips">'
                        f'<span class="pm-chip">{len(pm_all)} reporte(s)</span>'
                        f'<span class="pm-chip">{n_crit} de severidad crítica</span>'
                        f'<span class="pm-chip">Severidad máxima '
                        f'{float(pm_all["SEVERIDAD_SCORE"].max()):.2f}</span></div>')
            with k2:
                st.page_link(pg_postmortems, label="Ver reportes", icon=":material/arrow_forward:")


def pagina_eventos():
    encabezado("Eventos", "Eventos detectados",
               "Un evento reúne bloques atípicos cercanos en el tiempo. Aquí se muestra qué "
               "aspectos de la actividad cambiaron y el detalle de cada uno.")
    if EV.empty:
        with tarjeta():
            vacio("No se detectaron eventos con los filtros seleccionados.")
        return

    with tarjeta("Aspectos afectados en cada evento",
                 "Qué tanto cambió cada grupo de indicadores durante el evento, comparado con una "
                 "semana normal. Los tonos más oscuros indican un cambio mayor."):
        perfil = perfil_relevantes(WIN, REL, FAMILIAS[STREAM])
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
        chips = (f'<span class="chip alerta">{n} evento{"s" if n != 1 else ""} en el periodo</span>'
                 if n else '<span class="chip nada">Sin eventos en el periodo</span>')
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
        s = (RES_S.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])[["EVENTOS", "EVENTOS_RELEVANTES"]]
             .sum().reset_index().sort_values("WINDOW_SIZE_S"))
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
                       "hubo datos suficientes para esta comparación.")

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
    encabezado("Postmortems SOC", "Reportes forenses de los incidentes",
               "Cada reporte explica un evento relevante: qué pasó, cuándo, qué afectó y qué hacer. "
               "Selecciona uno en la tabla para ver su detalle y descargarlo.",
               pills=[("Periodo", periodo), ("Reportes", len(pm_all))])
    if pm_all.empty:
        vacio(f"Aún no hay reportes para {periodo}. Aparecerán aquí cuando el agente forense "
              "(cron_job_worker.py) procese los eventos relevantes del periodo.")
        return

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
        vacio("Ningún reporte coincide con los filtros.")
        return
    pm_f = pm_f.sort_values(["SEVERIDAD_SCORE", "FECHA_GENERACION"], ascending=False) \
        .reset_index(drop=True)

    # KPIs
    sev_max = float(pm_f["SEVERIDAD_SCORE"].max())
    n_crit = int((pm_f["SEVERIDAD_SCORE"] >= 0.70).sum())
    n_tacticas = int(pm_f["MITRE_TACTIC"].dropna().astype(str).nunique())
    apps = sorted({a for v in pm_f["APLICACIONES_AFECTADAS"] for a in _parse_lista_campo(v)})
    kpis = [
        ("Reportes", f"{len(pm_f)}", f"de {len(pm_all)} en el periodo", BRAND,
         "Reportes forenses que cumplen los filtros."),
        ("Severidad máxima", f"{sev_max:.2f}", f"{n_crit} crítico(s), 0.70 o más",
         CRITICO if sev_max >= 0.70 else "#E76500",
         "Puntuación de anomalía del modelo (de 0 a 1) del reporte más grave."),
        ("Tácticas MITRE", f"{n_tacticas}", "tácticas distintas", "#7858FF",
         "Tácticas MITRE ATT&CK / ATLAS asignadas a los incidentes."),
        ("Aplicaciones SAP", f"{len(apps)}", "afectadas en total", "#049F9A",
         ", ".join(apps) or "Sin aplicaciones registradas."),
    ]
    st.html('<div class="kpis kpis-4">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{html.escape(ayuda)}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    # Cola de incidentes (la fila seleccionada define el expediente) y severidad por expediente
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1, tarjeta("Cola de incidentes investigados",
                     "Ordenados de mayor a menor severidad. Haz clic en una fila para "
                     "abrir su expediente."):
        sel = st.dataframe(
            pm_f[["POSTMORTEM_ID", "TIPO_ATAQUE", "MITRE_TACTIC"]],
            hide_index=True, on_select="rerun", selection_mode="single-row", key="pm_tabla",
            column_config={
                "POSTMORTEM_ID": "Expediente",
                "TIPO_ATAQUE": "Tipo de ataque",
                "MITRE_TACTIC": "Táctica MITRE",
            })
    filas = sel.selection.rows
    i_sel = filas[0] if filas and filas[0] < len(pm_f) else 0
    fila = pm_f.iloc[i_sel]

    with c2, tarjeta("Severidad por expediente",
                     "Puntuación de anomalía de cada expediente, de 0 a 1. Resaltado, el que está abierto."):
        orden = pm_f.iloc[::-1]
        abierto = orden["POSTMORTEM_ID"] == fila["POSTMORTEM_ID"]
        fig = go.Figure(go.Bar(
            x=orden["SEVERIDAD_SCORE"], y=orden["POSTMORTEM_ID"], orientation="h",
            marker=dict(color=[COLOR_TIPO.get(t, BRAND) for t in orden["TIPO_ATAQUE"]],
                        opacity=[1 if a else .35 for a in abierto], cornerradius=4),
            text=[f"{v:.2f}" for v in orden["SEVERIDAD_SCORE"]], textposition="outside",
            textfont=dict(color=INK2, size=11), cliponaxis=False,
            customdata=orden["TIPO_ATAQUE"],
            hovertemplate="<b>%{y}</b><br>%{customdata}<br>Severidad %{x:.2f}<extra></extra>"))
        fig.update_xaxes(range=[0, 1.08], title_text=None)
        fig.update_yaxes(title_text=None)
        mostrar(estilo(fig, height=max(200, 44 * len(pm_f) + 50), legend=False,
                       margin=dict(l=8, r=12, t=8, b=28)))

    pm_id = str(fila.get("POSTMORTEM_ID") or "PM-N/A")
    sev = float(fila.get("SEVERIDAD_SCORE") or 0.0)
    sev_txt, sev_cls, _ = _etiqueta_severidad(sev)
    origen = STREAMS.get(str(fila.get("STREAM")).lower(), str(fila.get("STREAM")))
    meta = _parse_json_safe(fila.get("METADATA_JSON"))
    inc = meta.get("incident_meta") or {}
    apps_l = _parse_lista_campo(fila.get("APLICACIONES_AFECTADAS"))
    ips_l = _parse_lista_campo(fila.get("IPS_INVOLUCRADAS"))
    fecha, horas = _rango(fila.get("INICIO"), fila.get("FIN"))
    md = str(fila.get("REPORTE_MARKDOWN") or "Sin reporte registrado.")
    esc = html.escape

    # Expediente del incidente seleccionado
    with tarjeta("Expediente Forense del Incidente",
                 "Análisis que generó el agente forense con los registros de SAP HANA."):
        h1, h2 = st.columns([3, 1], vertical_alignment="center", gap="medium")
        with h1:
            st.html(
                f'<div class="pm-head"><span class="id">{esc(pm_id)}</span>'
                f'<span class="pm-pill {sev_cls}">Severidad {sev_txt.lower()} · {sev:.2f}</span>'
                f'<span class="pm-pill brand">{esc(origen)}</span>'
                f'<span class="pm-pill">Evento {int(fila.get("EVENTO_ID") or 0)}</span></div>'
                f'<p class="pm-sub"><b>{esc(str(fila.get("TIPO_ATAQUE") or "N/A"))}</b> · '
                f'{esc(str(fila.get("MITRE_TACTIC") or "Táctica pendiente"))} · '
                f'{esc(str(fila.get("SOURCE_TABLE") or "N/A"))} · '
                f'generado el {esc(_fmt_ts(fila.get("FECHA_GENERACION"))[:16])}</p>')
        with h2, st.container(key="pm_descargas", gap="small"):
            try:
                st.download_button("Descargar PDF", data=_construir_pdf_bytes(fila),
                                   file_name=f"{pm_id}.pdf", mime="application/pdf",
                                   icon=":material/download:", width="stretch",
                                   help="Descarga el reporte ejecutivo de este expediente en PDF.")
            except Exception as err:
                st.warning(f"No se pudo generar el PDF: {err}")
            st.download_button("Descargar Markdown", data=md.encode("utf-8"),
                               file_name=f"{pm_id}.md", mime="text/markdown",
                               icon=":material/download:", width="stretch",
                               help="Descarga el reporte completo de este expediente en Markdown.")

        z_feat = str(inc.get("z_max_feature") or "N/A")
        st.html(
            '<div class="pm-meta">'
            f'<div><p class="lbl">Cuándo</p><div class="val">{esc(fecha)}</div>'
            f'<div class="det">{esc(horas)}</div></div>'
            f'<div><p class="lbl">Indicador más afectado</p><div class="val">{esc(nombre(z_feat))}</div>'
            f'<div class="det">Intensidad {float(inc.get("z_max") or 0):.1f}</div></div>'
            f'<div><p class="lbl">Bloques atípicos</p><div class="val">{int(inc.get("n_ventanas") or 1)}</div>'
            f'<div class="det">Confianza {esc(str(inc.get("confianza") or "N/A"))}</div></div>'
            f'<div><p class="lbl">Alcance</p><div class="val">{len(apps_l)} aplicación(es)</div>'
            f'<div class="det">{len(ips_l)} IP(s) involucradas</div></div>'
            '</div>')
        st.html(f'<div class="pm-callout"><b>Resumen ejecutivo</b>'
                f'{esc(str(fila.get("RESUMEN_EJECUTIVO") or "Sin resumen registrado."))}</div>')

        t1, t2 = st.tabs(["Reporte completo", "Evidencia"])
        with t1, st.container(key="pm_reporte"):
            st.markdown(_sin_titulo(md))
        with t2:
            st.html(
                '<div class="pm-meta dos">'
                f'<div><p class="lbl">Aplicaciones SAP afectadas</p>'
                f'{_chips(apps_l, "Sin aplicaciones registradas.")}</div>'
                f'<div><p class="lbl">Direcciones IP involucradas</p>'
                f'{_chips(ips_l, "Sin IPs externas asociadas.")}</div>'
                '</div>')
            logs = (meta.get("evidence_logs_sample")
                    or (meta.get("raw_evidence") or {}).get("logs_muestra") or [])
            st.html('<p class="card-title">Muestra de registros</p>'
                    '<p class="card-sub">Registros originales que el agente consultó en SAP HANA.</p>')
            if logs and isinstance(logs, list):
                st.dataframe(pd.DataFrame(logs), hide_index=True)
            else:
                vacio("Este reporte no incluye una muestra de registros.")
            with st.expander("Ver metadatos completos (JSON)"):
                st.json(meta or {"nota": "Sin metadatos."})


# Navegación
pg_resumen = st.Page(pagina_resumen, title="Resumen", icon=":material/dashboard:",
                     url_path="resumen", default=True)
pg_eventos = st.Page(pagina_eventos, title="Eventos", icon=":material/timeline:", url_path="eventos")
pg_tipos = st.Page(pagina_tipos, title="Tipos de ataque", icon=":material/category:", url_path="tipos")
pg_postmortems = st.Page(pagina_postmortems, title="Postmortems SOC", icon=":material/security:",
                         url_path="postmortems")
pg_modelo = st.Page(pagina_modelo, title="Metodología", icon=":material/model_training:",
                    url_path="metodologia")

pg = st.navigation([pg_resumen, pg_eventos, pg_tipos, pg_postmortems, pg_modelo], position="top")
pg.run()
