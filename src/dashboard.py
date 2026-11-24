import re
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import hana_ml.dataframe as hana_df

from helpers.conexion import _params
from helpers.explicacion import FAMILIAS_SYSTEM, FAMILIAS_LLM
from helpers.clasificacion import FIRMAS

st.set_page_config(page_title="Anomalías SAP", page_icon=":material/shield:", layout="wide")

DARK = getattr(st.context, "theme", None) is not None and st.context.theme.type == "dark"

if DARK:
    SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]
    INK, INK2, MUTED, GRID, AXIS = "#e8e9ec", "#c3c2b7", "#8b8e96", "#262a31", "#353a42"
    NEUTRO, SURFACE, CARD_BORDER = "#5a5d63", "#16191e", "#2a2e35"
    DIV_MID = "#2c3038"
else:
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
    INK, INK2, MUTED, GRID, AXIS = "#1b1d21", "#4a4d55", "#8b8e96", "#eceef1", "#d5d8dd"
    NEUTRO, SURFACE, CARD_BORDER = "#c9ccd2", "#ffffff", "#e2e4e8"
    DIV_MID = "#f0f1f3"
CRITICO = "#d03b3b"
AZUL_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CONFIANZA_COLOR = {"alta": "#184f95", "media": "#3987e5", "baja": "#86b6ef"}

COLOR_TIPO = {
    # system
    "Contexto SAP": SERIES[0], "DDoS": SERIES[1], "DoS": SERIES[2], "Fuerza bruta": SERIES[3],
    "Explotación": SERIES[4], "Reconocimiento": SERIES[5], "Falla operativa": NEUTRO,
    # llm
    "Prompt injection": SERIES[0], "Degradación del servicio": SERIES[1],
    "Abuso de recursos": SERIES[2], "Acceso no autorizado": SERIES[3], "Automatización": SERIES[4],
}
SIMBOLO_TIPO = {t: s for t, s in zip(COLOR_TIPO, ["circle", "square", "diamond", "triangle-up",
                                                  "cross", "x", "circle-open", "circle", "square",
                                                  "diamond", "triangle-up", "cross"])}

STREAMS = {"system": "Sistema SAP", "llm": "Servicios de IA"}
VOLUMEN = {"system": "TOTAL_REQUESTS", "llm": "TOTAL_LLM_REQUESTS"}
FAMILIAS = {"system": FAMILIAS_SYSTEM, "llm": FAMILIAS_LLM}
FIRMA = {"system": FIRMAS["F_SYSTEM"], "llm": FIRMAS["F_LLM"]}
SIGNO = {"+": "Aumenta", "-": "Disminuye", "~": "Cambia"}
PLOT_CFG = {"displayModeBar": False}

# Qué mide cada variable de las ventanas
DESCRIPCION = {
    # system
    "TOTAL_REQUESTS": "Número de peticiones en el bloque de tiempo",
    "REQUEST_RATE_STD": "Qué tan irregular es el ritmo de las peticiones",
    "INTERVAL_CV": "Qué tan variable es el tiempo entre una petición y otra",
    "UNIQUE_IPS_COUNT": "Número de IPs distintas",
    "IP_ENTROPY": "Qué tan repartida está la actividad entre distintas IPs",
    "IP_TOP_SHARE": "Porcentaje de la actividad que viene de la IP más activa",
    "ERROR_RATE_400": "Porcentaje de respuestas con error del cliente (4xx)",
    "ERROR_RATE_500": "Porcentaje de respuestas con error del servidor (5xx)",
    "NON_200_RATE": "Porcentaje de respuestas que no fueron exitosas",
    "RATIO_ERROR": "Porcentaje de registros de error",
    "RATIO_WARN": "Porcentaje de registros de advertencia",
    "RATIO_SECURITY": "Porcentaje de registros de seguridad",
    "RATIO_INFO": "Porcentaje de registros informativos",
    "RATIO_AUDIT": "Porcentaje de registros de auditoría",
    "GET_VS_POST_RATIO": "Proporción de consultas (GET) frente a envíos de datos (POST)",
    "JS_HTTP_STATUS_CODE": "Qué tanto cambió la proporción de códigos de respuesta",
    "JS_HEADERS_HTTP_REQUEST_METHOD": "Qué tanto cambió la proporción de tipos de petición (GET, POST…)",
    "JS_SAP_FUNCTION_LOG_TYPE": "Qué tanto cambió la proporción de tipos de registro (informativo, error, auditoría…)",
    "JS_HEATHERS_REQUEST_PATH": "Qué tanto cambió la proporción de rutas solicitadas",
    "JS_SAP_FUNCTION_APPLICATION": "Qué tanto cambió la proporción de uso de cada aplicación SAP",
    "JS_SAP_SOURCE_TYPE": "Qué tanto cambió la proporción de canales de conexión (REST, RFC, OData…)",
    "JS_SAP_APP_ENV": "Qué tanto cambió la proporción de entornos (desarrollo, pruebas, producción…)",
    "JS_REGION_CODE": "Qué tanto cambió la proporción de actividad por región",
    "JS_HEADERS_CONTENT_TYPE": "Qué tanto cambió la proporción de formatos de contenido",
    # llm
    "TOTAL_LLM_REQUESTS": "Número de llamadas al LLM",
    "AVG_LLM_COST_USD": "Costo promedio por llamada (USD)",
    "INJECTION_COUNT": "Intentos de manipular al modelo con instrucciones maliciosas",
    "SAFETY_POLICY_COUNT": "Bloqueos por política de seguridad o contenido",
    "AUTH_FAILURE_COUNT": "Fallas de autenticación (API key, token, firma)",
    "ACCESS_DENIED_COUNT": "Accesos denegados (IP, región, permisos)",
    "RATE_ABUSE_COUNT": "Límites de tasa, cuota o presupuesto alcanzados",
    "DUPLICATE_REQUEST_COUNT": "Peticiones duplicadas",
    "MALFORMED_INPUT_COUNT": "Peticiones mal formadas o demasiado grandes",
    "AVG_TOKEN_RATIO": "Largo de la respuesta en relación con el de la consulta",
    "AVG_RESPONSE_TIME_MS": "Tiempo de respuesta promedio (ms)",
    "LLM_ERROR_RATE": "Porcentaje de llamadas con error",
    "LLM_TIMEOUT_RATE": "Porcentaje de llamadas que expiran",
    "TEMPERATURE_VARIANCE": "Qué tanto varía la configuración de creatividad del modelo",
    "PROMPT_CATEGORY_ENTROPY": "Qué tan variados son los temas de las consultas",
    "STREAM_RATIO": "Porcentaje de respuestas enviadas en tiempo real (streaming)",
    "AVG_PROMPT_LENGTH": "Largo promedio de las consultas",
    "MAX_PROMPT_LENGTH": "Largo de la consulta más extensa",
}

# Columnas categóricas de los logs crudos con las que se compara el evento contra su semana.
DIMENSIONES = {
    "system": {"SAP_FUNCTION_APPLICATION": "Aplicación SAP", "SAP_SOURCE_TYPE": "Canal de conexión",
               "SAP_APP_ENV": "Entorno", "SAP_FUNCTION_LOG_TYPE": "Tipo de registro",
               "REGION_CODE": "Región", "HTTP_STATUS_CODE": "Código de respuesta",
               "HEADERS_HTTP_REQUEST_METHOD": "Tipo de petición", "HEATHERS_REQUEST_PATH": "Ruta solicitada",
               "HEADERS_CONTENT_TYPE": "Formato de contenido"},
    "llm": {"LLM_MODEL_ID": "Modelo", "LLM_PROVIDER": "Proveedor",
            "LLM_PROMPT_CATEGORY": "Tema de la consulta", "LLM_STATUS": "Estado",
            "LLM_FINISH_REASON": "Motivo de finalización", "SAP_FUNCTION_APPLICATION": "Aplicación SAP",
            "REGION_CODE": "Región"},
}
INDICE = {"system": "sap-logs%", "llm": "llm-logs%"}

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


def nombre(f):
    return NOMBRE.get(f, f)


def texto_cambios(top3):
    """'IP_TOP_SHARE (+18.9), RATIO_AUDIT (-2.5)' -> 'Peso de la IP más activa ↑, Registros de auditoría ↓'."""
    if not isinstance(top3, str):
        return ""
    return ", ".join(f"{nombre(f)} {'↑' if float(z) > 0 else '↓'}"
                     for f, z in re.findall(r"([A-Z0-9_]+) \(([+-]?[\d.]+)\)", top3))


def otros_tipos(votos):
    """'Contexto SAP 1.00, Fuerza bruta 0.75, DoS 0.50' -> ['Fuerza bruta', 'DoS']."""
    if not isinstance(votos, str):
        return []
    return [t for t, _ in re.findall(r"\s*(.+?) ([\d.]+)(?:,|$)", votos)][1:3]

# Estilos: tarjetas blancas sobre fondo gris, títulos y encabezado de página
st.html(f"""
<style>
  [data-testid="stMainBlockContainer"] {{ padding-top: 4.2rem; padding-bottom: 3rem;
                                          max-width: 1480px; }}
  div[class*="st-key-card_"] {{
      background: {SURFACE}; border: 1px solid {CARD_BORDER} !important;
      border-radius: 14px; padding: 1.1rem 1.25rem 0.9rem;
      box-shadow: 0 1px 2px rgba(16,24,40,{0.25 if DARK else 0.04});
  }}
  .card-title {{ font-size: 1.02rem; font-weight: 650; color: {INK}; margin: 0; }}
  .card-sub {{ font-size: 0.84rem; color: {MUTED}; margin: 0.15rem 0 0.2rem; line-height: 1.4; }}
  .page-eyebrow {{ font-size: 0.78rem; font-weight: 600; letter-spacing: .08em;
                   text-transform: uppercase; color: {SERIES[0]}; margin-bottom: .15rem; }}
  .page-title {{ font-size: 1.85rem; font-weight: 700; color: {INK}; margin: 0;
                 line-height: 1.2; }}
  .page-sub {{ font-size: 0.95rem; color: {MUTED}; margin: .35rem 0 0; max-width: 820px; }}
  [data-testid="stMetric"] {{ background: {SURFACE}; }}
  [data-testid="stMetricLabel"] p {{ font-size: 0.82rem; color: {MUTED}; font-weight: 500; }}
  [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{
      white-space: normal !important; overflow: visible !important;
      text-overflow: clip !important; }}
  [data-testid="stMetricLabel"] {{ min-height: 2.5em; align-items: flex-start; }}
  [data-testid="stMetricValue"] {{ font-weight: 650; }}
  .side-brand {{ font-size: 1.15rem; font-weight: 700; color: {INK}; }}
  .side-sub {{ font-size: 0.8rem; color: {MUTED}; margin-bottom: .4rem; }}
  .paso {{ border-left: 3px solid {SERIES[0]}; padding: .1rem 0 .1rem .75rem; }}
  .paso b {{ color: {INK}; font-size: .92rem; }}
  .paso p {{ color: {MUTED}; font-size: .82rem; margin: .15rem 0 0; line-height: 1.4; }}
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


def encabezado(eyebrow, titulo, sub):
    st.html(f'<div class="page-eyebrow">{eyebrow}</div><p class="page-title">{titulo}</p>'
            f'<p class="page-sub">{sub}</p>')
    with st.container(horizontal=True, gap="small"):
        st.badge(MES.replace("_", "-"), icon=":material/calendar_month:", color="blue")
        st.badge(FUENTE, icon=":material/database:", color="gray")
        st.badge(STREAMS[STREAM], icon=":material/hub:", color="gray")
        st.badge(f"Bloques de {ETIQUETA[SIZE_S]}", icon=":material/schedule:", color="gray")
    st.write("")


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


def z_ventanas(win, feats):
    feats = [f for f in feats if f in win.columns]
    g = win.groupby("WEEK")[feats]
    mu = g.transform("mean")
    sd = g.transform(lambda s: s.std(ddof=0)).replace(0, np.nan)
    z = ((win[feats] - mu) / sd).fillna(0)
    m = (win["LABEL"] == -1) & (win["EVENTO_RELEVANTE"] == 1)
    return z[m].assign(EVENTO_ID=win.loc[m, "EVENTO_ID"].astype(int))


def perfil_familias(win, familias):
    z = z_ventanas(win, [f for fs in familias.values() for f in fs])
    if z.empty:
        return pd.DataFrame()
    por_ev = z.set_index("EVENTO_ID").abs().groupby("EVENTO_ID").mean()
    return pd.DataFrame({fam: por_ev[[f for f in fs if f in por_ev]].mean(axis=1)
                         for fam, fs in familias.items()})


def _iso(t):
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


@st.cache_data(ttl=3600, show_spinner="Analizando la actividad del evento…")
def mezcla_evento(tabla, indice, dims, ini, fin, sem_ini, sem_fin):
    """
    Conteo de logs por categoría de cada dimensión, durante el evento [ini, fin) y en el resto
    de su semana [sem_ini, sem_fin). Compara el texto ISO de TIMESTAMP (mismo formato en toda
    la tabla), así que no hace falta convertirlo.
    """
    base = (f'SELECT CASE WHEN "TIMESTAMP" >= \'{_iso(ini)}\' AND "TIMESTAMP" < \'{_iso(fin)}\' '
            f'THEN 1 ELSE 0 END AS EN_EVENTO, * FROM "{tabla}" '
            f'WHERE "_INDEX" LIKE \'{indice}\' AND "TIMESTAMP" >= \'{_iso(sem_ini)}\' '
            f'AND "TIMESTAMP" < \'{_iso(sem_fin)}\'')
    q = " UNION ALL ".join(
        f'SELECT \'{d}\' AS DIM, EN_EVENTO, TO_NVARCHAR("{d}") AS CAT, COUNT(*) AS N '
        f'FROM ({base}) GROUP BY EN_EVENTO, "{d}"' for d in dims)
    d = _conn().sql(q).collect()
    d["CAT"] = d["CAT"].fillna("(vacío)")
    d["PCT"] = d["N"] / d.groupby(["DIM", "EN_EVENTO"])["N"].transform("sum")
    p = d.pivot_table(index=["DIM", "CAT"], columns="EN_EVENTO", values="PCT", fill_value=0)
    p = p.reindex(columns=[0, 1], fill_value=0).rename(columns={0: "SEMANA", 1: "EVENTO"})
    p["DIF"] = p["EVENTO"] - p["SEMANA"]
    return p.reset_index()


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
    st.html('<div class="side-brand">Anomalías SAP</div>'
            '<div class="side-sub">Monitoreo de comportamiento inusual en los registros de SAP</div>')
    MES = st.selectbox("Periodo", meses, format_func=lambda m: m.replace("_", "-"), key="mes")
    eventos_all = leer("EVENTS", MES)
    resumen_all = leer("SUMMARY", MES)
    valid_all = leer("VALIDATION", MES)
    FUENTE = st.selectbox("Conjunto de datos", sorted(eventos_all["SOURCE_TABLE"].unique()), key="fuente")
    STREAM = st.segmented_control("Origen de los registros", list(STREAMS), default="system",
                                  format_func=STREAMS.get, key="stream",
                                  width="stretch") or "system"
    tam = (resumen_all[["WINDOW_SIZE_S", "WINDOW_LABEL"]].drop_duplicates()
           .sort_values("WINDOW_SIZE_S"))
    ETIQUETA = dict(zip(tam["WINDOW_SIZE_S"], tam["WINDOW_LABEL"]))
    SIZE_S = st.select_slider("Intervalo de análisis", list(ETIQUETA), value=list(ETIQUETA)[0],
                              format_func=ETIQUETA.get, key="size")
    st.divider()

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
    # ancho mínimo de una franja para que el evento se vea (≈ 0.6 % del periodo)
    span = (WIN["WINDOW_START"].max() - WIN["WINDOW_START"].min()) if len(WIN) else pd.Timedelta(0)
    piso = max(pd.Timedelta(minutes=40), span / 160)
    for e in REL.sort_values("INICIO").itertuples():
        ini, fin = e.INICIO, e.FIN + pd.Timedelta(seconds=int(SIZE_S))
        if fin - ini < piso:
            mid = ini + (fin - ini) / 2
            ini, fin = mid - piso / 2, mid + piso / 2
        fig.add_vrect(x0=ini, x1=fin, fillcolor=COLOR_TIPO.get(e.TIPO_ATAQUE, NEUTRO),
                      opacity=0.35, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=WIN["WINDOW_START"], y=WIN[col], mode="lines", name="Registros por bloque",
        line=dict(color=SERIES[0], width=1.3),
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


def fig_intensidad(height=380):
    fig = go.Figure()
    nr = EV[EV["RELEVANTE"] != 1]
    fig.add_trace(go.Scatter(
        x=nr["INICIO"], y=nr["Z_MAX"], mode="markers", name="No relevante",
        marker=dict(color=NEUTRO, size=8 + 2 * np.sqrt(nr["N_VENTANAS"]),
                    line=dict(color=SURFACE, width=2)),
        customdata=np.stack([nr["EVENTO_ID"], nr["N_VENTANAS"], nr["INDICADOR"]], -1)
        if len(nr) else None,
        hovertemplate="Evento %{customdata[0]} · %{x|%d-%b %H:%M}<br>Intensidad %{y:.1f}<br>"
                      "Indicador más afectado: %{customdata[2]}<extra></extra>"))
    for t, g in REL.groupby("TIPO_ATAQUE"):
        fig.add_trace(go.Scatter(
            x=g["INICIO"], y=g["Z_MAX"], mode="markers", name=t,
            marker=dict(color=COLOR_TIPO.get(t, NEUTRO), symbol=SIMBOLO_TIPO.get(t, "circle"),
                        size=8 + 2 * np.sqrt(g["N_VENTANAS"]),
                        line=dict(color=SURFACE, width=2)),
            customdata=np.stack([g["EVENTO_ID"], g["N_VENTANAS"], g["INDICADOR"],
                                 g["CONFIANZA"]], -1),
            hovertemplate="Evento %{customdata[0]} · " + t + "<br>%{x|%d-%b %H:%M}<br>"
                          "Intensidad %{y:.1f} · confianza %{customdata[3]}<br>"
                          "Indicador más afectado: %{customdata[2]}<extra></extra>"))
    fig.add_hline(y=Z_UMBRAL, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text=f"Mínimo para ser relevante ({Z_UMBRAL:g})",
                  annotation_font_color=MUTED, annotation_position="top left")
    fig.update_yaxes(title_text="Intensidad", title_font_color=MUTED, rangemode="tozero")
    return estilo(fig, height=height)


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

    n_win = int(RES["VENTANAS"].sum())
    n_anom = int(RES["VENTANAS_ANOMALAS"].sum())
    tipos_ataque = sorted(t for t in REL["TIPO_ATAQUE"].dropna().unique()
                          if t != "Falla operativa")
    k = st.columns([1, 1, 1, 1, 1.5], gap="small")
    k[0].metric("Bloques analizados", f"{n_win:,}", f"{len(RES)} semana(s)", delta_color="off",
                delta_arrow="off", border=True,
                help=f"La actividad se divide en bloques de {ETIQUETA[SIZE_S]} para analizarla.")
    k[1].metric("Bloques atípicos", f"{n_anom:,}",
                f"{n_anom / n_win:.1%} del total" if n_win else "—",
                delta_color="off", delta_arrow="off", border=True,
                help="Bloques cuyo comportamiento se aleja claramente del habitual.")
    k[2].metric("Eventos", f"{len(EV):,}", "detectados", delta_color="off",
                delta_arrow="off", border=True,
                help="Los bloques atípicos cercanos en el tiempo se agrupan en un solo evento.")
    k[3].metric("Eventos relevantes", f"{len(REL):,}", "requieren atención",
                delta_color="off", delta_arrow="off", border=True,
                help=f"Eventos con una intensidad de {Z_UMBRAL:g} o más.")
    k[4].metric("Tipos de ataque", len(tipos_ataque), ", ".join(tipos_ataque) or "ninguno",
                delta_color="off", delta_arrow="off", border=True)

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


def pagina_eventos():
    encabezado("Eventos", "Eventos detectados",
               "Un evento reúne bloques atípicos cercanos en el tiempo. Aquí se muestra su "
               "intensidad, qué aspectos de la actividad cambiaron y el detalle de cada uno.")
    if EV.empty:
        with tarjeta():
            vacio("No se detectaron eventos con los filtros seleccionados.")
        return

    with tarjeta("Intensidad de los eventos",
                 "La intensidad indica cuánto se alejó del comportamiento habitual el indicador más "
                 "afectado. Los eventos por encima de la línea punteada se consideran relevantes. "
                 "El tamaño del punto refleja la duración del evento."):
        mostrar(fig_intensidad(height=360))
    with tarjeta("Aspectos afectados en cada evento",
                 "Qué tanto cambió cada grupo de indicadores durante el evento, comparado con una "
                 "semana normal. Los tonos más oscuros indican un cambio mayor."):
        perfil = perfil_familias(WIN, FAMILIAS[STREAM])
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


def pagina_composicion():
    encabezado("Composición por tipo", "¿Qué caracteriza a cada tipo?",
               "Cada tipo de evento se reconoce por un patrón: un conjunto de indicadores que "
               "cambian de una forma particular. Selecciona un tipo y un evento para ver ese "
               "patrón y qué ocurrió en la actividad real.")
    tipos_rel = REL["TIPO_ATAQUE"].dropna().value_counts()
    if tipos_rel.empty:
        with tarjeta():
            vacio("No se detectaron eventos relevantes con los filtros seleccionados.")
        return

    opciones = list(tipos_rel.index)
    ev_tipo_de = lambda t: REL[REL["TIPO_ATAQUE"] == t].sort_values("Z_MAX", ascending=False)
    with tarjeta():
        c_t, c_e = st.columns([1, 2], gap="medium")
        tipo_sel = c_t.selectbox(
            "Tipo de evento", opciones,
            index=opciones.index("Contexto SAP") if "Contexto SAP" in opciones else 0,
            format_func=lambda t: f"{t} ({tipos_rel[t]} evento{'s' if tipos_rel[t] != 1 else ''})")
        ev_tipo = ev_tipo_de(tipo_sel)
        idx = ev_tipo.set_index("EVENTO_ID")
        ev_id = c_e.selectbox(
            "Evento", ev_tipo["EVENTO_ID"].astype(int).tolist(),
            format_func=lambda i: f"Evento {i} · {idx.loc[i, 'INICIO']:%d-%b %H:%M} · "
                                  f"intensidad {idx.loc[i, 'Z_MAX']:.1f} · "
                                  f"confianza {idx.loc[i, 'CONFIANZA']}")
    firma = {f: sg for f, tipos in FIRMA[STREAM].items() for t, sg in tipos.items()
             if t == tipo_sel}

    c5, c6 = st.columns([1.25, 1], gap="medium")
    with c5, tarjeta(f"Patrón que identifica «{tipo_sel}»",
                     "Indicadores que se revisan para asignar este tipo y el comportamiento "
                     "esperado en cada uno. «Cambia» significa que puede subir o bajar."):
        st.dataframe(
            pd.DataFrame([{"Indicador": nombre(f), "Comportamiento esperado": SIGNO[sg],
                           "Qué mide": DESCRIPCION.get(f, "")} for f, sg in firma.items()]),
            hide_index=True,
            column_config={"Comportamiento esperado": st.column_config.TextColumn(
                               "Esperado", width="small"),
                           "Qué mide": st.column_config.TextColumn(width="large")})
    with c6, tarjeta(f"Comportamiento del patrón en cada evento",
                     "Rojo: el indicador aumentó respecto a lo habitual. Azul: disminuyó. "
                     "El número indica la magnitud del cambio."):
        zf = z_ventanas(WIN, list(firma))
        zf = zf[zf["EVENTO_ID"].isin(ev_tipo["EVENTO_ID"].astype(int))]
        if zf.empty:
            vacio("No hay datos suficientes para este tipo.")
        else:
            zm = zf.groupby("EVENTO_ID")[[f for f in firma if f in zf.columns]].mean()
            zm = zm.loc[[i for i in ev_tipo["EVENTO_ID"].astype(int) if i in zm.index]]
            lim = max(1.0, float(np.abs(zm.values).max()))
            fig = go.Figure(go.Heatmap(
                z=zm.values, x=[nombre(f) for f in zm.columns],
                y=[f"Evento {i}" for i in zm.index], zmid=0, zmin=-lim, zmax=lim,
                colorscale=[[0, "#184f95"], [0.5, DIV_MID], [1, "#b42f2f"]],
                xgap=2, ygap=2, text=np.round(zm.values, 1), texttemplate="%{text}",
                hovertemplate="%{y}<br>%{x}: cambio de %{z:+.1f}<extra></extra>",
                colorbar=dict(title=dict(text="Cambio", font_color=MUTED), tickfont_color=MUTED,
                              thickness=10, outlinewidth=0)))
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_xaxes(side="top", showgrid=False, tickangle=-30)
            mostrar(estilo(fig, height=max(220, 36 * len(zm) + 150), legend=False,
                           margin=dict(l=8, r=8, t=110, b=8)))

    # --- Qué cambió en el tráfico durante el evento elegido
    e = idx.loc[ev_id]
    fin_ev = e.FIN + pd.Timedelta(seconds=int(SIZE_S))
    otros = otros_tipos(e.VOTOS)
    porque = (f"Se clasificó como «{tipo_sel}» porque los indicadores que más cambiaron fueron: "
              f"{e.CAMBIOS}.")
    if e.CONFIANZA != "alta" and otros:
        porque += (f" La confianza es {e.CONFIANZA} porque estos cambios también se parecen a "
                   f"{' y '.join(otros)}.")
    with tarjeta(f"Qué cambió en la actividad durante el evento {ev_id}",
                 f"Del {e.INICIO:%d-%b %H:%M} al {fin_ev:%d-%b %H:%M}, comparado con el resto de "
                 f"esa semana. {porque}"):
        sem = RES_S[RES_S["WEEK"] == e.WEEK].iloc[0]
        try:
            mz = mezcla_evento(FUENTE, INDICE[STREAM], tuple(DIMENSIONES[STREAM]), e.INICIO,
                               fin_ev, sem.WEEK_START, sem.WEEK_END + pd.Timedelta(seconds=1))
        except Exception as err:
            vacio(f"No fue posible consultar los registros originales de {FUENTE}. ({err})")
            return
        if mz.empty:
            vacio("No hay registros para este periodo.")
            return
        nombres = DIMENSIONES[STREAM]
        cambio = (mz["DIF"].abs().groupby(mz["DIM"]).sum() / 2).sort_values(ascending=False)

        # Los 3 cambios más grandes (categorías que ganaron peso), como tarjetas
        top = mz[mz["DIF"] > 0].sort_values("DIF", ascending=False).head(3)
        for col, r in zip(st.columns(3, gap="small"), top.itertuples()):
            with col:
                st.metric(f"{nombres[r.DIM]}: {r.CAT}", f"{r.EVENTO:.1%}",
                          f"{r.DIF * 100:+.1f} puntos", delta_color="off",
                          delta_description=f"frente a {r.SEMANA:.1%} habitual",
                          border=True, help="Porcentaje de la actividad durante el evento.")

        dims_top = list(cambio.index[:4])
        fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.2, vertical_spacing=0.2,
                            subplot_titles=[nombres[d] for d in dims_top])
        for k, d in enumerate(dims_top):
            r, c = k // 2 + 1, k % 2 + 1
            g = mz[mz["DIM"] == d].assign(A=lambda x: x["DIF"].abs()).nlargest(5, "A") \
                .sort_values("DIF")
            for colname, serie, color in (("SEMANA", "Semana habitual", NEUTRO),
                                          ("EVENTO", "Durante el evento", SERIES[0])):
                fig.add_trace(go.Bar(
                    y=g["CAT"], x=g[colname], orientation="h", name=serie, showlegend=k == 0,
                    legendgroup=serie,
                    marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                    hovertemplate=f"{nombres[d]}: %{{y}}<br>{serie}: %{{x:.1%}}<extra></extra>"),
                    row=r, col=c)
            fig.update_xaxes(tickformat=".0%", row=r, col=c)
            fig.update_yaxes(type="category", row=r, col=c)   # códigos HTTP son texto
        fig.update_layout(barmode="group", bargroupgap=0.06)
        fig.update_annotations(font=dict(size=13, color=INK2))
        estilo(fig, height=600, margin=dict(l=56, r=12, t=80, b=36))
        fig.update_layout(legend=dict(y=1.12))
        mostrar(fig)
        st.caption("Porcentaje de la actividad en cada categoría, durante el evento y en el resto "
                   "de la semana. Se muestran los cuatro aspectos que más cambiaron y, en cada uno, "
                   "las cinco categorías con mayor diferencia.")


def pagina_modelo():
    encabezado("Metodología", "Cómo se detectan y clasifican los eventos",
               "Los pasos del análisis y dos comprobaciones de calidad: cómo cambian los resultados "
               "según el intervalo de análisis y qué tan consistente es la clasificación.")
    with tarjeta("Proceso de análisis"):
        pasos = [
            ("1 · División en bloques", "La actividad se divide en bloques de tiempo, de 5 minutos "
                                        "a 3 horas, y en cada uno se miden indicadores como "
                                        "volumen, errores u origen de la actividad."),
            ("2 · Detección de lo atípico", "Un modelo de aprendizaje automático (Isolation "
                                            "Forest) identifica cada semana el 1 % de bloques "
                                            "con el comportamiento más inusual."),
            ("3 · Agrupación en eventos", "Los bloques atípicos cercanos se unen en un evento. "
                                          "Es relevante si algún indicador se alejó "
                                          f"{Z_UMBRAL:g} veces o más de lo habitual."),
            ("4 · Clasificación", "Los indicadores que más cambiaron se comparan con el patrón de "
                                  "cada tipo de ataque y se asigna el que mejor coincide."),
            ("5 · Validación", "Un segundo método agrupa los eventos de forma independiente y se "
                               "compara con la clasificación para medir su consistencia."),
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
                                       ("EVENTOS_RELEVANTES", "Eventos relevantes", SERIES[1])):
            fig.add_trace(go.Bar(
                x=s["WINDOW_LABEL"], y=s[colname], name=nombre,
                marker=dict(color=color, line=dict(color=SURFACE, width=2), cornerradius=4),
                hovertemplate="Intervalo de %{x}<br>" + nombre + ": %{y}<extra></extra>"))
        fig.update_layout(barmode="group", bargroupgap=0.08)
        fig.update_yaxes(title_text="Número de eventos", title_font_color=MUTED)
        fig.update_xaxes(title_text="Intervalo de análisis", title_font_color=MUTED)
        mostrar(estilo(fig, height=360))

    with c4, tarjeta("Consistencia de la clasificación",
                     "Compara la clasificación con un agrupamiento independiente de los mismos "
                     "eventos mediante dos índices (ARI y NMI). Un valor de 1 indica coincidencia "
                     "total y 0, coincidencia al azar."):
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
                                   ("NMI", SERIES[2], "Información mutua normalizada")):
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

# Navegación
pg = st.navigation([
    st.Page(pagina_resumen, title="Resumen", icon=":material/dashboard:",
            url_path="resumen", default=True),
    st.Page(pagina_eventos, title="Eventos", icon=":material/timeline:", url_path="eventos"),
    st.Page(pagina_composicion, title="Composición por tipo", icon=":material/category:",
            url_path="composicion"),
    st.Page(pagina_modelo, title="Metodología", icon=":material/model_training:",
            url_path="metodologia"),
], position="top")
pg.run()
