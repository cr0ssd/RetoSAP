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

import hana_ml.dataframe as hana_df

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
  @media (max-width: 1000px) {{ .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
  .kpi {{ background: {SURFACE}; border: 1px solid {CARD_BORDER}; border-top: 4px solid var(--c);
      border-radius: 14px; padding: .85rem 1rem .9rem; box-shadow: {SOMBRA}; }}
  .kpi .lab {{ display: flex; align-items: center; gap: .45rem; font-size: .8rem; color: {MUTED};
      font-weight: 600; }}
  .kpi .lab i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--c); }}
  .kpi .val {{ font-size: 1.9rem; font-weight: 700; color: {INK}; line-height: 1.15;
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
    pills = [("Periodo", MES.replace("_", "-")), ("Datos", FUENTE), ("Origen", STREAMS[STREAM]),
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


def perfil_relevantes(win, rel, familias):
    """
    Igual que la celda 17 de sandbox/demo_pipeline_adrian.ipynb: z por semana con
    explicacion.zscores (ventanas anómalas contra todas las de su semana) y |z| promedio por
    familia con explicacion.perfil_familias, solo para los eventos relevantes.
    """
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

# Navegación
pg = st.navigation([
    st.Page(pagina_resumen, title="Resumen", icon=":material/dashboard:",
            url_path="resumen", default=True),
    st.Page(pagina_eventos, title="Eventos", icon=":material/timeline:", url_path="eventos"),
    st.Page(pagina_tipos, title="Tipos de ataque", icon=":material/category:", url_path="tipos"),
    st.Page(pagina_modelo, title="Metodología", icon=":material/model_training:",
            url_path="metodologia"),
], position="top")
pg.run()
