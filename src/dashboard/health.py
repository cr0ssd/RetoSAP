"""MLOps · Health checks del sistema (infraestructura, datos, modelo y agente forense)."""
import html
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datos_hana import _conn, leer, leer_postmortems, _parse_json_safe
from helpers.clasificacion import validar_firmas
from helpers.IsolationForest import CONTAMINATION as CONTAMINATION_IF
from tema import BRAND, CRITICO, INK2, MUTED, NEUTRO, SERIES, STREAMS
import workflow as wf   # configuración y pasos del pipeline de producción

ESTADOS_HC = {  # estado -> (texto, clase CSS, color)
    "ok": ("Correcto", "ok", "#188918"),
    "warn": ("Atención", "warn", "#E76500"),
    "fail": ("Falla", "fail", CRITICO),
    "info": ("Informativo", "info", BRAND),
}
TABLAS_RESULTADO = ("EVENTS", "SUMMARY", "VALIDATION", "WINDOWS_SYSTEM", "WINDOWS_LLM")


@st.cache_data(ttl=60, show_spinner=False)
def inventario_hana():
    """Tablas de logs y de resultados del esquema, con registros y fecha de creación."""
    return _conn().sql(
        "SELECT t.TABLE_NAME, m.RECORD_COUNT, t.CREATE_TIME FROM TABLES t "
        "JOIN M_TABLES m ON m.SCHEMA_NAME = t.SCHEMA_NAME AND m.TABLE_NAME = t.TABLE_NAME "
        "WHERE t.SCHEMA_NAME = CURRENT_SCHEMA "
        "AND (t.TABLE_NAME LIKE 'LOGS%' OR t.TABLE_NAME LIKE 'RESULT%') ORDER BY t.TABLE_NAME"
    ).collect()


@st.cache_data(ttl=300, show_spinner=False)
def version_hana():
    return str(_conn().sql("SELECT VERSION FROM M_DATABASE").collect().iloc[0, 0])


@st.cache_data(ttl=60, show_spinner=False)
def metricas_pipeline(meses_proc):
    """Por mes procesado: resumen, validación, conteos por stream y postmortems."""
    res, val, pm = [], [], []
    for m in meses_proc:
        res.append(leer("SUMMARY", m).assign(MES=m))
        val.append(leer("VALIDATION", m).assign(MES=m))
        p = leer_postmortems(m)
        if not p.empty:
            pm.append(p.assign(MES=m))
    return (pd.concat(res, ignore_index=True), pd.concat(val, ignore_index=True),
            pd.concat(pm, ignore_index=True) if pm else pd.DataFrame())


def _ping_hana():
    t = time.perf_counter()
    cur = _conn().connection.cursor()
    cur.execute("SELECT 1 FROM DUMMY")
    cur.close()
    return (time.perf_counter() - t) * 1000


def health_checks():
    """Lista de verificaciones: (grupo, nombre, estado, valor, detalle)."""
    hc = []
    # Infraestructura
    try:
        ms = _ping_hana()
        hc.append(("Infraestructura", "Conexión a SAP HANA Cloud",
                   "ok" if ms < 500 else "warn" if ms < 2000 else "fail", f"{ms:.0f} ms",
                   "Tiempo de respuesta de una consulta mínima (SELECT 1 FROM DUMMY)."))
    except Exception as e:
        hc.append(("Infraestructura", "Conexión a SAP HANA Cloud", "fail", "Sin conexión", str(e)[:160]))
        return hc
    hc.append(("Infraestructura", "Versión de SAP HANA", "info", version_hana(),
               "Motor donde corren las ventanas, Isolation Forest y K-means (PAL)."))
    gemini = bool(os.getenv("GEMINI_API_KEY"))
    hc.append(("Infraestructura", "Credencial de Gemini en este entorno", "ok" if gemini else "info",
               "Configurada" if gemini else "No configurada",
               "Este entorno puede generar reportes con Gemini." if gemini else
               "El dashboard no la necesita: el worker de Cloud Foundry usa su propia llave "
               "(ver Redacción con Gemini)."))

    inv = inventario_hana()
    logs = inv[inv["TABLE_NAME"].str.startswith("LOGS")]
    res_tab = inv[inv["TABLE_NAME"].str.startswith("RESULT_")]
    meses_proc = sorted({t.rsplit("_", 2)[-2] + "_" + t.rsplit("_", 2)[-1]
                         for t in res_tab["TABLE_NAME"] if t.startswith("RESULT_SUMMARY_")})
    resumen, valid, pms = metricas_pipeline(tuple(meses_proc))

    # Datos
    procesadas = set(resumen["SOURCE_TABLE"].unique())
    pendientes = sorted(set(logs["TABLE_NAME"]) - procesadas)
    hc.append(("Datos", "Cobertura del pipeline",
               "ok" if not pendientes else "warn",
               f"{len(procesadas)} de {len(logs)} tablas",
               "Todas las tablas de logs tienen resultados." if not pendientes else
               "Sin procesar: " + ", ".join(pendientes) + "."))
    faltan, descuadres = [], []
    conteo = dict(zip(inv["TABLE_NAME"], inv["RECORD_COUNT"]))
    for m in meses_proc:
        faltan += [f"RESULT_{t}_{m}" for t in TABLAS_RESULTADO if f"RESULT_{t}_{m}" not in conteo]
        r = resumen[resumen["MES"] == m]
        for stx, tabla in (("system", f"RESULT_WINDOWS_SYSTEM_{m}"), ("llm", f"RESULT_WINDOWS_LLM_{m}")):
            esperado = int(r.loc[r["STREAM"] == stx, "VENTANAS"].sum())
            if tabla in conteo and int(conteo[tabla]) != esperado:
                descuadres.append(f"{tabla}: {int(conteo[tabla]):,} vs {esperado:,}")
    hc.append(("Datos", "Integridad de resultados", "fail" if faltan or descuadres else "ok",
               f"{len(meses_proc)} periodo(s)",
               ("Faltan: " + ", ".join(faltan) + ". " if faltan else "")
               + ("Ventanas no cuadran: " + "; ".join(descuadres) + "." if descuadres else "")
               or "Las 5 tablas por periodo existen y las ventanas guardadas cuadran con el resumen."))
    ult = pd.to_datetime(res_tab.loc[~res_tab["TABLE_NAME"].str.startswith("RESULT_POSTMORTEMS"),
                                     "CREATE_TIME"]).max()
    dias = (pd.Timestamp.now() - ult).days if pd.notna(ult) else None
    hc.append(("Datos", "Frescura de resultados",
               "ok" if dias is not None and dias <= 7 else "warn",
               f"hace {dias} día(s)" if dias is not None else "N/A",
               f"Última carga del pipeline: {ult:%d/%m/%Y %H:%M}. Se espera un lote nuevo cada semana."
               if pd.notna(ult) else "No hay tablas de resultados."))

    # Modelo
    try:
        for stx in STREAMS:
            validar_firmas(wf.FEATURES[stx], firmas=wf.FIRMAS[stx])
        hc.append(("Modelo", "Firmas de clasificación", "ok", "24 + 18 variables",
                   "Cada variable del Isolation Forest tiene firma y no sobran firmas."))
    except Exception as e:
        hc.append(("Modelo", "Firmas de clasificación", "fail", "Inconsistentes", str(e)[:160]))
    tasa = resumen.groupby("STREAM")[["VENTANAS_ANOMALAS", "VENTANAS"]].sum()
    tasa = (tasa["VENTANAS_ANOMALAS"] / tasa["VENTANAS"]).to_dict()
    fuera = {k: v for k, v in tasa.items() if not 0.005 <= v <= 0.02}
    hc.append(("Modelo", "Tasa de anomalías", "ok" if not fuera else "warn",
               "\n".join(f"{STREAMS[k]} {v:.2%}" for k, v in tasa.items()),
               f"Debe rondar la contaminación configurada ({CONTAMINATION_IF:.0%}). Tolerancia: 0.5 % a 2 %."))
    ev_ok = valid[valid["ESTADO"] == "evaluable"]
    ari = ev_ok.groupby("STREAM")["ARI"].mean().to_dict()
    ari_min = min(ari.values()) if ari else None
    hc.append(("Modelo", "Consistencia tipos vs. clústeres",
               "info" if ari_min is None else "ok" if ari_min >= 0.5 else "warn" if ari_min >= 0.3 else "fail",
               "\n".join(f"{STREAMS[k]} ARI {v:.2f}" for k, v in ari.items()) or "Sin semanas evaluables",
               f"Promedio sobre {len(ev_ok)} semana(s) evaluables. Correcto con ARI ≥ 0.5."))

    # Agente forense
    if pms.empty:
        hc.append(("Agente forense", "Reportes postmortem", "warn", "0", "Aún no hay reportes generados."))
    else:
        gen = pms["METADATA_JSON"].map(lambda j: _parse_json_safe(j).get("generador", "N/A"))
        n_llm = int((gen == "gemini_api").sum())
        ultimo = pd.to_datetime(pms["FECHA_GENERACION"]).max()
        hc.append(("Agente forense", "Reportes postmortem", "ok", f"{len(pms)} reportes",
                   f"En {pms['MES'].nunique()} periodo(s). Último generado el {ultimo:%d/%m/%Y %H:%M}."))
        hc.append(("Agente forense", "Redacción con Gemini",
                   "ok" if n_llm == len(pms) else "warn", f"{n_llm} de {len(pms)}",
                   "Reportes redactados por Gemini; el resto usó el motor determinístico de respaldo."))
    return hc


def _html_checks(hc):
    grupos = {}
    for g, *resto in hc:
        grupos.setdefault(g, []).append(resto)
    out = '<div class="hc-grid">'
    for g, filas in grupos.items():
        out += f'<div class="hc-group"><p class="hc-gtitle">{html.escape(g)}</p>'
        for nombre, estado, valor, detalle in filas:
            txt, cls, _ = ESTADOS_HC[estado]
            out += (f'<div class="hc-row"><i class="hc-dot {cls}"></i>'
                    f'<div class="hc-main"><b>{html.escape(nombre)}</b>'
                    f'<span>{html.escape(detalle)}</span></div>'
                    f'<div class="hc-val"><b>{html.escape(str(valor))}</b>'
                    f'<span class="hc-pill {cls}">{txt}</span></div></div>')
        out += '</div>'
    return out + '</div>'


def vista_health(tarjeta, estilo, mostrar):
    """Vista completa de Health checks. Recibe los helpers de UI del dashboard
    (tarjeta, estilo, mostrar) para no depender del script principal."""
    t0 = time.perf_counter()
    with st.spinner("Revisando el sistema…"):
        hc = health_checks()
    cuenta = {k: sum(1 for c in hc if c[2] == k) for k in ESTADOS_HC}
    general = "fail" if cuenta["fail"] else "warn" if cuenta["warn"] else "ok"
    inv = inventario_hana()
    logs = inv[inv["TABLE_NAME"].str.startswith("LOGS")]
    lat = next((c[3] for c in hc if c[1] == "Conexión a SAP HANA Cloud"), "N/A")
    kpis = [
        ("Estado general", ESTADOS_HC[general][0],
         f"{cuenta['ok']} correctas · {cuenta['warn']} con atención · {cuenta['fail']} con falla",
         ESTADOS_HC[general][2], "Resultado combinado de todas las verificaciones."),
        ("Latencia de HANA", lat, "consulta mínima", BRAND, "Tiempo de respuesta de SAP HANA Cloud."),
        ("Registros en HANA", f"{logs['RECORD_COUNT'].sum() / 1e6:,.1f} M",
         f"en {len(logs)} tablas de logs", "#7858FF", "Registros de logs cargados en el esquema."),
        ("Última revisión", time.strftime("%H:%M:%S"), f"tomó {time.perf_counter() - t0:.1f} s", "#049F9A",
         "Las verificaciones se recalculan al abrir la vista o con el botón Volver a revisar."),
    ]
    st.html('<div class="kpis kpis-4">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{html.escape(ayuda)}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    with tarjeta("Verificaciones del sistema",
                 "Estado de la infraestructura, los datos, el modelo y el agente forense."):
        if st.button("Volver a revisar", icon=":material/refresh:", key="hc_refrescar"):
            inventario_hana.clear()
            metricas_pipeline.clear()
            leer_postmortems.clear()
            st.rerun()
        st.html(_html_checks(hc))

    meses_proc = sorted({t[len("RESULT_SUMMARY_"):] for t in inv["TABLE_NAME"]
                         if t.startswith("RESULT_SUMMARY_")})
    resumen, _, _ = metricas_pipeline(tuple(meses_proc))
    c1, c2 = st.columns(2, gap="medium")
    with c1, tarjeta("Tasa de anomalías por intervalo",
                     "Proporción de ventanas marcadas como anómalas. La línea punteada es la "
                     "contaminación configurada en el modelo (1 %)."):
        t = (resumen.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL", "STREAM"])
             [["VENTANAS_ANOMALAS", "VENTANAS"]].sum().reset_index().sort_values("WINDOW_SIZE_S"))
        t["TASA"] = t["VENTANAS_ANOMALAS"] / t["VENTANAS"]
        fig = go.Figure()
        for i, stx in enumerate(STREAMS):
            d = t[t["STREAM"] == stx]
            fig.add_trace(go.Bar(x=d["WINDOW_LABEL"], y=d["TASA"], name=STREAMS[stx],
                                 marker=dict(color=SERIES[i], cornerradius=4),
                                 hovertemplate="%{x}: %{y:.2%}<extra>" + STREAMS[stx] + "</extra>"))
        fig.add_hline(y=CONTAMINATION_IF, line=dict(color=CRITICO, dash="dot", width=1.5))
        fig.update_yaxes(tickformat=".1%", rangemode="tozero")
        mostrar(estilo(fig, height=300))
    with c2, tarjeta("Volumen y cobertura de los logs",
                     "Registros por tabla de logs en HANA. En azul, las que ya pasaron por el "
                     "pipeline; en gris, las pendientes."):
        proc = set(resumen["SOURCE_TABLE"])
        d = logs.sort_values("TABLE_NAME", ascending=False)
        fig = go.Figure(go.Bar(
            y=d["TABLE_NAME"], x=d["RECORD_COUNT"] / 1e6, orientation="h",
            marker=dict(color=[BRAND if n in proc else NEUTRO for n in d["TABLE_NAME"]], cornerradius=4),
            text=[f"{v / 1e6:.1f} M" for v in d["RECORD_COUNT"]], textposition="outside",
            textfont=dict(color=INK2, size=11), cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f} M registros<extra></extra>"))
        fig.update_xaxes(title_text="Millones de registros", title_font_color=MUTED)
        mostrar(estilo(fig, height=300, legend=False, margin=dict(l=8, r=40, t=8, b=40)))

    with tarjeta("Inventario de tablas en SAP HANA",
                 "Tablas de logs y de resultados del esquema, con su número de registros."):
        inv2 = inv.assign(TIPO=np.where(inv["TABLE_NAME"].str.startswith("LOGS"), "Logs", "Resultados"))
        st.dataframe(inv2[["TABLE_NAME", "TIPO", "RECORD_COUNT", "CREATE_TIME"]], hide_index=True,
                     column_config={
                         "TABLE_NAME": "Tabla", "TIPO": "Tipo",
                         "RECORD_COUNT": st.column_config.NumberColumn("Registros", format="localized"),
                         "CREATE_TIME": st.column_config.DatetimeColumn("Creada", format="DD/MM/YYYY HH:mm"),
                     })
