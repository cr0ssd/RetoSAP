"""MLOps · Health checks del sistema (infraestructura, datos, modelo y agente forense)."""
import html
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datos_hana import _conn, ping_hana_rapido, reiniciar_conexion, leer, leer_postmortems, _parse_json_safe
from helpers.clasificacion import validar_firmas
from helpers.email_sender import estado_servicio_correo
from helpers.IsolationForest import CONTAMINATION as CONTAMINATION_IF
from tema import BRAND, CRITICO, INK2, MUTED, NEUTRO, SERIES, STREAMS
import workflow as wf   # configuración y pasos del pipeline de producción


def tr(es, en):
    """Texto en el idioma elegido en el dashboard (se lee en cada llamada, no al importar)."""
    return en if st.session_state.get("idioma", "es") == "en" else es


def streams():
    """Nombre de cada origen de logs en el idioma elegido."""
    return {"system": tr(STREAMS["system"], "SAP System"), "llm": tr(STREAMS["llm"], "AI Services")}


def estados_hc():
    """estado -> (texto, clase CSS, color), en el idioma elegido."""
    return {
        "ok": (tr("Correcto", "OK"), "ok", "#188918"),
        "warn": (tr("Atención", "Warning"), "warn", "#E76500"),
        "fail": (tr("Falla", "Failure"), "fail", CRITICO),
        "info": (tr("Informativo", "Info"), "info", BRAND),
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
    ping_hana_rapido.clear()
    return ping_hana_rapido()


def hc_conexion():
    return tr("Conexión a SAP HANA Cloud", "SAP HANA Cloud connection")


def _es_hana_apagada(err_text):
    """Detecta si el error de SAP HANA corresponde a una instancia detenida (codigo 1890)."""
    low = str(err_text or "").lower()
    return "1890" in low or "stopped" in low or "inactive" in low


def _check_chatbot():
    """Verifica el endpoint /health del backend Node.js de Sapy."""
    chat_url = os.getenv("CHATBOT_URL", "http://localhost:3000/api/chat")
    health_url = (
        chat_url.rsplit("/api/chat", 1)[0] + "/health"
        if "/api/chat" in chat_url
        else chat_url.rstrip("/") + "/health"
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8") or "{}")
        ms = (time.perf_counter() - t) * 1000
        if payload.get("ok") is True:
            model = str(payload.get("model") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"))
            return (
                "ok",
                f"{ms:.0f} ms",
                tr(f"Servidor Node de Sapy activo (modelo {model}).", f"Sapy Node server active (model {model})."),
            )
        return (
            "warn",
            tr("Respuesta inesperada", "Unexpected reply"),
            tr(f"El endpoint {health_url} respondió sin confirmación.", f"Endpoint {health_url} replied without ok=true."),
        )
    except Exception:
        return (
            "warn",
            tr("Sin respuesta", "Unreachable"),
            tr(f"El servidor de Sapy no responde en {health_url}.", f"Sapy server is not responding at {health_url}."),
        )


def _check_email_service():
    """Verifica si el proveedor transaccional de correo (Resend/Brevo/SMTP) esta listo."""
    est = estado_servicio_correo()
    if est["configured"] and not est.get("sandbox"):
        return ("ok", est["provider_label"], est["hint"])
    if est["configured"] and est.get("sandbox"):
        return ("warn", f"{est['provider_label']} (sandbox)", est["hint"])
    return ("warn", tr("No configurado", "Not configured"), est["hint"])


def health_checks(hana_error_previo=None):
    """Lista de verificaciones: (grupo, nombre, estado, valor, detalle)."""
    hc = []
    hana_ok = True
    hana_err = None

    # Infraestructura: SAP HANA + servicios externos/locales del entorno
    if hana_error_previo:
        hana_ok = False
        hana_err = str(hana_error_previo)
    else:
        try:
            ms = _ping_hana()
            hc.append((tr("Infraestructura", "Infrastructure"), hc_conexion(),
                       "ok" if ms < 500 else "warn" if ms < 2000 else "fail", f"{ms:.0f} ms",
                       tr("Tiempo de respuesta de una consulta mínima (SELECT 1 FROM DUMMY).",
                          "Response time of a minimal query (SELECT 1 FROM DUMMY).")))
        except Exception as e:
            hana_ok = False
            hana_err = str(e)

    if not hana_ok:
        if _es_hana_apagada(hana_err):
            hc.append((
                tr("Infraestructura", "Infrastructure"),
                hc_conexion(),
                "fail",
                tr("Instancia apagada", "Instance stopped"),
                tr(
                    "La instancia de SAP HANA Cloud está apagada (código 1890). Enciéndela desde SAP HANA Cloud Central y presiona Volver a revisar.",
                    "The SAP HANA Cloud instance is stopped (code 1890). Start it from SAP HANA Cloud Central and click Check again.",
                ),
            ))
        else:
            hc.append((tr("Infraestructura", "Infrastructure"), hc_conexion(), "fail",
                       tr("Sin conexión", "No connection"), str(hana_err)[:180]))

    if hana_ok:
        hc.append((tr("Infraestructura", "Infrastructure"), tr("Versión de SAP HANA", "SAP HANA version"), "info", version_hana(),
                   tr("Motor donde corren las ventanas, Isolation Forest y K-means (PAL).",
                      "Engine running the windows, Isolation Forest and K-means (PAL).")))

    cb_status, cb_val, cb_det = _check_chatbot()
    hc.append((tr("Infraestructura", "Infrastructure"), tr("Asistente Sapy (Chatbot)", "Sapy Assistant (Chatbot)"),
               cb_status, cb_val, cb_det))

    em_status, em_val, em_det = _check_email_service()
    hc.append((tr("Infraestructura", "Infrastructure"), tr("Servicio de correo (Postmortems PDF)", "Email service (Postmortem PDFs)"),
               em_status, em_val, em_det))

    gemini = bool(os.getenv("GEMINI_API_KEY"))
    hc.append((tr("Infraestructura", "Infrastructure"), tr("Credencial de Gemini en este entorno", "Gemini credential in this environment"),
               "ok" if gemini else "info",
               tr("Configurada", "Configured") if gemini else tr("No configurada", "Not configured"),
               tr("Este entorno puede generar reportes y operar Sapy con Gemini.", "This environment can generate reports and run Sapy with Gemini.")
               if gemini else
               tr("El dashboard no la necesita para leer datos; el backend de Sapy y los reportes con Gemini sí la requieren.",
                  "The dashboard does not need it to read data; the Sapy backend and Gemini reports require it.")))

    # Modelo (verificacion en Python, independiente de HANA)
    try:
        for stx in STREAMS:
            validar_firmas(wf.FEATURES[stx], firmas=wf.FIRMAS[stx])
        hc.append((tr("Modelo", "Model"), tr("Firmas de clasificación", "Classification signatures"), "ok", "24 + 18 variables",
                   tr("Cada variable del Isolation Forest tiene firma y no sobran firmas.",
                      "Every Isolation Forest variable has a signature and there are no extra signatures.")))
    except Exception as e:
        hc.append((tr("Modelo", "Model"), tr("Firmas de clasificación", "Classification signatures"), "fail",
                   tr("Inconsistentes", "Inconsistent"), str(e)[:160]))

    if not hana_ok:
        hc.append((
            tr("Datos", "Data"),
            tr("Cobertura e integridad en HANA", "Coverage and integrity in HANA"),
            "warn",
            tr("En espera de HANA", "Waiting for HANA"),
            tr(
                "No se pueden consultar tablas LOGS_* ni RESULT_* mientras SAP HANA Cloud esté apagada o sin conexión.",
                "Cannot query LOGS_* or RESULT_* tables while SAP HANA Cloud is stopped or unreachable.",
            ),
        ))
        hc.append((
            tr("Agente forense", "Forensic agent"),
            tr("Reportes postmortem en HANA", "Postmortem reports in HANA"),
            "warn",
            tr("En espera de HANA", "Waiting for HANA"),
            tr(
                "Las métricas de RESULT_POSTMORTEMS_* se mostrarán automáticamente al encender SAP HANA Cloud.",
                "RESULT_POSTMORTEMS_* metrics will appear automatically once SAP HANA Cloud is running.",
            ),
        ))
        return hc

    inv = inventario_hana()
    logs = inv[inv["TABLE_NAME"].str.startswith("LOGS")]
    res_tab = inv[inv["TABLE_NAME"].str.startswith("RESULT_")]
    meses_proc = sorted({t.rsplit("_", 2)[-2] + "_" + t.rsplit("_", 2)[-1]
                         for t in res_tab["TABLE_NAME"] if t.startswith("RESULT_SUMMARY_")})
    resumen, valid, pms = metricas_pipeline(tuple(meses_proc))

    # Datos
    procesadas = set(resumen["SOURCE_TABLE"].unique())
    pendientes = sorted(set(logs["TABLE_NAME"]) - procesadas)
    hc.append((tr("Datos", "Data"), tr("Cobertura del pipeline", "Pipeline coverage"),
               "ok" if not pendientes else "warn",
               tr(f"{len(procesadas)} de {len(logs)} tablas", f"{len(procesadas)} of {len(logs)} tables"),
               tr("Todas las tablas de logs tienen resultados.", "All log tables have results.") if not pendientes else
               tr("Sin procesar: ", "Not processed: ") + ", ".join(pendientes) + "."))
    faltan, descuadres = [], []
    conteo = dict(zip(inv["TABLE_NAME"], inv["RECORD_COUNT"]))
    for m in meses_proc:
        faltan += [f"RESULT_{t}_{m}" for t in TABLAS_RESULTADO if f"RESULT_{t}_{m}" not in conteo]
        r = resumen[resumen["MES"] == m]
        for stx, tabla in (("system", f"RESULT_WINDOWS_SYSTEM_{m}"), ("llm", f"RESULT_WINDOWS_LLM_{m}")):
            esperado = int(r.loc[r["STREAM"] == stx, "VENTANAS"].sum())
            if tabla in conteo and int(conteo[tabla]) != esperado:
                descuadres.append(f"{tabla}: {int(conteo[tabla]):,} vs {esperado:,}")
    hc.append((tr("Datos", "Data"), tr("Integridad de resultados", "Results integrity"),
               "fail" if faltan or descuadres else "ok",
               tr(f"{len(meses_proc)} periodo(s)", f"{len(meses_proc)} period(s)"),
               (tr("Faltan: ", "Missing: ") + ", ".join(faltan) + ". " if faltan else "")
               + (tr("Ventanas no cuadran: ", "Windows do not match: ") + "; ".join(descuadres) + "." if descuadres else "")
               or tr("Las 5 tablas por periodo existen y las ventanas guardadas cuadran con el resumen.",
                     "The 5 tables per period exist and the saved windows match the summary.")))
    ult = pd.to_datetime(res_tab.loc[~res_tab["TABLE_NAME"].str.startswith("RESULT_POSTMORTEMS"),
                                     "CREATE_TIME"]).max()
    dias = (pd.Timestamp.now() - ult).days if pd.notna(ult) else None
    hc.append((tr("Datos", "Data"), tr("Frescura de resultados", "Results freshness"),
               "ok" if dias is not None and dias <= 7 else "warn",
               tr(f"hace {dias} día(s)", f"{dias} day(s) ago") if dias is not None else "N/A",
               tr(f"Última carga del pipeline: {ult:%d/%m/%Y %H:%M}. Se espera un lote nuevo cada semana.",
                  f"Last pipeline load: {ult:%d/%m/%Y %H:%M}. A new batch is expected every week.")
               if pd.notna(ult) else tr("No hay tablas de resultados.", "There are no result tables.")))

    # Modelo (metricas calculadas sobre HANA)
    tasa = resumen.groupby("STREAM")[["VENTANAS_ANOMALAS", "VENTANAS"]].sum()
    tasa = (tasa["VENTANAS_ANOMALAS"] / tasa["VENTANAS"]).to_dict()
    fuera = {k: v for k, v in tasa.items() if not 0.005 <= v <= 0.02}
    hc.append((tr("Modelo", "Model"), tr("Tasa de anomalías", "Anomaly rate"), "ok" if not fuera else "warn",
               "\n".join(f"{streams()[k]} {v:.2%}" for k, v in tasa.items()),
               tr(f"Debe rondar la contaminación configurada ({CONTAMINATION_IF:.0%}). Tolerancia: 0.5 % a 2 %.",
                  f"Should be close to the configured contamination ({CONTAMINATION_IF:.0%}). Tolerance: 0.5% to 2%.")))
    ev_ok = valid[valid["ESTADO"] == "evaluable"]
    ari = ev_ok.groupby("STREAM")["ARI"].mean().to_dict()
    ari_min = min(ari.values()) if ari else None
    hc.append((tr("Modelo", "Model"), tr("Consistencia tipos vs. clústeres", "Types vs. clusters consistency"),
               "info" if ari_min is None else "ok" if ari_min >= 0.5 else "warn" if ari_min >= 0.3 else "fail",
               "\n".join(f"{streams()[k]} ARI {v:.2f}" for k, v in ari.items()) or tr("Sin semanas evaluables", "No evaluable weeks"),
               tr(f"Promedio sobre {len(ev_ok)} semana(s) evaluables. Correcto con ARI ≥ 0.5.",
                  f"Average over {len(ev_ok)} evaluable week(s). OK with ARI ≥ 0.5.")))

    # Agente forense
    if pms.empty:
        hc.append((tr("Agente forense", "Forensic agent"), tr("Reportes postmortem", "Postmortem reports"), "warn", "0",
                   tr("Aún no hay reportes generados.", "No reports have been generated yet.")))
    else:
        gen = pms["METADATA_JSON"].map(lambda j: _parse_json_safe(j).get("generador", "N/A"))
        n_llm = int((gen == "gemini_api").sum())
        ultimo = pd.to_datetime(pms["FECHA_GENERACION"]).max()
        hc.append((tr("Agente forense", "Forensic agent"), tr("Reportes postmortem", "Postmortem reports"), "ok",
                   tr(f"{len(pms)} reportes", f"{len(pms)} reports"),
                   tr(f"En {pms['MES'].nunique()} periodo(s). Último generado el {ultimo:%d/%m/%Y %H:%M}.",
                      f"In {pms['MES'].nunique()} period(s). Last generated on {ultimo:%d/%m/%Y %H:%M}.")))
        hc.append((tr("Agente forense", "Forensic agent"), tr("Redacción con Gemini", "Written with Gemini"),
                   "ok" if n_llm == len(pms) else "warn", tr(f"{n_llm} de {len(pms)}", f"{n_llm} of {len(pms)}"),
                   tr("Reportes redactados por Gemini; el resto usó el motor determinístico de respaldo.",
                      "Reports written by Gemini; the rest used the deterministic fallback engine.")))
        if "REPORTE_MARKDOWN_EN" in pms.columns:
            n_en = int((pms["REPORTE_MARKDOWN_EN"].notna() & (pms["REPORTE_MARKDOWN_EN"].astype(str).str.strip() != "")).sum())
        else:
            n_en = 0
        hc.append((tr("Agente forense", "Forensic agent"), tr("Versión en inglés", "English version"),
                   "ok" if n_en == len(pms) else "warn", tr(f"{n_en} de {len(pms)}", f"{n_en} of {len(pms)}"),
                   tr("Reportes con traducción al inglés (src/traducir_postmortems.py completa los que faltan).",
                      "Reports with an English translation (src/traducir_postmortems.py fills in the missing ones).")))
    return hc


def _html_checks(hc):
    grupos = {}
    for g, *resto in hc:
        grupos.setdefault(g, []).append(resto)
    out = '<div class="hc-grid">'
    for g, filas in grupos.items():
        out += f'<div class="hc-group"><p class="hc-gtitle">{html.escape(g)}</p>'
        for nombre, estado, valor, detalle in filas:
            txt, cls, _ = estados_hc()[estado]
            out += (f'<div class="hc-row"><i class="hc-dot {cls}"></i>'
                    f'<div class="hc-main"><b>{html.escape(nombre)}</b>'
                    f'<span>{html.escape(detalle)}</span></div>'
                    f'<div class="hc-val"><b>{html.escape(str(valor))}</b>'
                    f'<span class="hc-pill {cls}">{txt}</span></div></div>')
        out += '</div>'
    return out + '</div>'


def vista_health(tarjeta, estilo, mostrar, hana_error=None):
    """Vista completa de Health checks. Recibe los helpers de UI del dashboard
    (tarjeta, estilo, mostrar) para no depender del script principal."""
    t0 = time.perf_counter()
    with st.spinner(tr("Revisando el sistema…", "Checking the system…")):
        hc = health_checks(hana_error_previo=hana_error)
    cuenta = {k: sum(1 for c in hc if c[2] == k) for k in estados_hc()}
    general = "fail" if cuenta["fail"] else "warn" if cuenta["warn"] else "ok"
    hana_item = next((c for c in hc if c[1] == hc_conexion()), None)
    hana_ok = bool(hana_item and hana_item[2] != "fail")

    if not hana_ok:
        detalle_hana = hana_error or (hana_item[4] if hana_item else "")
        apagada = _es_hana_apagada(detalle_hana)
        if apagada:
            st.error(
                tr(
                    "La instancia de SAP HANA Cloud está apagada (código 1890). "
                    "Enciéndela desde SAP HANA Cloud Central y presiona «Volver a revisar» para desbloquear el dashboard.",
                    "The SAP HANA Cloud instance is stopped (code 1890). "
                    "Start it from SAP HANA Cloud Central and click «Check again» to unlock the dashboard.",
                ),
                icon=":material/power_off:",
            )
        else:
            st.error(
                tr(
                    "No fue posible conectar con SAP HANA Cloud. Verifica el estado de la instancia y presiona «Volver a revisar».",
                    "Could not connect to SAP HANA Cloud. Verify the instance status and click «Check again».",
                ),
                icon=":material/error:",
            )

        otros = [
            c for c in hc
            if c[1] != hc_conexion() and c[0] in (tr("Infraestructura", "Infrastructure"), tr("Modelo", "Model"))
        ]
        otros_ok = sum(1 for c in otros if c[2] == "ok")
        kpis = [
            (
                tr("Estado general", "Overall status"),
                estados_hc()[general][0],
                tr(
                    f"{cuenta['ok']} correctas · {cuenta['warn']} con atención · {cuenta['fail']} con falla",
                    f"{cuenta['ok']} OK · {cuenta['warn']} warnings · {cuenta['fail']} failures",
                ),
                estados_hc()[general][2],
                tr("Resultado combinado de todas las verificaciones.", "Combined result of all checks."),
            ),
            (
                "SAP HANA Cloud",
                tr("Apagada", "Stopped") if apagada else tr("Sin conexión", "Offline"),
                tr("Instancia detenida (1890)", "Instance stopped (1890)") if apagada else tr("No responde", "Unreachable"),
                CRITICO,
                tr("Base de datos principal requerida para las vistas de datos.", "Primary database required for analytics views."),
            ),
            (
                tr("Otros servicios", "Other services"),
                tr(f"{otros_ok} de {len(otros)} activos", f"{otros_ok} of {len(otros)} active"),
                tr("Chatbot Sapy, Correo PDF, Gemini y Modelo", "Sapy Chatbot, PDF Email, Gemini & Model"),
                "#188918" if otros_ok == len(otros) else "#E76500",
                tr("Servicios independientes de HANA verificados en vivo.", "HANA-independent services verified live."),
            ),
            (
                tr("Última revisión", "Last check"),
                time.strftime("%H:%M:%S"),
                tr(f"tomó {time.perf_counter() - t0:.1f} s", f"took {time.perf_counter() - t0:.1f} s"),
                "#049F9A",
                tr("Presiona Volver a revisar una vez encendida la instancia.", "Click Check again once the instance is running."),
            ),
        ]
        st.html('<div class="kpis kpis-4">' + "".join(
            f'<div class="kpi" style="--c:{c}" title="{html.escape(ayuda)}"><div class="lab"><i></i>{lab}</div>'
            f'<div class="val">{val}</div><div class="det">{det}</div></div>'
            for lab, val, det, c, ayuda in kpis) + '</div>')

        with tarjeta(tr("Verificaciones del sistema", "System checks"),
                     tr("Estado de SAP HANA Cloud y de los demás servicios de la plataforma.",
                        "Status of SAP HANA Cloud and all other platform services.")):
            if st.button(tr("Volver a revisar", "Check again"), icon=":material/refresh:",
                         type="primary", key="hc_refrescar_offline"):
                reiniciar_conexion()
                st.cache_data.clear()
                st.rerun()
            st.html(_html_checks(hc))
        return

    inv = inventario_hana()
    logs = inv[inv["TABLE_NAME"].str.startswith("LOGS")]
    lat = hana_item[3] if hana_item else "N/A"
    kpis = [
        (tr("Estado general", "Overall status"), estados_hc()[general][0],
         tr(f"{cuenta['ok']} correctas · {cuenta['warn']} con atención · {cuenta['fail']} con falla",
            f"{cuenta['ok']} OK · {cuenta['warn']} warnings · {cuenta['fail']} failures"),
         estados_hc()[general][2], tr("Resultado combinado de todas las verificaciones.", "Combined result of all checks.")),
        (tr("Latencia de HANA", "HANA latency"), lat, tr("consulta mínima", "minimal query"), BRAND,
         tr("Tiempo de respuesta de SAP HANA Cloud.", "SAP HANA Cloud response time.")),
        (tr("Registros en HANA", "Records in HANA"), f"{logs['RECORD_COUNT'].sum() / 1e6:,.1f} M",
         tr(f"en {len(logs)} tablas de logs", f"in {len(logs)} log tables"), "#7858FF",
         tr("Registros de logs cargados en el esquema.", "Log records loaded in the schema.")),
        (tr("Última revisión", "Last check"), time.strftime("%H:%M:%S"),
         tr(f"tomó {time.perf_counter() - t0:.1f} s", f"took {time.perf_counter() - t0:.1f} s"), "#049F9A",
         tr("Las verificaciones se recalculan al abrir la vista o con el botón Volver a revisar.",
            "Checks are recalculated when opening the view or with the Check again button.")),
    ]
    st.html('<div class="kpis kpis-4">' + "".join(
        f'<div class="kpi" style="--c:{c}" title="{html.escape(ayuda)}"><div class="lab"><i></i>{lab}</div>'
        f'<div class="val">{val}</div><div class="det">{det}</div></div>'
        for lab, val, det, c, ayuda in kpis) + '</div>')

    with tarjeta(tr("Verificaciones del sistema", "System checks"),
                 tr("Estado de la infraestructura, los datos, el modelo y el agente forense.",
                    "Status of the infrastructure, the data, the model and the forensic agent.")):
        if st.button(tr("Volver a revisar", "Check again"), icon=":material/refresh:", key="hc_refrescar"):
            reiniciar_conexion()
            st.cache_data.clear()
            st.rerun()
        st.html(_html_checks(hc))

    meses_proc = sorted({t[len("RESULT_SUMMARY_"):] for t in inv["TABLE_NAME"]
                         if t.startswith("RESULT_SUMMARY_")})
    resumen, _, _ = metricas_pipeline(tuple(meses_proc))
    c1, c2 = st.columns(2, gap="medium")
    with c1, tarjeta(tr("Tasa de anomalías por intervalo", "Anomaly rate by interval"),
                     tr("Proporción de ventanas marcadas como anómalas. La línea punteada es la "
                        "contaminación configurada en el modelo (1 %).",
                        "Share of windows flagged as anomalous. The dotted line is the contamination "
                        "configured in the model (1%).")):
        t = (resumen.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL", "STREAM"])
             [["VENTANAS_ANOMALAS", "VENTANAS"]].sum().reset_index().sort_values("WINDOW_SIZE_S"))
        t["TASA"] = t["VENTANAS_ANOMALAS"] / t["VENTANAS"]
        fig = go.Figure()
        for i, stx in enumerate(STREAMS):
            d = t[t["STREAM"] == stx]
            fig.add_trace(go.Bar(x=d["WINDOW_LABEL"], y=d["TASA"], name=streams()[stx],
                                 marker=dict(color=SERIES[i], cornerradius=4),
                                 hovertemplate="%{x}: %{y:.2%}<extra>" + streams()[stx] + "</extra>"))
        fig.add_hline(y=CONTAMINATION_IF, line=dict(color=CRITICO, dash="dot", width=1.5))
        fig.update_yaxes(tickformat=".1%", rangemode="tozero")
        mostrar(estilo(fig, height=300))
    with c2, tarjeta(tr("Volumen y cobertura de los logs", "Log volume and coverage"),
                     tr("Registros por tabla de logs en HANA. En azul, las que ya pasaron por el "
                        "pipeline; en gris, las pendientes.",
                        "Records per log table in HANA. Blue: already processed by the pipeline; "
                        "gray: pending.")):
        proc = set(resumen["SOURCE_TABLE"])
        d = logs.sort_values("TABLE_NAME", ascending=False)
        fig = go.Figure(go.Bar(
            y=d["TABLE_NAME"], x=d["RECORD_COUNT"] / 1e6, orientation="h",
            marker=dict(color=[BRAND if n in proc else NEUTRO for n in d["TABLE_NAME"]], cornerradius=4),
            text=[f"{v / 1e6:.1f} M" for v in d["RECORD_COUNT"]], textposition="outside",
            textfont=dict(color=INK2, size=11), cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f} M " + tr("registros", "records") + "<extra></extra>"))
        fig.update_xaxes(title_text=tr("Millones de registros", "Millions of records"), title_font_color=MUTED)
        mostrar(estilo(fig, height=300, legend=False, margin=dict(l=8, r=40, t=8, b=40)))

    with tarjeta(tr("Inventario de tablas en SAP HANA", "SAP HANA table inventory"),
                 tr("Tablas de logs y de resultados del esquema, con su número de registros.",
                    "Log and result tables in the schema, with their record counts.")):
        inv2 = inv.assign(TIPO=np.where(inv["TABLE_NAME"].str.startswith("LOGS"), "Logs", tr("Resultados", "Results")))
        st.dataframe(inv2[["TABLE_NAME", "TIPO", "RECORD_COUNT", "CREATE_TIME"]], hide_index=True,
                     column_config={
                         "TABLE_NAME": tr("Tabla", "Table"), "TIPO": tr("Tipo", "Type"),
                         "RECORD_COUNT": st.column_config.NumberColumn(tr("Registros", "Records"), format="localized"),
                         "CREATE_TIME": st.column_config.DatetimeColumn(tr("Creada", "Created"), format="DD/MM/YYYY HH:mm"),
                     })
