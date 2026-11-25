import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

DIR = Path(__file__).resolve().parent
AVATAR = DIR / "componentes" / "sapy.png"

# Backend Node (src/chatbot/server.js)
CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:3000/api/chat")
CHATBOT_TIMEOUT = 45        # segundos; el servidor corta a Gemini a los 30 s, así que siempre responde antes


# Lo que Sapy sabe: guía fija del dashboard (sapy_guia.md) + los datos que se están viendo
GUIA = DIR / "sapy_guia.md"
MAX_EVENTOS, MAX_POSTMORTEMS = 40, 30


def _txt(valor, limite):
    if valor is None or (not isinstance(valor, (list, tuple, dict)) and pd.isna(valor)):
        return ""
    return " ".join(str(valor).split())[:limite]


def _fecha(valor):
    return "" if pd.isna(valor) else pd.Timestamp(valor).strftime("%Y-%m-%d %H:%M")


def _meta(raw):
    """METADATA_JSON de un postmortem -> dict (vacío si no se puede leer)."""
    try:
        d = raw if isinstance(raw, dict) else json.loads(str(raw))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def _sec_guia(d):
    texto = GUIA.read_text(encoding="utf-8") if GUIA.exists() else ""
    return "=== GUÍA DEL DASHBOARD ===\n" + texto.replace("{Z_UMBRAL}", f"{d['z_umbral']:g}")


def _sec_tipos(d):
    flecha = {"+": "aumenta", "-": "disminuye", "~": "cambia"}
    lineas = ["=== TIPOS DE ATAQUE POSIBLES EN EL ORIGEN SELECCIONADO ==="]
    for t in d["tipos"]:
        senales = ", ".join(f"{d['nombre'](f)} {flecha[sg[t]]}" for f, sg in d["firma"].items() if t in sg)
        desc = d["descripcion_tipo"].get(t, "")
        lineas.append(f"- {d['tipo_txt'](t)}: {desc} Indicadores que lo señalan: {senales or 'n/d'}.")
    return "\n".join(lineas)


def _sec_estado(d):
    win, ev, rel, res = d["ventanas"], d["eventos"], d["relevantes"], d["resumen"]
    if d["periodo"]:
        n_win, n_anom = len(win), int((win["LABEL"] == -1).sum())
        n_ev, n_rel = len(ev), len(rel)
        tipos = sorted(set(rel["TIPO_ATAQUE"].dropna()) - {"Falla operativa"})
        alcance = f"del {d['periodo']} (periodo seleccionado con zoom)"
    else:
        n_win, n_anom = int(res["VENTANAS"].sum()), int(res["VENTANAS_ANOMALAS"].sum())
        n_ev, n_rel = int(res["EVENTOS"].sum()), int(res["EVENTOS_RELEVANTES"].sum())
        tipos = sorted({t.strip() for x in res["TIPOS_LISTA"].dropna() for t in x.split(",")} - {"Falla operativa"})
        alcance = "del mes completo"
    pct = f" ({n_anom / n_win:.1%} del total)" if n_win else ""
    return "\n".join([
        "=== ESTADO ACTUAL DEL DASHBOARD ===",
        f"Mes: {d['mes']}. Origen de los registros: {d['origen']}. Intervalo de análisis: {d['intervalo']}.",
        f"Alcance de los datos: {alcance}. Umbral de intensidad para un evento relevante: {d['z_umbral']:g}.",
        f"Bloques analizados: {n_win:,}. Bloques atípicos: {n_anom:,}{pct}. Eventos detectados: {n_ev:,}. "
        f"Eventos relevantes: {n_rel:,}.",
        f"Tipos de ataque en eventos relevantes (sin contar Falla operativa): "
        f"{', '.join(d['tipo_txt'](t) for t in tipos) or 'ninguno'}.",
    ])


def _sec_eventos(d):
    ev, rel = d["eventos"], d["relevantes"]
    if ev.empty:
        return "=== EVENTOS ===\nNo hay eventos detectados con la selección actual."
    top = rel.sort_values("Z_MAX", ascending=False).head(MAX_EVENTOS)
    lineas = [f"=== EVENTOS RELEVANTES (más intensos primero; {len(top)} de {len(rel)}) ==="]
    for _, r in top.iterrows():
        lineas.append(
            f"- Evento {int(r['EVENTO_ID'])} | tipo={d['tipo_txt'](r['TIPO_ATAQUE'])} | "
            f"confianza={_txt(r.get('CONFIANZA'), 10)} | inicio={_fecha(r['INICIO'])} | "
            f"fin={_fecha(r['FIN_REAL'])} | intensidad={r['Z_MAX']:.1f} | "
            f"indicador más afectado={_txt(r.get('INDICADOR'), 60)} | cambios={_txt(r.get('CAMBIOS'), 160)}")
    lineas.append(f"Eventos no relevantes: {len(ev) - len(rel)} (no se listan; son los de menor intensidad).")
    return "\n".join(lineas)


def _lista(valor, mostrar=8):
    ##Lista guardada como JSON o CSV -> 'N elementos (a, b, c…)'.
    if valor is None or (not isinstance(valor, (list, tuple)) and pd.isna(valor)):
        return "0"
    if isinstance(valor, (list, tuple)):
        items = [str(x).strip() for x in valor]
    else:
        txt = str(valor).strip()
        try:
            cargado = json.loads(txt) if txt.startswith("[") else None
        except Exception:
            cargado = None
        items = [str(x).strip() for x in cargado] if isinstance(cargado, list) else [x.strip() for x in txt.split(",")]
    items = [x for x in dict.fromkeys(items) if x]
    resto = f"… y {len(items) - mostrar} más" if len(items) > mostrar else ""
    return f"{len(items)} ({', '.join(items[:mostrar])}{resto})" if items else "0"


def _num(valor, decimales=2):
    try:
        return f"{float(valor):.{decimales}f}"
    except (TypeError, ValueError):
        return ""


def _sec_postmortems(d):
    df = d["postmortems"]
    if df is None or df.empty:
        return "=== POSTMORTEMS ===\nNo hay postmortems para el mes/periodo seleccionado."
    df = df.sort_values("INICIO", ascending=False, na_position="last")
    lineas = [f"=== POSTMORTEMS (reportes forenses; {min(len(df), MAX_POSTMORTEMS)} más recientes de {len(df)}) ===",
              "La severidad es un puntaje: a mayor valor, más grave. El número de evento de un postmortem es el del "
              "intervalo que aparece en su ID (p. ej. 30M = bloques de 30 min), que puede no coincidir con el "
              "intervalo seleccionado en el dashboard; no lo confundas con los eventos de la lista anterior."]
    for _, r in df.head(MAX_POSTMORTEMS).iterrows():
        inc = _meta(r.get("METADATA_JSON")).get("incident_meta") or {}
        z = inc.get("z_max")
        lineas.append(
            f"- {_txt(r.get('POSTMORTEM_ID'), 40)} | evento={_txt(r.get('EVENTO_ID'), 10)} | "
            f"origen={_txt(r.get('STREAM'), 20)} | tipo={_txt(r.get('TIPO_ATAQUE'), 60)} | "
            f"MITRE={_txt(r.get('MITRE_TACTIC'), 60)} | severidad={_num(r.get('SEVERIDAD_SCORE'))} | "
            f"inicio={_fecha(r.get('INICIO'))} | fin={_fecha(r.get('FIN'))} | "
            f"indicador={_txt(d['nombre'](str(inc.get('z_max_feature') or '')), 60)}"
            f"{f' (Z={float(z):.1f})' if z else ''} | "
            f"aplicaciones={_lista(r.get('APLICACIONES_AFECTADAS'))} | IPs={_lista(r.get('IPS_INVOLUCRADAS'))}\n"
            f"  resumen: {_txt(r.get('RESUMEN_EJECUTIVO'), 400)}")
    return "\n".join(lineas)


def _sec_intervalos(d):
    res = d["resumen_s"]
    if res.empty:
        return ""
    g = (res.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])[["EVENTOS", "EVENTOS_RELEVANTES"]].sum()
         .reset_index().sort_values("WINDOW_SIZE_S"))
    lineas = ["=== RESULTADOS SEGÚN EL INTERVALO DE ANÁLISIS (mes completo, origen seleccionado) ==="]
    lineas += [f"- {r['WINDOW_LABEL']}: {int(r['EVENTOS'])} eventos, {int(r['EVENTOS_RELEVANTES'])} relevantes"
               for _, r in g.iterrows()]
    return "\n".join(lineas)


def _sec_consistencia(d):
    v = d["validacion"]
    v = v[v["ESTADO"] == "evaluable"] if len(v) else v
    if v.empty:
        return ("=== CONSISTENCIA DE LA CLASIFICACIÓN ===\nNo hay datos suficientes para evaluarla en este "
                "periodo (se necesitan al menos dos tipos de evento).")
    g = (v.groupby(["WINDOW_SIZE_S", "WINDOW_LABEL"])[["ARI", "NMI"]].mean()
         .reset_index().sort_values("WINDOW_SIZE_S"))
    lineas = ["=== CONSISTENCIA DE LA CLASIFICACIÓN (promedio por intervalo; 1 = coincidencia total, 0 = azar) ==="]
    lineas += [f"- {r['WINDOW_LABEL']}: ARI={r['ARI']:.2f}, NMI={r['NMI']:.2f}" for _, r in g.iterrows()]
    return "\n".join(lineas)


def construir_contexto(d):
    """Texto completo que se envía a Gemini. Si una sección falla, se omite: el dashboard nunca se cae por Sapy."""
    partes = []
    for seccion in (_sec_guia, _sec_tipos, _sec_estado, _sec_eventos, _sec_postmortems,
                    _sec_intervalos, _sec_consistencia):
        try:
            partes.append(seccion(d))
        except Exception:
            continue
    return "\n\n".join(p for p in partes if p)


# Lógica del chatbot: envía la pregunta al servidor Node, que consulta a Gemini
def responder(historial, pregunta, idioma="es", contexto=""):
    en = idioma == "en"
    # El servidor guarda el historial por sessionId, por eso no se reenvía `historial`.
    sesion = st.session_state.setdefault("sapy_session_id", str(uuid4()))
    peticion = urllib.request.Request(
        CHATBOT_URL,
        data=json.dumps({"message": pregunta, "sessionId": sesion, "context": contexto,
                         "language": idioma}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(peticion, timeout=CHATBOT_TIMEOUT) as r:
            return json.load(r).get("reply") or ("I got an empty answer." if en else "No recibí una respuesta.")
    except urllib.error.HTTPError as e:
        try:
            detalle = json.load(e).get("error", "")
        except Exception:
            detalle = ""
        return f"⚠️ {detalle or (f'Server error ({e.code}).' if en else f'Error del servidor ({e.code}).')}"
    except urllib.error.URLError:
        return ("⚠️ I can't reach the chatbot server. Start it with `npm start` in `src/chatbot`."
                if en else
                "⚠️ No puedo conectarme al servidor del chatbot. Inícialo con `npm start` en `src/chatbot`.")
    except TimeoutError:
        return "⚠️ The request took too long. Try again." if en else "⚠️ La solicitud tardó demasiado. Intenta de nuevo."


# Interfaz: botón flotante en la esquina y panel de chat
def _css(avatar_b64, tr):
    st.html(f"""
<style>
  /* Botón flotante con la cara de Sapy */
  .st-key-sapy_fab {{ position: fixed; right: 26px; bottom: 26px; z-index: 999; width: auto; }}
  .st-key-sapy_fab button {{
      width: 68px; height: 68px; min-height: 68px; border-radius: 50%; padding: 0;
      background: #fff url("data:image/png;base64,{avatar_b64}") center 30% / 118% no-repeat !important;
      border: 3px solid #7c5cc4 !important;
      box-shadow: 0 6px 18px rgba(60, 40, 120, .28); transition: transform .15s, box-shadow .15s; }}
  .st-key-sapy_fab button:hover {{ transform: translateY(-2px) scale(1.04);
      box-shadow: 0 10px 24px rgba(60, 40, 120, .35); }}
  .st-key-sapy_fab button p {{ display: none; }}
  /* Globito "¿Te ayudo?" */
  .st-key-sapy_fab::before {{ content: "{tr('¿Te ayudo?', 'Need help?')}"; position: absolute; right: 80px; bottom: 20px;
      white-space: nowrap; background: #fff; color: #3b2a6e; font-size: .82rem; font-weight: 650;
      padding: .38rem .75rem; border-radius: 14px 14px 4px 14px; border: 1px solid #e3dcf5;
      box-shadow: 0 4px 12px rgba(60, 40, 120, .14); }}
  /* Panel de chat en la esquina inferior derecha */
  [data-testid="stDialog"]:has(.st-key-sapy_panel) {{ background: rgba(29, 45, 62, .18); }}
  section[role="dialog"]:has(.st-key-sapy_panel) {{ position: fixed !important; right: 26px; bottom: 108px;
      top: auto !important; left: auto !important; margin: 0 !important; transform: none !important;
      width: min(420px, calc(100vw - 32px)) !important;
      height: min(600px, calc(100vh - 140px)); max-height: none !important; overflow: hidden !important;
      border-radius: 18px !important; border-top: 5px solid #7c5cc4; background: #ffffff !important; opacity: 1 !important;
      box-shadow: 0 18px 48px rgba(40, 30, 90, .28); padding-top: 1rem !important; }}
  section[role="dialog"]:has(.st-key-sapy_panel) > h2 {{ display: none; }}   /* el encabezado propio ya dice Sapy */
  /* La X de cerrar queda por encima del encabezado propio (si no, el encabezado la tapa y no responde) */
  section[role="dialog"]:has(.st-key-sapy_panel) > button {{ z-index: 10; }}
  .sapy-head {{ display: flex; align-items: center; gap: .7rem; margin: -.2rem 0 .3rem; padding-right: 2.2rem; }}
  .sapy-head img {{ width: 46px; height: 46px; border-radius: 50%; object-fit: cover;
      object-position: center 30%; border: 2px solid #7c5cc4; background: #fff; }}
  .sapy-head b {{ display: block; font-size: 1.05rem; color: #1D2D3E; }}
  .sapy-head span {{ font-size: .8rem; color: #556B82; }}
  /* Alto fijo: el panel no cambia de tamaño; solo se desplazan los mensajes y la caja de texto queda abajo */
  /* El panel reparte su alto: la caja de texto toma lo que necesita y los mensajes el resto */
  section[role="dialog"]:has(.st-key-sapy_panel) {{ display: flex !important; flex-direction: column; }}
  section[role="dialog"]:has(.st-key-sapy_panel) > div:has(.st-key-sapy_panel) {{
      flex: 1 1 auto; min-height: 0; display: flex; flex-direction: column; }}
  section[role="dialog"] [data-testid="stVerticalBlock"]:has(.st-key-sapy_panel),
  section[role="dialog"] [data-testid="stLayoutWrapper"]:has(.st-key-sapy_panel), .st-key-sapy_panel {{
      flex: 1 1 auto !important; min-height: 0 !important; display: flex; flex-direction: column; }}
  [data-testid="stLayoutWrapper"]:has(> .st-key-sapy_mensajes) {{
      flex: 1 1 0 !important; min-height: 0 !important; height: auto !important; max-height: none !important; }}
  .st-key-sapy_mensajes {{ height: 100% !important; max-height: 100% !important; flex: none !important; }}
  .st-key-sapy_mensajes [data-testid="stChatMessage"] {{ padding: .35rem .2rem; }}
  /* Caja de texto: crece hasta 3 renglones y después se desplaza por dentro; el panel no se mueve */
  section[role="dialog"]:has(.st-key-sapy_panel) [data-testid="stChatInputTextArea"],
  section[role="dialog"]:has(.st-key-sapy_panel) [data-testid="stChatInput"] textarea {{
      max-height: 4.6rem !important; overflow-y: auto !important; }}
  /* Sugerencias y respuestas rápidas como "chips" */
  .st-key-sapy_mensajes .stButton button {{ border-radius: 999px; min-height: 0; padding: .22rem .75rem;
      font-size: .8rem; border: 1px solid #d9cff3; color: #4a3a86; background: #faf8ff; }}
  .st-key-sapy_mensajes .stButton button:hover {{ background: #0070F2; border-color: #0070F2; color: #fff; }}
  .st-key-sapy_mensajes .stButton button:hover * {{ color: #fff; }}
  .st-key-sapy_mensajes [data-testid="stHorizontalBlock"] {{ gap: .4rem; flex-wrap: wrap; }}
  .st-key-sapy_mensajes [data-testid="stColumn"] {{ width: auto !important; flex: 0 0 auto !important; min-width: 0; }}
  /* Ícono del usuario en el azul del dashboard (Streamlit lo pinta rojo por defecto) */
  .st-key-sapy_mensajes [data-testid="stChatMessageAvatarUser"] {{ background-color: #0070F2 !important;
      color: #fff !important; }}
  .st-key-sapy_mensajes [data-testid="stChatMessageContent"] p,
  .st-key-sapy_mensajes [data-testid="stChatMessageContent"] li {{ font-size: .88rem; line-height: 1.5; }}
</style>""")


DESPEDIDAS = ("es todo", "eso es todo", "seria todo", "sería todo", "nada mas", "nada más", "no gracias",
              "no, gracias", "ya es todo", "es todo gracias", "that's all", "thats all", "that is all",
              "no thanks", "no, thanks", "nothing else", "i'm done", "im done")


def _es_despedida(texto):
    t = " ".join(str(texto).lower().replace("¡", "").replace("!", "").replace(".", "").split())
    return t in ("listo", "ya", "no", "gracias", "ok gracias", "thanks", "done") or any(d in t for d in DESPEDIDAS)


ESPERA_CIERRE = 45      # segundos sin preguntar antes de "¿Es todo o necesitas ayuda con otra cosa?"
REVISION_CIERRE = 5     # cada cuánto se revisa (solo mientras el chat está abierto)


def mostrar_sapy(idioma, tr, datos):
    """Dibuja el botón flotante; al pulsarlo abre el chat. tr(es, en) elige el texto según el idioma.
    datos: lo que el dashboard está mostrando (ver construir_contexto); Sapy responde con eso."""
    if not AVATAR.exists():
        return
    contexto = construir_contexto(datos)
    avatar_b64 = base64.b64encode(AVATAR.read_bytes()).decode()
    _css(avatar_b64, tr)
    st.session_state.setdefault("sapy_historial", [])
    av = str(AVATAR)

    def _pedir(texto):          # botones (sugerencias / respuestas rápidas) -> pregunta pendiente
        st.session_state["sapy_pendiente"] = texto

    def _seguir():              # "Tengo otra pregunta": oculta la pregunta de cierre hasta la próxima respuesta
        st.session_state["sapy_sin_cierre"] = len(st.session_state["sapy_historial"])

    def _limpiar():
        st.session_state["sapy_historial"] = []
        st.session_state.pop("sapy_session_id", None)     # conversación nueva también en el servidor

    @st.dialog("Sapy", width="medium")
    def panel():
        with st.container(key="sapy_panel"):
            st.html(f'<div class="sapy-head"><img src="data:image/png;base64,{avatar_b64}">'
                    f'<div><b>Sapy</b><span>{tr("Tu guía del dashboard de anomalías SAP", "Your guide to the SAP anomaly dashboard")}'
                    '</span></div></div>')
            hist = st.session_state["sapy_historial"]
            mensajes = st.container(height=360, key="sapy_mensajes", border=False)
            with mensajes:
                with st.chat_message("assistant", avatar=av):
                    st.markdown(tr("¡Hola! Soy **Sapy**. Puedo explicarte qué significa cualquier dato del "
                                   "dashboard, dónde encontrar algo o cómo interpretar una gráfica.",
                                   "Hi! I'm **Sapy**. I can explain what any data in the dashboard means, "
                                   "where to find something or how to read a chart."))
                    sug_slot = st.empty()       # se vacía en cuanto se hace la primera pregunta
                    if not hist:
                        sugerencias = [tr("¿Qué es un evento relevante?", "What is a relevant event?"),
                                       tr("¿Dónde descargo el reporte en PDF?", "Where do I download the PDF report?"),
                                       tr("¿Cómo leo la gráfica de actividad?", "How do I read the activity chart?")]
                        with sug_slot.container():
                            for k, sug in enumerate(sugerencias):
                                st.button(sug, key=f"sapy_sug_{k}", on_click=_pedir, args=(sug,))
                for m in hist:
                    with st.chat_message(m["rol"], avatar=av if m["rol"] == "assistant" else None):
                        st.markdown(m["texto"])

            c1, c2 = st.columns([7, 1], vertical_alignment="bottom", gap="small")
            escrita = c1.chat_input(tr("Pregúntale a Sapy…", "Ask Sapy…"), key="sapy_input")
            c2.button(":material/delete_sweep:", key="sapy_limpiar", on_click=_limpiar, disabled=not hist,
                      help=tr("Nueva conversación", "New conversation"))
            pregunta = escrita or st.session_state.pop("sapy_pendiente", None)

            if pregunta == "__fin__" or (pregunta and _es_despedida(pregunta)):
                sug_slot.empty()
                despedida = tr("¡Con gusto! Aquí estaré cuando necesites algo más. 👋",
                               "You're welcome! I'll be here whenever you need anything else. 👋")
                nuevos = ([] if pregunta == "__fin__" else [{"rol": "user", "texto": pregunta}]) + \
                         [{"rol": "assistant", "texto": despedida, "fin": True}]
                with mensajes:
                    for m in nuevos:
                        with st.chat_message(m["rol"], avatar=av if m["rol"] == "assistant" else None):
                            st.markdown(m["texto"])
                st.session_state["sapy_historial"] = (hist + nuevos)[-40:]
                time.sleep(1.6)
                st.rerun()                     # recarga la app: el chat se cierra; se puede volver a abrir
            elif pregunta:
                sug_slot.empty()
                with mensajes:
                    with st.chat_message("user"):
                        st.markdown(pregunta)
                    with st.chat_message("assistant", avatar=av):
                        with st.spinner(tr("Sapy está pensando…", "Sapy is thinking…")):
                            texto = responder([m for m in hist if not m.get("fin")], pregunta, idioma, contexto)
                        st.markdown(texto)
                hist = hist + [{"rol": "user", "texto": pregunta}, {"rol": "assistant", "texto": texto}]
                st.session_state["sapy_historial"] = hist[-40:]      # lo que se ve; al modelo van menos

                st.session_state["sapy_ultima_resp"] = time.time()

            with mensajes:
                cierre()

    @st.fragment(run_every=f"{REVISION_CIERRE}s")
    def cierre():
       ## Pregunta de cierre: solo aparece si pasan ESPERA_CIERRE segundos sin otra pregunta.
        ## Así no interrumpe una conversación fluida. No se guarda en el historial.
        hist = st.session_state["sapy_historial"]
        ultimo = hist[-1] if hist else None
        quieto = time.time() - st.session_state.get("sapy_ultima_resp", time.time()) >= ESPERA_CIERRE
        if not (quieto and ultimo and ultimo["rol"] == "assistant" and not ultimo.get("fin")
                and st.session_state.get("sapy_sin_cierre") != len(hist)):
            return
        with st.chat_message("assistant", avatar=str(AVATAR)):
            st.markdown(tr("¿Es todo o necesitas ayuda con otra cosa?",
                           "Is that all, or do you need help with something else?"))
            b1, b2 = st.columns(2)
            fin = b1.button(tr("Es todo, gracias", "That's all, thanks"), key="sapy_fin")
            b2.button(tr("Tengo otra pregunta", "I have another question"), key="sapy_otra",
                      on_click=_seguir)
            if fin:
                despedida = tr("¡Con gusto! Aquí estaré cuando necesites algo más. 👋",
                               "You're welcome! I'll be here whenever you need anything else. 👋")
                st.session_state["sapy_historial"] = (hist + [{"rol": "assistant", "texto": despedida,
                                                              "fin": True}])[-40:]
                st.markdown(despedida)
                time.sleep(1.6)
                st.rerun()                     # recarga la app: el chat se cierra

    with st.container(key="sapy_fab"):
        if st.button("Sapy", key="sapy_abrir", help=tr("Pregúntale a Sapy", "Ask Sapy")):
            panel()
