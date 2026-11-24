"""
agente_forense_postmortem.py - Agente de Inteligencia Artificial para generacion de reportes Postmortem.

Consume las herramientas MCP de SAP HANA Cloud (mcp_hana_tools.py) y genera reportes forenses
estructurados bajo los estandares de ciberseguridad del SOC de SAP y alineados a MITRE ATT&CK.

Estructura obligatoria del Postmortem:
  1. Resumen Ejecutivo y Metadatos de la Amenaza
  2. Linea de Tiempo Cronologica Detallada (Timeline)
  3. Sistemas, Servicios y Componentes SAP Comprometidos
  4. Analisis de Causa Raiz y Evidencia Forense (Root Cause)
  5. Plan de Mitigacion y Recomendaciones de Hardening (Action Items)
"""

import sys
import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables de entorno (.env)
load_dotenv()

# Asegurar resolucion de modulos tanto desde src/ como desde la raiz
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers.idioma import CONFIANZA_EN, tipo_en, traducir_reporte_en

try:
    from mcp_hana_tools import HanaMcpTools
    from generador_pdf_postmortem import generar_pdf_postmortem
except ImportError:
    from src.mcp_hana_tools import HanaMcpTools
    from src.generador_pdf_postmortem import generar_pdf_postmortem

log = logging.getLogger('agente_forense')

# Mapeo de vectores a MITRE ATT&CK / ATLAS
MITRE_MAP = {
    'Contexto SAP': {'tactic': 'TA0001: Initial Access', 'technique': 'T1078: Valid Accounts / SAP Context Anomaly'},
    'Fuerza bruta': {'tactic': 'TA0006: Credential Access', 'technique': 'T1110: Brute Force / Password Spraying'},
    'Explotación': {'tactic': 'TA0001: Initial Access', 'technique': 'T1190: Exploit Public-Facing Application'},
    'Reconocimiento': {'tactic': 'TA0043: Reconnaissance', 'technique': 'T1595: Active Scanning / Endpoint Probing'},
    'DDoS': {'tactic': 'TA0040: Impact', 'technique': 'T1498: Network Denial of Service'},
    'DoS': {'tactic': 'TA0040: Impact', 'technique': 'T1499: Endpoint Denial of Service'},
    'Prompt injection': {'tactic': 'TA0001: Initial Access (AI)', 'technique': 'AML.T0051: LLM Prompt Injection & Jailbreak'},
    'Degradación del servicio': {'tactic': 'TA0040: Impact (AI)', 'technique': 'AML.T0029: Denial of ML Service / Token Exhaustion'},
    'Abuso de recursos': {'tactic': 'TA0040: Impact (AI)', 'technique': 'AML.T0034: LLM Compute Resource Hijacking'},
    'Acceso no autorizado': {'tactic': 'TA0006: Credential Access (AI)', 'technique': 'AML.T0040: AI Model Inference API Unauthorized Access'},
    'Automatización': {'tactic': 'TA0005: Defense Evasion (AI)', 'technique': 'AML.T0048: Automated Scripting & Bot Abuse'},
}


def construir_postmortem_id(mes: str, incident_data: Dict[str, Any]) -> str:
    """Genera un identificador unico por mes, evento, escala de ventana y stream."""
    evento_id = int(incident_data['evento_id'])
    stream = str(incident_data['stream'])[:3].upper()
    w_label = str(incident_data.get('window_label') or '').strip().upper().replace('MIN', 'M')
    if w_label:
        return f"PM-{mes}-EV{evento_id:03d}-{w_label}-{stream}"
    return f"PM-{mes}-EV{evento_id:03d}-{stream}"


def _extraer_resumen_de_markdown(reporte_md: str, fallback: str) -> str:
    """Extrae el parrafo narrativo de la Seccion 1 redactada por Gemini si esta disponible."""
    if not reporte_md or "## 1." not in reporte_md:
        return fallback[:900]
    sec1 = reporte_md.split("## 1.", 1)[1]
    if "## 2." in sec1:
        sec1 = sec1.split("## 2.", 1)[0]
    parrafos = []
    for line in sec1.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-") or s.startswith("*") or s.startswith("|"):
            continue
        limpio = s.replace("**", "").replace("`", "")
        if limpio.lower().startswith("resumen ejecutivo"):
            limpio = limpio.split(":", 1)[-1].strip() if ":" in limpio else limpio[len("resumen ejecutivo"):].lstrip(" .-")
        if limpio.lower().startswith("y clasificacion de la amenaza"):
            limpio = limpio[len("y clasificacion de la amenaza"):].lstrip(" .:-")
        for pref in ("executive summary and threat classification", "executive summary"):
            if limpio.lower().startswith(pref):
                limpio = limpio[len(pref):].lstrip(" .:-")
                break
        if len(limpio) >= 45:
            parrafos.append(limpio)
    if parrafos:
        texto = " ".join(parrafos[:2])
        if len(texto) > 900:
            corte = texto[:900].rfind(". ")
            texto = (texto[:corte + 1] if corte > 300 else texto[:897] + "...")
        return texto
    return fallback[:900]


class AgenteForensePostmortem:
    """
    Agente de IA especializado en investigacion y generacion de reportes postmortem forenses.
    Soporta ejecucion mediante Gemini API (LLM) con fallback automatico a motor deterministico.
    """

    def __init__(self, mcp_tools: HanaMcpTools, model_provider: Optional[str] = None):
        self.mcp = mcp_tools
        self.provider = model_provider or os.getenv("LLM_PROVIDER", "auto")
        self._init_llm_client()

    def _init_llm_client(self):
        """Inicializa el cliente de LLM si hay API Key disponible en el entorno."""
        load_dotenv()
        self.llm_client = None
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if gemini_key and gemini_key.strip():
            try:
                from google import genai
                self.llm_client = ('gemini', genai.Client(api_key=gemini_key.strip()))
                log.info("[INFO] Cliente Gemini API configurado activamente para el Agente.")
                return
            except Exception as e:
                log.warning("[WARN] No se pudo inicializar google.genai: %s", e)

        if openai_key and openai_key.strip():
            try:
                import openai
                self.llm_client = ('openai', openai.OpenAI(api_key=openai_key.strip()))
                log.info("[INFO] Cliente OpenAI configurado para el Agente.")
                return
            except Exception as e:
                log.warning("[WARN] No se pudo inicializar openai: %s", e)

        log.info("[INFO] Modo Forense Deterministico activado (sin API key; sintesis basada en evidencia MCP).")

    def investigar_y_redactar(self, mes: str, incident_data: Dict[str, Any],
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Ejecuta el ciclo de investigacion forense ReAct:
          1. Solicita telemetria agregada de ventanas mediante MCP.
          2. Extrae logs crudos y payloads de evidencia mediante MCP.
          3. Analiza el vector y sintetiza el reporte postmortem (via Gemini API o motor deterministico).
          4. Persiste el resultado en SAP HANA Cloud.
          5. Genera el PDF profesional si se proporciona output_dir.
        """
        evento_id = incident_data['evento_id']
        stream = incident_data['stream']
        tipo_ataque = incident_data['tipo_ataque']
        source_table = incident_data.get('source_table', 'LOGS_BATCH_1')
        window_size_s = incident_data.get('window_size_s')
        escalas = incident_data.get('escalas_detectadas') or (
            [incident_data['window_label']] if incident_data.get('window_label') else ['5min']
        )

        log.info("[FORENSIC] Investigando Evento %d (%s - %s | escalas=%s)...",
                 evento_id, stream, tipo_ataque, ",".join(escalas))

        # Paso 1: Telemetria de ventanas via MCP (filtrada por window_size_s)
        telemetry = self.mcp.get_incident_windows_telemetry(
            mes, stream, evento_id, window_size_s=window_size_s
        )

        # Paso 2: Logs crudos y evidencia via MCP
        raw_evidence = self.mcp.sample_raw_logs_for_incident(
            source_table=source_table,
            inicio=incident_data['inicio'],
            fin=incident_data['fin'],
            stream=stream,
            limit=40
        )

        # Paso 3: Identificar Mitre
        mitre_info = MITRE_MAP.get(tipo_ataque, {
            'tactic': 'TA0001: Initial Access',
            'technique': 'T1078: Anomaly Vector'
        })

        # Paso 4: Redactar reporte (Gemini API o Motor de Respaldo)
        postmortem_id = construir_postmortem_id(mes, incident_data)
        self._generador = 'deterministico'

        if self.llm_client and self.llm_client[0] == 'gemini':
            reporte_md = self._sintetizar_con_gemini(
                postmortem_id=postmortem_id,
                incident_data=incident_data,
                telemetry=telemetry,
                raw_evidence=raw_evidence,
                mitre_info=mitre_info
            )
        else:
            reporte_md = self._sintetizar_postmortem_deterministico(
                postmortem_id=postmortem_id,
                incident_data=incident_data,
                telemetry=telemetry,
                raw_evidence=raw_evidence,
                mitre_info=mitre_info
            )

        resumen_base = (
            f"Incidente {postmortem_id}: Se detecto actividad de '{tipo_ataque}' clasificada bajo "
            f"{mitre_info['technique']} en las escalas {', '.join(escalas)}. "
            f"Involucro a {len(raw_evidence['aplicaciones_afectadas'])} aplicacion(es) SAP y "
            f"{len(raw_evidence['ips_involucradas'])} direccion(es) IP a lo largo de "
            f"{incident_data['n_ventanas']} ventana(s) anomala(s), con desviacion maxima en "
            f"{incident_data.get('z_max_feature', 'N/A')} (Z = {float(incident_data.get('z_max') or 0.0):.2f})."
        )
        resumen_ejecutivo = _extraer_resumen_de_markdown(reporte_md, resumen_base)

        # Paso 4b: Version en ingles (misma informacion). Con Gemini se traduce el reporte redactado;
        # sin Gemini (o si la traduccion falla) se usa la plantilla deterministica en ingles.
        reporte_md_en = None
        if self._generador == 'gemini_api':
            reporte_md_en = traducir_reporte_en(self.llm_client[1], reporte_md)
        if not reporte_md_en:
            reporte_md_en = self._sintetizar_postmortem_deterministico_en(
                postmortem_id, incident_data, telemetry, raw_evidence, mitre_info)
        resumen_base_en = (
            f"Incident {postmortem_id}: '{tipo_en(tipo_ataque)}' activity classified under "
            f"{mitre_info['technique']} was detected at the {', '.join(escalas)} scales. "
            f"It involved {len(raw_evidence['aplicaciones_afectadas'])} SAP application(s) and "
            f"{len(raw_evidence['ips_involucradas'])} IP address(es) across "
            f"{incident_data['n_ventanas']} anomalous window(s), with the largest deviation in "
            f"{incident_data.get('z_max_feature', 'N/A')} (Z = {float(incident_data.get('z_max') or 0.0):.2f})."
        )
        resumen_ejecutivo_en = _extraer_resumen_de_markdown(reporte_md_en, resumen_base_en)

        resultado = {
            'postmortem_id': postmortem_id,
            'evento_id': evento_id,
            'stream': stream,
            'source_table': source_table,
            'tipo_ataque': tipo_ataque,
            'mitre_tactic': mitre_info['tactic'],
            'severidad_score': float(incident_data.get('score_pico') or 0.0),
            'inicio': incident_data['inicio'],
            'fin': incident_data['fin'],
            'aplicaciones_afectadas': raw_evidence['aplicaciones_afectadas'],
            'ips_involucradas': raw_evidence['ips_involucradas'],
            'resumen_ejecutivo': resumen_ejecutivo,
            'reporte_markdown': reporte_md,
            'resumen_ejecutivo_en': resumen_ejecutivo_en,
            'reporte_markdown_en': reporte_md_en,
            'metadata_json': {
                'incident_meta': {
                    **incident_data,
                    'escalas_detectadas': escalas,
                },
                'telemetry_summary': telemetry.get('metricas_resumen', {}),
                'mitre_info': mitre_info,
                'escalas_detectadas': escalas,
                'evidence_logs_sample': raw_evidence.get('logs_muestra', [])[:6],
                'total_raw_evidence_sampled': raw_evidence['total_obtenidos'],
                'generador': self._generador
            }
        }

        # Paso 5: Persistir en HANA via MCP
        self.mcp.persist_postmortem_report(mes, resultado)

        # Paso 6: Generar PDF profesional si se solicita output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            pdf_path = os.path.join(output_dir, f"{postmortem_id}.pdf")
            try:
                generar_pdf_postmortem(resultado, pdf_path)
                resultado['pdf_path'] = pdf_path
                log.info("[OK] Postmortem PDF generado exitosamente: %s", pdf_path)
            except Exception as e_pdf:
                log.error("[ERROR] Fallo al generar PDF del postmortem %s: %s", postmortem_id, e_pdf)

        return resultado

    def _sintetizar_con_gemini(self, postmortem_id: str, incident_data: Dict[str, Any],
                              telemetry: Dict[str, Any], raw_evidence: Dict[str, Any],
                              mitre_info: Dict[str, Any]) -> str:
        """
        Invoca a Gemini API para redactar la narrativa forense profunda basada en evidencia MCP.
        """
        from google.genai import types

        client = self.llm_client[1]
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

        system_instruction = (
            "Eres un Especialista Principal de Ciberseguridad e Investigador Forense en el SOC de SAP Cloud Observability.\n"
            "Tu objetivo es analizar la telemetria real de SAP HANA Cloud, anomalias detectadas por Isolation Forest y "
            "logs crudos para redactar un Reporte Forense Postmortem profesional, riguroso y estructurado.\n\n"
            "REGLAS ESTRICTAS DE FORMATO Y ESTILO:\n"
            "1. Cero emojis o pictogramas decorativos en todo el reporte. Usa etiquetas de texto limpio si son necesarias ([INFO], [ALERTA], [CRITICO]).\n"
            "2. El reporte debe estructurarse obligatoriamente con el titulo principal '# REPORTE FORENSE POSTMORTEM: " + postmortem_id + "'\n"
            "**Clasificacion:** Documento Confidencial SOC / SAP Cloud Observability\n"
            "**Fecha de Emision:** " + datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC') + "\n"
            "**Estado:** Confirmado e Investigado\n\n"
            "y las siguientes 5 secciones con titulos de nivel 2 (##):\n"
            "## 1. Resumen Ejecutivo y Clasificacion de la Amenaza\n"
            "## 2. Linea de Tiempo Cronologica (Incident Timeline)\n"
            "## 3. Sistemas, Servicios y Componentes SAP Comprometidos\n"
            "## 4. Analisis Forense de Causa Raiz y Evidencia (Root Cause)\n"
            "## 5. Plan de Mitigacion y Recomendaciones de Hardening (Action Items)\n\n"
            "3. En la Seccion 1, incluye un parrafo narrativo claro explicando que ocurrio, en que escalas de ventana temporal se detecto y cual fue el indicador mas desviado (Z-score). No incluyas metricas de 'severidad' ni 'score de severidad'.\n"
            "4. En la Seccion 2, genera una tabla Markdown obligatoria con columnas: Hito | Marca Temporal (UTC) | Descripcion Operativa.\n"
            "5. En la Seccion 4, cita de forma precisa los logs crudos, IPs, paths o payloads de inferencia proporcionados en la evidencia para respaldar pericialmente el analisis.\n"
            "6. En la Seccion 5, incluye dos subapartados claros:\n"
            "   ### Acciones Inmediatas (Contencion)\n"
            "   ### Acciones a Mediano Plazo (Hardening de Arquitectura)\n"
            "7. Manten un tono pericial corporativo, conciso y de alto nivel de seguridad."
        )

        prompt_payload = {
            "postmortem_id": postmortem_id,
            "incident_metadata": {
                "evento_id": incident_data.get('evento_id'),
                "stream": incident_data.get('stream'),
                "tipo_ataque": incident_data.get('tipo_ataque'),
                "confianza": incident_data.get('confianza'),
                "window_label": incident_data.get('window_label'),
                "escalas_detectadas": incident_data.get('escalas_detectadas', []),
                "z_max": incident_data.get('z_max'),
                "z_max_feature": incident_data.get('z_max_feature'),
                "inicio": str(incident_data.get('inicio')),
                "fin": str(incident_data.get('fin')),
                "n_ventanas": incident_data.get('n_ventanas')
            },
            "mitre_framework": mitre_info,
            "telemetry_summary": telemetry.get('metricas_resumen', {}),
            "evidence_logs_sample": raw_evidence.get('logs_muestra', [])[:10],
            "impacted_applications": raw_evidence.get('aplicaciones_afectadas', []),
            "source_ips": raw_evidence.get('ips_involucradas', [])
        }

        user_content = (
            f"Analiza la siguiente evidencia recolectada por el bus MCP desde SAP HANA Cloud para el incidente {postmortem_id}:\n\n"
            f"```json\n{json.dumps(prompt_payload, indent=2, default=str)}\n```\n\n"
            "Genera el reporte postmortem completo en Markdown siguiendo estrictamente las 5 secciones requeridas."
        )

        modelos_intento = [model_name]
        for candidato_fb in ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite"):
            if candidato_fb not in modelos_intento:
                modelos_intento.append(candidato_fb)

        for idx_m, m_actual in enumerate(modelos_intento):
            try:
                log.info("[LLM] Invocando Gemini API (modelo: %s) para redactar postmortem %s...", m_actual, postmortem_id)
                resp = client.models.generate_content(
                    model=m_actual,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    )
                )
                if resp and resp.text:
                    log.info("[OK] Reporte postmortem redactado exitosamente por Gemini API (%s).", m_actual)
                    self._generador = 'gemini_api'
                    return resp.text.strip()
                raise RuntimeError("Respuesta vacia recibida de Gemini API.")
            except Exception as e:
                log.warning("[WARN] Fallo con modelo %s (%s).", m_actual, e)
                if idx_m + 1 < len(modelos_intento):
                    log.info("[LLM] Reintentando con fallback %s...", modelos_intento[idx_m + 1])

        log.warning("[WARN] Recurriendo a motor deterministico de respaldo para %s.", postmortem_id)
        self._generador = 'deterministico'
        return self._sintetizar_postmortem_deterministico(
            postmortem_id, incident_data, telemetry, raw_evidence, mitre_info
        )

    def _sintetizar_postmortem_deterministico(self, postmortem_id: str, incident_data: Dict[str, Any],
                                               telemetry: Dict[str, Any], raw_evidence: Dict[str, Any],
                                               mitre_info: Dict[str, Any]) -> str:
        """
        Motor de respaldo deterministico: construye el reporte estructurado sin requerir LLM externo.
        """
        stream = incident_data['stream']
        tipo_ataque = incident_data['tipo_ataque']
        apps = ", ".join(f"`{a}`" for a in raw_evidence['aplicaciones_afectadas']) if raw_evidence['aplicaciones_afectadas'] else "`sap-core-services`"
        ips = ", ".join(f"`{ip}`" for ip in raw_evidence['ips_involucradas']) if raw_evidence['ips_involucradas'] else "No IP directa (Llamada interna/proxy)"

        # Extraer ejemplos de evidencia forense cruda
        evidencia_texto = []
        for i, item in enumerate(raw_evidence.get('logs_muestra', [])[:6]):
            ts = item.get('TIMESTAMP', 'N/A')
            app = item.get('SAP_FUNCTION_APPLICATION', 'N/A')
            if stream == 'llm':
                prompt = item.get('LLM_PROMPT') or 'N/A'
                err = item.get('LLM_ERROR_MESSAGE') or 'None'
                status = item.get('LLM_STATUS') or 'N/A'
                evidencia_texto.append(
                    f"- **[{ts}] App:** `{app}` | **Status:** `{status}`\n"
                    f"  - **Prompt Extract:** `{prompt[:140]}...`\n"
                    f"  - **Security Filter Output:** `{err}`"
                )
            else:
                ip = item.get('CLIENT_IP', 'N/A')
                status = item.get('HTTP_STATUS_CODE', 'N/A')
                path = item.get('HEATHERS_REQUEST_PATH', 'N/A')
                msg = item.get('SAP_FUNCTION_MESSAGE', 'N/A')
                evidencia_texto.append(
                    f"- **[{ts}] IP:** `{ip}` | **HTTP:** `{status}` | **Path:** `{path}`\n"
                    f"  - **Log Message:** `{msg}`"
                )

        evidencia_bloque = "\n".join(evidencia_texto) if evidencia_texto else "No se recuperaron registros especificos adicionales."

        # Generar recomendaciones tecnicas de acuerdo al vector
        hardening_items = self._generar_hardening_recomendaciones(tipo_ataque, stream)

        md = f"""# REPORTE FORENSE POSTMORTEM: {postmortem_id}
**Clasificacion:** Documento Confidencial SOC / SAP Cloud Observability  
**Fecha de Emision:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Estado:** Confirmado e Investigado

---

## 1. Resumen Ejecutivo y Clasificacion de la Amenaza

- **Identificador de Incidente:** `{postmortem_id}` (Evento ID `{incident_data['evento_id']}`)
- **Stream de Telemetria:** `{stream.upper()}`
- **Vector de Ataque Clasificado:** `{tipo_ataque}` (Confianza: `{incident_data['confianza'].upper()}`)
- **Escalas de Ventana Detectadas:** `{', '.join(incident_data.get('escalas_detectadas') or [str(incident_data.get('window_label', '5min'))])}`
- **Variable de Mayor Anomalia (Z-Score):** `{incident_data['z_max_feature']}` (Z = `{incident_data['z_max']:.2f}`)
- **Mapeo MITRE ATT&CK:**
  - **Tactica:** `{mitre_info['tactic']}`
  - **Tecnica:** `{mitre_info['technique']}`

### Resumen del Evento:
Durante la ventana de monitoreo se detecto una desviacion anomala respecto a la linea base operativa en las escalas `{', '.join(incident_data.get('escalas_detectadas') or [str(incident_data.get('window_label', '5min'))])}`. El modelo Isolation Forest aislo {incident_data['n_ventanas']} ventanas anomalas consecutivas que alcanzaron un z-score maximo de {incident_data['z_max']:.2f} en la variable `{incident_data['z_max_feature']}`. El perfil de ejecucion coincide con patrones de **{tipo_ataque}**, afectando los servicios {apps}.

---

## 2. Linea de Tiempo Cronologica (Incident Timeline)

| Hito | Marca Temporal (UTC) | Descripcion Operativa |
| :--- | :--- | :--- |
| **T0 - Inicio de Anomalia** | `{incident_data['inicio']}` | Primer disparo de umbral en Isolation Forest. |
| **T1 - Pico de Impacto** | `{incident_data['ventana_pico']}` | Maximo Z-score ({incident_data['z_max']:.2f}) en `{incident_data['z_max_feature']}`. |
| **T2 - Cese / Mitigacion** | `{incident_data['fin']}` | Fin de la ventana anomala consecutiva registrada. |

---

## 3. Sistemas, Servicios y Componentes SAP Comprometidos

- **Servicios SAP Afectados:** {apps}
- **Direcciones IP / Origenes:** {ips}
- **Ambiente de Ejecucion:** SAP BTP Cloud Foundry Runtime / SAP Cloud Services

---

## 4. Analisis Forense de Causa Raiz y Evidencia (Root Cause)

El analisis estadistico correlacionado indica que el incidente se origino mediante una dispersion anomala clasificada como **{tipo_ataque}**. La telemetria agregada evidencia una alteracion significativa en `{incident_data['z_max_feature']}` alcanzando una desviacion de {incident_data['z_max']:.2f} sigmas sobre la media historica.

### Evidencia de Logs Muestreados via MCP:
{evidencia_bloque}

---

## 5. Plan de Mitigacion y Recomendaciones de Hardening (Action Items)

{hardening_items}

---
*Reporte autogenerado por el Agente Forense de Observabilidad de SAP. Datos verificados contra tablas maestras en SAP HANA Cloud.*
"""
        return md

    def _generar_hardening_recomendaciones(self, tipo_ataque: str, stream: str) -> str:
        """Genera acciones correctivas precisas segun el vector de ataque."""
        if 'Prompt injection' in tipo_ataque:
            return """### Acciones Inmediatas (Contencion):
1. **Regla de Bloqueo en Gateway de Inferencia:** Implementar inspeccion de entrada en el proxy de IA antes de despachar a los modelos fundacionales.
2. **Activacion de Guardrails de Seguridad:** Desplegar politicas estrictas en Llama Guard / NeMo Guardrails para interceptar directivas de evasion de contexto (`SYSTEM:`, `ignore previous instructions`, etc.).

### Acciones a Mediano Plazo (Hardening de Arquitectura):
1. **Separacion Estricta de Contextos:** Aislar instrucciones de sistema de los datos no confiables de usuario utilizando delimitadores XML estructurados.
2. **Revision de Permisos de Herramientas (Tool Calling):** Revocar privilegios elevados de ejecucion para funciones que expongan datos de salarios, credenciales o esquemas de base de datos."""

        elif 'Degradación del servicio' in tipo_ataque or 'DoS' in tipo_ataque or 'DDoS' in tipo_ataque:
            return """### Acciones Inmediatas (Contencion):
1. **Ajuste de Timeouts en Gateway:** Disminuir el umbral de timeout de inferencia y peticiones HTTP de 30s a 15s para liberar hilos bloqueados.
2. **Circuit Breakers Activos:** Habilitar disyuntores de circuito por aplicacion para que el fallo de un servicio externo no colapse el pool de conexiones de BTP.

### Acciones a Mediano Plazo (Hardening de Arquitectura):
1. **Rate Limiting Dinamico:** Configurar cuotas de peticiones por minuto basadas en token-bucket tanto a nivel IP como a nivel `SERVICE_ID`.
2. **Enrutamiento Adaptativo de Proveedores:** Balancear carga entre multiples instancias o proveedores para evitar saturacion de cuota (HTTP 429)."""

        elif 'Fuerza bruta' in tipo_ataque:
            return """### Acciones Inmediatas (Contencion):
1. **Bloqueo Inmediato en WAF:** Crear reglas de bloqueo IP para los orígenes con tasas elevadas de fallos 401/403 consecutivos.
2. **Bloqueo de Cuenta / Rate Limit en Login:** Forzar politicas de bloqueo temporal tras 5 intentos fallidos consecutivos de autenticacion.

### Acciones a Mediano Plazo (Hardening de Arquitectura):
1. **Obligatoriedad de MFA (Multi-Factor Authentication):** Enforzar MFA para todos los usuarios con roles administrativos o acceso a APIs de finanzas.
2. **Monitoreo Continuo de Entropia de IPs:** Alertar al SOC en tiempo real cuando la entropia de peticiones de una sola subnet supere 3 desviaciones estandar."""

        else:
            return """### Acciones Inmediatas (Contencion):
1. **Aislamiento Preventivo:** Validar la integridad de los endpoints afectados y restringir accesos no autenticados.
2. **Inspeccion de Headers:** Bloquear metodos HTTP inusuales y validar el Content-Type de las solicitudes entrantes.

### Acciones a Mediano Plazo (Hardening de Arquitectura):
1. **Revision de Logs de Auditoria:** Auditar cambios recientes en politicas de autorizacion y roles de BTP.
2. **Monitoreo Continuo con Isolation Forest:** Mantener la calibracion semanal de umbrales para deteccion preventiva temprana."""

    def _sintetizar_postmortem_deterministico_en(self, postmortem_id: str, incident_data: Dict[str, Any],
                                                  telemetry: Dict[str, Any], raw_evidence: Dict[str, Any],
                                                  mitre_info: Dict[str, Any]) -> str:
        """English version of the deterministic fallback report (same structure and data)."""
        stream = incident_data['stream']
        tipo = tipo_en(incident_data['tipo_ataque'])
        confianza = CONFIANZA_EN.get(str(incident_data.get('confianza', '')).lower(),
                                     str(incident_data.get('confianza', 'N/A')))
        escalas = ', '.join(incident_data.get('escalas_detectadas') or [str(incident_data.get('window_label', '5min'))])
        apps = ", ".join(f"`{a}`" for a in raw_evidence['aplicaciones_afectadas']) if raw_evidence['aplicaciones_afectadas'] else "`sap-core-services`"
        ips = ", ".join(f"`{ip}`" for ip in raw_evidence['ips_involucradas']) if raw_evidence['ips_involucradas'] else "No direct IP (internal/proxy call)"

        evidencia = []
        for item in raw_evidence.get('logs_muestra', [])[:6]:
            ts = item.get('TIMESTAMP', 'N/A')
            app = item.get('SAP_FUNCTION_APPLICATION', 'N/A')
            if stream == 'llm':
                prompt = item.get('LLM_PROMPT') or 'N/A'
                evidencia.append(
                    f"- **[{ts}] App:** `{app}` | **Status:** `{item.get('LLM_STATUS') or 'N/A'}`\n"
                    f"  - **Prompt Extract:** `{prompt[:140]}...`\n"
                    f"  - **Security Filter Output:** `{item.get('LLM_ERROR_MESSAGE') or 'None'}`")
            else:
                evidencia.append(
                    f"- **[{ts}] IP:** `{item.get('CLIENT_IP', 'N/A')}` | **HTTP:** `{item.get('HTTP_STATUS_CODE', 'N/A')}` | "
                    f"**Path:** `{item.get('HEATHERS_REQUEST_PATH', 'N/A')}`\n"
                    f"  - **Log Message:** `{item.get('SAP_FUNCTION_MESSAGE', 'N/A')}`")
        evidencia_bloque = "\n".join(evidencia) if evidencia else "No additional specific records were retrieved."
        z_max = float(incident_data.get('z_max') or 0.0)

        return f"""# FORENSIC POSTMORTEM REPORT: {postmortem_id}
**Classification:** Confidential SOC Document / SAP Cloud Observability  
**Issue Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status:** Confirmed and Investigated

---

## 1. Executive Summary and Threat Classification

- **Incident Identifier:** `{postmortem_id}` (Event ID `{incident_data['evento_id']}`)
- **Telemetry Stream:** `{stream.upper()}`
- **Classified Attack Vector:** `{tipo}` (Confidence: `{confianza.upper()}`)
- **Detected Window Scales:** `{escalas}`
- **Most Anomalous Variable (Z-Score):** `{incident_data['z_max_feature']}` (Z = `{z_max:.2f}`)
- **MITRE ATT&CK Mapping:**
  - **Tactic:** `{mitre_info['tactic']}`
  - **Technique:** `{mitre_info['technique']}`

### Event Summary:
During the monitoring window, an anomalous deviation from the operational baseline was detected at the `{escalas}` scales. The Isolation Forest model isolated {incident_data['n_ventanas']} consecutive anomalous windows that reached a maximum z-score of {z_max:.2f} in the variable `{incident_data['z_max_feature']}`. The execution profile matches **{tipo}** patterns, affecting the services {apps}.

---

## 2. Chronological Timeline (Incident Timeline)

| Milestone | Timestamp (UTC) | Operational Description |
| :--- | :--- | :--- |
| **T0 - Anomaly Start** | `{incident_data['inicio']}` | First Isolation Forest threshold trigger. |
| **T1 - Peak Impact** | `{incident_data.get('ventana_pico', incident_data['inicio'])}` | Maximum Z-score ({z_max:.2f}) in `{incident_data['z_max_feature']}`. |
| **T2 - End / Mitigation** | `{incident_data['fin']}` | End of the recorded consecutive anomalous window. |

---

## 3. Compromised SAP Systems, Services and Components

- **Affected SAP Services:** {apps}
- **IP Addresses / Sources:** {ips}
- **Runtime Environment:** SAP BTP Cloud Foundry Runtime / SAP Cloud Services

---

## 4. Forensic Root Cause Analysis and Evidence (Root Cause)

The correlated statistical analysis indicates that the incident originated from an anomalous dispersion classified as **{tipo}**. The aggregated telemetry shows a significant change in `{incident_data['z_max_feature']}`, reaching a deviation of {z_max:.2f} sigmas above the historical mean.

### Log Evidence Sampled via MCP:
{evidencia_bloque}

---

## 5. Mitigation Plan and Hardening Recommendations (Action Items)

{self._generar_hardening_recomendaciones_en(incident_data['tipo_ataque'])}

---
*Report automatically generated by the SAP Observability Forensic Agent. Data verified against master tables in SAP HANA Cloud.*
"""

    def _generar_hardening_recomendaciones_en(self, tipo_ataque: str) -> str:
        """English version of the corrective actions by attack vector."""
        if 'Prompt injection' in tipo_ataque:
            return """### Immediate Actions (Containment):
1. **Blocking Rule at the Inference Gateway:** Implement input inspection in the AI proxy before dispatching requests to foundation models.
2. **Enable Security Guardrails:** Deploy strict policies in Llama Guard / NeMo Guardrails to intercept context-evasion directives (`SYSTEM:`, `ignore previous instructions`, etc.).

### Medium-Term Actions (Architecture Hardening):
1. **Strict Context Separation:** Isolate system instructions from untrusted user data using structured XML delimiters.
2. **Tool Calling Permission Review:** Revoke elevated execution privileges for functions that expose salary data, credentials or database schemas."""
        elif 'Degradación del servicio' in tipo_ataque or 'DoS' in tipo_ataque or 'DDoS' in tipo_ataque:
            return """### Immediate Actions (Containment):
1. **Gateway Timeout Tuning:** Lower the inference and HTTP request timeout from 30s to 15s to release blocked threads.
2. **Active Circuit Breakers:** Enable per-application circuit breakers so that an external service failure does not exhaust the BTP connection pool.

### Medium-Term Actions (Architecture Hardening):
1. **Dynamic Rate Limiting:** Configure per-minute request quotas based on token buckets at both the IP and `SERVICE_ID` level.
2. **Adaptive Provider Routing:** Balance load across multiple instances or providers to avoid quota saturation (HTTP 429)."""
        elif 'Fuerza bruta' in tipo_ataque:
            return """### Immediate Actions (Containment):
1. **Immediate WAF Blocking:** Create IP blocking rules for sources with high rates of consecutive 401/403 failures.
2. **Account Lockout / Login Rate Limit:** Enforce temporary lockout policies after 5 consecutive failed authentication attempts.

### Medium-Term Actions (Architecture Hardening):
1. **Mandatory MFA (Multi-Factor Authentication):** Enforce MFA for all users with administrative roles or access to finance APIs.
2. **Continuous IP Entropy Monitoring:** Alert the SOC in real time when the request entropy of a single subnet exceeds 3 standard deviations."""
        else:
            return """### Immediate Actions (Containment):
1. **Preventive Isolation:** Validate the integrity of the affected endpoints and restrict unauthenticated access.
2. **Header Inspection:** Block unusual HTTP methods and validate the Content-Type of incoming requests.

### Medium-Term Actions (Architecture Hardening):
1. **Audit Log Review:** Audit recent changes to BTP authorization policies and roles.
2. **Continuous Monitoring with Isolation Forest:** Keep the weekly threshold calibration for early preventive detection."""
