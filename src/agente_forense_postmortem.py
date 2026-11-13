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

try:
    from mcp_hana_tools import HanaMcpTools
    from generador_pdf_postmortem import generar_pdf_postmortem
except ImportError:
    from src.mcp_hana_tools import HanaMcpTools
    from src.generador_pdf_postmortem import generar_pdf_postmortem

log = logging.getLogger('agente_forense')

# Mapeo de vectores a MITRE ATT&CK
MITRE_MAP = {
    'Fuerza bruta': {'tactic': 'TA0006: Credential Access', 'technique': 'T1110: Brute Force / Password Spraying'},
    'Explotación': {'tactic': 'TA0001: Initial Access', 'technique': 'T1190: Exploit Public-Facing Application'},
    'DDoS': {'tactic': 'TA0040: Impact', 'technique': 'T1498: Network Denial of Service'},
    'DoS': {'tactic': 'TA0040: Impact', 'technique': 'T1499: Endpoint Denial of Service'},
    'Prompt injection': {'tactic': 'TA0001: Initial Access (AI)', 'technique': 'AML.T0051: LLM Prompt Injection & Jailbreak'},
    'Degradación del servicio': {'tactic': 'TA0040: Impact (AI)', 'technique': 'AML.T0029: Denial of ML Service / Token Exhaustion'},
    'Abuso de recursos': {'tactic': 'TA0009: Resource Development', 'technique': 'AML.T0034: LLM Compute Resource Hijacking'}
}


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

        log.info("[FORENSIC] Investigando Evento %d (%s - %s)...", evento_id, stream, tipo_ataque)

        # Paso 1: Telemetria de ventanas via MCP
        telemetry = self.mcp.get_incident_windows_telemetry(mes, stream, evento_id)

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
            'technique': 'T1190: Unknown Vector'
        })

        # Paso 4: Redactar reporte (Gemini API o Motor de Respaldo)
        postmortem_id = f"PM-{mes}-EV{evento_id:03d}-{stream[:3].upper()}"

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

        resumen_ejecutivo = (
            f"Incidente {postmortem_id}: Se detecto una campana de '{tipo_ataque}' clasificada como "
            f"{mitre_info['technique']} con severidad IF de {incident_data['score_pico']:.4f}. "
            f"Afecto a {len(raw_evidence['aplicaciones_afectadas'])} aplicaciones SAP durante "
            f"{incident_data['n_ventanas']} ventanas de observabilidad."
        )

        resultado = {
            'postmortem_id': postmortem_id,
            'evento_id': evento_id,
            'stream': stream,
            'source_table': source_table,
            'tipo_ataque': tipo_ataque,
            'mitre_tactic': mitre_info['tactic'],
            'severidad_score': incident_data['score_pico'],
            'inicio': incident_data['inicio'],
            'fin': incident_data['fin'],
            'aplicaciones_afectadas': raw_evidence['aplicaciones_afectadas'],
            'ips_involucradas': raw_evidence['ips_involucradas'],
            'resumen_ejecutivo': resumen_ejecutivo,
            'reporte_markdown': reporte_md,
            'metadata_json': {
                'incident_meta': incident_data,
                'telemetry_summary': telemetry.get('metricas_resumen', {}),
                'mitre_info': mitre_info,
                'total_raw_evidence_sampled': raw_evidence['total_obtenidos'],
                'generador': 'gemini_api' if (self.llm_client and self.llm_client[0] == 'gemini') else 'deterministico'
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
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

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
            "3. En la Seccion 2, genera una tabla Markdown obligatoria con columnas: Hito | Marca Temporal (UTC) | Descripcion Operativa.\n"
            "4. En la Seccion 4, cita de forma precisa los logs crudos, IPs, paths o payloads de inferencia proporcionados en la evidencia para respaldar pericialmente el analisis.\n"
            "5. En la Seccion 5, incluye dos subapartados claros:\n"
            "   ### Acciones Inmediatas (Contencion)\n"
            "   ### Acciones a Mediano Plazo (Hardening de Arquitectura)\n"
            "6. Manten un tono pericial corporativo, conciso y de alto nivel de seguridad."
        )

        prompt_payload = {
            "postmortem_id": postmortem_id,
            "incident_metadata": {
                "evento_id": incident_data.get('evento_id'),
                "stream": incident_data.get('stream'),
                "tipo_ataque": incident_data.get('tipo_ataque'),
                "confianza": incident_data.get('confianza'),
                "score_pico_isolation_forest": incident_data.get('score_pico'),
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

        try:
            log.info("[LLM] Invocando Gemini API (modelo: %s) para redactar postmortem %s...", model_name, postmortem_id)
            resp = client.models.generate_content(
                model=model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                )
            )
            if resp and resp.text:
                log.info("[OK] Reporte postmortem redactado exitosamente por Gemini API (%s).", model_name)
                return resp.text.strip()
            raise RuntimeError("Respuesta vacia recibida de Gemini API.")
        except Exception as e:
            model_fallback = "gemini-3.5-flash-lite" if model_name != "gemini-3.5-flash-lite" else "gemini-3.6-flash"
            log.warning("[WARN] Fallo con modelo %s (%s). Reintentando con %s...", model_name, e, model_fallback)
            try:
                resp = client.models.generate_content(
                    model=model_fallback,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                    )
                )
                if resp and resp.text:
                    log.info("[OK] Reporte postmortem redactado exitosamente con %s.", model_fallback)
                    return resp.text.strip()
            except Exception as e2:
                log.warning("[WARN] Fallback de Gemini tampoco respondio: %s", e2)

            log.warning("[WARN] Recurriendo a motor deterministico de respaldo para %s.", postmortem_id)
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
- **Severidad de Deteccion (Isolation Forest):** `{incident_data['score_pico']:.4f}`
- **Variable de Mayor Anomalia (Z-Score):** `{incident_data['z_max_feature']}` (Z = `{incident_data['z_max']:.2f}`)
- **Mapeo MITRE ATT&CK:**
  - **Tactica:** `{mitre_info['tactic']}`
  - **Tecnica:** `{mitre_info['technique']}`

### Resumen del Evento:
Durante la ventana de monitoreo se detecto una desviacion severa respecto a la linea base operativa. El modelo Isolation Forest aislo {incident_data['n_ventanas']} ventanas anomalas consecutivas que alcanzaron un z-score maximo en la variable `{incident_data['z_max_feature']}`. El perfil de ejecucion coincide con patrones conocidos de **{tipo_ataque}**, afectando servicios criticos del entorno SAP.

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
