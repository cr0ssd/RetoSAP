import logging
import os

log = logging.getLogger(__name__)

# Tipos de ataque (CLASIFICACION_TIPOS_ATAQUE.md) en inglés
TIPOS_EN = {
    "DoS": "DoS",
    "DDoS": "DDoS",
    "Fuerza bruta": "Brute force",
    "Reconocimiento": "Reconnaissance",
    "Explotación": "Exploitation",
    "Contexto SAP": "SAP context",
    "Falla operativa": "Operational failure",
    "Prompt injection": "Prompt injection",
    "Acceso no autorizado": "Unauthorized access",
    "Abuso de recursos": "Resource abuse",
    "Automatización": "Automation",
    "Degradación del servicio": "Service degradation",
}

CONFIANZA_EN = {"alta": "high", "media": "medium", "baja": "low"}

MODELOS_GEMINI = ("gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.1-flash-lite")

INSTRUCCION_TRADUCCION = (
    "You are a professional translator specialized in cybersecurity and SAP forensic reports.\n"
    "Translate the following Spanish SOC postmortem report into English.\n\n"
    "STRICT RULES:\n"
    "1. Keep exactly the same Markdown structure: headings, numbering (## 1. to ## 5.), bullet lists, "
    "bold text and tables (same columns and rows).\n"
    "2. Do NOT translate or change: incident IDs, IP addresses, URLs, paths, timestamps, numbers, "
    "Z-scores, feature/column names in UPPER_CASE, code in backticks, MITRE ATT&CK / ATLAS IDs "
    "(TA0001, T1110, AML.T0051...) or application names (sap-*).\n"
    "3. Translate only what is in the original: do not add, remove or reorder sections, headings, "
    "rows or content. Never create empty sections.\n"
    "   Heading glossary (use it only for headings that appear in the original):\n"
    "   'REPORTE FORENSE POSTMORTEM' = 'FORENSIC POSTMORTEM REPORT'; "
    "'Resumen Ejecutivo y Clasificacion de la Amenaza' = 'Executive Summary and Threat Classification'; "
    "'Linea de Tiempo Cronologica' = 'Chronological Timeline'; "
    "'Sistemas, Servicios y Componentes SAP Comprometidos' = 'Compromised SAP Systems, Services and Components'; "
    "'Analisis Forense de Causa Raiz y Evidencia' = 'Forensic Root Cause Analysis and Evidence'; "
    "'Plan de Mitigacion y Recomendaciones de Hardening' = 'Mitigation Plan and Hardening Recommendations'; "
    "'Acciones Inmediatas (Contencion)' = 'Immediate Actions (Containment)'; "
    "'Acciones a Mediano Plazo (Hardening de Arquitectura)' = 'Medium-Term Actions (Architecture Hardening)'.\n"
    "4. Translate attack types as: " + ", ".join(f"{k} -> {v}" for k, v in TIPOS_EN.items()) + ".\n"
    "5. Keep a professional, concise forensic tone. No emojis. Return only the translated Markdown."
)


def tipo_en(tipo):
    """Nombre del tipo de ataque en inglés (o el mismo texto si no está en la tabla)."""
    return TIPOS_EN.get(str(tipo), str(tipo))


def traducir_reporte_en(client, texto, modelo=None):
    """
    Traduce `texto` (Markdown en español) al inglés con Gemini. Prueba el modelo configurado y
    después los de respaldo. Devuelve el texto traducido o None si ningún modelo respondió.
    """
    if not texto or not str(texto).strip():
        return ""
    from google.genai import types

    modelos = [modelo or os.getenv("GEMINI_MODEL", MODELOS_GEMINI[0])]
    modelos += [m for m in MODELOS_GEMINI if m not in modelos]
    for m in modelos:
        try:
            resp = client.models.generate_content(
                model=m, contents=str(texto),
                config=types.GenerateContentConfig(system_instruction=INSTRUCCION_TRADUCCION,
                                                   temperature=0.1))
            if resp and resp.text and resp.text.strip():
                return resp.text.strip()
        except Exception as e:
            log.warning("[WARN] Traducción con %s falló: %s", m, e)
    return None
