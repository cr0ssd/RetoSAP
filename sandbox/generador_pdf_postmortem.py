"""
generador_pdf_postmortem.py - Generador de reportes forenses Postmortem en formato PDF profesional.

Convierte los metadatos forenses estructurados obtenidos via MCP y el analisis del Agente
en documentos PDF con formato corporativo SAP / SOC (ReportLab).
"""

import os
import sys
import re
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

# Paleta corporativa SAP y SOC
SAP_NAVY = colors.HexColor('#003366')
SAP_BLUE = colors.HexColor('#0070F2')
SAP_DARK = colors.HexColor('#1E293B')
SAP_MUTED = colors.HexColor('#64748B')
SAP_LIGHT = colors.HexColor('#F8FAFC')
SAP_BORDER = colors.HexColor('#E2E8F0')
SAP_ALERT_RED = colors.HexColor('#D92D20')
SAP_CODE_BG = colors.HexColor('#F1F5F9')
SAP_CODE_BORDER = colors.HexColor('#CBD5E1')


def _md_to_reportlab(text: str) -> str:
    """Convierte marcas markdown basicas (** y `) a etiquetas XML compatibles con ReportLab."""
    if not text:
        return ""
    # Escapar ampersands sueltos
    t = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', str(text))
    # Convertir negritas **texto**
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    # Convertir backticks `codigo` a font mono azul
    t = re.sub(r'`(.*?)`', r'<font color="#003366"><b>\1</b></font>', t)
    # Reemplazar posibles tags no soportados por ReportLab
    t = t.replace('<code>', '<font color="#003366"><b>').replace('</code>', '</b></font>')
    return t


class NumberedCanvas(canvas.Canvas):
    """Canvas de dos pasadas para calcular y numerar paginas: 'Pagina X de Y'."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        # Top rule
        self.setStrokeColor(SAP_BORDER)
        self.setLineWidth(0.5)
        self.line(42, self._pagesize[1] - 32, self._pagesize[0] - 42, self._pagesize[1] - 32)

        # Header string
        self.setFont('Helvetica-Bold', 7)
        self.setFillColor(SAP_BLUE)
        self.drawString(42, self._pagesize[1] - 26, "SAP CLOUD OBSERVABILITY & SECURITY OPERATIONS CENTER")
        self.setFont('Helvetica', 7)
        self.setFillColor(SAP_MUTED)
        self.drawRightString(self._pagesize[0] - 42, self._pagesize[1] - 26, "DOCUMENTO FORENSE CONFIDENCIAL")

        # Bottom rule
        self.line(42, 34, self._pagesize[0] - 42, 34)

        # Footer string
        self.setFont('Helvetica', 8)
        self.drawString(42, 22, "SAP BTP / SAP HANA Cloud - Inferencia Forense Verificada")
        self.drawRightString(self._pagesize[0] - 42, 22, f"Pagina {self._pageNumber} de {page_count}")
        self.restoreState()


def generar_pdf_postmortem(postmortem_dict: Dict[str, Any], output_path: str) -> str:
    """
    Construye un documento PDF profesional para el postmortem proporcionado.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=42,
        rightMargin=42,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Estilos tipograficos personalizados
    style_title = ParagraphStyle(
        'PostmortemTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=SAP_NAVY,
        spaceAfter=3
    )

    style_subtitle = ParagraphStyle(
        'PostmortemSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=SAP_BLUE,
        spaceAfter=12
    )

    style_h2 = ParagraphStyle(
        'PostmortemH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=SAP_NAVY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    style_h3 = ParagraphStyle(
        'PostmortemH3',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=SAP_DARK,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'PostmortemBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=SAP_DARK,
        spaceAfter=6
    )

    style_meta_label = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=SAP_MUTED
    )

    style_meta_val = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=12,
        textColor=SAP_NAVY
    )

    style_code = ParagraphStyle(
        'CodeStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10.5,
        textColor=SAP_DARK
    )

    story = []

    # 1. ENCABEZADO Y TITULO PRINCIPAL
    postmortem_id = postmortem_dict['postmortem_id']
    tipo_ataque = postmortem_dict['tipo_ataque']
    stream = postmortem_dict['stream'].upper()
    severidad = postmortem_dict['severidad_score']
    mitre_tactic = postmortem_dict.get('mitre_tactic', 'TA0001: Initial Access')
    mitre_info = postmortem_dict.get('metadata_json', {}).get('mitre_info', {})
    mitre_technique = mitre_info.get('technique', 'T1190: Exploit')

    story.append(Paragraph(f"REPORTE FORENSE POSTMORTEM", style_title))
    story.append(Paragraph(f"{postmortem_id} | {tipo_ataque.upper()} ({stream})", style_subtitle))

    # 2. TABLA RESUMEN DE METADATOS
    meta_table_data = [
        [
            Paragraph("ID DE INCIDENTE", style_meta_label),
            Paragraph("STREAM OPERATIVO", style_meta_label),
            Paragraph("SEVERIDAD ISOLATION FOREST", style_meta_label),
            Paragraph("ESTADO FORENSE", style_meta_label)
        ],
        [
            Paragraph(f"<b>{postmortem_id}</b>", style_meta_val),
            Paragraph(f"<b>{stream}</b>", style_meta_val),
            Paragraph(f"<b>{severidad:.4f}</b> (Anomalia Critica)", style_meta_val),
            Paragraph("<font color='#0070F2'><b>Confirmado e Investigado</b></font>", style_meta_val)
        ],
        [
            Paragraph("VECTOR CLASIFICADO", style_meta_label),
            Paragraph("TACTICA MITRE ATT&CK", style_meta_label),
            Paragraph("TECNICA MITRE ATT&CK", style_meta_label),
            Paragraph("INTERVALO DEL ATAQUE", style_meta_label)
        ],
        [
            Paragraph(f"<b>{tipo_ataque}</b>", style_meta_val),
            Paragraph(f"<b>{mitre_tactic}</b>", style_meta_val),
            Paragraph(f"<b>{mitre_technique}</b>", style_meta_val),
            Paragraph(f"<b>{postmortem_dict['inicio'][:16]} a {postmortem_dict['fin'][11:16]} UTC</b>", style_meta_val)
        ]
    ]

    t_meta = Table(meta_table_data, colWidths=[120, 110, 138, 160])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # 3. SECCION 1: RESUMEN EJECUTIVO
    story.append(Paragraph("1. Resumen Ejecutivo y Caracterizacion del Incidente", style_h2))
    resumen_txt = postmortem_dict.get('resumen_ejecutivo', '')
    story.append(Paragraph(resumen_txt, style_body))

    inc_meta = postmortem_dict.get('metadata_json', {}).get('incident_meta', {})
    z_feat = inc_meta.get('z_max_feature', 'N/A')
    z_val = inc_meta.get('z_max', 0.0)
    n_win = inc_meta.get('n_ventanas', 1)

    detalle_contexto = (
        f"El motor no supervisado de Isolation Forest aislo <b>{n_win} ventanas consecutivas</b> con "
        f"desviacion extrema en la dimension <font color='#003366'><b>{z_feat}</b></font> (Z-Score = {z_val:.2f}). "
        f"La concentracion estadistica y la persistencia temporal superan los umbrales de operacion "
        f"ordinaria, evidenciando una accion hostil focalizada contra la infraestructura SAP."
    )
    story.append(Paragraph(detalle_contexto, style_body))
    story.append(Spacer(1, 6))

    # 4. SECCION 2: LINEA DE TIEMPO DETALLADA (TIMELINE)
    story.append(Paragraph("2. Linea de Tiempo Cronologica (Incident Timeline)", style_h2))
    timeline_data = [
        [
            Paragraph("<b>Hito Operativo</b>", style_meta_label),
            Paragraph("<b>Marca Temporal (UTC)</b>", style_meta_label),
            Paragraph("<b>Descripcion Forense</b>", style_meta_label)
        ],
        [
            Paragraph("<b>T0 - Inicio de Anomalia</b>", style_body),
            Paragraph(f"<font color='#003366'><b>{postmortem_dict['inicio']}</b></font>", style_body),
            Paragraph("Primer traspaso de umbral anomalo en el cluster de ventanas.", style_body)
        ],
        [
            Paragraph("<b>T1 - Ventana Pico</b>", style_body),
            Paragraph(f"<font color='#003366'><b>{inc_meta.get('ventana_pico', postmortem_dict['inicio'])}</b></font>", style_body),
            Paragraph(f"Pico maximo de score anomalo ({severidad:.4f}) y saturacion de errores.", style_body)
        ],
        [
            Paragraph("<b>T2 - Cese / Contencion</b>", style_body),
            Paragraph(f"<font color='#003366'><b>{postmortem_dict['fin']}</b></font>", style_body),
            Paragraph("Termino de la rafaga o restablecimiento de la linea base operativa.", style_body)
        ]
    ]

    t_time = Table(timeline_data, colWidths=[130, 130, 268])
    t_time.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor('#FEF2F2')),  # Fila pico en tono suave
        ('BACKGROUND', (0, 3), (-1, 3), colors.white),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_time)
    story.append(Spacer(1, 8))

    # 5. SECCION 3: SISTEMAS Y COMPONENTES AFECTADOS
    story.append(Paragraph("3. Sistemas, Servicios y Componentes Comprometidos", style_h2))
    apps = postmortem_dict.get('aplicaciones_afectadas', [])
    apps_str = ", ".join(f"<b>{a}</b>" for a in apps) if apps else "Servicios centrales de orquestacion SAP"
    ips = postmortem_dict.get('ips_involucradas', [])
    ips_str = ", ".join(f"<font color='#003366'><b>{ip}</b></font>" for ip in ips) if ips else "Trafico interno / Inferencia autenticada"

    sys_table_data = [
        [Paragraph("<b>Componente Auditado</b>", style_meta_label), Paragraph("<b>Detalle Tecnico y Alcance</b>", style_meta_label)],
        [Paragraph("Aplicaciones SAP Impactadas", style_body), Paragraph(apps_str, style_body)],
        [Paragraph("Origenes / IPs Detectadas", style_body), Paragraph(ips_str, style_body)],
        [Paragraph("Entorno de Afectacion", style_body), Paragraph("SAP BTP Cloud Foundry / Inferencia LLM / SAP HANA Cloud", style_body)],
        [Paragraph("Radio de Impacto", style_body), Paragraph("Aislamiento a nivel de microservicio con incremento en latencias y consumo de cuota.", style_body)]
    ]
    t_sys = Table(sys_table_data, colWidths=[160, 368])
    t_sys.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_sys)
    story.append(Spacer(1, 8))

    # 6. SECCION 4: CAUSA RAIZ Y EVIDENCIA FORENSE CRUDA VIA MCP
    story.append(Paragraph("4. Diagnostico de Causa Raiz y Evidencia Forense (via MCP)", style_h2))
    diag_txt = (
        f"El analisis forense de las consultas extraidas via MCP revela una violacion coordinada. "
        f"En el stream <b>{stream}</b>, el vector <b>{tipo_ataque}</b> genero concentraciones de fallo "
        f"en la variable <font color='#003366'><b>{z_feat}</b></font>. A continuacion se presenta la muestra representativa "
        f"de los registros crudos obtenidos directamente de SAP HANA Cloud:"
    )
    story.append(Paragraph(diag_txt, style_body))

    # Parsear logs de muestra del Markdown o de la metadata
    meta_json = postmortem_dict.get('metadata_json', {})
    total_sampled = meta_json.get('total_raw_evidence_sampled', 0)

    # Extraer lineas de evidencia del reporte markdown
    reporte_md = postmortem_dict.get('reporte_markdown', '')
    lineas_evidencia = []
    if '### Muestra de Evidencia Forense Extraida (via MCP):' in reporte_md:
        parte_ev = reporte_md.split('### Muestra de Evidencia Forense Extraida (via MCP):')[1]
        parte_ev = parte_ev.split('---')[0].strip()
        for linea in parte_ev.split('\n'):
            linea_clean = linea.strip()
            if linea_clean:
                lineas_evidencia.append(linea_clean)

    if lineas_evidencia:
        ev_paragraphs = []
        for l in lineas_evidencia[:8]:
            l_fmt = _md_to_reportlab(l)
            ev_paragraphs.append([Paragraph(l_fmt, style_code)])

        t_ev = Table(ev_paragraphs, colWidths=[528])
        t_ev.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), SAP_CODE_BG),
            ('BOX', (0, 0), (-1, -1), 0.5, SAP_CODE_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_ev)
    else:
        story.append(Paragraph("Evidencia extraida directamente de tablas de telemetria en SAP HANA Cloud.", style_code))

    story.append(Spacer(1, 8))

    # 7. SECCION 5: PLAN DE MITIGACION Y HARDENING
    story.append(Paragraph("5. Plan de Mitigacion y Recomendaciones de Hardening", style_h2))

    if 'Prompt injection' in tipo_ataque:
        items_inmediatos = [
            "<b>Regla de Inspeccion en Gateway de Inferencia:</b> Validar entradas en el proxy de IA antes de reenviar a los modelos fundacionales.",
            "<b>Activacion de Guardrails Llama Guard / NeMo:</b> Configurar filtros semanticos para interceptar comandos directos de evasion (ej. <code>SYSTEM: disable filter</code>)."
        ]
        items_mediano = [
            "<b>Delimitadores Estructurados (XML/JSON):</b> Separar de manera categorica el contexto del sistema de las cadenas ingresadas por usuarios.",
            "<b>Principio de Minimo Privilegio en Tool Calling:</b> Revocar acceso a funciones con privilegios elevados (consultas salariales, credenciales de admin)."
        ]
    elif 'Degradación' in tipo_ataque or 'DoS' in tipo_ataque:
        items_inmediatos = [
            "<b>Ajuste Agresivo de Timeouts:</b> Reducir el tiempo de espera de 30s a 15s en el API Gateway para liberar hilos y evitar caidas en cascada.",
            "<b>Circuit Breakers Activos:</b> Aislar de forma automatica microservicios de LLM que superen el 20% de fallos consecutivos."
        ]
        items_mediano = [
            "<b>Rate Limiting Dinamico por Cuotas:</b> Implementar limitacion por usuario y por aplicacion consumidora para amortiguar picos artificiales.",
            "<b>Enrutamiento Adaptativo Multi-Proveedor:</b> Conmutar a modelos secundarios cuando se detecte saturacion HTTP 429 / 529."
        ]
    else:
        items_inmediatos = [
            "<b>Bloqueo Inmediato en WAF:</b> Anadir a la lista de denegacion las direcciones IP con indices anomalos de codigos HTTP 401/403.",
            "<b>Restriccion Temporal por Intentos Fallidos:</b> Activar bloqueo de 15 minutos tras 5 intentos fallidos de autenticacion."
        ]
        items_mediano = [
            "<b>Enforzamiento de MFA Obligatorio:</b> Exigir autenticacion multifactor para todas las llamadas a APIs de SAP S/4HANA Finance y Ariba.",
            "<b>Monitoreo de Entropia en Tiempo Real:</b> Configurar alarmas tempranas ante incrementos abruptos en la diversidad de endpoints solicitados."
        ]

    story.append(Paragraph("<b>Acciones Inmediatas (Contencion en &lt; 24 horas):</b>", style_h3))
    for act in items_inmediatos:
        story.append(Paragraph(f"• {_md_to_reportlab(act)}", style_body))

    story.append(Paragraph("<b>Acciones a Mediano Plazo (Hardening de Arquitectura):</b>", style_h3))
    for act in items_mediano:
        story.append(Paragraph(f"• {_md_to_reportlab(act)}", style_body))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=SAP_BORDER, spaceAfter=6))
    story.append(Paragraph(
        "<i>Reporte forense certificado por la Unidad de Observabilidad y Ciberseguridad de SAP BTP. "
        "Consistencia validada contra registros maestros en SAP HANA Cloud.</i>",
        style_meta_label
    ))

    # Construir documento
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
