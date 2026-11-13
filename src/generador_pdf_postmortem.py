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
        self.pages = []

    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self.pages)
        for page in self.pages:
            self.__dict__.update(page)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, total_pages):
        w, h = letter
        self.saveState()

        # Header corporativo SAP
        self.setFillColor(SAP_NAVY)
        self.rect(0, h - 8, w, 8, fill=1, stroke=0)
        self.setFillColor(SAP_BLUE)
        self.rect(0, h - 8, 120, 8, fill=1, stroke=0)

        # Header texto
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(SAP_NAVY)
        self.drawString(36, h - 22, "SAP CLOUD OBSERVABILITY & SECURITY OPERATIONS")

        self.setFont("Helvetica", 8)
        self.setFillColor(SAP_MUTED)
        self.drawRightString(w - 36, h - 22, "REPORTE FORENSE POSTMORTEM | CONFIDENCIAL")

        self.setStrokeColor(SAP_BORDER)
        self.setLineWidth(0.5)
        self.line(36, h - 26, w - 36, h - 26)

        # Footer
        self.line(36, 32, w - 36, 32)
        self.setFont("Helvetica", 8)
        self.setFillColor(SAP_MUTED)
        self.drawString(36, 20, "Generado por Agente Forense Autónomo (Model Context Protocol + SAP HANA Cloud)")
        
        page_str = f"Página {self._pageNumber} de {total_pages}"
        self.drawRightString(w - 36, 20, page_str)

        self.restoreState()


def generar_pdf_postmortem(postmortem_data: Dict[str, Any], output_path: str):
    """
    Compila el reporte estructurado en un documento PDF de 2 paginas con diseno ejecutivo.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=SAP_NAVY
    )
    meta_style = ParagraphStyle(
        'MetaText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=SAP_MUTED
    )
    sec_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=SAP_NAVY,
        spaceBefore=6,
        spaceAfter=3
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=SAP_DARK
    )
    exec_summary_style = ParagraphStyle(
        'ExecSummary',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=SAP_DARK
    )
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=SAP_DARK
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=SAP_NAVY
    )
    code_box_style = ParagraphStyle(
        'CodeBox',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=6.5,
        leading=8.5,
        textColor=SAP_NAVY
    )

    story = []

    pm_id = postmortem_data['postmortem_id']
    evento_id = postmortem_data['evento_id']
    stream = postmortem_data['stream'].upper()
    tipo_ataque = postmortem_data['tipo_ataque']
    mitre_tactic = postmortem_data.get('mitre_tactic', 'N/A')
    severidad = postmortem_data.get('severidad_score', 0.0)
    inicio = postmortem_data['inicio']
    fin = postmortem_data['fin']
    resumen = postmortem_data.get('resumen_ejecutivo', '')

    meta = postmortem_data.get('metadata_json', {})
    inc_meta = meta.get('incident_meta', {})
    z_max = inc_meta.get('z_max', 0.0)
    z_feat = inc_meta.get('z_max_feature', 'N/A')
    confianza = inc_meta.get('confianza', 'alta').upper()
    n_ventanas = inc_meta.get('n_ventanas', 1)

    # -------------------------------------------------------------
    # CABECERA EJECUTIVA
    # -------------------------------------------------------------
    header_table_data = [
        [
            Paragraph(f"REPORTE FORENSE POSTMORTEM: {pm_id}", title_style),
            Paragraph(f"<b>SEVERIDAD (IF):</b> <font color='#D92D20'>{severidad:.4f}</font><br/><b>CONFIANZA:</b> {confianza}", table_cell_bold)
        ],
        [
            Paragraph(f"<b>Evento ID:</b> {evento_id} &nbsp;|&nbsp; <b>Stream:</b> {stream} &nbsp;|&nbsp; <b>Vector:</b> {tipo_ataque}", meta_style),
            Paragraph(f"<b>Intervalo:</b> {inicio} &rarr; {fin}", meta_style)
        ]
    ]
    t_header = Table(header_table_data, colWidths=[410, 130])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 4))
    story.append(HRFlowable(width="100%", thickness=1, color=SAP_BLUE, spaceBefore=2, spaceAfter=6))

    # -------------------------------------------------------------
    # SECCION 1: RESUMEN EJECUTIVO Y CLASIFICACION MITRE
    # -------------------------------------------------------------
    story.append(Paragraph("1. Resumen Ejecutivo y Clasificacion de la Amenaza", sec_title_style))

    c1_meta = [
        [Paragraph("<b>Vector Clasificado:</b>", table_cell_bold), Paragraph(tipo_ataque, table_cell_style)],
        [Paragraph("<b>Marco MITRE ATT&CK:</b>", table_cell_bold), Paragraph(mitre_tactic, table_cell_style)],
        [Paragraph("<b>Metrica Mas Desviada:</b>", table_cell_bold), Paragraph(f"{z_feat} (Z = {z_max:.2f})", table_cell_style)],
        [Paragraph("<b>Ventanas Impactadas:</b>", table_cell_bold), Paragraph(f"{n_ventanas} ventanas consecutivas", table_cell_style)],
    ]
    t_c1_meta = Table(c1_meta, colWidths=[130, 400])
    t_c1_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ]))
    story.append(t_c1_meta)
    story.append(Spacer(1, 4))

    card_resumen = [
        [Paragraph(f"<b>Dictamen Forense:</b> {_md_to_reportlab(resumen)}", exec_summary_style)]
    ]
    t_resumen = Table(card_resumen, colWidths=[540])
    t_resumen.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F7FF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BAE6FD')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_resumen)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECCION 2: TIMELINE CRONOLOGICO
    # -------------------------------------------------------------
    story.append(Paragraph("2. Linea de Tiempo Cronologica (Incident Timeline)", sec_title_style))

    pico_ts = inc_meta.get('ventana_pico', inicio)
    timeline_rows = [
        [Paragraph("Hito Operativo", table_cell_bold), Paragraph("Marca Temporal (UTC)", table_cell_bold), Paragraph("Detalle Forense Observado", table_cell_bold)],
        [Paragraph("T0 - Inicio Anomalia", table_cell_bold), Paragraph(str(inicio), table_cell_style), Paragraph("Primer cruce de umbral detectado por Isolation Forest.", table_cell_style)],
        [Paragraph("T1 - Pico de Impacto", table_cell_bold), Paragraph(str(pico_ts), table_cell_style), Paragraph(f"Maximo z-score ({z_max:.2f}) en {z_feat}. Severidad: {severidad:.4f}.", table_cell_style)],
        [Paragraph("T2 - Fin de Intervalo", table_cell_bold), Paragraph(str(fin), table_cell_style), Paragraph("Cese de la actividad anomala o restablecimiento del baseline.", table_cell_style)],
    ]
    t_timeline = Table(timeline_rows, colWidths=[110, 120, 310])
    t_timeline.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_timeline)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECCION 3: SISTEMAS Y COMPONENTES COMPROMETIDOS
    # -------------------------------------------------------------
    story.append(Paragraph("3. Sistemas, Servicios y Componentes SAP Comprometidos", sec_title_style))

    apps = postmortem_data.get('aplicaciones_afectadas', [])
    apps_str = ", ".join(f"<b>{a}</b>" for a in apps) if apps else "Servicios internos de plataforma SAP"
    ips = postmortem_data.get('ips_involucradas', [])
    ips_str = ", ".join(ips[:8]) if ips else "Trafico interno / Gateway proxy"
    if len(ips) > 8:
        ips_str += f" (+{len(ips) - 8} adicionales)"

    comp_data = [
        [Paragraph("<b>Aplicaciones SAP:</b>", table_cell_bold), Paragraph(apps_str, table_cell_style)],
        [Paragraph("<b>Origenes IP Muestreados:</b>", table_cell_bold), Paragraph(ips_str, table_cell_style)],
    ]
    t_comp = Table(comp_data, colWidths=[130, 400])
    t_comp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_comp)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECCION 4: CAUSA RAIZ Y EVIDENCIA FORENSE (LOGS MUESTRA)
    # -------------------------------------------------------------
    story.append(Paragraph("4. Analisis de Causa Raiz y Evidencia Forense de Logs", sec_title_style))

    reporte_md = postmortem_data.get('reporte_markdown', '')
    p_seccion4 = ""
    if "## 4." in reporte_md:
        partes = reporte_md.split("## 4.")[1]
        p_seccion4 = partes.split("## 5.")[0].strip() if "## 5." in partes else partes[:600]

    lineas_limpias = []
    for l in p_seccion4.split('\n')[:8]:
        l_str = l.strip()
        if l_str and not l_str.startswith('|') and not l_str.startswith('#'):
            lineas_limpias.append(_md_to_reportlab(l_str))

    analisis_txt = "<br/>".join(lineas_limpias) if lineas_limpias else "Analisis pericial basado en telemetria correlacionada."
    story.append(Paragraph(analisis_txt, body_style))
    story.append(Spacer(1, 4))

    # Callout Box con muestra representativa de logs extraidos via MCP
    logs_raw_sample = meta.get('evidence_logs_sample', []) or meta.get('raw_evidence', {}).get('logs_muestra', [])
    if logs_raw_sample:
        story.append(Paragraph("<b>Muestra de Evidencia en Logs Crudos (MCP Sample):</b>", ParagraphStyle('SubSub', parent=sec_title_style, fontSize=8.5)))
        log_lines = []
        for item in logs_raw_sample[:4]:
            ts = item.get('TIMESTAMP', 'N/A')
            app = item.get('SAP_FUNCTION_APPLICATION', 'N/A')
            if stream == 'LLM':
                prompt = (item.get('LLM_PROMPT') or '')[:80]
                status = item.get('LLM_STATUS') or 'N/A'
                err = item.get('LLM_ERROR_MESSAGE') or 'None'
                log_lines.append(f"[{ts}] APP={app} STATUS={status} ERR={err}\n  PROMPT={prompt}...")
            else:
                ip = item.get('CLIENT_IP', 'N/A')
                code = item.get('HTTP_STATUS_CODE', 'N/A')
                path = (item.get('HEATHERS_REQUEST_PATH') or '')[:40]
                log_lines.append(f"[{ts}] IP={ip} HTTP={code} PATH={path}")

        txt_logs = "\n".join(log_lines)
        t_box = Table([[Paragraph(txt_logs.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>'), code_box_style)]], colWidths=[540])
        t_box.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), SAP_CODE_BG),
            ('BOX', (0, 0), (-1, -1), 0.5, SAP_CODE_BORDER),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_box)
        story.append(Spacer(1, 6))

    # -------------------------------------------------------------
    # SECCION 5: ACCIONES Y HARDENING (ACTION ITEMS)
    # -------------------------------------------------------------
    story.append(Paragraph("5. Plan de Mitigacion y Recomendaciones de Hardening", sec_title_style))

    p_seccion5 = ""
    if "## 5." in reporte_md:
        p_seccion5 = reporte_md.split("## 5.")[1].strip()

    mitigaciones = []
    for l in p_seccion5.split('\n'):
        l_str = l.strip()
        if l_str and (l_str.startswith('1.') or l_str.startswith('2.') or l_str.startswith('-') or l_str.startswith('*')):
            mitigaciones.append(_md_to_reportlab(l_str))
        if len(mitigaciones) >= 5:
            break

    if not mitigaciones:
        mitigaciones = [
            "1. <b>Contencion Inmediata:</b> Aislamiento de sesiones activas y bloqueo de rangos IP anomalos en WAF.",
            "2. <b>Rate Limiting y Timeouts:</b> Restriccion temporal de concurrencia sobre los endpoints afectados.",
            "3. <b>Revision de Autorizaciones:</b> Auditoria estricta de politicas RBAC y llamadas a herramientas del LLM.",
            "4. <b>Monitoreo Reforzado:</b> Calibracion continua del Isolation Forest para deteccion temprana de recurrencias."
        ]

    items_tabla = [[Paragraph(m, body_style)] for m in mitigaciones]
    t_mitiga = Table(items_tabla, colWidths=[540])
    t_mitiga.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEFCE8')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF08A')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF9C3')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_mitiga)

    doc.build(story, canvasmaker=NumberedCanvas)
