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
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable, PageBreak
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
    """Convierte marcas markdown basicas (** y `) a etiquetas XML seguras para ReportLab."""
    if not text:
        return ""
    t = str(text).replace('<code>', '\x00CODE\x00').replace('</code>', '\x00/CODE\x00')
    t = re.sub(r'\$Z_\{?max\}?\$', 'Z_max', t)
    t = re.sub(r'\$([^$]+)\$', r'\1', t)
    t = re.sub(r'&(?!(?:amp|lt|gt|quot|apos);)', '&amp;', t)
    t = t.replace('<', '&lt;').replace('>', '&gt;')
    t = t.replace('\x00CODE\x00', '<font color="#003366"><b>').replace('\x00/CODE\x00', '</b></font>')
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    t = t.replace('**', '')
    t = re.sub(r'`(.*?)`', r'<font color="#003366"><b>\1</b></font>', t)
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


def _build_styles() -> Dict[str, ParagraphStyle]:
    """Estilos tipograficos corporativos compartidos para reportes individuales y mensuales."""
    styles = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'DocTitle', parent=styles['Normal'], fontName='Helvetica-Bold',
            fontSize=15, leading=18, textColor=SAP_NAVY,
        ),
        'meta': ParagraphStyle(
            'MetaText', parent=styles['Normal'], fontName='Helvetica',
            fontSize=8, leading=11, textColor=SAP_MUTED,
        ),
        'sec_title': ParagraphStyle(
            'SectionTitle', parent=styles['Normal'], fontName='Helvetica-Bold',
            fontSize=10, leading=13, textColor=SAP_NAVY, spaceBefore=6, spaceAfter=3,
            keepWithNext=True,
        ),
        'sub_title': ParagraphStyle(
            'SubSub', parent=styles['Normal'], fontName='Helvetica-Bold',
            fontSize=8.5, leading=11, textColor=SAP_NAVY, spaceBefore=5, spaceAfter=2,
            keepWithNext=True,
        ),
        'body': ParagraphStyle(
            'BodyDark', parent=styles['Normal'], fontName='Helvetica',
            fontSize=8, leading=11, textColor=SAP_DARK,
        ),
        'exec_summary': ParagraphStyle(
            'ExecSummary', parent=styles['Normal'], fontName='Helvetica',
            fontSize=8.5, leading=12, textColor=SAP_DARK,
        ),
        'cell': ParagraphStyle(
            'TableCell', parent=styles['Normal'], fontName='Helvetica',
            fontSize=7.5, leading=9.5, textColor=SAP_DARK,
        ),
        'cell_bold': ParagraphStyle(
            'TableCellBold', parent=styles['Normal'], fontName='Helvetica-Bold',
            fontSize=7.5, leading=9.5, textColor=SAP_NAVY,
        ),
        'code_box': ParagraphStyle(
            'CodeBox', parent=styles['Normal'], fontName='Courier',
            fontSize=6.5, leading=8.5, textColor=SAP_NAVY,
        ),
    }


def _append_incident_story(story: List[Any], postmortem_data: Dict[str, Any], st: Dict[str, ParagraphStyle]):
    """Agrega las 5 secciones ejecutivas de un incidente al flujo ReportLab."""
    title_style = st['title']
    meta_style = st['meta']
    sec_title_style = st['sec_title']
    body_style = st['body']
    exec_summary_style = st['exec_summary']
    table_cell_style = st['cell']
    table_cell_bold = st['cell_bold']
    code_box_style = st['code_box']

    pm_id = postmortem_data['postmortem_id']
    evento_id = postmortem_data['evento_id']
    stream = str(postmortem_data['stream']).upper()
    tipo_ataque = str(postmortem_data['tipo_ataque'])
    mitre_tactic = str(postmortem_data.get('mitre_tactic', 'N/A'))
    severidad = float(postmortem_data.get('severidad_score', 0.0) or 0.0)
    inicio = postmortem_data['inicio']
    fin = postmortem_data['fin']
    resumen = postmortem_data.get('resumen_ejecutivo', '')

    meta = postmortem_data.get('metadata_json', {}) or {}
    inc_meta = meta.get('incident_meta', {}) or {}
    z_max = float(inc_meta.get('z_max', 0.0) or 0.0)
    z_feat = str(inc_meta.get('z_max_feature', 'N/A'))
    confianza = str(inc_meta.get('confianza', 'alta')).upper()
    n_ventanas = int(inc_meta.get('n_ventanas', 1) or 1)
    escalas = (
        meta.get('escalas_detectadas')
        or inc_meta.get('escalas_detectadas')
        or ([str(inc_meta['window_label'])] if inc_meta.get('window_label') else ['5min'])
    )
    escalas_str = ", ".join(str(x) for x in escalas)

    header_table_data = [
        [
            Paragraph(f"REPORTE FORENSE POSTMORTEM: {_md_to_reportlab(pm_id)}", title_style),
            Paragraph(f"<b>CONFIANZA:</b> {_md_to_reportlab(confianza)}<br/><b>ESCALAS:</b> {_md_to_reportlab(escalas_str)}", table_cell_bold)
        ],
        [
            Paragraph(f"<b>Evento ID:</b> {evento_id} &nbsp;|&nbsp; <b>Stream:</b> {_md_to_reportlab(stream)} &nbsp;|&nbsp; <b>Vector:</b> {_md_to_reportlab(tipo_ataque)}", meta_style),
            Paragraph(f"<b>Intervalo:</b> {_md_to_reportlab(str(inicio))} &rarr; {_md_to_reportlab(str(fin))}", meta_style)
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

    story.append(Paragraph("1. Resumen Ejecutivo y Clasificacion de la Amenaza", sec_title_style))

    c1_meta = [
        [Paragraph("<b>Vector Clasificado:</b>", table_cell_bold), Paragraph(_md_to_reportlab(tipo_ataque), table_cell_style)],
        [Paragraph("<b>Marco MITRE ATT&amp;CK:</b>", table_cell_bold), Paragraph(_md_to_reportlab(mitre_tactic), table_cell_style)],
        [Paragraph("<b>Escalas de Ventana:</b>", table_cell_bold), Paragraph(_md_to_reportlab(escalas_str), table_cell_style)],
        [Paragraph("<b>Metrica Mas Desviada:</b>", table_cell_bold), Paragraph(f"{_md_to_reportlab(z_feat)} (Z = {z_max:.2f})", table_cell_style)],
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

    story.append(Paragraph("2. Linea de Tiempo Cronologica (Incident Timeline)", sec_title_style))

    pico_ts = inc_meta.get('ventana_pico', inicio)
    timeline_rows = [
        [Paragraph("Hito Operativo", table_cell_bold), Paragraph("Marca Temporal (UTC)", table_cell_bold), Paragraph("Detalle Forense Observado", table_cell_bold)],
        [Paragraph("T0 - Inicio Anomalia", table_cell_bold), Paragraph(_md_to_reportlab(str(inicio)), table_cell_style), Paragraph("Primer cruce de umbral detectado por Isolation Forest.", table_cell_style)],
        [Paragraph("T1 - Pico de Impacto", table_cell_bold), Paragraph(_md_to_reportlab(str(pico_ts)), table_cell_style), Paragraph(f"Maximo z-score ({z_max:.2f}) en {_md_to_reportlab(z_feat)}.", table_cell_style)],
        [Paragraph("T2 - Fin de Intervalo", table_cell_bold), Paragraph(_md_to_reportlab(str(fin)), table_cell_style), Paragraph("Cese de la actividad anomala o restablecimiento del baseline.", table_cell_style)],
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

    story.append(Paragraph("3. Sistemas, Servicios y Componentes SAP Comprometidos", sec_title_style))

    apps = postmortem_data.get('aplicaciones_afectadas', [])
    apps_str = ", ".join(f"<b>{_md_to_reportlab(a)}</b>" for a in apps) if apps else "Servicios internos de plataforma SAP"
    ips = postmortem_data.get('ips_involucradas', [])
    ips_str = ", ".join(_md_to_reportlab(ip) for ip in ips[:8]) if ips else "Trafico interno / Gateway proxy"
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

    story.append(Paragraph("4. Analisis de Causa Raiz y Evidencia Forense de Logs", sec_title_style))

    reporte_md = postmortem_data.get('reporte_markdown', '') or ''
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

    logs_raw_sample = meta.get('evidence_logs_sample', []) or meta.get('raw_evidence', {}).get('logs_muestra', [])
    if logs_raw_sample:
        story.append(Paragraph("<b>Muestra de Evidencia en Logs Crudos (MCP Sample):</b>", st['sub_title']))
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


def generar_pdf_postmortem(postmortem_data: Dict[str, Any], output_path: Any):
    """
    Compila el reporte estructurado de un incidente en un documento PDF ejecutivo.
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42
    )
    st = _build_styles()
    story: List[Any] = []
    _append_incident_story(story, postmortem_data, st)
    doc.build(story, canvasmaker=NumberedCanvas)


def _clasificar_severidad_incidente(z_max: float, severidad_score: float = 0.0):
    """Clasifica cualitativamente la severidad del incidente con color corporativo para tablas PDF."""
    if z_max >= 20.0 or severidad_score >= 0.82:
        return "Crítico", "#B42318", colors.HexColor("#FEE4E2")
    if z_max >= 10.0 or severidad_score >= 0.74:
        return "Grave", "#B54708", colors.HexColor("#FEF3C7")
    return "Moderado", "#0057D2", colors.HexColor("#E0F2FE")


def _extraer_causa_raiz_breve(reporte_md: str) -> str:
    """Extrae una sintesis concisa de Causa Raiz (Seccion 4) de un reporte Markdown."""
    if not reporte_md or "## 4." not in reporte_md:
        return ""
    sec4 = reporte_md.split("## 4.", 1)[1]
    if "## 5." in sec4:
        sec4 = sec4.split("## 5.", 1)[0]
    frases = []
    for l in sec4.splitlines():
        s = l.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("-") or s.startswith("*"):
            continue
        limpio = s.replace("**", "").replace("`", "")
        limpio = re.sub(
            r'^(?:4\.\s*)?(?:An[aá]lisis\s+Forense\s+de\s+Causa\s+Ra[ií]z\s+y\s+Evidencia(?:\s*\(Root\s+Cause\))?)[:\.\-\s]*',
            '',
            limpio,
            flags=re.IGNORECASE,
        ).strip()
        if len(limpio) >= 40:
            frases.append(limpio)
        if len(frases) >= 2:
            break
    txt = " ".join(frases)
    if len(txt) > 380:
        corte = txt[:380].rfind(". ")
        txt = txt[:corte + 1] if corte > 160 else txt[:377] + "..."
    return txt


def generar_pdf_postmortem_mensual(postmortems: List[Dict[str, Any]], periodo: str, output_path: Any):
    """
    Compila un Dossier Forense Mensual Unificado (~5-8 paginas) estilo SAP SOC:
      1. Resumen Ejecutivo Consolidado y Distribucion por Campana / Vector MITRE
      2. Cronologia Maestra de Incidentes Multi-Escala del Periodo
      3. Analisis Forense Unificado por Familia de Amenaza (con sintesis por incidente y evidencia MCP)
      4. Matriz de Correlacion Multi-Escala y Superficie de Impacto en Aplicaciones SAP
      5. Plan Maestro Consolidado de Contencion y Hardening de Arquitectura
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=42,
    )
    st = _build_styles()
    story: List[Any] = []

    ordenados = sorted(
        postmortems,
        key=lambda x: str(x.get('inicio', '')),
    )

    n_total = len(ordenados)
    streams_set = sorted({str(p.get('stream', 'N/A')).upper() for p in ordenados})
    vectores_set = sorted({str(p.get('tipo_ataque', 'N/A')) for p in ordenados})
    tacticas_set = sorted({str(p.get('mitre_tactic', 'N/A')) for p in ordenados if p.get('mitre_tactic')})
    apps_all = sorted({a for p in ordenados for a in (p.get('aplicaciones_afectadas') or [])})
    ips_all = sorted({ip for p in ordenados for ip in (p.get('ips_involucradas') or [])})

    n_sys = sum(1 for p in ordenados if str(p.get('stream', '')).lower() == 'system')
    n_llm = sum(1 for p in ordenados if str(p.get('stream', '')).lower() == 'llm')
    ini_mes = str(ordenados[0].get('inicio', 'N/A'))[:16] if ordenados else 'N/A'
    fin_mes = max((str(p.get('fin', 'N/A'))[:16] for p in ordenados), default='N/A')

    # -------------------------------------------------------------------------
    # CABECERA CORPORATIVA DEL DOSSIER MENSUAL
    # -------------------------------------------------------------------------
    header_data = [
        [
            Paragraph(f"DOSSIER FORENSE POSTMORTEM MENSUAL: {_md_to_reportlab(periodo)}", st['title']),
            Paragraph(
                f"<b>INCIDENTES UNICOS:</b> {n_total}<br/>"
                f"<b>TACTICAS MITRE:</b> {len(tacticas_set)}",
                st['cell_bold'],
            ),
        ],
        [
            Paragraph(
                f"<b>Cobertura Temporal:</b> {_md_to_reportlab(ini_mes)} &rarr; {_md_to_reportlab(fin_mes)} UTC &nbsp;|&nbsp; "
                f"<b>Corrientes:</b> SYSTEM ({n_sys}) / LLM ({n_llm})",
                st['meta'],
            ),
            Paragraph(f"<b>Apps SAP:</b> {len(apps_all)} &nbsp;|&nbsp; <b>IPs:</b> {len(ips_all)}", st['meta']),
        ],
    ]
    t_hdr = Table(header_data, colWidths=[410, 130])
    t_hdr.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 1),
    ]))
    story.append(t_hdr)
    story.append(Spacer(1, 3))
    story.append(HRFlowable(width="100%", thickness=1.2, color=SAP_BLUE, spaceBefore=2, spaceAfter=6))

    # -------------------------------------------------------------------------
    # SECCION 1: RESUMEN EJECUTIVO Y PANORAMA DE CAMPANAS DEL MES
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Resumen Ejecutivo Consolidado y Panorama de Amenazas del Mes", st['sec_title']))

    # Agrupar por vector de ataque (campana)
    por_vector: Dict[str, List[Dict[str, Any]]] = {}
    for p in ordenados:
        vec = str(p.get('tipo_ataque', 'Anomalia'))
        por_vector.setdefault(vec, []).append(p)
    campanas_ordenadas = sorted(por_vector.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    resumen_campanas_txt = ", ".join(f"<b>{_md_to_reportlab(v)}</b> ({len(lista)})" for v, lista in campanas_ordenadas)

    dictamen_mes = (
        f"Durante el periodo <b>{_md_to_reportlab(periodo)}</b> ({_md_to_reportlab(ini_mes)} a {_md_to_reportlab(fin_mes)} UTC), "
        f"el pipeline de observabilidad con Isolation Forest en SAP HANA Cloud y el Agente Forense MCP correlacionaron "
        f"<b>{n_total}</b> incidentes relevantes unicos tras deduplicacion multi-escala (5min, 15min, 30min, 1h y 3h): "
        f"<b>{n_sys}</b> sobre infraestructura y aplicaciones core SAP (stream SYSTEM) y <b>{n_llm}</b> sobre "
        f"servicios de inferencia de IA generativa (stream LLM). "
        f"La actividad maliciosa y anomala se agrupo en <b>{len(campanas_ordenadas)}</b> familias de amenaza: "
        f"{resumen_campanas_txt}, mapeadas a <b>{len(tacticas_set)}</b> tactica(s) MITRE ATT&amp;CK / ATLAS e "
        f"impactando un total de <b>{len(apps_all)}</b> aplicacion(es) SAP y <b>{len(ips_all)}</b> origen(es) IP unicos."
    )
    t_dict = Table([[Paragraph(f"<b>Dictamen Forense Consolidado del Periodo:</b> {dictamen_mes}", st['exec_summary'])]], colWidths=[540])
    t_dict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0F7FF')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#BAE6FD')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_dict)
    story.append(Spacer(1, 5))

    # Tabla 1B: Sintesis por Campana / Vector de Ataque
    orden_esc = {'5min': 1, '15min': 2, '30min': 3, '1h': 4, '3h': 5}
    camp_rows = [[
        Paragraph("Campana / Vector", st['cell_bold']),
        Paragraph("Stream", st['cell_bold']),
        Paragraph("Tactica y Tecnica MITRE", st['cell_bold']),
        Paragraph("Inc.", st['cell_bold']),
        Paragraph("Escalas Detectadas", st['cell_bold']),
        Paragraph("Indicador Dominante (Z-Max)", st['cell_bold']),
        Paragraph("Servicios SAP Impactados", st['cell_bold']),
    ]]
    for vec, lista in campanas_ordenadas:
        st_vec = ", ".join(sorted({str(x.get('stream', '')).upper() for x in lista}))
        mitre_vec = str(lista[0].get('mitre_tactic', 'N/A'))
        meta0 = lista[0].get('metadata_json') or {}
        tech0 = (meta0.get('mitre_info') or {}).get('technique')
        if tech0:
            mitre_vec = f"{mitre_vec} ({tech0.split(':')[0]})"
        esc_set = set()
        best_feat, best_z = "N/A", 0.0
        apps_vec = set()
        for x in lista:
            mx = x.get('metadata_json') or {}
            incx = mx.get('incident_meta') or {}
            for e_lbl in (mx.get('escalas_detectadas') or incx.get('escalas_detectadas') or ['5min']):
                esc_set.add(str(e_lbl))
            zx = float(incx.get('z_max') or 0.0)
            if zx >= best_z:
                best_z = zx
                best_feat = str(incx.get('z_max_feature') or 'N/A')
            for a in (x.get('aplicaciones_afectadas') or []):
                apps_vec.add(str(a))
        esc_ordenadas = sorted(esc_set, key=lambda s: orden_esc.get(s, 99))
        apps_vec_list = sorted(apps_vec)
        apps_vec_str = ", ".join(apps_vec_list[:4]) + (f" (+{len(apps_vec_list) - 4})" if len(apps_vec_list) > 4 else "")
        camp_rows.append([
            Paragraph(f"<b>{_md_to_reportlab(vec)}</b>", st['cell']),
            Paragraph(_md_to_reportlab(st_vec), st['cell']),
            Paragraph(_md_to_reportlab(mitre_vec), st['cell']),
            Paragraph(f"<b>{len(lista)}</b>", st['cell']),
            Paragraph(_md_to_reportlab(", ".join(esc_ordenadas)), st['cell']),
            Paragraph(_md_to_reportlab(f"{best_feat} (Z={best_z:.1f})"), st['cell']),
            Paragraph(_md_to_reportlab(apps_vec_str or "Plataforma SAP"), st['cell']),
        ])

    t_camp = Table(camp_rows, colWidths=[78, 46, 104, 28, 66, 98, 120])
    t_camp.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_camp)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECCION 2: CRONOLOGIA MAESTRA DE INCIDENTES MULTI-ESCALA DEL PERIODO
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Cronologia Maestra de Incidentes Multi-Escala del Periodo", st['sec_title']))
    matriz_rows = [[
        Paragraph("Expediente", st['cell_bold']),
        Paragraph("Origen", st['cell_bold']),
        Paragraph("Vector / Ataque", st['cell_bold']),
        Paragraph("Ventana Temporal (UTC)", st['cell_bold']),
    ]]
    for p in ordenados:
        ini_s = str(p.get('inicio', 'N/A'))[:16]
        fin_s = str(p.get('fin', 'N/A'))
        fin_short = fin_s[11:16] if len(fin_s) >= 16 and fin_s[:10] == ini_s[:10] else fin_s[:16]
        ventana_str = f"{ini_s} -> {fin_short}"
        matriz_rows.append([
            Paragraph(f"<b>{_md_to_reportlab(str(p.get('postmortem_id', 'N/A')))}</b>", st['cell']),
            Paragraph(_md_to_reportlab(str(p.get('stream', '')).upper()), st['cell']),
            Paragraph(_md_to_reportlab(str(p.get('tipo_ataque', 'N/A'))), st['cell']),
            Paragraph(_md_to_reportlab(ventana_str), st['cell']),
        ])

    t_mat = Table(matriz_rows, colWidths=[145, 70, 155, 170], repeatRows=1)
    t_mat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECCION 3: ANALISIS FORENSE UNIFICADO POR FAMILIA / CAMPANA DE AMENAZA
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. Analisis Forense Unificado por Familia de Amenaza y Evidencia MCP", st['sec_title']))
    story.append(Paragraph(
        "A continuacion se consolida el analisis pericial de causa raiz, las sintesis ejecutivas redactadas "
        "por Gemini para cada incidente y las muestras representativas de logs crudos extraidos via MCP desde "
        "SAP HANA Cloud, agrupados por vector de ataque para eliminar redundancia operativa.",
        st['body'],
    ))
    story.append(Spacer(1, 4))

    cell_center_bold = ParagraphStyle(
        'TableCellCenterBold', parent=st['cell_bold'], alignment=1,
    )
    cell_center = ParagraphStyle(
        'TableCellCenter', parent=st['cell'], alignment=1,
    )

    for idx_c, (vec, lista) in enumerate(campanas_ordenadas, start=1):
        st_vec = ", ".join(sorted({str(x.get('stream', '')).upper() for x in lista}))
        mitre_t = str(lista[0].get('mitre_tactic', 'N/A'))
        meta_first = lista[0].get('metadata_json') or {}
        mitre_tech = (meta_first.get('mitre_info') or {}).get('technique', '')
        apps_camp = sorted({a for x in lista for a in (x.get('aplicaciones_afectadas') or [])})
        ips_camp = sorted({ip for x in lista for ip in (x.get('ips_involucradas') or [])})
        esc_camp = sorted(
            {
                str(e)
                for x in lista
                for e in (
                    (x.get('metadata_json') or {}).get('escalas_detectadas')
                    or ((x.get('metadata_json') or {}).get('incident_meta') or {}).get('escalas_detectadas')
                    or ['5min']
                )
            },
            key=lambda s: orden_esc.get(s, 99),
        )

        # Sub-encabezado de la Campana
        story.append(Paragraph(
            f"3.{idx_c} Campana: {_md_to_reportlab(vec)} ({len(lista)} incidente(s) | Stream: {_md_to_reportlab(st_vec)} | "
            f"Escalas: {_md_to_reportlab(', '.join(esc_camp))})",
            st['sub_title'],
        ))

        # Causa raiz consolidada de la campana
        causa_principal = ""
        for x in lista:
            c_breve = _extraer_causa_raiz_breve(str(x.get('reporte_markdown') or ''))
            if c_breve:
                causa_principal = c_breve
                break
        if not causa_principal:
            causa_principal = str(lista[0].get('resumen_ejecutivo') or 'Patron anomalo detectado por Isolation Forest.')

        ips_muestra_txt = ", ".join(ips_camp[:10]) + (f" (+{len(ips_camp) - 10} adicionales)" if len(ips_camp) > 10 else "")
        resumen_camp_rows = [
            [
                Paragraph("<b>Marco MITRE:</b>", st['cell_bold']),
                Paragraph(_md_to_reportlab(f"{mitre_t} — {mitre_tech}" if mitre_tech else mitre_t), st['cell']),
            ],
            [
                Paragraph("<b>Aplicaciones SAP:</b>", st['cell_bold']),
                Paragraph(_md_to_reportlab(", ".join(apps_camp) or "Servicios internos SAP"), st['cell']),
            ],
            [
                Paragraph("<b>IPs Involucradas:</b>", st['cell_bold']),
                Paragraph(_md_to_reportlab(ips_muestra_txt or "Trafico interno / Gateway proxy"), st['cell']),
            ],
            [
                Paragraph("<b>Mecanismo y Causa Raiz:</b>", st['cell_bold']),
                Paragraph(_md_to_reportlab(causa_principal), st['cell']),
            ],
        ]
        t_rc = Table(resumen_camp_rows, colWidths=[105, 435])
        t_rc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), SAP_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ]))
        story.append(t_rc)
        story.append(Spacer(1, 3))

        # Tabla compacta con el desglose de cada incidente dentro de esta campana
        inc_rows = [[
            Paragraph("Expediente y Ventana (UTC)", st['cell_bold']),
            Paragraph("Severidad", cell_center_bold),
            Paragraph("Hallazgo Forense Puntual (Sintesis Gemini)", st['cell_bold']),
        ]]
        inc_table_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F7FF')),
            ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('VALIGN', (1, 1), (1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ]
        for row_idx, x in enumerate(lista, start=1):
            mx = x.get('metadata_json') or {}
            incx = mx.get('incident_meta') or {}
            zv_x = float(incx.get('z_max') or 0.0)
            sev_s = float(x.get('severidad_score') or 0.0)
            sev_lbl, sev_fg, sev_bg = _clasificar_severidad_incidente(zv_x, sev_s)
            inc_table_styles.append(('BACKGROUND', (1, row_idx), (1, row_idx), sev_bg))
            ini_x = str(x.get('inicio', 'N/A'))[:16]
            fin_x = str(x.get('fin', 'N/A'))[:16]
            res_x = str(x.get('resumen_ejecutivo') or '').strip()
            if len(res_x) > 260:
                corte = res_x[:260].rfind(". ")
                res_x = res_x[:corte + 1] if corte > 110 else res_x[:257] + "..."
            inc_rows.append([
                Paragraph(
                    f"<b>{_md_to_reportlab(str(x.get('postmortem_id', 'N/A')))}</b><br/>"
                    f"<font color='#64748B'>{_md_to_reportlab(ini_x)} &rarr; {_md_to_reportlab(fin_x)}</font>",
                    st['cell'],
                ),
                Paragraph(
                    f"<b><font color='{sev_fg}'>{sev_lbl}</font></b>",
                    cell_center,
                ),
                Paragraph(_md_to_reportlab(res_x), st['cell']),
            ])
        t_inc_camp = Table(inc_rows, colWidths=[126, 72, 342], repeatRows=1)
        t_inc_camp.setStyle(TableStyle(inc_table_styles))
        story.append(t_inc_camp)

        # Muestra representativa de logs crudos MCP para la campana (maximo 3 lineas consolidadas)
        log_lines = []
        seen_log_keys = set()
        for x in lista:
            mx = x.get('metadata_json') or {}
            raw_list = mx.get('evidence_logs_sample') or (mx.get('raw_evidence') or {}).get('logs_muestra') or []
            st_x = str(x.get('stream', 'system')).upper()
            for item in raw_list:
                ts = str(item.get('TIMESTAMP', 'N/A'))[:19]
                app = str(item.get('SAP_FUNCTION_APPLICATION', 'N/A'))
                if st_x == 'LLM':
                    prompt = str(item.get('LLM_PROMPT') or '')[:75].replace('\n', ' ')
                    status = str(item.get('LLM_STATUS') or 'N/A')
                    err = str(item.get('LLM_ERROR_MESSAGE') or 'None')[:45]
                    line = f"[{ts}] ({x.get('postmortem_id')}) APP={app} STATUS={status} ERR={err} | PROMPT={prompt}..."
                else:
                    ip = str(item.get('CLIENT_IP', 'N/A'))
                    code = str(item.get('HTTP_STATUS_CODE', 'N/A'))
                    path = str(item.get('HEATHERS_REQUEST_PATH') or '')[:38]
                    msg = str(item.get('SAP_FUNCTION_MESSAGE') or '')[:48]
                    line = f"[{ts}] ({x.get('postmortem_id')}) APP={app} IP={ip} HTTP={code} PATH={path} MSG={msg}"
                k = (app, line[:55])
                if k not in seen_log_keys:
                    seen_log_keys.add(k)
                    log_lines.append(line)
                if len(log_lines) >= 3:
                    break
            if len(log_lines) >= 3:
                break

        if log_lines:
            story.append(Spacer(1, 2))
            txt_logs = "Evidencia MCP (Logs Crudos):\n" + "\n".join(log_lines)
            safe_logs = txt_logs.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('\n', '<br/>')
            t_box = Table([[Paragraph(safe_logs, st['code_box'])]], colWidths=[540])
            t_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), SAP_CODE_BG),
                ('BOX', (0, 0), (-1, -1), 0.5, SAP_CODE_BORDER),
                ('TOPPADDING', (0, 0), (-1, -1), 2.5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
                ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t_box)

        story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECCION 4: MATRIZ DE CORRELACION MULTI-ESCALA Y SUPERFICIE SAP
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Correlacion Multi-Escala (5min a 3h) y Superficie de Impacto SAP", st['sec_title']))

    todas_escalas = ['5min', '15min', '30min', '1h', '3h']
    esc_stats_rows = [[
        Paragraph("Longitud de Ventana", st['cell_bold']),
        Paragraph("Incidentes Detectados", st['cell_bold']),
        Paragraph("Exclusivos de esta Escala", st['cell_bold']),
        Paragraph("Vectores Observados en la Escala", st['cell_bold']),
    ]]
    for esc_name in todas_escalas:
        inc_en_esc = []
        exclusivos = []
        vecs_esc = set()
        for p in ordenados:
            mp = p.get('metadata_json') or {}
            incp = mp.get('incident_meta') or {}
            escs = [str(e) for e in (mp.get('escalas_detectadas') or incp.get('escalas_detectadas') or ['5min'])]
            if esc_name in escs:
                inc_en_esc.append(str(p.get('postmortem_id')))
                vecs_esc.add(str(p.get('tipo_ataque', 'N/A')))
                if len(escs) == 1:
                    exclusivos.append(str(p.get('postmortem_id')))
        if inc_en_esc:
            esc_stats_rows.append([
                Paragraph(f"<b>{_md_to_reportlab(esc_name)}</b>", st['cell']),
                Paragraph(f"{len(inc_en_esc)} de {n_total}", st['cell']),
                Paragraph(
                    f"{len(exclusivos)} ({_md_to_reportlab(', '.join(exclusivos[:3]))})" if exclusivos else "0 (correlacionados)",
                    st['cell'],
                ),
                Paragraph(_md_to_reportlab(", ".join(sorted(vecs_esc))), st['cell']),
            ])

    t_esc = Table(esc_stats_rows, colWidths=[95, 95, 150, 200])
    t_esc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), SAP_LIGHT),
        ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
    ]))
    story.append(t_esc)
    story.append(Spacer(1, 4))

    # Impacto por Aplicacion SAP
    app_map: Dict[str, Dict[str, Any]] = {}
    for p in ordenados:
        for a in (p.get('aplicaciones_afectadas') or []):
            info = app_map.setdefault(str(a), {'count': 0, 'vectores': set()})
            info['count'] += 1
            info['vectores'].add(str(p.get('tipo_ataque', 'N/A')))
    if app_map:
        app_rows = [[
            Paragraph("Aplicacion / Servicio SAP", st['cell_bold']),
            Paragraph("Incidentes Asociados", st['cell_bold']),
            Paragraph("Vectores de Amenaza Observados sobre el Activo", st['cell_bold']),
        ]]
        for app_name, info in sorted(app_map.items(), key=lambda kv: (-kv[1]['count'], kv[0])):
            app_rows.append([
                Paragraph(f"<b>{_md_to_reportlab(app_name)}</b>", st['cell']),
                Paragraph(f"{info['count']} incidente(s)", st['cell']),
                Paragraph(_md_to_reportlab(", ".join(sorted(info['vectores']))), st['cell']),
            ])
        t_apps = Table(app_rows, colWidths=[150, 95, 295])
        t_apps.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), SAP_LIGHT),
            ('BOX', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, SAP_BORDER),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2.2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ]))
        story.append(t_apps)
    story.append(Spacer(1, 6))

    # -------------------------------------------------------------------------
    # SECCION 5: PLAN MAESTRO CONSOLIDADO DE CONTENCION Y HARDENING DEL MES
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Plan Maestro Consolidado de Contencion y Hardening del Mes", st['sec_title']))
    acciones_por_camp = []
    for vec, lista in campanas_ordenadas:
        acciones_vec = []
        for p in lista:
            md = str(p.get('reporte_markdown') or '')
            if "## 5." in md:
                for l in md.split("## 5.")[1].splitlines():
                    ls = l.strip()
                    if ls and (ls.startswith('1.') or ls.startswith('2.') or ls.startswith('-') or ls.startswith('*')):
                        limpio = _md_to_reportlab(ls.lstrip('-*0123456789. ').strip())
                        if limpio and limpio not in acciones_vec:
                            acciones_vec.append(limpio)
                        if len(acciones_vec) >= 2:
                            break
            if len(acciones_vec) >= 2:
                break
        if acciones_vec:
            acciones_por_camp.append(
                f"<b>[{_md_to_reportlab(vec)} — {len(lista)} inc.]:</b> " + " &nbsp;|&nbsp; ".join(acciones_vec)
            )

    if not acciones_por_camp:
        acciones_por_camp = [
            "<b>[Proteccion Perimetral y WAF]:</b> Bloqueo automatizado de rangos IP anomalos y rate-limiting dinamico por endpoint SAP.",
            "<b>[Seguridad de Identidad y Contexto SAP]:</b> Enforzar MFA, rotacion de sesiones y validacion estricta de cabeceras y payloads.",
            "<b>[Guardrails de IA Generativa / LLM]:</b> Aislar directivas de sistema, filtrar Prompt Injection y limitar cuotas de tokens por SERVICE_ID.",
        ]

    t_act = Table([[Paragraph(a, st['body'])] for a in acciones_por_camp], colWidths=[540])
    t_act.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FEFCE8')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF08A')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FEF9C3')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_act)

    doc.build(story, canvasmaker=NumberedCanvas)
