"""
email_sender.py - Envio transaccional de reportes Postmortem PDF por correo electronico.

Soporta tres proveedores usando solo la libreria estandar de Python (sin dependencias extra):
  1. Resend REST API (HTTPS 443, recomendado con dominio verificado o pruebas a tu correo).
  2. Brevo REST API  (HTTPS 443, permite enviar a cualquier destinatario verificando un solo correo).
  3. SMTP (STARTTLS/SSL, por ejemplo Gmail con App Password).

Variables de entorno soportadas (.env o Railway/CF):
  - EMAIL_PROVIDER  : auto (default) | resend | brevo | smtp
  - EMAIL_FROM      : Remitente (ej. "SAP SOC <reportes@tudominio.com>" o correo verificado)
  - RESEND_API_KEY  : API key de Resend (re_...)
  - BREVO_API_KEY   : API key v3 de Brevo (xkeysib-...)
  - SMTP_HOST       : Host SMTP (default: smtp.gmail.com)
  - SMTP_PORT       : Puerto SMTP (default: 587)
  - SMTP_USER       : Usuario SMTP
  - SMTP_PASSWORD   : Contrasena o App Password SMTP
"""

import base64
import html
import json
import os
import re
import smtplib
import ssl
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=False)

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
_DEFAULT_RESEND_FROM = "Forestia SOC Observability <no-reply@forestia.cloud>"
_USER_AGENT = "forestia-soc-observability-dashboard/1.0"
_MAX_RECIPIENTS = 10


class EmailSendError(RuntimeError):
    """Error controlado y legible al enviar un reporte por correo."""


def _ssl_context() -> ssl.SSLContext:
    """Crea contexto TLS usando certifi si esta disponible (evita fallos de CA en macOS)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def parsear_destinatarios(raw: str) -> Tuple[List[str], List[str]]:
    """Separa y valida correos ingresados por coma, punto y coma o salto de linea."""
    if not raw or not str(raw).strip():
        return [], []
    tokens = [t.strip() for t in re.split(r"[,;\n]+", str(raw)) if t.strip()]
    validos: List[str] = []
    invalidos: List[str] = []
    vistos = set()
    for token in tokens:
        low = token.lower()
        if low in vistos:
            continue
        vistos.add(low)
        if _EMAIL_RE.match(token):
            validos.append(token)
        else:
            invalidos.append(token)
    return validos, invalidos


def _resolver_proveedor() -> Optional[str]:
    """Determina el proveedor activo segun EMAIL_PROVIDER y credenciales cargadas."""
    load_dotenv(override=False)
    explicit = (os.getenv("EMAIL_PROVIDER") or "auto").strip().lower()
    if explicit in ("resend", "brevo", "smtp"):
        return explicit
    if os.getenv("RESEND_API_KEY", "").strip():
        return "resend"
    if os.getenv("BREVO_API_KEY", "").strip():
        return "brevo"
    if os.getenv("SMTP_USER", "").strip() and os.getenv("SMTP_PASSWORD", "").strip():
        return "smtp"
    return None


def estado_servicio_correo() -> Dict[str, Any]:
    """Devuelve el estado actual de configuracion sin exponer secretos."""
    prov = _resolver_proveedor()
    if prov == "resend":
        key_ok = bool(os.getenv("RESEND_API_KEY", "").strip())
        sender = (os.getenv("EMAIL_FROM") or os.getenv("RESEND_FROM") or _DEFAULT_RESEND_FROM).strip()
        sandbox = "onboarding@resend.dev" in sender.lower()
        hint = (
            "Modo sandbox Resend (onboarding@resend.dev): solo permite enviar al correo dueno de tu cuenta "
            "hasta que verifiques un dominio en Resend."
            if sandbox
            else f"Remitente configurado: {sender}"
        )
        return {
            "configured": key_ok,
            "provider": "resend",
            "provider_label": "Resend API",
            "sender": sender,
            "sandbox": sandbox,
            "hint": hint if key_ok else "Falta RESEND_API_KEY en .env o variables de entorno.",
        }
    if prov == "brevo":
        key_ok = bool(os.getenv("BREVO_API_KEY", "").strip())
        sender = (os.getenv("EMAIL_FROM") or os.getenv("BREVO_SENDER_EMAIL") or "").strip()
        ready = key_ok and bool(sender)
        return {
            "configured": ready,
            "provider": "brevo",
            "provider_label": "Brevo API",
            "sender": sender or "(sin EMAIL_FROM)",
            "sandbox": False,
            "hint": (
                f"Remitente verificado: {sender}"
                if ready
                else "Configura BREVO_API_KEY y EMAIL_FROM (correo verificado en Brevo) en .env."
            ),
        }
    if prov == "smtp":
        user = os.getenv("SMTP_USER", "").strip()
        pwd = os.getenv("SMTP_PASSWORD", "").strip()
        host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
        sender = (os.getenv("EMAIL_FROM") or user).strip()
        ready = bool(user and pwd)
        return {
            "configured": ready,
            "provider": "smtp",
            "provider_label": f"SMTP ({host})",
            "sender": sender or "(sin SMTP_USER)",
            "sandbox": False,
            "hint": (
                f"Servidor SMTP configurado: {host} ({sender})"
                if ready
                else "Configura SMTP_USER y SMTP_PASSWORD en .env."
            ),
        }
    return {
        "configured": False,
        "provider": None,
        "provider_label": "No configurado",
        "sender": "",
        "sandbox": False,
        "hint": (
            "Agrega RESEND_API_KEY (o BREVO_API_KEY + EMAIL_FROM) en tu archivo .env y en Railway "
            "para habilitar el envio en vivo."
        ),
    }


def _separar_nombre_y_correo(sender: str) -> Tuple[str, str]:
    """Extrae (nombre, correo) de 'Nombre <correo@dominio>' o devuelve ('Forestia SOC Observability', correo)."""
    m = re.match(r"^\s*(.*?)\s*<\s*([^<>]+)\s*>\s*$", sender)
    if m:
        name = m.group(1).strip().strip('"') or "Forestia SOC Observability"
        addr = m.group(2).strip()
        return name, addr
    return "Forestia SOC Observability", sender.strip()


def _construir_cuerpo_correo(
    periodo: str,
    n_incidentes: int,
    n_tacticas: int,
    n_apps: int,
    filename: str,
    nota_adicional: str = "",
) -> Tuple[str, str]:
    """Genera cuerpo HTML corporativo estilo SAP Horizon y respaldo en texto plano."""
    p_esc = html.escape(periodo)
    f_esc = html.escape(filename)
    nota_clean = (nota_adicional or "").strip()
    nota_html = ""
    nota_txt = ""
    if nota_clean:
        nota_html = (
            '<div style="margin:16px 0;padding:12px 14px;background:#F0F7FF;'
            'border-left:3px solid #0070F2;border-radius:6px;color:#1D2D3E;font-size:13px;">'
            f'<div style="font-size:11px;font-weight:700;letter-spacing:.04em;'
            f'text-transform:uppercase;color:#0070F2;margin-bottom:4px;">Nota del analista</div>'
            f'{html.escape(nota_clean).replace(chr(10), "<br/>")}</div>'
        )
        nota_txt = f"\nNota del analista:\n{nota_clean}\n"

    html_body = f"""<!doctype html>
<html lang="es">
<body style="margin:0;padding:24px;background:#F5F6F7;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;color:#1D2D3E;">
  <div style="max-width:600px;margin:0 auto;background:#FFFFFF;border:1px solid #E1E5EA;border-radius:10px;overflow:hidden;">
    <div style="background:#003366;padding:18px 24px;border-bottom:4px solid #0070F2;">
      <div style="font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#A6E0FF;">
        Forestia SOC Observability &middot; SAP Cloud Security
      </div>
      <div style="font-size:20px;font-weight:700;color:#FFFFFF;margin-top:4px;">
        Dossier Forense Postmortem — {p_esc}
      </div>
    </div>
    <div style="padding:22px 24px;font-size:14px;line-height:1.55;color:#1D2D3E;">
      <p style="margin:0 0 14px;">
        Se adjunta el reporte forense consolidado en formato PDF generado por el Agente Forense Autonomo
        (Isolation Forest en SAP HANA Cloud + Model Context Protocol).
      </p>
      <table role="presentation" style="width:100%;border-collapse:collapse;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px;margin:14px 0;">
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:12px;color:#556B82;font-weight:600;">Periodo analizado</td>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:13px;font-weight:700;color:#003366;">{p_esc}</td>
        </tr>
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:12px;color:#556B82;font-weight:600;">Incidentes incluidos</td>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:13px;font-weight:700;color:#003366;">{n_incidentes} incidente(s) unicos</td>
        </tr>
        <tr>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:12px;color:#556B82;font-weight:600;">Tacticas MITRE / Apps SAP</td>
          <td style="padding:10px 14px;border-bottom:1px solid #E2E8F0;font-size:13px;color:#1D2D3E;">{n_tacticas} tactica(s) · {n_apps} aplicacion(es) SAP</td>
        </tr>
        <tr>
          <td style="padding:10px 14px;font-size:12px;color:#556B82;font-weight:600;">Archivo adjunto</td>
          <td style="padding:10px 14px;font-size:13px;font-family:monospace;color:#003366;">{f_esc}</td>
        </tr>
      </table>
      {nota_html}
      <p style="margin:16px 0 0;font-size:12px;color:#64748B;">
        Documento confidencial de operaciones de seguridad (SOC). Generado automaticamente desde Forestia · SAP Cloud Observability.
      </p>
    </div>
  </div>
</body>
</html>"""

    text_body = (
        f"Forestia SOC Observability | SAP Cloud Security\n"
        f"Dossier Forense Postmortem — {periodo}\n\n"
        f"Se adjunta el reporte forense consolidado en PDF ({filename}).\n"
        f"- Periodo analizado: {periodo}\n"
        f"- Incidentes incluidos: {n_incidentes}\n"
        f"- Tacticas MITRE: {n_tacticas}\n"
        f"- Aplicaciones SAP afectadas: {n_apps}\n"
        f"{nota_txt}\n"
        f"Documento confidencial generado desde Forestia · SAP Cloud Observability."
    )
    return html_body, text_body


def _enviar_via_resend(
    destinatarios: List[str],
    asunto: str,
    html_body: str,
    text_body: str,
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    api_key = os.getenv("RESEND_API_KEY", "").strip()
    if not api_key:
        raise EmailSendError("RESEND_API_KEY no esta configurada en el entorno.")
    sender = (os.getenv("EMAIL_FROM") or os.getenv("RESEND_FROM") or _DEFAULT_RESEND_FROM).strip()

    payload = {
        "from": sender,
        "to": destinatarios,
        "subject": asunto,
        "html": html_body,
        "text": text_body,
        "attachments": [
            {
                "filename": filename,
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25, context=_ssl_context()) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
            return {
                "ok": True,
                "provider": "Resend API",
                "message_id": str(body.get("id") or ""),
                "recipients": destinatarios,
                "sender": sender,
            }
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            err_json = json.loads(raw_err)
            err_msg = str(err_json.get("message") or raw_err)
        except Exception:
            err_msg = raw_err
        low = err_msg.lower()
        if err.code == 403 and ("testing emails" in low or "verify a domain" in low or "own email" in low):
            raise EmailSendError(
                "Resend rechazo el envio porque la cuenta sigue en modo sandbox (onboarding@resend.dev), "
                "que solo permite enviar a tu propio correo registrado. Para enviar a cualquier correo "
                "verifica un dominio en Resend y define EMAIL_FROM, o configura BREVO_API_KEY + EMAIL_FROM."
            ) from err
        raise EmailSendError(f"Error HTTP {err.code} de Resend: {err_msg}") from err
    except urllib.error.URLError as err:
        raise EmailSendError(f"No se pudo conectar con api.resend.com: {err.reason}") from err


def _enviar_via_brevo(
    destinatarios: List[str],
    asunto: str,
    html_body: str,
    text_body: str,
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    api_key = os.getenv("BREVO_API_KEY", "").strip()
    sender_raw = (os.getenv("EMAIL_FROM") or os.getenv("BREVO_SENDER_EMAIL") or "").strip()
    if not api_key or not sender_raw:
        raise EmailSendError(
            "Para usar Brevo debes configurar BREVO_API_KEY y EMAIL_FROM (correo verificado en Brevo)."
        )
    sender_name, sender_email = _separar_nombre_y_correo(sender_raw)
    payload = {
        "sender": {"name": sender_name, "email": sender_email},
        "to": [{"email": addr} for addr in destinatarios],
        "subject": asunto,
        "htmlContent": html_body,
        "textContent": text_body,
        "attachment": [
            {
                "name": filename,
                "content": base64.b64encode(pdf_bytes).decode("ascii"),
            }
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=data,
        method="POST",
        headers={
            "api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=25, context=_ssl_context()) as resp:
            body = json.loads(resp.read().decode("utf-8") or "{}")
            return {
                "ok": True,
                "provider": "Brevo API",
                "message_id": str(body.get("messageId") or ""),
                "recipients": destinatarios,
                "sender": sender_email,
            }
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            err_msg = str(json.loads(raw_err).get("message") or raw_err)
        except Exception:
            err_msg = raw_err
        raise EmailSendError(f"Error HTTP {err.code} de Brevo: {err_msg}") from err
    except urllib.error.URLError as err:
        raise EmailSendError(f"No se pudo conectar con api.brevo.com: {err.reason}") from err


def _enviar_via_smtp(
    destinatarios: List[str],
    asunto: str,
    html_body: str,
    text_body: str,
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    host = (os.getenv("SMTP_HOST") or "smtp.gmail.com").strip()
    port = int(os.getenv("SMTP_PORT") or "587")
    user = os.getenv("SMTP_USER", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    sender = (os.getenv("EMAIL_FROM") or user).strip()
    if not user or not password:
        raise EmailSendError("Para usar SMTP debes configurar SMTP_USER y SMTP_PASSWORD.")

    msg = EmailMessage()
    msg["Subject"] = asunto
    msg["From"] = sender
    msg["To"] = ", ".join(destinatarios)
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename=filename,
    )

    ctx = _ssl_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=25, context=ctx) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=25) as server:
                server.ehlo()
                server.starttls(context=ctx)
                server.ehlo()
                server.login(user, password)
                server.send_message(msg)
        return {
            "ok": True,
            "provider": f"SMTP ({host})",
            "message_id": "",
            "recipients": destinatarios,
            "sender": sender,
        }
    except Exception as err:
        raise EmailSendError(f"Falla al enviar por SMTP ({host}:{port}): {err}") from err


def enviar_postmortem_pdf(
    destinatarios_raw: str,
    pdf_bytes: bytes,
    filename: str,
    periodo: str,
    n_incidentes: int,
    n_tacticas: int = 0,
    n_apps: int = 0,
    asunto: Optional[str] = None,
    nota_adicional: str = "",
) -> Dict[str, Any]:
    """Valida destinatarios, arma el correo HTML+PDF y despacha por el proveedor configurado."""
    validos, invalidos = parsear_destinatarios(destinatarios_raw)
    if invalidos:
        raise EmailSendError(
            f"Direccion(es) de correo invalida(s): {', '.join(invalidos)}. Revisa el formato."
        )
    if not validos:
        raise EmailSendError("Ingresa al menos una direccion de correo electronico valida.")
    if len(validos) > _MAX_RECIPIENTS:
        raise EmailSendError(f"Maximo {_MAX_RECIPIENTS} destinatarios por envio.")
    if not pdf_bytes:
        raise EmailSendError("El archivo PDF generado esta vacio y no puede adjuntarse.")

    prov = _resolver_proveedor()
    if not prov:
        raise EmailSendError(
            "No hay ningun proveedor de correo configurado. Agrega RESEND_API_KEY (o BREVO_API_KEY + EMAIL_FROM) "
            "en tu archivo .env y en las variables de Railway."
        )

    subject = (asunto or "").strip() or f"Reporte Forense Postmortem SOC — {periodo} ({n_incidentes} incidentes)"
    html_body, text_body = _construir_cuerpo_correo(
        periodo=periodo,
        n_incidentes=n_incidentes,
        n_tacticas=n_tacticas,
        n_apps=n_apps,
        filename=filename,
        nota_adicional=nota_adicional,
    )

    if prov == "resend":
        return _enviar_via_resend(validos, subject, html_body, text_body, pdf_bytes, filename)
    if prov == "brevo":
        return _enviar_via_brevo(validos, subject, html_body, text_body, pdf_bytes, filename)
    if prov == "smtp":
        return _enviar_via_smtp(validos, subject, html_body, text_body, pdf_bytes, filename)
    raise EmailSendError(f"Proveedor de correo no soportado: {prov}")
