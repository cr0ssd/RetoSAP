"""Acceso a SAP HANA compartido por el dashboard y el módulo de health checks."""
import json

import pandas as pd
import streamlit as st

import hana_ml.dataframe as hana_df

from helpers.conexion import _params


@st.cache_resource(show_spinner=False)
def _crear_conexion():
    # Tiempo límite de 2 min por consulta: si la red se cae, falla en lugar de quedarse colgada.
    return hana_df.ConnectionContext(**_params(), communicationTimeout=120000)


def reiniciar_conexion():
    """Limpia la conexion en cache para forzar una reconexion limpia con SAP HANA."""
    _crear_conexion.clear()


def _conn():
    """Conexión guardada entre recargas; si dejó de responder (p. ej. tras horas sin uso), se rehace."""
    c = _crear_conexion()
    try:
        cur = c.connection.cursor()
        cur.execute("SELECT 1 FROM DUMMY")
        cur.close()
    except Exception:
        _crear_conexion.clear()
        c = _crear_conexion()
    return c


@st.cache_data(ttl=600, show_spinner="Cargando datos…")
def leer(tipo, mes):
    df = _conn().sql(f'SELECT * FROM "RESULT_{tipo}_{mes}"').collect()
    for c in ("WEEK_START", "WEEK_END", "WINDOW_START", "INICIO", "FIN", "VENTANA_PICO"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


_POSTMORTEM_COLS = [
    "POSTMORTEM_ID", "EVENTO_ID", "STREAM", "SOURCE_TABLE", "TIPO_ATAQUE",
    "MITRE_TACTIC", "SEVERIDAD_SCORE", "INICIO", "FIN", "APLICACIONES_AFECTADAS",
    "IPS_INVOLUCRADAS", "RESUMEN_EJECUTIVO", "REPORTE_MARKDOWN", "METADATA_JSON",
    "FECHA_GENERACION",
]


def _parse_json_safe(raw):
    """Convierte METADATA_JSON de HANA a dict de forma segura."""
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=120, show_spinner="Cargando reportes postmortem…")
def leer_postmortems(mes):
    """Lee RESULT_POSTMORTEMS_<mes> si existe en el esquema activo; si no, retorna vacío."""
    tabla = f"RESULT_POSTMORTEMS_{mes}"
    existe = _conn().sql(
        f"SELECT TABLE_NAME FROM M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
        f"AND TABLE_NAME = '{tabla}'"
    ).collect()
    if existe.empty:
        return pd.DataFrame(columns=_POSTMORTEM_COLS)
    df = _conn().sql(
        f'SELECT * FROM "{tabla}" ORDER BY SEVERIDAD_SCORE DESC, FECHA_GENERACION DESC'
    ).collect()
    for c in ("INICIO", "FIN", "FECHA_GENERACION"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df
