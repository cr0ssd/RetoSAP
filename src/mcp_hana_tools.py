"""
mcp_hana_tools.py - Herramientas MCP (Model Context Protocol) para investigacion forense en SAP HANA Cloud.

Expone metodos estandarizados que un Agente de IA invoca para recopilar evidencia forense
sin cargar datasets masivos en el contexto del LLM:
  1. get_candidate_incidents: Identifica incidentes de seguridad de alta severidad.
  2. get_incident_windows_telemetry: Recupera la trayectoria temporal de las ventanas anomalas.
  3. sample_raw_logs_for_incident: Extrae logs crudos especificos del intervalo de tiempo del incidente.
  4. persist_postmortem_report: Guarda el reporte forense final en la tabla RESULT_POSTMORTEMS_<mes>.
"""

import sys
import os
import json
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Asegurar resolucion de modulos tanto desde src/ como desde la raiz
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, '..'))
for p in (REPO_ROOT, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from helpers.conexion import conexion
from helpers.clasificacion import TIPOS_NO_ATAQUE

log = logging.getLogger('mcp_hana_tools')


class HanaMcpTools:
    """
    Capa de herramientas MCP conectada a SAP HANA Cloud.
    """

    def __init__(self, conn=None):
        self.conn = conn if conn is not None else conexion()
        self._owns_conn = conn is None

    def close(self):
        if self._owns_conn and self.conn is not None:
            try:
                self.conn.close()
            except Exception as e:
                log.warning("[WARN] Error al cerrar conexion HANA: %s", e)

    # --------------------------------------------------------------------------
    # Tool 1: get_candidate_incidents
    # --------------------------------------------------------------------------
    def get_candidate_incidents(self, mes: str, stream: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Consulta los incidentes de seguridad mas criticos de la tabla RESULT_EVENTS_<mes>.
        Filtra RELEVANTE == 1 y excluye clasificaciones de no-ataque.
        """
        tabla_eventos = f"RESULT_EVENTS_{mes}"
        cond_stream = f"AND STREAM = '{stream}'" if stream else ""
        
        sql = f"""
        SELECT EVENTO_ID, STREAM, SOURCE_TABLE, WEEK, INICIO, FIN, N_VENTANAS,
               VENTANA_PICO, SCORE_PICO, Z_MAX, Z_MAX_FEATURE, RELEVANTE,
               TIPO_ATAQUE, MARGEN, CONFIANZA, TOP3
        FROM {tabla_eventos}
        WHERE RELEVANTE = 1
          {cond_stream}
        ORDER BY SCORE_PICO DESC, N_VENTANAS DESC
        """
        try:
            df = self.conn.sql(sql).collect()
        except Exception as e:
            log.error("[ERROR] Fallo al consultar %s: %s", tabla_eventos, e)
            return []

        if df.empty:
            return []

        # Filtrar no ataques
        df_ataques = df[~df['TIPO_ATAQUE'].isin(TIPOS_NO_ATAQUE)].copy()
        if df_ataques.empty:
            df_ataques = df.copy()

        candidatos = []
        for _, row in df_ataques.head(top_k).iterrows():
            candidatos.append({
                'evento_id': int(row['EVENTO_ID']),
                'stream': str(row['STREAM']),
                'source_table': str(row['SOURCE_TABLE']),
                'week': int(row['WEEK']) if pd.notna(row['WEEK']) else None,
                'inicio': str(row['INICIO']),
                'fin': str(row['FIN']),
                'n_ventanas': int(row['N_VENTANAS']),
                'ventana_pico': str(row['VENTANA_PICO']),
                'score_pico': float(row['SCORE_PICO']) if pd.notna(row['SCORE_PICO']) else 0.0,
                'z_max': float(row['Z_MAX']) if pd.notna(row['Z_MAX']) else 0.0,
                'z_max_feature': str(row['Z_MAX_FEATURE']) if pd.notna(row['Z_MAX_FEATURE']) else 'N/A',
                'tipo_ataque': str(row['TIPO_ATAQUE']) if pd.notna(row['TIPO_ATAQUE']) else 'Indeterminado',
                'confianza': str(row['CONFIANZA']) if pd.notna(row['CONFIANZA']) else 'baja',
                'top3': str(row['TOP3']) if pd.notna(row['TOP3']) else ''
            })
        return candidatos

    # --------------------------------------------------------------------------
    # Tool 2: get_incident_windows_telemetry
    # --------------------------------------------------------------------------
    def get_incident_windows_telemetry(self, mes: str, stream: str, evento_id: int) -> Dict[str, Any]:
        """
        Obtiene las ventanas asociadas al incidente desde RESULT_WINDOWS_<STREAM>_<mes>.
        Calcula metricas agregadas para contextualizar la dimension del impacto.
        """
        tabla_ventanas = f"RESULT_WINDOWS_{stream.upper()}_{mes}"
        sql = f"""
        SELECT *
        FROM {tabla_ventanas}
        WHERE EVENTO_ID = {evento_id}
        ORDER BY WINDOW_START ASC
        """
        try:
            df = self.conn.sql(sql).collect()
        except Exception as e:
            log.error("[ERROR] Fallo al consultar ventanas %s: %s", tabla_ventanas, e)
            return {'total_ventanas': 0, 'ventanas': [], 'metricas_resumen': {}}

        if df.empty:
            return {'total_ventanas': 0, 'ventanas': [], 'metricas_resumen': {}}

        # Resumen cuantitativo de metricas
        columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        resumen = {}
        for c in columnas_numericas:
            if c not in ('WINDOW_INDEX', 'WINDOW_SIZE_S', 'WEEK', 'EVENTO_ID', 'CLUSTER_ID'):
                resumen[c] = {
                    'mean': float(df[c].mean()) if pd.notna(df[c].mean()) else 0.0,
                    'max': float(df[c].max()) if pd.notna(df[c].max()) else 0.0,
                    'min': float(df[c].min()) if pd.notna(df[c].min()) else 0.0
                }

        # Extraer registros representativos de la serie de tiempo
        ventanas_info = []
        for _, row in df.iterrows():
            ventanas_info.append({
                'window_start': str(row.get('WINDOW_START', '')),
                'score': float(row['SCORE']) if 'SCORE' in row and pd.notna(row['SCORE']) else None,
                'label': int(row['LABEL']) if 'LABEL' in row and pd.notna(row['LABEL']) else None,
                'cluster_id': int(row['CLUSTER_ID']) if 'CLUSTER_ID' in row and pd.notna(row['CLUSTER_ID']) else None,
            })

        return {
            'total_ventanas': len(df),
            'ventanas': ventanas_info,
            'metricas_resumen': resumen
        }

    # --------------------------------------------------------------------------
    # Tool 3: sample_raw_logs_for_incident
    # --------------------------------------------------------------------------
    def sample_raw_logs_for_incident(self, source_table: str, inicio: str, fin: str,
                                     stream: str = 'system', limit: int = 30) -> Dict[str, Any]:
        """
        Extrae una muestra acotada de logs crudos de la tabla fuente (ej. LOGS_BATCH_1)
        dentro del intervalo temporal del incidente.
        """
        cols_query = """
            "TIMESTAMP", "SAP_FUNCTION_APPLICATION", "CLIENT_IP", "HTTP_STATUS_CODE",
            "HEATHERS_REQUEST_PATH", "SAP_FUNCTION_MESSAGE", "SAP_FUNCTION_LOG_TYPE",
            "LLM_PROMPT", "LLM_RESPONSE", "LLM_STATUS", "LLM_ERROR_MESSAGE",
            "LLM_MODEL", "LLM_TOTAL_TOKENS", "LLM_TOTAL_COST_USD"
        """

        # Filtro adicional segun stream para capturar evidencia real
        filtro_extra = ""
        if stream == 'llm':
            filtro_extra = "AND (LLM_STATUS IN ('error', 'timeout') OR LLM_ERROR_MESSAGE IS NOT NULL OR LLM_PROMPT IS NOT NULL)"
        else:
            filtro_extra = "AND (HTTP_STATUS_CODE >= 400 OR SAP_FUNCTION_LOG_TYPE IN ('ERROR', 'SECURITY', 'WARN'))"

        sql = f"""
        SELECT {cols_query}
        FROM "{source_table}"
        WHERE "TIMESTAMP" >= '{inicio}' AND "TIMESTAMP" <= '{fin}'
          {filtro_extra}
        ORDER BY "TIMESTAMP" ASC
        LIMIT {limit}
        """

        try:
            df = self.conn.sql(sql).collect()
        except Exception as e:
            # Fallback a consulta simple sin filtro extra
            log.warning("[WARN] Fallback en consulta de logs crudos: %s", e)
            sql_fallback = f"""
            SELECT {cols_query}
            FROM "{source_table}"
            WHERE "TIMESTAMP" >= '{inicio}' AND "TIMESTAMP" <= '{fin}'
            ORDER BY "TIMESTAMP" ASC
            LIMIT {limit}
            """
            try:
                df = self.conn.sql(sql_fallback).collect()
            except Exception as e2:
                log.error("[ERROR] Fallo total al recuperar logs crudos: %s", e2)
                return {'total_obtenidos': 0, 'logs_muestra': [], 'aplicaciones_afectadas': [], 'ips_involucradas': []}

        if df.empty:
            return {'total_obtenidos': 0, 'logs_muestra': [], 'aplicaciones_afectadas': [], 'ips_involucradas': []}

        apps_afectadas = [str(a) for a in df['SAP_FUNCTION_APPLICATION'].dropna().unique().tolist()]
        ips = [str(ip) for ip in df['CLIENT_IP'].dropna().unique().tolist() if str(ip) != '']

        # Limpiar diccionario para envio al LLM
        registros = []
        for _, r in df.iterrows():
            item = {}
            for col in df.columns:
                val = r[col]
                if pd.notna(val) and val is not None and str(val).strip() != '':
                    item[col] = str(val) if not isinstance(val, (int, float)) else val
            registros.append(item)

        return {
            'total_obtenidos': len(registros),
            'aplicaciones_afectadas': apps_afectadas,
            'ips_involucradas': ips[:15],
            'logs_muestra': registros
        }

    # --------------------------------------------------------------------------
    # Tool 4: persist_postmortem_report
    # --------------------------------------------------------------------------
    def persist_postmortem_report(self, mes: str, postmortem_dict: Dict[str, Any]) -> bool:
        """
        Crea (si no existe) y persiste el reporte Postmortem en la tabla RESULT_POSTMORTEMS_<mes>.
        Maneja insercion / actualizacion idempotente mediante UPSERT en SAP HANA Cloud.
        """
        tabla_pm = f"RESULT_POSTMORTEMS_{mes}"

        # 1. Asegurar tabla en HANA
        ddl = f"""
        CREATE COLUMN TABLE "{tabla_pm}" (
            "POSTMORTEM_ID" VARCHAR(100) NOT NULL,
            "EVENTO_ID" INTEGER,
            "STREAM" VARCHAR(20),
            "SOURCE_TABLE" VARCHAR(100),
            "TIPO_ATAQUE" VARCHAR(100),
            "MITRE_TACTIC" VARCHAR(250),
            "SEVERIDAD_SCORE" DOUBLE,
            "INICIO" TIMESTAMP,
            "FIN" TIMESTAMP,
            "APLICACIONES_AFECTADAS" NVARCHAR(1000),
            "IPS_INVOLUCRADAS" NVARCHAR(2000),
            "RESUMEN_EJECUTIVO" NVARCHAR(3000),
            "REPORTE_MARKDOWN" NCLOB,
            "METADATA_JSON" NCLOB,
            "FECHA_GENERACION" TIMESTAMP DEFAULT CURRENT_UTCTIMESTAMP,
            PRIMARY KEY ("POSTMORTEM_ID")
        )
        """
        try:
            cur = self.conn.connection.cursor()
            cur.execute(f"SELECT TABLE_NAME FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = '{tabla_pm}'")
            if not cur.fetchone():
                log.info("[DB] Creando tabla oficial de postmortems: %s", tabla_pm)
                cur.execute(ddl)
                self.conn.connection.commit()
            cur.close()
        except Exception as e:
            log.warning("[WARN] Verificacion DDL de %s: %s", tabla_pm, e)

        # 2. Preparar valores
        pm_id = postmortem_dict['postmortem_id']
        evento_id = int(postmortem_dict['evento_id'])
        stream = str(postmortem_dict['stream'])
        source_table = str(postmortem_dict.get('source_table', 'LOGS_BATCH_1'))
        tipo_ataque = str(postmortem_dict['tipo_ataque'])
        mitre_tactic = str(postmortem_dict.get('mitre_tactic', 'N/A'))
        severidad = float(postmortem_dict.get('severidad_score', 0.0))
        inicio = str(postmortem_dict['inicio'])
        fin = str(postmortem_dict['fin'])
        apps = ", ".join(postmortem_dict.get('aplicaciones_afectadas', []))[:990]
        ips = ", ".join(postmortem_dict.get('ips_involucradas', []))[:1990]
        resumen = str(postmortem_dict.get('resumen_ejecutivo', ''))[:2990]
        reporte_md = str(postmortem_dict.get('reporte_markdown', ''))
        metadata_str = json.dumps(postmortem_dict.get('metadata_json', {}), ensure_ascii=False)

        upsert_sql = f"""
        UPSERT "{tabla_pm}" (
            "POSTMORTEM_ID", "EVENTO_ID", "STREAM", "SOURCE_TABLE", "TIPO_ATAQUE",
            "MITRE_TACTIC", "SEVERIDAD_SCORE", "INICIO", "FIN", "APLICACIONES_AFECTADAS",
            "IPS_INVOLUCRADAS", "RESUMEN_EJECUTIVO", "REPORTE_MARKDOWN", "METADATA_JSON",
            "FECHA_GENERACION"
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_UTCTIMESTAMP
        ) WITH PRIMARY KEY
        """

        try:
            cur = self.conn.connection.cursor()
            cur.execute(upsert_sql, (
                pm_id, evento_id, stream, source_table, tipo_ataque,
                mitre_tactic, severidad, inicio, fin, apps,
                ips, resumen, reporte_md, metadata_str
            ))
            self.conn.connection.commit()
            cur.close()
            log.info("[OK] Postmortem %s persistido exitosamente en %s", pm_id, tabla_pm)
            return True
        except Exception as e:
            log.error("[ERROR] Fallo al hacer UPSERT de %s en %s: %s", pm_id, tabla_pm, e)
            return False
