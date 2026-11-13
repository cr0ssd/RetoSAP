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

# Asegurar acceso a modulos de src/helpers si se ejecuta desde sandbox o raiz
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
SRC_DIR = os.path.join(REPO_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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
        Recupera el perfil temporal y estadistico de las ventanas anomalas de ese evento.
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
            log.warning("[WARN] No se pudo leer %s: %s", tabla_ventanas, e)
            return {'error': str(e), 'ventanas': 0}

        if df.empty:
            return {'ventanas': 0, 'metricas_resumen': {}}

        cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
        metricas = {}
        for col in cols_numericas:
            if col not in ['WINDOW_SIZE_S', 'WINDOW_INDEX', 'WEEK', 'LABEL', 'EVENTO_ID', 'CLUSTER_ID']:
                metricas[col] = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean())
                }

        resumen = {
            'evento_id': evento_id,
            'stream': stream,
            'ventanas_afectadas': len(df),
            'inicio_absoluto': str(df['WINDOW_START'].iloc[0]),
            'fin_absoluto': str(df['WINDOW_START'].iloc[-1]),
            'metricas_resumen': metricas
        }
        return resumen

    # --------------------------------------------------------------------------
    # Tool 3: sample_raw_logs_for_incident
    # --------------------------------------------------------------------------
    def sample_raw_logs_for_incident(self, source_table: str, inicio: str, fin: str,
                                     stream: str, limit: int = 35) -> Dict[str, Any]:
        """
        Consulta la tabla cruda fuente para extraer logs reales y patrones de evidencia.
        """
        ts_ini = inicio.replace(' ', 'T')
        if not ts_ini.endswith('Z') and '.' not in ts_ini:
            ts_ini += '.000Z'
        ts_fin = fin.replace(' ', 'T')
        if not ts_fin.endswith('Z') and '.' not in ts_fin:
            ts_fin += '.999Z'

        if stream == 'llm':
            cols = ["TIMESTAMP", "CLIENT_IP", "SAP_FUNCTION_APPLICATION", "LLM_MODEL_ID",
                    "LLM_PROMPT", "LLM_ERROR_MESSAGE", "SAP_FUNCTION_LOG_TYPE", "LLM_STATUS", "LLM_COST_USD"]
            filtro_adicional = "AND (LLM_PROMPT IS NOT NULL OR LLM_ERROR_MESSAGE IS NOT NULL)"
        else:
            cols = ["TIMESTAMP", "CLIENT_IP", "HTTP_STATUS_CODE", "HEADERS_HTTP_REQUEST_METHOD",
                    "HEATHERS_REQUEST_PATH", "SAP_FUNCTION_APPLICATION", "SAP_FUNCTION_LOG_TYPE",
                    "SAP_FUNCTION_MESSAGE", "REGION_CODE"]
            filtro_adicional = ""

        cols_str = ", ".join(cols)
        sql = f"""
        SELECT {cols_str}
        FROM {source_table}
        WHERE TIMESTAMP >= '{ts_ini}' AND TIMESTAMP <= '{ts_fin}'
          {filtro_adicional}
        LIMIT {limit}
        """

        try:
            df = self.conn.sql(sql).collect()
        except Exception as e:
            log.warning("[WARN] Fallo query con ISO format en %s: %s", source_table, e)
            sql_fallback = f"""
            SELECT {cols_str}
            FROM {source_table}
            WHERE TIMESTAMP >= '{inicio}' AND TIMESTAMP <= '{fin}'
              {filtro_adicional}
            LIMIT {limit}
            """
            try:
                df = self.conn.sql(sql_fallback).collect()
            except Exception as e2:
                log.error("[ERROR] Fallo definitivo muestreo logs: %s", e2)
                return {'logs': [], 'total_obtenidos': 0, 'error': str(e2)}

        logs_muestra = []
        for _, row in df.iterrows():
            item = {}
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    item[col] = None
                elif isinstance(val, (np.integer, int)):
                    item[col] = int(val)
                elif isinstance(val, (np.floating, float)):
                    item[col] = float(val)
                else:
                    item[col] = str(val)
            logs_muestra.append(item)

        # Analisis agregado de la muestra
        ips_distintas = list(set(log_item.get('CLIENT_IP') for log_item in logs_muestra if log_item.get('CLIENT_IP')))
        apps_afectadas = list(set(log_item.get('SAP_FUNCTION_APPLICATION') for log_item in logs_muestra if log_item.get('SAP_FUNCTION_APPLICATION')))
        
        return {
            'source_table': source_table,
            'inicio': inicio,
            'fin': fin,
            'total_obtenidos': len(logs_muestra),
            'ips_involucradas': ips_distintas[:10],
            'aplicaciones_afectadas': apps_afectadas,
            'logs_muestra': logs_muestra
        }

    # --------------------------------------------------------------------------
    # Tool 4: persist_postmortem_report
    # --------------------------------------------------------------------------
    def persist_postmortem_report(self, mes: str, report_data: Dict[str, Any]) -> bool:
        """
        Crea (si no existe) y guarda el registro forense en RESULT_POSTMORTEMS_<mes>.
        """
        tabla_destino = f"RESULT_POSTMORTEMS_{mes}"
        
        # 1. Asegurar tabla en HANA
        sql_create = f"""
        CREATE COLUMN TABLE {tabla_destino} (
            POSTMORTEM_ID VARCHAR(64) PRIMARY KEY,
            EVENTO_ID INTEGER,
            STREAM VARCHAR(16),
            SOURCE_TABLE VARCHAR(64),
            TIPO_ATAQUE VARCHAR(64),
            MITRE_TACTIC VARCHAR(64),
            SEVERIDAD_SCORE DOUBLE,
            INICIO TIMESTAMP,
            FIN TIMESTAMP,
            APLICACIONES_AFECTADAS VARCHAR(256),
            IPS_INVOLUCRADAS VARCHAR(512),
            RESUMEN_EJECUTIVO VARCHAR(1000),
            REPORTE_MARKDOWN NCLOB,
            METADATA_JSON NCLOB,
            FECHA_GENERACION TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            cursor = self.conn.connection.cursor()
            # Verificar si existe
            cursor.execute(f"SELECT COUNT(*) FROM SYS.M_TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA AND TABLE_NAME = '{tabla_destino}'")
            if cursor.fetchone()[0] == 0:
                log.info("[INFO] Creando tabla destino: %s", tabla_destino)
                cursor.execute(sql_create)
            
            # Borrar si ya existia para ese POSTMORTEM_ID (idempotencia)
            cursor.execute(f"DELETE FROM {tabla_destino} WHERE POSTMORTEM_ID = '{report_data['postmortem_id']}'")
            
            # Insertar registro
            sql_insert = f"""
            INSERT INTO {tabla_destino} (
                POSTMORTEM_ID, EVENTO_ID, STREAM, SOURCE_TABLE, TIPO_ATAQUE,
                MITRE_TACTIC, SEVERIDAD_SCORE, INICIO, FIN, APLICACIONES_AFECTADAS,
                IPS_INVOLUCRADAS, RESUMEN_EJECUTIVO, REPORTE_MARKDOWN, METADATA_JSON
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                report_data['postmortem_id'],
                int(report_data['evento_id']),
                report_data['stream'],
                report_data.get('source_table', 'N/A'),
                report_data['tipo_ataque'],
                report_data.get('mitre_tactic', 'TA0001: Initial Access'),
                float(report_data['severidad_score']),
                report_data['inicio'],
                report_data['fin'],
                ", ".join(report_data.get('aplicaciones_afectadas', []))[:250],
                ", ".join(report_data.get('ips_involucradas', []))[:500],
                report_data['resumen_ejecutivo'][:990],
                report_data['reporte_markdown'],
                json.dumps(report_data.get('metadata_json', {}))
            )
            
            cursor.execute(sql_insert, params)
            self.conn.connection.commit()
            log.info("[OK] Postmortem %s persistido exitosamente en %s", report_data['postmortem_id'], tabla_destino)
            return True
        except Exception as e:
            log.error("[ERROR] Fallo al persistir postmortem en HANA: %s", e)
            return False
