"""
workflow.py - Pipeline de producción: detección de anomalías en logs SAP.

Desde una tabla de logs ya cargada en HANA hasta las tablas RESULT_<TIPO>_<AAAA>_<MM>.
Una sola corrida procesa todos los tamaños de ventana sin releer la BD por cada tamaño.

Uso (desde la raíz del repo):
    python src/workflow.py                       # usa TABLA por defecto
    python src/workflow.py --tabla LOGS_2026_11
    python src/workflow.py --tabla LOGS_2026_10 --reset   # recrea las tablas de los meses

Resultados en memoria: R[size_s][stream] con
    'if'      ventanas con SCORE, LABEL y EVENTO_ID
    'z'       z-scores (largo) de las ventanas anómalas
    'ev'      eventos con relevancia y clasificación
    'ds'      ventanas anómalas de eventos relevantes (dataset multiclase)
    'mapa'    WEEK, WINDOW_START, EVENTO_ID, TIPO_ATAQUE, CLUSTER_ID de esas ventanas
    'X'       'ds' normalizado por semana (features del IF)
    'tipos'   clasificación por evento relevante
    'estados' estado del clustering por semana
"""

# ============================================================
# 1. IMPORTACIONES
# ============================================================

import argparse
import logging
import time

from helpers.imports import pd

from helpers.conexion import conexion
from helpers.semanas import split_into_weeks
from helpers import ventanas
from helpers.IsolationForest import ModelIForest, STREAMS
from helpers.eventos import agrupar_eventos, resumen_eventos
from helpers.relevantes import marcar_relevantes, dataset_multiclase
from helpers.normalizacion import normalizar_por_semana
from helpers.clasificacion import validar_firmas, clasificar_eventos, resumen_tipos
from helpers import clustering
from helpers import resultados


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

TABLA = 'LOGS_BATCH_1'
RESET = False        # True tira y recrea las tablas de los meses de la corrida (borra TODOS sus datasets)

WEEK_DAYS = 7
MIN_LAST_DAYS = 3    # un sobrante de menos de 3 días se pega a la semana anterior

SIZES = ventanas.WINDOW_SIZES_S          # (300, 900, 1800, 3600, 10800): 5 min ... 3 h

FEATURES_SYSTEM = [
    'TOTAL_REQUESTS', 'REQUEST_RATE_STD', 'INTERVAL_CV',
    'UNIQUE_IPS_COUNT', 'IP_ENTROPY', 'IP_TOP_SHARE',
    'ERROR_RATE_400', 'ERROR_RATE_500', 'NON_200_RATE',
    'RATIO_ERROR', 'RATIO_WARN', 'RATIO_SECURITY', 'RATIO_INFO',
    'RATIO_AUDIT', 'GET_VS_POST_RATIO',
    'JS_HTTP_STATUS_CODE', 'JS_HEADERS_HTTP_REQUEST_METHOD',
    'JS_SAP_FUNCTION_LOG_TYPE', 'JS_HEATHERS_REQUEST_PATH',
    'JS_SAP_FUNCTION_APPLICATION', 'JS_SAP_SOURCE_TYPE',
    'JS_SAP_APP_ENV', 'JS_REGION_CODE', 'JS_HEADERS_CONTENT_TYPE',
]

FEATURES_LLM = [
    'TOTAL_LLM_REQUESTS', 'AVG_LLM_COST_USD', 'INJECTION_COUNT',
    'SAFETY_POLICY_COUNT', 'AUTH_FAILURE_COUNT', 'ACCESS_DENIED_COUNT',
    'RATE_ABUSE_COUNT', 'DUPLICATE_REQUEST_COUNT',
    'MALFORMED_INPUT_COUNT', 'AVG_TOKEN_RATIO',
    'AVG_RESPONSE_TIME_MS', 'LLM_ERROR_RATE', 'LLM_TIMEOUT_RATE',
    'TEMPERATURE_VARIANCE', 'PROMPT_CATEGORY_ENTROPY',
    'STREAM_RATIO', 'AVG_PROMPT_LENGTH', 'MAX_PROMPT_LENGTH',
]

FEATURES = {'system': FEATURES_SYSTEM, 'llm': FEATURES_LLM}
FIRMAS = {'system': 'F_SYSTEM', 'llm': 'F_LLM'}

# En ventanas consecutivas; con traslape del 50 %, 3 pasos = 1.5 tamaños de ventana.
MAX_GAP = {s: {'system': 3, 'llm': 3} for s in SIZES}

# |z| máximo que debe alcanzar un evento para ser relevante; uno por tamaño
# (el |z| máximo posible es sqrt(n-1) y n cambia mucho con el tamaño).
Z_UMBRAL = {s: 5 for s in SIZES}

CLIP_Z = None        # p. ej. 10 para recortar colas largas; None = sin recorte

COLS_DROP_LLM = ['HTTP_STATUS_CODE', 'SERVICE_ID', 'CLIENT_IP']

COLS_DROP_SYSTEM = [
    'LLM_ERROR_MESSAGE', 'LLM_RESPONSE_SIZE_BYTES', 'SAP_LLM_RESPONSE_SIZE',
    'SAP_LLM_RESPONSE_TIME', 'LLM_PROMPT_TOKENS', 'LLM_COMPLETION_TOKENS',
    'LLM_TOTAL_TOKENS', 'LLM_RESPONSE_TIME_MS', 'LLM_MODEL_ID', 'LLM_STATUS',
    'LLM_COST_USD', 'LLM_PROMPT_CATEGORY', 'LLM_TEMPERATURE', 'LLM_TOP_P',
    'LLM_STREAM', 'LLM_FINISH_REASON', 'LLM_PROMPT_ID', 'LLM_PROVIDER', 'LLM_PROMPT',
]

log = logging.getLogger('workflow')


def _iter(R):
    """Recorre R como (tamaño, stream, R[tamaño][stream])."""
    for s in SIZES:
        for st in STREAMS:
            yield s, st, R[s][st]


# ============================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def cargar_datos(conn, tabla):
    """Lee los registros de HANA y separa System y LLM."""
    dataset = conn.table(tabla).drop(['_IGNORED'])
    df_llm = dataset.filter('"_INDEX" LIKE \'llm-logs%\'').drop(COLS_DROP_LLM)
    df_system = dataset.filter('"_INDEX" LIKE \'sap-logs%\'').drop(COLS_DROP_SYSTEM)
    return df_system, df_llm


def preparar_semanas(conn, df_system, df_llm, tabla):
    """Divide system y LLM en semanas compartidas y devuelve la tabla SEMANAS."""
    system_weeks, llm_weeks, semanas = split_into_weeks(
        df_system=df_system, df_llm=df_llm, conn=conn, tabla=tabla,
        week_days=WEEK_DAYS, min_last_days=MIN_LAST_DAYS,
    )
    log.info('%d semana(s): %s -> %s', len(semanas),
             semanas['WEEK_START'].iloc[0], semanas['WEEK_END'].iloc[-1])
    return system_weeks, llm_weeks, semanas


# ============================================================
# 4. DETECCIÓN DE ANOMALÍAS (ventanas + Isolation Forest)
# ============================================================

def detectar_anomalias(conn, semanas, system_weeks, llm_weeks):
    """Ventanas de todos los tamaños e IF por semana x tamaño x stream. Devuelve R."""
    return ModelIForest(conn, len(semanas), system_weeks, llm_weeks, FEATURES, SIZES)


# ============================================================
# 5. GENERACIÓN DE EVENTOS
# ============================================================

def generar_eventos(R):
    """Agrupa las ventanas anómalas en eventos (no cruzan semanas)."""
    for s, st, d in _iter(R):
        d['if'] = agrupar_eventos(d['if'], MAX_GAP[s][st])
        d['ev'] = resumen_eventos(d['if'])


# ============================================================
# 6. EVENTOS RELEVANTES Y DATASET MULTICLASE
# ============================================================

def seleccionar_relevantes(R):
    """Marca eventos relevantes (Z_MAX >= Z_UMBRAL[tamaño]) y arma el dataset multiclase."""
    for s, st, d in _iter(R):
        d['ev'] = marcar_relevantes(d['if'], d['z'], d['ev'], Z_UMBRAL[s])
        d['ds'], d['mapa'] = dataset_multiclase(d['if'], d['ev'])


# ============================================================
# 7. NORMALIZACIÓN
# ============================================================

def normalizar_datasets(R):
    """Z-score con signo por semana de las features del IF (para el clustering)."""
    for s, st, d in _iter(R):
        feats = FEATURES[st]
        d['X'] = normalizar_por_semana(d['ds'], d['if'], feats, CLIP_Z)
        assert len(d['X']) == len(d['ds']) == len(d['mapa']), f'{s} {st}: filas desalineadas'
        assert list(d['X'].columns) == feats, f'{s} {st}: columnas distintas a las del IF'
        assert not d['X'].isna().any().any(), f'{s} {st}: hay NaN'


# ============================================================
# 8. CLASIFICACIÓN DEL TIPO DE ATAQUE
# ============================================================

NUEVAS = ['TIPO_ATAQUE', 'MARGEN', 'CONFIANZA', 'SIN_SIGNO', 'TOP3', 'VOTOS']


def clasificar(R):
    """Clasifica por reglas los eventos relevantes y agrega el tipo a eventos y mapa."""
    for st in STREAMS:
        validar_firmas(FEATURES[st], firmas=FIRMAS[st])

    for s, st, d in _iter(R):
        d['tipos'] = clasificar_eventos(d['if'], d['z'], d['ev'], firmas=FIRMAS[st])
        d['ev'] = (d['ev'].drop(columns=NUEVAS, errors='ignore')
                   .merge(d['tipos'], on='EVENTO_ID', how='left'))
        # El tipo va al mapa, no a X (X solo lleva las features del IF).
        d['mapa']['TIPO_ATAQUE'] = d['mapa']['EVENTO_ID'].astype(int).map(
            d['tipos'].set_index('EVENTO_ID')['TIPO_ATAQUE'])


# ============================================================
# 9. CLUSTERING DE CONSISTENCIA (K-means PAL) Y VALIDACIÓN
# ============================================================

def clusterizar(conn, R):
    """
    Un K-means PAL por semana (K = tipos presentes) sobre X, y métricas de acuerdo
    tipo vs. cluster. Devuelve la tabla `validacion` que espera resultados.construir.
    """
    partes = []
    for s, st, d in _iter(R):
        d['mapa']['CLUSTER_ID'], d['estados'] = clustering.clusterizar_semanas(conn, d['X'], d['mapa'])
        partes.append(clustering.validar_semanas(d['mapa'], d['estados'])
                      .assign(tamaño=ventanas.size_label(s), stream=st))
    return pd.concat(partes, ignore_index=True)


# ============================================================
# 10. CARGA DE RESULTADOS A HANA
# ============================================================

def cargar_resultados(conn, tabla, R, semanas, validacion, reset=RESET):
    """Arma las tablas RESULT_* por mes y las carga de forma incremental por SOURCE_TABLE."""
    tablas = resultados.construir(tabla, R, SIZES, semanas, Z_UMBRAL, MAX_GAP, validacion)
    carga = resultados.guardar(conn, tablas, tabla, reset=reset)
    if not carga['ok'].all():
        raise RuntimeError('La carga no coincide con HANA:\n%s' % carga[~carga['ok']].to_string())
    return carga


# ============================================================
# 11. LOGGING
# ============================================================

def resumen_corrida(R):
    """Conteos por tamaño y stream (solo para el log)."""
    filas = []
    for s, st, d in _iter(R):
        k = resumen_tipos(d['tipos'])
        ev = d['estados']
        filas.append({
            'tamaño': ventanas.size_label(s), 'stream': st,
            'ventanas': len(d['if']), 'anomalías': int((d['if']['LABEL'] == -1).sum()),
            'eventos': len(d['ev']), 'relevantes': int(d['ev']['RELEVANTE'].sum()),
            'tipos': k['tipos'], 'ataques': k['ataques'],
            'semanas_evaluables': int((ev['estado'] == 'evaluable').sum()),
        })
    return pd.DataFrame(filas)


# ============================================================
# 12. MAIN
# ============================================================

def main(tabla=TABLA, reset=RESET):
    t_total = time.perf_counter()
    conn = conexion()          # una sola conexión: las tablas temporales viven en la sesión

    try:
        log.info('Tabla fuente: %s', tabla)

        # 1. Carga y separación system / LLM
        df_system, df_llm = cargar_datos(conn, tabla)

        # 2. Semanas
        system_weeks, llm_weeks, semanas = preparar_semanas(conn, df_system, df_llm, tabla)

        # 3. Ventanas + Isolation Forest (todas las semanas, tamaños y streams)
        R = detectar_anomalias(conn, semanas, system_weeks, llm_weeks)

        # 4. Eventos
        generar_eventos(R)

        # 5. Relevancia y dataset multiclase
        seleccionar_relevantes(R)

        # 6. Normalización
        normalizar_datasets(R)

        # 7. Clasificación
        clasificar(R)

        # 8. Clustering y validación
        validacion = clusterizar(conn, R)

        with pd.option_context('display.width', 200, 'display.max_columns', None):
            log.info('Resumen de la corrida:\n%s', resumen_corrida(R).to_string(index=False))

        # 9. Carga a HANA
        carga = cargar_resultados(conn, tabla, R, semanas, validacion, reset=reset)
        log.info('Carga a HANA:\n%s', carga.to_string(index=False))

        log.info('Listo en %.0f s', time.perf_counter() - t_total)
        return {'R': R, 'semanas': semanas, 'validacion': validacion, 'carga': carga}

    finally:
        conn.close()


def _args():
    p = argparse.ArgumentParser(description='Detección de anomalías en logs SAP (producción).')
    p.add_argument('--tabla', default=TABLA, help='tabla de logs en HANA (default: %(default)s)')
    p.add_argument('--reset', action='store_true',
                   help='tira y recrea las tablas RESULT_* de los meses de la corrida')
    return p.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        force=True)   # hana_ml puede registrar handlers antes y ocultar el log
    logging.getLogger('hana_ml').setLevel(logging.WARNING)   # sin el SQL de cada llamada a PAL
    args = _args()
    salida = main(tabla=args.tabla, reset=args.reset or RESET)
