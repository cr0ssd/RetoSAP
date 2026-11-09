# ============================================================
# 1. IMPORTACIONES
# ============================================================

from helpers.imports import pd

from helpers.conexion import conexion
from helpers.semanas import split_into_weeks
from helpers.IsolationForest import ModelIForest
from helpers.eventos import agrupar_eventos, resumen_eventos

from helpers.explicacion import (
    dominantes_por_ventana,
    explicar_eventos,
    perfil_familias,
    FAMILIAS_SYSTEM,
    FAMILIAS_LLM
)

from helpers.relevantes import marcar_relevantes, dataset_multiclase
from helpers.normalizacion import normalizar_por_semana
from helpers.clasificacion import (
    validar_firmas,
    clasificar_eventos,
    resumen_tipos
)


# ============================================================
# 2. CONFIGURACIÓN
# ============================================================

TABLA = 'LOGS_2026_10'

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

MAX_GAP = {'system': 3, 'llm': 3}

Z_UMBRAL = 5
CLIP_Z = None


# ============================================================
# 3. CARGA Y PREPARACIÓN DE DATOS
# ============================================================

def cargar_datos(conn, tabla):
    """Lee los registros de HANA y separa System y LLM."""

    dataset = conn.table(tabla)

    cols_drop_llm = [
        'HTTP_STATUS_CODE', 'SERVICE_ID', 'CLIENT_IP'
    ]

    cols_drop_system = [
        'LLM_ERROR_MESSAGE', 'LLM_RESPONSE_SIZE_BYTES',
        'SAP_LLM_RESPONSE_SIZE', 'SAP_LLM_RESPONSE_TIME',
        'LLM_PROMPT_TOKENS', 'LLM_COMPLETION_TOKENS',
        'LLM_TOTAL_TOKENS', 'LLM_RESPONSE_TIME_MS',
        'LLM_MODEL_ID', 'LLM_STATUS', 'LLM_COST_USD',
        'LLM_PROMPT_CATEGORY', 'LLM_TEMPERATURE', 'LLM_TOP_P',
        'LLM_STREAM', 'LLM_FINISH_REASON', 'LLM_PROMPT_ID',
        'LLM_PROVIDER', 'LLM_PROMPT',
    ]

    dataset = dataset.drop(['_IGNORED'])

    df_llm = dataset.filter(
        '"_INDEX" LIKE \'llm-logs%\''
    ).drop(cols_drop_llm)

    df_system = dataset.filter(
        '"_INDEX" LIKE \'sap-logs%\''
    ).drop(cols_drop_system)

    return df_system, df_llm


def preparar_semanas(conn, df_system, df_llm, tabla):
    """Divide los DataFrames en semanas."""

    system_weeks, llm_weeks, metadata = split_into_weeks(
        df_system=df_system,
        df_llm=df_llm,
        conn=conn,
        tabla=tabla,
        week_days=7,
        min_last_days=3
    )

    return system_weeks, llm_weeks, metadata


# ============================================================
# 4. DETECCIÓN DE ANOMALÍAS
# ============================================================

def detectar_anomalias(
    conn, metadata, system_weeks, llm_weeks, df_system
):
    """Entrena Isolation Forest y genera resultados y explicaciones."""

    res = ModelIForest(
        conn,
        metadata['n_weeks'],
        system_weeks,
        llm_weeks,
        df_system,
        FEATURES_SYSTEM,
        FEATURES_LLM
    )

    if_system, cf_system, z_system = (
        pd.concat(res['system'][k], ignore_index=True)
        for k in ('r', 'cf', 'z')
    )

    if_llm, cf_llm, z_llm = (
        pd.concat(res['llm'][k], ignore_index=True)
        for k in ('r', 'cf', 'z')
    )

    return if_system, cf_system, z_system, if_llm, cf_llm, z_llm


# ============================================================
# 5. GENERACIÓN DE EVENTOS
# ============================================================

def generar_eventos(if_system, if_llm):
    """Agrupa las ventanas anómalas en eventos."""

    if_system = agrupar_eventos(if_system, MAX_GAP['system'])
    if_llm = agrupar_eventos(if_llm, MAX_GAP['llm'])

    eventos_system = resumen_eventos(if_system)
    eventos_llm = resumen_eventos(if_llm)

    return if_system, if_llm, eventos_system, eventos_llm


# ============================================================
# 6. EXPLICACIÓN DE EVENTOS
# ============================================================

def explicar_resultados(
    if_system, cf_system, z_system,
    if_llm, cf_llm, z_llm,
    eventos_system, eventos_llm
):
    """Añade explicaciones por ventana y por evento."""

    # Explicación por ventana
    if_system = if_system.merge(
        dominantes_por_ventana(cf_system, z_system),
        on=['WEEK', 'WINDOW_INDEX'],
        how='left'
    )

    if_llm = if_llm.merge(
        dominantes_por_ventana(cf_llm, z_llm),
        on=['WEEK', 'WINDOW_INDEX'],
        how='left'
    )

    # Explicación por evento
    eventos_system = eventos_system.merge(
        explicar_eventos(if_system, cf_system, z_system),
        on='EVENTO_ID',
        how='left'
    )

    eventos_llm = eventos_llm.merge(
        explicar_eventos(if_llm, cf_llm, z_llm),
        on='EVENTO_ID',
        how='left'
    )

    return if_system, if_llm, eventos_system, eventos_llm


# ============================================================
# 7. IDENTIFICACIÓN DE EVENTOS RELEVANTES
# ============================================================

def seleccionar_relevantes(
    if_system, z_system, eventos_system,
    if_llm, z_llm, eventos_llm
):
    """Marca eventos relevantes y construye los datasets multiclase."""

    eventos_system = marcar_relevantes(
        if_system, z_system, eventos_system, Z_UMBRAL
    )

    eventos_llm = marcar_relevantes(
        if_llm, z_llm, eventos_llm, Z_UMBRAL
    )

    ds_multi_system, mapa_multi_system = dataset_multiclase(
        if_system, eventos_system
    )

    ds_multi_llm, mapa_multi_llm = dataset_multiclase(
        if_llm, eventos_llm
    )

    return (
        eventos_system, eventos_llm,
        ds_multi_system, ds_multi_llm,
        mapa_multi_system, mapa_multi_llm
    )


# ============================================================
# 8. PERFILES DE EVENTOS RELEVANTES
# ============================================================

def crear_perfiles_relevantes(
    eventos_system, if_system, z_system,
    eventos_llm, if_llm, z_llm
):
    """
    Calcula los perfiles por familia de los eventos relevantes.
    No realiza visualizaciones.
    """

    base_cols = [
        'EVENTO_ID', 'WEEK', 'INICIO', 'FIN',
        'N_VENTANAS', 'SCORE_PICO', 'Z_MAX'
    ]

    perfil_system = eventos_system.loc[
        eventos_system['RELEVANTE'], base_cols
    ].merge(
        perfil_familias(if_system, z_system, FAMILIAS_SYSTEM),
        on='EVENTO_ID',
        how='left'
    )

    perfil_llm = eventos_llm.loc[
        eventos_llm['RELEVANTE'], base_cols
    ].merge(
        perfil_familias(if_llm, z_llm, FAMILIAS_LLM),
        on='EVENTO_ID',
        how='left'
    )

    return perfil_system, perfil_llm


# ============================================================
# 9. NORMALIZACIÓN
# ============================================================

def normalizar_datasets(
    ds_multi_system, ds_multi_llm, if_system, if_llm
):
    """Normaliza los datasets multiclase por semana."""

    X_multi_system = normalizar_por_semana(
        ds_multi_system,
        if_system,
        FEATURES_SYSTEM,
        CLIP_Z
    )

    X_multi_llm = normalizar_por_semana(
        ds_multi_llm,
        if_llm,
        FEATURES_LLM,
        CLIP_Z
    )

    return X_multi_system, X_multi_llm


# ============================================================
# 10. CLASIFICACIÓN
# ============================================================

def clasificar(
    if_system, z_system, eventos_system,
    if_llm, z_llm, eventos_llm,
    mapa_multi_system, mapa_multi_llm
):
    """Clasifica los eventos y añade sus tipos a los mapas."""

    validar_firmas(FEATURES_SYSTEM, firmas='F_SYSTEM')
    validar_firmas(FEATURES_LLM, firmas='F_LLM')

    tipos_system = clasificar_eventos(
        if_system, z_system, eventos_system, firmas='F_SYSTEM'
    )

    tipos_llm = clasificar_eventos(
        if_llm, z_llm, eventos_llm, firmas='F_LLM'
    )

    nuevas = [
        'TIPO_ATAQUE', 'MARGEN', 'CONFIANZA',
        'SIN_SIGNO', 'TOP3', 'VOTOS'
    ]

    eventos_system = eventos_system.drop(
        columns=nuevas, errors='ignore'
    ).merge(tipos_system, on='EVENTO_ID', how='left')

    eventos_llm = eventos_llm.drop(
        columns=nuevas, errors='ignore'
    ).merge(tipos_llm, on='EVENTO_ID', how='left')

    # Se añade el tipo de ataque al mapa, no a X_multi.
    for mapa, tipos in (
        (mapa_multi_system, tipos_system),
        (mapa_multi_llm, tipos_llm)
    ):
        mapa['TIPO_ATAQUE'] = mapa['EVENTO_ID'].astype(int).map(
            tipos.set_index('EVENTO_ID')['TIPO_ATAQUE']
        )

    K_TIPOS = {
        'system': resumen_tipos(tipos_system),
        'llm': resumen_tipos(tipos_llm)
    }

    return eventos_system, eventos_llm, tipos_system, tipos_llm, K_TIPOS


# ============================================================
# 11. MAIN
# ============================================================

def main():

    # La conexión se crea una sola vez.
    conn = conexion()

    try:
        # 1. Carga de datos
        df_system, df_llm = cargar_datos(conn, TABLA)

        # 2. Preparación de semanas
        system_weeks, llm_weeks, metadata = preparar_semanas(
            conn, df_system, df_llm, TABLA
        )

        # 3. Detección de anomalías
        (
            if_system, cf_system, z_system,
            if_llm, cf_llm, z_llm
        ) = detectar_anomalias(
            conn, metadata, system_weeks, llm_weeks, df_system
        )

        # 4. Generación de eventos
        (
            if_system, if_llm,
            eventos_system, eventos_llm
        ) = generar_eventos(if_system, if_llm)

        # 5. Explicación de eventos
        (
            if_system, if_llm,
            eventos_system, eventos_llm
        ) = explicar_resultados(
            if_system, cf_system, z_system,
            if_llm, cf_llm, z_llm,
            eventos_system, eventos_llm
        )

        # 6. Selección de eventos relevantes
        (
            eventos_system, eventos_llm,
            ds_multi_system, ds_multi_llm,
            mapa_multi_system, mapa_multi_llm
        ) = seleccionar_relevantes(
            if_system, z_system, eventos_system,
            if_llm, z_llm, eventos_llm
        )

        # 7. Perfiles (opcional)
        # perfil_system, perfil_llm = crear_perfiles_relevantes(
        #     eventos_system, if_system, z_system,
        #     eventos_llm, if_llm, z_llm
        # )

        # 8. Normalización
        X_multi_system, X_multi_llm = normalizar_datasets(
            ds_multi_system, ds_multi_llm,
            if_system, if_llm
        )

        # 9. Clasificación
        (
            eventos_system, eventos_llm,
            tipos_system, tipos_llm, K_TIPOS
        ) = clasificar(
            if_system, z_system, eventos_system,
            if_llm, z_llm, eventos_llm,
            mapa_multi_system, mapa_multi_llm
        )

        # Se devuelven los resultados para utilizarlos fuera de main.
        return {
            'if_system': if_system,
            'if_llm': if_llm,
            'eventos_system': eventos_system,
            'eventos_llm': eventos_llm,
            'tipos_system': tipos_system,
            'tipos_llm': tipos_llm,
            'X_multi_system': X_multi_system,
            'X_multi_llm': X_multi_llm,
            'mapa_multi_system': mapa_multi_system,
            'mapa_multi_llm': mapa_multi_llm,
            'K_TIPOS': K_TIPOS,
            'metadata': metadata
        }

    finally:
        conn.close()


if __name__ == '__main__':
    resultados = main()