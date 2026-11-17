# explicacion.py
"""
Z-scores de las ventanas anómalas contra todas las ventanas de su semana y
perfiles por familia de features usados en producción y el dashboard.

El contrafactual y diagnóstico interactivo siguen en sandbox/explicacion.py.
"""

from helpers.imports import pd, np

FAMILIAS_SYSTEM = {
    'Volumen y ritmo': ['TOTAL_REQUESTS', 'REQUEST_RATE_STD', 'INTERVAL_CV'],
    'Origen (IPs)': ['UNIQUE_IPS_COUNT', 'IP_ENTROPY', 'IP_TOP_SHARE'],
    'Tipo de log': ['RATIO_ERROR', 'RATIO_WARN', 'RATIO_SECURITY', 'RATIO_INFO', 'RATIO_AUDIT',
                    'JS_SAP_FUNCTION_LOG_TYPE'],
    'Respuestas HTTP': ['ERROR_RATE_400', 'ERROR_RATE_500', 'NON_200_RATE', 'JS_HTTP_STATUS_CODE'],
    'Peticiones': ['GET_VS_POST_RATIO', 'JS_HEADERS_HTTP_REQUEST_METHOD',
                   'JS_HEATHERS_REQUEST_PATH', 'JS_HEADERS_CONTENT_TYPE'],
    'Contexto SAP': ['JS_SAP_FUNCTION_APPLICATION', 'JS_SAP_SOURCE_TYPE', 'JS_SAP_APP_ENV',
                     'JS_REGION_CODE'],
}

FAMILIAS_LLM = {
    'Volumen y recursos': ['TOTAL_LLM_REQUESTS', 'AVG_LLM_COST_USD', 'AVG_TOKEN_RATIO',
                           'AVG_RESPONSE_TIME_MS'],
    'Fallos': ['LLM_ERROR_RATE', 'LLM_TIMEOUT_RATE'],
    'Contenido del prompt': ['AVG_PROMPT_LENGTH', 'MAX_PROMPT_LENGTH', 'PROMPT_CATEGORY_ENTROPY'],
    'Parámetros de la llamada': ['TEMPERATURE_VARIANCE', 'STREAM_RATIO'],
    'Mensajes de seguridad': ['INJECTION_COUNT', 'SAFETY_POLICY_COUNT', 'AUTH_FAILURE_COUNT',
                              'ACCESS_DENIED_COUNT', 'RATE_ABUSE_COUNT', 'DUPLICATE_REQUEST_COUNT',
                              'MALFORMED_INPUT_COUNT'],
}


def zscores(win_pdf, anomalous_idx, features, key='WINDOW_INDEX'):
    """
    Z-score de cada feature en las ventanas anómalas, con media y desviación (ddof=0) de TODAS
    las ventanas del batch. sd = 0 -> z = 0. Formato largo: WINDOW_INDEX, FEATURE, Z.
    """
    mu = win_pdf[features].mean()
    sd = win_pdf[features].std(ddof=0).replace(0, np.nan)
    a = win_pdf[win_pdf[key].isin(anomalous_idx)]
    z = ((a[features] - mu) / sd).fillna(0)
    z[key] = a[key].values
    out = z.melt(id_vars=key, var_name='FEATURE', value_name='Z')
    return out.rename(columns={key: 'WINDOW_INDEX'}).reset_index(drop=True)


def perfil_familias(if_df, z, familias, top=3):
    """
    Perfil descriptivo por evento: z-score promedio en valor absoluto de cada familia y
    TOP_FEATURES, las `top` features más desviadas con su z con signo (+ alto, - bajo).
    Requiere EVENTO_ID en if_df y WEEK en z.
    """
    keys = ['WEEK', 'WINDOW_INDEX']
    m = if_df.loc[if_df['LABEL'] == -1, keys + ['EVENTO_ID']]
    if z.empty or m.empty:
        return pd.DataFrame()
    ze = z.merge(m, on=keys)
    ze['ABSZ'] = ze['Z'].abs()
    feature_a_familia = {f: n for n, fs in familias.items() for f in fs}
    ze['FAMILIA'] = ze['FEATURE'].map(feature_a_familia)

    perfil = (ze.dropna(subset=['FAMILIA']).groupby(['EVENTO_ID', 'FAMILIA'])['ABSZ'].mean()
              .unstack('FAMILIA').reindex(columns=list(familias)))

    g = (ze.groupby(['EVENTO_ID', 'FEATURE']).agg(ABSZ=('ABSZ', 'mean'), Z=('Z', 'mean'))
         .reset_index().sort_values(['EVENTO_ID', 'ABSZ'], ascending=[True, False]))
    g['_t'] = g['FEATURE'] + ' (' + g['Z'].map('{:+.1f}'.format) + ')'
    top_txt = g.groupby('EVENTO_ID')['_t'].apply(lambda s: ', '.join(s.head(top))).rename('TOP_FEATURES')
    return perfil.join(top_txt).round(2).reset_index()
