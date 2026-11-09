"""
ventanas.py - Generación de ventanas sobre SAP HANA (solo generación, sin diagnóstico).

Ventanas de tamaño configurable con traslape fijo del 50 % (paso = tamaño / 2): cada log cae en
2 ventanas consecutivas y se descartan las ventanas parciales de los bordes del batch. Todo se
calcula en HANA con SQL; solo se descarga al cliente la tabla final de ventanas.

Flujo (un batch = una semana, ya filtrada en un DataFrame de hana_ml):

    import ventanas
    hist = ventanas.prepare_system_history(conn, df_system)          # una vez por corrida
    for w in range(n_weeks):
        ventanas.prepare_system_base(conn, system_weeks[w])          # una vez por semana
        ventanas.prepare_llm_base(conn, llm_weeks[w])
        for size_s in ventanas.WINDOW_SIZES_S:                       # una vez por tamaño
            win = ventanas.build_system_windows(conn, size_s, hist)
            win = ventanas.materialize(conn, win, '#WIN_SYSTEM')     # o win.collect()
            ...

Qué se comparte y qué no:
  * Por corrida: primera aparición de IPs y rutas en TODO el historial (novel_*).
  * Por semana: parseo de timestamps, banderas y conteos de mensajes LLM (tablas #BASE_*).
  * Por tamaño: expansión a ventanas, intervalos y agregación (tablas #EV_*).
Las features no son aditivas entre tamaños (únicos, entropías, CV, divergencias), por eso las
ventanas grandes se recalculan desde los eventos y no se derivan de las de 5 min.

Decisiones de diseño:
  * js_* comparan cada ventana contra la distribución de SU PROPIO batch (semana) y tamaño.
  * novel_ips / novel_paths usan el historial completo: la primera aparición y el calentamiento
    de 24 h se miden desde el inicio de todos los datos, no del batch.
  * Las tablas temporales son locales a la sesión. El DataFrame que devuelve build_* las
    referencia, así que debe consumirse (collect/materialize) antes de pedir el siguiente
    tamaño o semana, que las reemplaza.
Los nombres de columna de salida van en MAYÚSCULAS (convención de HANA).
"""

WINDOW_SIZES_S = (300, 900, 1800, 3600, 10800)   # 5 min, 15 min, 30 min, 1 h, 3 h
WARMUP_S = 24 * 3600         # calentamiento para novel_*

# Columnas categóricas con las que se calcula js_* (system).
CATEGORICAL_COLS_SYSTEM = [
    'HTTP_STATUS_CODE', 'HEADERS_HTTP_REQUEST_METHOD', 'SAP_FUNCTION_LOG_TYPE',
    'HEATHERS_REQUEST_PATH', 'SAP_FUNCTION_APPLICATION', 'SAP_SOURCE_TYPE',
    'SAP_APP_ENV', 'REGION_CODE', 'HEADERS_HTTP_HOST', 'HEADERS_CONTENT_TYPE',
]

_EPOCH = "TO_TIMESTAMP('1970-01-01 00:00:00', 'YYYY-MM-DD HH24:MI:SS')"
_TS_FMT = "'YYYY-MM-DD HH24:MI:SS.FF3'"
SYSTEM_BASE_TABLE = '#BASE_SYSTEM'
LLM_BASE_TABLE = '#BASE_LLM'
SYSTEM_EVENTS_TABLE = '#EV_SYSTEM'
LLM_EVENTS_TABLE = '#EV_LLM'
FIRST_SEEN_TABLES = {'CLIENT_IP': '#FS_IP', 'HEATHERS_REQUEST_PATH': '#FS_PATH'}

# Conteos por ventana de mensajes LLM por tema (patrones LIKE sobre el mensaje en mayúsculas).
# Cada clave se convierte en una feature (suma de logs de la ventana que coinciden). Los temas se
# agruparon por significado a partir de los textos de LLM_ERROR; se pueden editar sin tocar el SQL.
LLM_MESSAGE_THEMES = {
    'INJECTION_COUNT': ['%INJECTION%', '%ADVERSARIAL%'],
    'SAFETY_POLICY_COUNT': ['%SAFETY%', '%CONTENT POLICY%', '%CONTENT MODERATION%', '%CONTENT FILTER%',
                            '%ACCEPTABLE USE%', '%TERMS OF SERVICE%'],
    'AUTH_FAILURE_COUNT': ['%API KEY%', '%TOKEN EXPIRED%', '%TOKEN HAS EXPIRED%', '%CREDENTIALS%',
                           '%SIGNATURE%', '%HMAC%'],
    'ACCESS_DENIED_COUNT': ['%WHITELIST%', '%ALLOWED LIST%', '%IP ADDRESS BLOCKED%', '%GEOGRAPHIC%',
                            '%REGIONAL COMPLIANCE%', '%PERMISSION%', '%DOES NOT ALLOW ACCESS%'],
    'RATE_ABUSE_COUNT': ['%RATE LIMIT%', '%TOO MANY SIMULTANEOUS%', '%CONCURRENT REQUEST%',
                         '%PARALLEL REQUEST%', '%API CALL LIMIT%', '%REQUEST QUOTA%',
                         '%INSUFFICIENT QUOTA%', '%ACCOUNT QUOTA%', '%BILLING LIMIT%',
                         '%TOKEN ALLOWANCE%', '%SESSION TOKEN LIMIT%', '%TOKEN BUDGET%'],
    'DUPLICATE_REQUEST_COUNT': ['%DUPLICATE REQUEST%', '%IDENTICAL REQUEST%', '%ALREADY PROCESSED%'],
    'MALFORMED_INPUT_COUNT': ['%MALFORMED%', '%REQUEST JSON%', '%NOT VALID JSON%', '%PAYLOAD SIZE%',
                              '%REQUEST BODY EXCEEDS%', '%PAYLOAD TOO LARGE%',
                              '%UNSUPPORTED MEDIA TYPE%', '%CONTENT-TYPE HEADER%'],
}


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def step_of(size_s):
    """Paso de la ventana: el traslape es siempre exactamente la mitad."""
    if size_s <= 0 or size_s % 2:
        raise ValueError('El tamaño de ventana debe ser un número par de segundos, no %r.' % size_s)
    return size_s // 2


def size_label(size_s):
    """300 -> '5min', 3600 -> '1h' (para nombres de tablas y gráficas)."""
    return '%dh' % (size_s // 3600) if size_s % 3600 == 0 else '%dmin' % (size_s // 60)


def _q(name):
    return '"%s"' % name


def _ts(col):
    """Texto ISO ('2026-10-01T00:10:20.000Z') -> TIMESTAMP. Nulo o vacío -> NULL."""
    c = _q(col)
    return ("CASE WHEN %s IS NULL OR %s = '' THEN NULL ELSE "
            "TO_TIMESTAMP(REPLACE(REPLACE(%s,'T',' '),'Z',''), %s) END" % (c, c, c, _TS_FMT))


def _num(col):
    """Texto o número -> DOUBLE; lo que no sea numérico -> NULL (como to_numeric coerce)."""
    s = "TO_NVARCHAR(%s)" % _q(col)
    return ("CASE WHEN %s LIKE_REGEXPR '^[-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?$' THEN TO_DOUBLE(%s) END" % (s, s))


def _floor_step(col, step_s):
    return ("ADD_SECONDS(%s, FLOOR(SECONDS_BETWEEN(%s, %s) / %d.0) * %d)"
            % (_EPOCH, _EPOCH, col, step_s, step_s))


def _run(conn, sql):
    """Ejecuta DDL levantando el error (execute_sql de hana_ml solo lo registra)."""
    with conn.connection.cursor() as cur:
        cur.execute(sql)
    if not conn.connection.getautocommit():
        conn.connection.commit()


def drop_temp(conn, table=SYSTEM_EVENTS_TABLE):
    """Borra la tabla temporal si existe."""
    try:
        _run(conn, 'DROP TABLE %s' % _q(table))
    except Exception:
        pass


def _create_temp(conn, table, select_sql):
    drop_temp(conn, table)
    _run(conn, 'CREATE LOCAL TEMPORARY COLUMN TABLE %s AS (%s)' % (_q(table), select_sql))
    return table


def _check_timestamps(conn, table):
    n_bad = conn.sql('SELECT COUNT(*) AS N FROM %s WHERE "TS" IS NULL' % _q(table)).collect().iloc[0, 0]
    if n_bad:
        raise ValueError('Hay %d logs sin un TIMESTAMP válido.' % n_bad)


# --------------------------------------------------------------------------
# 1) Expansión a ventanas solapadas (equivale a expand_to_windows)
# --------------------------------------------------------------------------
def _expand_sql(l1, size_s):
    """
    `l1` debe traer TS y WS0 (inicio de la ventana base de un paso).
    Duplica cada fila (WS = WS0 y WS = WS0 - paso) y conserva solo las ventanas
    completas: [WS, WS + tamaño) dentro del rango observado del batch.
    """
    step = step_of(size_s)
    return (
        'SELECT e.* FROM ('
        'SELECT t.*, t."WS0" AS "WS" FROM (%(l1)s) t '
        'UNION ALL '
        'SELECT t.*, ADD_SECONDS(t."WS0", -%(step)d) AS "WS" FROM (%(l1)s) t'
        ') e CROSS JOIN ('
        'SELECT MIN("WS0") AS "TMIN", '
        'ADD_SECONDS(%(epoch)s, CEIL(SECONDS_BETWEEN(%(epoch)s, MAX("TS")) / %(step)d.0) * %(step)d) AS "TMAX" '
        'FROM (%(l1)s) m'
        ') r WHERE e."WS" >= r."TMIN" AND ADD_SECONDS(e."WS", %(size)d) <= r."TMAX"'
        % {'l1': l1, 'step': step, 'size': size_s, 'epoch': _EPOCH}
    )


def _lag_sql(ev):
    """Intervalos (s) entre solicitudes consecutivas dentro de cada ventana."""
    return (
        'SELECT x.*, '
        'NANO100_BETWEEN(LAG(x."TS") OVER (PARTITION BY x."WS" ORDER BY x."TS"), x."TS") / 10000000.0 AS "D_TS", '
        'NANO100_BETWEEN(LAG(x."ET") OVER (PARTITION BY x."WS" ORDER BY x."ET"), x."ET") / 10000000.0 AS "D_ET" '
        'FROM (%s) x' % ev
    )


def _base_with_ws0(base, size_s):
    return 'SELECT b.*, %s AS "WS0" FROM %s b' % (_floor_step('b."TS"', step_of(size_s)), _q(base))


# --------------------------------------------------------------------------
# 2) Preparación de eventos (system): historial, base por semana y expansión por tamaño
# --------------------------------------------------------------------------
def prepare_system_history(conn, df_history):
    """
    Una vez por corrida. Calcula el inicio de todos los datos y la primera aparición de cada IP
    y de cada ruta en TODO el historial (tablas temporales #FS_*). Devuelve el contexto que
    build_system_windows necesita.
    """
    hist = 'SELECT %s AS "TS", "CLIENT_IP", "HEATHERS_REQUEST_PATH" FROM (%s) h0' % (
        _ts('TIMESTAMP'), df_history.select_statement)
    t0 = conn.sql('SELECT MIN("TS") AS T0 FROM (%s) z' % hist).collect().iloc[0, 0]
    t0_sql = "TO_TIMESTAMP('%s', 'YYYY-MM-DD HH24:MI:SS.FF3')" % t0.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    for col, table in FIRST_SEEN_TABLES.items():
        _create_temp(conn, table,
                     'SELECT %s AS "V", MIN("TS") AS "FS" FROM (%s) h WHERE %s IS NOT NULL GROUP BY %s'
                     % (_q(col), hist, _q(col), _q(col)))
    return {'t0_sql': t0_sql}


def prepare_system_base(conn, df_week, table=SYSTEM_BASE_TABLE):
    """Una vez por semana: eventos system con timestamps parseados y banderas, sin expandir."""
    l0 = (
        'SELECT %s AS "TS", %s AS "ET", "CLIENT_IP", %s AS "HTTP_STATUS_CODE", '
        '"HEADERS_HTTP_REQUEST_METHOD", "SAP_FUNCTION_LOG_TYPE", "HEATHERS_REQUEST_PATH", '
        '"SAP_FUNCTION_APPLICATION", "SAP_SOURCE_TYPE", "SAP_APP_ENV", "REGION_CODE", '
        '"HEADERS_HTTP_HOST", "HEADERS_CONTENT_TYPE" FROM (%s) s'
        % (_ts('TIMESTAMP'), _ts('EVENT_TIME_REQUESTED'), _num('HTTP_STATUS_CODE'),
           df_week.select_statement)
    )
    lt = 'UPPER("SAP_FUNCTION_LOG_TYPE")'
    base = (
        'SELECT s.*, '
        'CASE WHEN "HTTP_STATUS_CODE" IS NOT NULL AND "HTTP_STATUS_CODE" <> 200 THEN 1 ELSE 0 END AS "IS_NON200", '
        'CASE WHEN "HTTP_STATUS_CODE" BETWEEN 400 AND 499 THEN 1 ELSE 0 END AS "IS_4XX", '
        'CASE WHEN "HTTP_STATUS_CODE" BETWEEN 500 AND 599 THEN 1 ELSE 0 END AS "IS_5XX", '
        'CASE WHEN "SAP_FUNCTION_LOG_TYPE" IS NOT NULL THEN 1 ELSE 0 END AS "HAS_LT", '
        'CASE WHEN %s = \'ERROR\' THEN 1 ELSE 0 END AS "IS_ERROR", '
        'CASE WHEN %s = \'WARNING\' THEN 1 ELSE 0 END AS "IS_WARNING", '
        'CASE WHEN %s = \'SECURITY\' THEN 1 ELSE 0 END AS "IS_SECURITY", '
        'CASE WHEN %s = \'AUDIT\' THEN 1 ELSE 0 END AS "IS_AUDIT", '
        'CASE WHEN %s = \'INFO\' THEN 1 ELSE 0 END AS "IS_INFO", '
        'CASE WHEN UPPER("HEADERS_HTTP_REQUEST_METHOD") = \'GET\' THEN 1 ELSE 0 END AS "IS_GET" '
        'FROM (%s) s' % (lt, lt, lt, lt, lt, l0)
    )
    _create_temp(conn, table, base)
    _check_timestamps(conn, table)
    return table


def expand_system_events(conn, size_s, base=SYSTEM_BASE_TABLE, table=SYSTEM_EVENTS_TABLE):
    """Por tamaño: expande los eventos base a ventanas solapadas y calcula los intervalos."""
    return _create_temp(conn, table, _lag_sql(_expand_sql(_base_with_ws0(base, size_s), size_s)))


# --------------------------------------------------------------------------
# 3) Bloques de features (system)
# --------------------------------------------------------------------------
def _main_agg_sql(t):
    d = lambda a, b: 'TO_DOUBLE(SUM("%s")) / NULLIF(SUM("%s"), 0)' % (a, b)
    return (
        'SELECT "WS", COUNT(*) AS "TOTAL_REQUESTS", '
        'CASE WHEN COUNT("D_TS") > 1 THEN STDDEV("D_TS") END AS "REQUEST_RATE_STD", '
        'COUNT(DISTINCT "CLIENT_IP") AS "UNIQUE_IPS_COUNT", '
        '%s AS "ERROR_RATE_400", %s AS "ERROR_RATE_500", '
        '%s AS "RATIO_ERROR", %s AS "RATIO_WARN", %s AS "RATIO_SECURITY", '
        '%s AS "RATIO_INFO", %s AS "RATIO_AUDIT", '
        'TO_DOUBLE(SUM("IS_GET")) / COUNT(*) AS "GET_VS_POST_RATIO", '
        'TO_DOUBLE(SUM("IS_NON200")) / COUNT(*) AS "NON_200_RATE", '
        'CASE WHEN COUNT("D_ET") > 1 THEN STDDEV("D_ET") / NULLIF(AVG("D_ET"), 0) END AS "INTERVAL_CV" '
        'FROM %s GROUP BY "WS"'
        % (d('IS_4XX', 'IS_NON200'), d('IS_5XX', 'IS_NON200'),
           d('IS_ERROR', 'HAS_LT'), d('IS_WARNING', 'HAS_LT'), d('IS_SECURITY', 'HAS_LT'),
           d('IS_INFO', 'HAS_LT'), d('IS_AUDIT', 'HAS_LT'), _q(t))
    )


def _ip_sql(t):
    """Entropía de Shannon normalizada (0 si <=1 IP) y concentración de la IP más activa."""
    return (
        'SELECT "WS", '
        'CASE WHEN COUNT(*) > 1 THEN (-SUM("P" * LN("P"))) / LN(COUNT(*)) ELSE 0 END AS "IP_ENTROPY", '
        'MAX("P") AS "IP_TOP_SHARE" FROM ('
        'SELECT "WS", TO_DOUBLE("C") / SUM("C") OVER (PARTITION BY "WS") AS "P" FROM ('
        'SELECT "WS", "CLIENT_IP", COUNT(*) AS "C" FROM %s WHERE "CLIENT_IP" IS NOT NULL '
        'GROUP BY "WS", "CLIENT_IP") g) pp GROUP BY "WS"' % _q(t)
    )


def _js_sql(t, col):
    """
    Divergencia de Jensen-Shannon (bits, 0..1) entre la mezcla de categorías de cada
    ventana (P) y la del batch completo (Q). Las categorías que no aparecen en la ventana
    aportan 0.5 * Q cada una, de ahí el término 0.5 * (1 - SUM(Q en la ventana)).
    Para usar otra línea base (p. ej. un periodo fijo) basta cambiar cómo se calcula Q aquí.
    """
    cat = "COALESCE(TO_NVARCHAR(%s), '__NA__')" % _q(col)
    return (
        'SELECT "WS", SUM(0.5 * "P" * LN(2 * "P" / ("P" + "Q")) / LN(2) '
        '+ 0.5 * "Q" * LN(2 * "Q" / ("P" + "Q")) / LN(2)) + 0.5 * (1 - SUM("Q")) AS "JS_%s" FROM ('
        'SELECT "WS", "CAT", TO_DOUBLE("C") / SUM("C") OVER (PARTITION BY "WS") AS "P", '
        'TO_DOUBLE(SUM("C") OVER (PARTITION BY "CAT")) / SUM("C") OVER () AS "Q" FROM ('
        'SELECT "WS", %s AS "CAT", COUNT(*) AS "C" FROM %s GROUP BY "WS", %s) g) pq GROUP BY "WS"'
        % (col, cat, _q(t), cat)
    )


def _novel_sql(t, first_seen, t0_sql, col, alias):
    """Valores de `col` vistos por primera vez (en TODO el historial) dentro de la ventana."""
    return (
        'SELECT e."WS", COUNT(*) AS "%s" FROM ('
        'SELECT DISTINCT "WS", %s AS "V" FROM %s WHERE %s IS NOT NULL) e '
        'JOIN %s f ON f."V" = e."V" '
        'WHERE f."FS" >= e."WS" AND e."WS" >= ADD_SECONDS(%s, %d) GROUP BY e."WS"'
        % (alias, _q(col), _q(t), _q(col), _q(first_seen), t0_sql, WARMUP_S)
    )


# --------------------------------------------------------------------------
# 4) Ventanas system
# --------------------------------------------------------------------------
def build_system_windows(conn, size_s, history, base=SYSTEM_BASE_TABLE, table=SYSTEM_EVENTS_TABLE,
                         categorical_cols=CATEGORICAL_COLS_SYSTEM):
    """
    Genera las ventanas system de `size_s` segundos (paso = size_s / 2) para la semana ya
    preparada con prepare_system_base y devuelve un DataFrame de hana_ml (perezoso) con una fila
    por ventana, sin orden garantizado (ordenar por WINDOW_START al descargar, o usar materialize()).

    history: lo que devuelve prepare_system_history (una vez por corrida).
    """
    expand_system_events(conn, size_s, base, table)

    blocks = [('A', _main_agg_sql(table)), ('IP', _ip_sql(table))]
    for i, col in enumerate(categorical_cols):
        blocks.append(('J%d' % i, _js_sql(table, col)))
    blocks.append(('N1', _novel_sql(table, FIRST_SEEN_TABLES['CLIENT_IP'], history['t0_sql'],
                                    'CLIENT_IP', 'NOVEL_IPS')))
    blocks.append(('N2', _novel_sql(table, FIRST_SEEN_TABLES['HEATHERS_REQUEST_PATH'], history['t0_sql'],
                                    'HEATHERS_REQUEST_PATH', 'NOVEL_PATHS')))

    select = ['A."WS" AS "WINDOW_START"', 'A."TOTAL_REQUESTS"', 'A."REQUEST_RATE_STD"',
              'A."UNIQUE_IPS_COUNT"', 'A."ERROR_RATE_400"', 'A."ERROR_RATE_500"',
              'A."RATIO_ERROR"', 'A."RATIO_WARN"', 'A."RATIO_SECURITY"', 'A."RATIO_INFO"',
              'A."RATIO_AUDIT"', 'COALESCE(IP."IP_ENTROPY", 0) AS "IP_ENTROPY"',
              'A."GET_VS_POST_RATIO"']
    select += ['J%d."JS_%s"' % (i, c) for i, c in enumerate(categorical_cols)]
    select += ['IP."IP_TOP_SHARE"',
               'COALESCE(N1."NOVEL_IPS", 0) AS "NOVEL_IPS"',
               'COALESCE(N2."NOVEL_PATHS", 0) AS "NOVEL_PATHS"',
               'A."NON_200_RATE"', 'A."INTERVAL_CV"']

    joins = ' '.join('LEFT JOIN (%s) %s ON %s."WS" = A."WS"' % (sql, name, name)
                     for name, sql in blocks[1:])
    final = 'SELECT %s FROM (%s) A %s' % (', '.join(select), blocks[0][1], joins)
    return conn.sql(final)


# --------------------------------------------------------------------------
# 5) Ventanas LLM
# --------------------------------------------------------------------------
def _theme_flags(msg_expr):
    """Una bandera 0/1 por tema de LLM_MESSAGE_THEMES: el mensaje coincide con algún patrón."""
    cols = []
    for name, pats in LLM_MESSAGE_THEMES.items():
        cond = ' OR '.join("%s LIKE '%s'" % (msg_expr, p) for p in pats)
        cols.append('CASE WHEN %s THEN 1 ELSE 0 END AS "F_%s"' % (cond, name))
    return ', '.join(cols)


def prepare_llm_base(conn, df_week, table=LLM_BASE_TABLE):
    """
    Una vez por semana: eventos LLM con timestamps parseados, banderas y conteos de mensajes,
    sin expandir. No arrastra el texto del prompt, solo su longitud.
    """
    msg = 'UPPER(COALESCE("SAP_FUNCTION_MESSAGE", \'\'))'
    l0 = (
        'SELECT %s AS "TS", %s AS "COST", %s AS "COMP_TOK", %s AS "PROMPT_TOK", %s AS "RESP_MS", '
        '%s AS "TEMP", UPPER("LLM_STATUS") AS "STATUS", "LLM_PROMPT_CATEGORY" AS "CAT", '
        'LOWER(TO_NVARCHAR("LLM_STREAM")) AS "STREAM", LENGTH("LLM_PROMPT") AS "PROMPT_LEN", %s '
        'FROM (%s) s'
        % (_ts('TIMESTAMP'), _num('LLM_COST_USD'), _num('LLM_COMPLETION_TOKENS'),
           _num('LLM_PROMPT_TOKENS'), _num('LLM_RESPONSE_TIME_MS'), _num('LLM_TEMPERATURE'),
           _theme_flags(msg), df_week.select_statement)
    )
    base = (
        'SELECT s.*, '
        'CASE WHEN "STATUS" = \'ERROR\' THEN 1 ELSE 0 END AS "IS_ERROR", '
        'CASE WHEN "STATUS" = \'TIMEOUT\' THEN 1 ELSE 0 END AS "IS_TIMEOUT", '
        'CASE WHEN "STREAM" = \'true\' THEN 1 ELSE 0 END AS "IS_STREAM", '
        '"COMP_TOK" / NULLIF("PROMPT_TOK", 0) AS "TOKEN_RATIO" '
        'FROM (%s) s' % l0
    )
    _create_temp(conn, table, base)
    _check_timestamps(conn, table)
    return table


def expand_llm_events(conn, size_s, base=LLM_BASE_TABLE, table=LLM_EVENTS_TABLE):
    """Por tamaño: expande los eventos LLM base a ventanas solapadas."""
    return _create_temp(conn, table, _expand_sql(_base_with_ws0(base, size_s), size_s))


def _entropy_sql(t, col, alias):
    """Entropía de Shannon normalizada de `col` por ventana (0 si hay <=1 categoría)."""
    return (
        'SELECT "WS", CASE WHEN COUNT(*) > 1 THEN (-SUM("P" * LN("P"))) / LN(COUNT(*)) ELSE 0 END AS "%s" FROM ('
        'SELECT "WS", TO_DOUBLE("C") / SUM("C") OVER (PARTITION BY "WS") AS "P" FROM ('
        'SELECT "WS", %s, COUNT(*) AS "C" FROM %s WHERE %s IS NOT NULL GROUP BY "WS", %s) g) pp GROUP BY "WS"'
        % (alias, _q(col), _q(t), _q(col), _q(col))
    )


def build_llm_windows(conn, size_s, base=LLM_BASE_TABLE, table=LLM_EVENTS_TABLE):
    """
    Genera las ventanas LLM de `size_s` segundos (paso = size_s / 2) para la semana ya preparada
    con prepare_llm_base y devuelve un DataFrame de hana_ml (perezoso) con una fila por ventana,
    sin orden garantizado. Mismas precauciones que build_system_windows.
    """
    expand_llm_events(conn, size_s, base, table)
    t = _q(table)
    temas_agg = ', '.join('SUM("F_%s") AS "%s"' % (n, n) for n in LLM_MESSAGE_THEMES)
    temas_sel = ', '.join('A."%s"' % n for n in LLM_MESSAGE_THEMES)
    agg = (
        'SELECT "WS", COUNT(*) AS "TOTAL_LLM_REQUESTS", AVG("COST") AS "AVG_LLM_COST_USD", '
        + temas_agg + ', '
        'AVG("TOKEN_RATIO") AS "AVG_TOKEN_RATIO", AVG("RESP_MS") AS "AVG_RESPONSE_TIME_MS", '
        'TO_DOUBLE(SUM("IS_ERROR")) / COUNT(*) AS "LLM_ERROR_RATE", '
        'TO_DOUBLE(SUM("IS_TIMEOUT")) / COUNT(*) AS "LLM_TIMEOUT_RATE", '
        'CASE WHEN COUNT("TEMP") > 1 THEN POWER(STDDEV("TEMP"), 2) END AS "TEMPERATURE_VARIANCE", '
        'TO_DOUBLE(SUM("IS_STREAM")) / COUNT(*) AS "STREAM_RATIO", '
        'AVG("PROMPT_LEN") AS "AVG_PROMPT_LENGTH", MAX("PROMPT_LEN") AS "MAX_PROMPT_LENGTH" '
        'FROM ' + t + ' GROUP BY "WS"'
    )
    final = (
        'SELECT A."WS" AS "WINDOW_START", A."TOTAL_LLM_REQUESTS", A."AVG_LLM_COST_USD", '
        + temas_sel + ', '
        'A."AVG_TOKEN_RATIO", A."AVG_RESPONSE_TIME_MS", A."LLM_ERROR_RATE", '
        'A."LLM_TIMEOUT_RATE", A."TEMPERATURE_VARIANCE", '
        'COALESCE(E."PROMPT_CATEGORY_ENTROPY", 0) AS "PROMPT_CATEGORY_ENTROPY", '
        'A."STREAM_RATIO", A."AVG_PROMPT_LENGTH", A."MAX_PROMPT_LENGTH" '
        'FROM (' + agg + ') A LEFT JOIN (' + _entropy_sql(table, 'CAT', 'PROMPT_CATEGORY_ENTROPY')
        + ') E ON E."WS" = A."WS"'
    )
    return conn.sql(final)


# --------------------------------------------------------------------------
# 6) Materializar ventanas para el modelo
# --------------------------------------------------------------------------
def materialize(conn, win_df, name, index_col='WINDOW_INDEX'):
    """
    Guarda las ventanas en una tabla temporal local, con un índice entero 1..N ordenado
    por WINDOW_START (clave que pide el Isolation Forest). Las features se convierten a DOUBLE. Evita recalcular la consulta
    en cada paso del modelo y no deja tablas permanentes. Devuelve un DataFrame de hana_ml.
    """
    drop_temp(conn, name)
    # PAL solo acepta INTEGER/DOUBLE: COUNT() sale BIGINT y AVG() de enteros sale DECIMAL.
    feats = ', '.join('CAST(w.%s AS DOUBLE) AS %s' % (_q(c), _q(c))
                      for c in win_df.columns if c != 'WINDOW_START')
    sql = ('SELECT CAST(ROW_NUMBER() OVER (ORDER BY "WINDOW_START") AS INTEGER) AS "%s", '
           'w."WINDOW_START", %s FROM (%s) w' % (index_col, feats, win_df.select_statement))
    _run(conn, 'CREATE LOCAL TEMPORARY COLUMN TABLE %s AS (%s)' % (_q(name), sql))
    return conn.sql('SELECT * FROM %s' % _q(name))
