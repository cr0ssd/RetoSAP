"""
Sección 8: ensambla las tablas de resultados del pipeline y las carga a SAP HANA.

Una tabla por tipo de resultado y por mes: RESULT_<TIPO>_<AAAA>_<MM>. El mes sale de los datos
(WEEK_START, inicio absoluto de cada semana); una semana va completa al mes en que empieza, así
que un dataset que cruza dos meses escribe en las dos tablas. Los tamaños de ventana van juntos
y se distinguen con WINDOW_SIZE_S / WINDOW_LABEL.

Carga incremental por dataset fuente (SOURCE_TABLE):
  * si la tabla del mes no existe, se crea;
  * si existe, se borran solo las filas de esa SOURCE_TABLE y se insertan las nuevas, así que
    volver a correr un dataset lo reemplaza y uno nuevo se agrega sin tocar a los demás;
  * reset=True tira y recrea las tablas del mes (p. ej. si cambiaron las features).
No se guardan diagnósticos (contrafactual, z por ventana, PCA).
"""
import numpy as np
import pandas as pd

import ventanas

STREAMS = ('system', 'llm')
TIPOS = ('WINDOWS_SYSTEM', 'WINDOWS_LLM', 'EVENTS', 'SUMMARY', 'VALIDATION')
NO_ATAQUE = {'Falla operativa'}          # igual que clasificacion.TIPOS_NO_ATAQUE
_MAX_STR = 4900                          # VARCHAR(5000) en hana_ml

# Tipos fijos de las columnas que pueden venir vacías en una corrida y llenas en otra: así el
# esquema de la tabla no depende de qué dataset la creó.
_ENTEROS = {'WINDOW_SIZE_S', 'WEEK', 'WINDOW_INDEX', 'LABEL', 'EVENTO_ID', 'EVENTO_RELEVANTE',
            'CLUSTER_ID', 'N_VENTANAS', 'RELEVANTE', 'SIN_SIGNO', 'K', 'VENTANAS', 'EVENTOS',
            'VENTANAS_ANOMALAS', 'EVENTOS_RELEVANTES', 'VENTANAS_RELEVANTES', 'MAX_GAP', 'TIPOS',
            'ATAQUES'}
_REALES = {'SCORE', 'SCORE_PICO', 'Z_MAX', 'Z_UMBRAL', 'MARGEN', 'ARI', 'NMI', 'HOMOGENEIDAD',
           'COMPLETITUD', 'ARI_EVENTOS', 'DIAS_CUBIERTOS'}
_FECHAS = {'WEEK_START', 'WEEK_END', 'WINDOW_START', 'WINDOW_END', 'INICIO', 'FIN', 'VENTANA_PICO'}
_TEXTOS = {'SOURCE_TABLE', 'WINDOW_LABEL', 'STREAM', 'Z_MAX_FEATURE', 'TIPO_ATAQUE', 'CONFIANZA',
           'TOP3', 'VOTOS', 'ESTADO', 'MOTIVO', 'TIPOS_LISTA'}

_EVENT_COLS = ['EVENTO_ID', 'WEEK', 'INICIO', 'FIN', 'N_VENTANAS', 'VENTANA_PICO', 'SCORE_PICO',
               'Z_MAX', 'Z_MAX_FEATURE', 'RELEVANTE', 'TIPO_ATAQUE', 'MARGEN', 'CONFIANZA',
               'SIN_SIGNO', 'TOP3', 'VOTOS']
_NO_PERSISTIR = ('CF_', 'Z_', 'NOVEL_')   # prefijos de diagnóstico en las tablas de ventanas


# --------------------------------------------------------------------------
# Nombres y tipos
# --------------------------------------------------------------------------
def nombre_tabla(tipo, mes):
    """tipo: uno de TIPOS; mes: 'AAAA_MM'."""
    return f'RESULT_{tipo}_{mes}'


def _mes(fechas):
    return pd.to_datetime(fechas).dt.strftime('%Y_%m')


def _entero(col):
    return pd.array([None if pd.isna(x) else int(x) for x in col], dtype='Int64')


def _texto(col):
    return col.map(lambda x: None if x is None or (not isinstance(x, str) and pd.isna(x))
                   else str(x)[:_MAX_STR]).astype(object)


def _limpiar(df):
    """Deja tipos que hana_ml sabe crear: BIGINT, DOUBLE, TIMESTAMP, VARCHAR. NaN/NA -> NULL."""
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    for c in df.columns:
        col = df[c]
        if c in _ENTEROS or pd.api.types.is_bool_dtype(col) or pd.api.types.is_integer_dtype(col):
            df[c] = _entero(col)
        elif c in _FECHAS or pd.api.types.is_datetime64_any_dtype(col):
            col = pd.to_datetime(col)
            df[c] = col.dt.tz_localize(None) if col.dt.tz is not None else col
        elif c in _REALES or pd.api.types.is_float_dtype(col):
            df[c] = pd.to_numeric(col, errors='coerce').astype('float64').replace([np.inf, -np.inf], np.nan)
        else:
            df[c] = _texto(col)
    return df


# --------------------------------------------------------------------------
# Tablas en pandas
# --------------------------------------------------------------------------
def _ids(d, columna='EVENTO_ID'):
    d[columna] = _entero(d[columna])
    return d


def _con_semana(df, semanas):
    """Agrega WEEK_START después de WEEK (llave absoluta entre datasets)."""
    out = df.merge(semanas[['WEEK', 'WEEK_START']], on='WEEK', how='left')
    assert out['WEEK_START'].notna().all(), 'hay semanas sin WEEK_START'
    cols = list(df.columns)
    cols.insert(cols.index('WEEK') + 1, 'WEEK_START')
    return out[cols]


def tabla_ventanas(R, sizes, stream):
    """Todas las ventanas (anómalas o no) de un stream, de todos los tamaños. Sin diagnósticos."""
    partes = []
    for s in sizes:
        d = R[s][stream]
        w = d['if']
        w = _ids(w[[c for c in w.columns if not c.startswith(_NO_PERSISTIR)]].copy())
        ev = _ids(d['ev'][['EVENTO_ID', 'RELEVANTE']].copy()).rename(columns={'RELEVANTE': 'EVENTO_RELEVANTE'})
        w = w.merge(ev, on='EVENTO_ID', how='left')
        m = d['mapa'].reindex(columns=['WEEK', 'WINDOW_START', 'TIPO_ATAQUE', 'CLUSTER_ID'])
        w = w.merge(m, on=['WEEK', 'WINDOW_START'], how='left')
        w.insert(0, 'WINDOW_LABEL', ventanas.size_label(s))
        w.insert(0, 'WINDOW_SIZE_S', s)
        partes.append(w)
    return pd.concat(partes, ignore_index=True)


def tabla_eventos(R, sizes, z_umbral):
    partes = []
    for s in sizes:
        for st in STREAMS:
            e = R[s][st]['ev'].reindex(columns=_EVENT_COLS)   # columnas fijas aunque no haya eventos
            e.insert(0, 'Z_UMBRAL', float(z_umbral[s]))
            e.insert(0, 'STREAM', st)
            e.insert(0, 'WINDOW_LABEL', ventanas.size_label(s))
            e.insert(0, 'WINDOW_SIZE_S', s)
            partes.append(e)
    llenas = [e for e in partes if len(e)]
    return pd.concat(llenas, ignore_index=True) if llenas else partes[0]


def tabla_resumen(R, sizes, semanas, z_umbral, max_gap):
    """Una fila por tamaño, stream y semana."""
    filas = []
    for s in sizes:
        for st in STREAMS:
            d = R[s][st]
            for sem in semanas.itertuples():
                w = d['if'][d['if']['WEEK'] == sem.WEEK]
                e = d['ev'][d['ev']['WEEK'] == sem.WEEK]
                rel = e[e['RELEVANTE'].fillna(False).astype(bool)]
                tipos = sorted(rel['TIPO_ATAQUE'].dropna().unique()) if 'TIPO_ATAQUE' in rel else []
                filas.append({
                    'WINDOW_SIZE_S': s, 'WINDOW_LABEL': ventanas.size_label(s), 'STREAM': st,
                    'WEEK': sem.WEEK, 'WEEK_END': sem.WEEK_END, 'DIAS_CUBIERTOS': sem.DIAS_CUBIERTOS,
                    'VENTANAS': len(w), 'VENTANAS_ANOMALAS': int((w['LABEL'] == -1).sum()),
                    'EVENTOS': len(e), 'EVENTOS_RELEVANTES': len(rel),
                    'VENTANAS_RELEVANTES': int(rel['N_VENTANAS'].sum()),
                    'Z_UMBRAL': float(z_umbral[s]), 'MAX_GAP': int(max_gap[s][st]),
                    'TIPOS': len(tipos), 'ATAQUES': sum(t not in NO_ATAQUE for t in tipos),
                    'TIPOS_LISTA': ', '.join(tipos) or None,
                })
    return pd.DataFrame(filas)


def tabla_validacion(validacion):
    v = validacion.rename(columns={'tamaño': 'WINDOW_LABEL', 'stream': 'STREAM'}).copy()
    v.columns = [c.upper() for c in v.columns]
    etiqueta = {ventanas.size_label(s): s for s in ventanas.WINDOW_SIZES_S}
    v.insert(0, 'WINDOW_SIZE_S', v['WINDOW_LABEL'].map(etiqueta))
    return v


def construir(tabla, R, sizes, semanas, z_umbral, max_gap, validacion):
    """
    Devuelve {nombre_de_tabla_en_HANA: DataFrame limpio}, ya repartido por mes según WEEK_START.
    tabla: nombre de la tabla fuente (SOURCE_TABLE). semanas: DataFrame WEEK, WEEK_START,
    WEEK_END, DIAS_CUBIERTOS de la sección 2.
    """
    brutas = {
        'WINDOWS_SYSTEM': tabla_ventanas(R, sizes, 'system'),
        'WINDOWS_LLM': tabla_ventanas(R, sizes, 'llm'),
        'EVENTS': tabla_eventos(R, sizes, z_umbral),
        'SUMMARY': tabla_resumen(R, sizes, semanas, z_umbral, max_gap),
        'VALIDATION': tabla_validacion(validacion),
    }
    meses = sorted(_mes(semanas['WEEK_START']).unique())
    out = {}
    for tipo, df in brutas.items():
        df = _con_semana(df, semanas)
        df.insert(0, 'SOURCE_TABLE', tabla)
        df = _limpiar(df)
        mes = _mes(df['WEEK_START'])
        for m in meses:   # aunque un mes no tenga filas de este tipo, se incluye para borrar lo viejo
            out[nombre_tabla(tipo, m)] = df[mes == m].reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Carga a HANA
# --------------------------------------------------------------------------
def _columnas_hana(conn, nombre):
    return list(conn.sql(f'SELECT * FROM "{nombre}" LIMIT 0').collect().columns)


def _ejecutar(conn, sql, params=()):
    cur = conn.connection.cursor()
    try:
        cur.execute(sql, params)
    finally:
        cur.close()
    if not conn.connection.getautocommit():
        conn.connection.commit()


def guardar(conn, tablas, fuente, reset=False, create_fn=None):
    """
    Carga incremental por SOURCE_TABLE (ver docstring del módulo). Devuelve un resumen con las
    filas escritas y las que HANA tiene de esta fuente después de cargar.
    """
    if create_fn is None:
        from hana_ml.dataframe import create_dataframe_from_pandas as create_fn

    # Primero se valida todo, para no dejar la carga a medias por un esquema distinto
    existe = {n: (not reset) and conn.has_table(n) for n in tablas}
    for n, df in tablas.items():
        if existe[n] and len(df):
            faltan = set(df.columns) ^ set(_columnas_hana(conn, n))
            if faltan:
                raise ValueError(f'{n}: las columnas no coinciden con la tabla existente '
                                 f'({sorted(faltan)}). Usa reset=True para recrearla.')

    filas = []
    for n, df in tablas.items():
        if existe[n]:
            _ejecutar(conn, f'DELETE FROM "{n}" WHERE "SOURCE_TABLE" = ?', (fuente,))
            if len(df):
                create_fn(conn, df, n, force=False, append=True, replace=False,
                          allow_bigint=True, disable_progressbar=True)
            accion = 'reemplazo'
        elif len(df):
            create_fn(conn, df, n, force=True, replace=False, allow_bigint=True,
                      disable_progressbar=True)
            accion = 'recreada' if reset else 'creada'
        else:
            if reset and conn.has_table(n):
                _ejecutar(conn, f'DROP TABLE "{n}"')
            filas.append({'tabla': n, 'acción': 'sin filas, no se crea', 'filas': 0,
                          'filas_en_HANA': 0, 'total_en_tabla': 0, 'ok': True})
            continue
        r = conn.sql(f'SELECT COUNT(*) AS T, SUM(CASE WHEN "SOURCE_TABLE" = \'{fuente}\' THEN 1 ELSE 0 END) '
                     f'AS N FROM "{n}"').collect()
        n_hana = int(r.loc[0, 'N'] or 0)
        filas.append({'tabla': n, 'acción': accion, 'filas': len(df), 'filas_en_HANA': n_hana,
                      'total_en_tabla': int(r.loc[0, 'T']), 'ok': n_hana == len(df)})
    return pd.DataFrame(filas)
