"""
Mejoras propuestas para el notebook de generación de ventanas (system).

Mantiene el esquema actual: ventanas de 5 min con paso de 2.5 min.
Todo se calcula de forma agregada por ventana (no se revisan logs uno por uno).

Uso (dentro del notebook, después de crear `logs` con la columna `timestamp`):

    events = expand_to_windows(logs)                  # sin ventanas parciales
    extra  = build_extra_features(events)             # features nuevas
    df_batch_system = df_batch_system.join(extra, how='inner')
    df_batch_system = df_batch_system.drop(columns=['request_rate_avg', 'unique_paths_ratio'])

Para `logs`, incluye además las columnas categóricas adicionales que quieras
comparar (ver CATEGORICAL_COLS) y '@event_time_requested'.
"""
import numpy as np
import pandas as pd

WINDOW_SIZE = pd.Timedelta(minutes=5)
WINDOW_STEP = pd.Timedelta(minutes=2.5)

# Columnas categóricas del log original (lista de variables compartida).
# Hoy solo se usan 3 de ellas; el resto no entra en ninguna feature.
CATEGORICAL_COLS = [
    'http_status_code', 'headers_http_request_method', 'sap_function_log_type',
    'heathers_request_path', 'sap_function_application', 'sap_source_type',
    'sap_app_env', 'region_code', 'headers_http_host', 'headers_content_type',
    'sap_function_message',            # solo se usa si tiene pocas plantillas distintas
]


# --------------------------------------------------------------------------
# 1) Ventanas: cada log en 2 ventanas, descartando las parciales de los bordes
# --------------------------------------------------------------------------
def expand_to_windows(logs, window_size=WINDOW_SIZE, window_step=WINDOW_STEP):
    ts = logs['timestamp']
    starts = ts.dt.floor(window_step)
    logs = logs.assign(_event_id=np.arange(len(logs)))
    events = pd.concat(
        [logs.assign(window_start=starts),
         logs.assign(window_start=starts - window_size + window_step)],
        ignore_index=True,
    )
    # Solo ventanas completas: [ws, ws + size) debe caer dentro del rango observado.
    t_min = ts.min().floor(window_step)
    t_max = ts.max().ceil(window_step)
    ws = events['window_start']
    return events[(ws >= t_min) & (ws + window_size <= t_max)].copy()


# --------------------------------------------------------------------------
# 2) Utilidades de distribución
# --------------------------------------------------------------------------
def _counts(events, col):
    """Matriz ventana x categoría (los nulos cuentan como categoría propia)."""
    cat = events[col].astype('string').fillna('__NA__')
    return events.assign(_cat=cat).groupby(['window_start', '_cat']).size().unstack(fill_value=0)


def normalized_entropy(counts):
    c = counts.to_numpy(float)
    tot = c.sum(1, keepdims=True)
    p = np.divide(c, tot, out=np.zeros_like(c), where=tot > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(1)
    k = (c > 0).sum(1)
    return pd.Series(np.where(k > 1, h / np.log2(np.maximum(k, 2)), 0.0), index=counts.index)


def js_divergence(counts, baseline_counts=None):
    """
    Divergencia de Jensen-Shannon (en bits, 0..1) entre la distribución de cada
    ventana y la distribución base (por defecto, la global). Es una métrica de
    'desviación' por columna: sube si cambia la mezcla de categorías, incluyendo
    categorías nuevas. Su ruido base depende del tamaño de ventana y del número
    de categorías, así que compara ventanas entre sí, no columnas entre sí.
    """
    c = counts.to_numpy(float)
    q = (counts.sum(0) if baseline_counts is None else baseline_counts.reindex(columns=counts.columns, fill_value=0).sum(0)).to_numpy(float)
    q = q / q.sum()
    p = c / c.sum(1, keepdims=True)
    m = 0.5 * (p + q)
    with np.errstate(divide='ignore', invalid='ignore'):
        kl_pm = np.where(p > 0, p * np.log2(p / m), 0.0).sum(1)
        kl_qm = np.where(q > 0, q * np.log2(q / m), 0.0).sum(1)
    return pd.Series(0.5 * kl_pm + 0.5 * kl_qm, index=counts.index)


def novelty_count(events, col, warmup=pd.Timedelta(hours=24)):
    """
    Valores distintos de `col` cuya PRIMERA aparición en todo el dataset cae dentro
    de la ventana (p. ej. IPs o rutas nunca vistas). Se ignora un periodo de
    calentamiento inicial donde todo es 'nuevo'. Es causal: solo mira el pasado.
    """
    first_seen = events.groupby(col)['timestamp'].transform('min')
    t0 = events['timestamp'].min()
    is_new = (first_seen >= events['window_start']) & (events['window_start'] >= t0 + warmup)
    out = events[is_new].groupby('window_start')[col].nunique()
    return out.reindex(events['window_start'].unique(), fill_value=0).sort_index()


# --------------------------------------------------------------------------
# 3) Features nuevas
# --------------------------------------------------------------------------
def build_extra_features(events, categorical_cols=CATEGORICAL_COLS, max_cardinality=500,
                         time_col='@event_time_requested', baseline_mask=None):
    """
    Devuelve un DataFrame indexado por window_start con:
      - js_<col>           : desviación de la mezcla de categorías vs. la base
      - ip_top_share       : fracción de la ventana generada por la IP más activa
      - novel_ips/novel_paths : IPs / rutas vistas por primera vez en la ventana
      - non_200_rate       : fracción de requests no-200 sobre el TOTAL (no sobre los no-200)
      - interval_cv        : coef. de variación de los tiempos entre requests (ráfagas)
    `baseline_mask`: serie booleana sobre window_start para elegir ventanas base.
    """
    out = {}

    for col in categorical_cols:
        if col not in events.columns:
            continue
        if events[col].nunique(dropna=False) > max_cardinality:
            continue                                   # texto libre: usar plantillas (Drain) aparte
        counts = _counts(events, col)
        base = counts[baseline_mask.reindex(counts.index, fill_value=False)] if baseline_mask is not None else None
        out[f'js_{col}'] = js_divergence(counts, base)

    # Concentración de IPs (complementa a ip_entropy)
    ip_counts = events.dropna(subset=['client_ip']).groupby(['window_start', 'client_ip']).size()
    out['ip_top_share'] = ip_counts.groupby(level=0).max() / ip_counts.groupby(level=0).sum()

    # Novedad
    out['novel_ips'] = novelty_count(events, 'client_ip')
    if 'heathers_request_path' in events.columns:
        out['novel_paths'] = novelty_count(events, 'heathers_request_path')

    # Fallos sobre el total (error_rate_400/500 son composición entre no-200)
    status = pd.to_numeric(events['http_status_code'], errors='coerce')
    out['non_200_rate'] = (status.notna() & status.ne(200)).groupby(events['window_start']).mean()

    # Ráfagas con precisión de milisegundos (@timestamp viene redondeado al segundo)
    if time_col in events.columns:
        t = pd.to_datetime(events[time_col], utc=True, errors='coerce')
        tmp = events.assign(_t=t).sort_values(['window_start', '_t'], kind='mergesort')
        iv = tmp.groupby('window_start')['_t'].diff().dt.total_seconds()
        g = iv.groupby(tmp['window_start'])
        out['interval_cv'] = g.std() / g.mean()

    return pd.DataFrame(out).sort_index()


# --------------------------------------------------------------------------
# 4) Diagnóstico: ¿dónde están las anomalías inyectadas?
# --------------------------------------------------------------------------
def category_burst_report(logs, cols=CATEGORICAL_COLS + ['client_ip'], bucket='5min', max_cardinality=500, top=30):
    """
    Para cada categoría de cada columna calcula su frecuencia global y qué tan
    concentrada está en el tiempo. Categorías raras Y concentradas en pocos
    buckets son los mejores candidatos a eventos inyectados.
    Es una exploración agregada (por categoría), no revisión log por log.
    """
    b = logs['timestamp'].dt.floor(bucket)
    rows = []
    for col in cols:
        if col not in logs.columns or logs[col].nunique(dropna=False) > max_cardinality:
            continue
        cat = logs[col].astype('string').fillna('__NA__')
        ct = pd.crosstab(b, cat)
        for name in ct.columns:
            s = ct[name]
            rows.append({
                'columna': col, 'categoria': name, 'logs': int(s.sum()),
                'share_global': s.sum() / len(logs),
                'buckets_activos': int((s > 0).sum()), 'buckets_total': len(s),
                'share_en_top3_buckets': s.nlargest(3).sum() / s.sum(),
            })
    rep = pd.DataFrame(rows)
    rep['rareza_x_concentracion'] = rep['share_en_top3_buckets'] * (1 - rep['buckets_activos'] / rep['buckets_total'])
    return rep.sort_values(['rareza_x_concentracion', 'share_global'], ascending=[False, True]).head(top)
