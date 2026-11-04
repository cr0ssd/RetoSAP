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
import re
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
    if rep.empty:
        return rep
    rep['rareza_x_concentracion'] = rep['share_en_top3_buckets'] * (1 - rep['buckets_activos'] / rep['buckets_total'])
    return rep.sort_values(['rareza_x_concentracion', 'share_global'], ascending=[False, True]).head(top)


# --------------------------------------------------------------------------
# 5) Diagnóstico de columnas que hoy no entran en ninguna ventana
# --------------------------------------------------------------------------
def check_unused_columns(df_system, bucket='5min', max_cardinality=500):
    """
    Revisa, de forma agregada por bucket de tiempo, las columnas del log crudo que
    no se usan en las ventanas: _score (numérica), event_hash (duplicados/replays),
    service_id, event_code_version, @version, el retraso entre @timestamp y
    @event_time_requested, y las plantillas de sap_function_message.
    Para cada bloque reporta el z-score máximo de un bucket frente a los demás
    (|z| muy por encima de ~4 es raro) o el reporte de ráfagas por categoría.
    Recibe df_system (el DataFrame crudo de system) y devuelve un dict de resultados.
    """
    d = df_system.copy()
    d['timestamp'] = pd.to_datetime(d['@timestamp'], utc=True, errors='coerce')
    b = d['timestamp'].dt.floor(bucket)
    res = {}

    # Buckets casi vacíos (p. ej. el bucket suelto al final del periodo) dan medias y
    # desviaciones ruidosas y producen z-scores enormes sin significado: se ignoran.
    n_b = d.groupby(b).size()
    ok = n_b >= 0.5 * n_b.median()
    res['buckets_con_pocos_logs'] = n_b[~ok]

    def max_z(s):
        s = s[ok.reindex(s.index, fill_value=False)].dropna()
        return float(((s - s.mean()) / s.std()).abs().max())

    # a) Retraso entre el evento y su registro
    if '@event_time_requested' in d:
        lag = (d['timestamp'] - pd.to_datetime(d['@event_time_requested'], utc=True, errors='coerce')).dt.total_seconds()
        res['lag_describe'] = lag.describe(percentiles=[.01, .5, .99]).round(3)
        res['lag_max_z_bucket_mean'] = max_z(lag.groupby(b).mean())

    # b) _score: ¿cambia con el tiempo o según tipo de log / código HTTP?
    if '_score' in d:
        sc = pd.to_numeric(d['_score'], errors='coerce')
        res['score_describe'] = sc.describe().round(4)
        res['score_max_z_bucket_mean'] = max_z(sc.groupby(b).mean())
        res['score_max_z_bucket_std'] = max_z(sc.groupby(b).std())
        for col in ('sap_function_log_type', 'http_status_code'):
            if col in d:
                res[f'score_mean_by_{col}'] = sc.groupby(d[col]).mean().round(4)

    # c) event_hash: eventos repetidos (posibles replays)
    if 'event_hash' in d:
        dup = d.duplicated('event_hash', keep=False)
        share = dup.groupby(b).mean()
        res['hash_dup_share_global'] = float(dup.mean())
        res['hash_dup_share_max_bucket'] = float(share.max())
        res['hash_dup_max_z_bucket'] = max_z(share) if share.std() > 0 else 0.0

    # d) Categorías no usadas: ¿alguna aparece solo en ciertos momentos?
    others = [c for c in ('service_id', 'event_code_version', '@version') if c in d]
    if others:
        res['burst_otras_columnas'] = category_burst_report(d, cols=others, bucket=bucket, top=10)

    # e) Mensajes: plantillas (números -> <N>) y su concentración temporal
    if 'sap_function_message' in d:
        d['msg_template'] = d['sap_function_message'].astype('string').str.replace(r'\d+', '<N>', regex=True)
        res['n_mensajes'] = int(d['sap_function_message'].nunique())
        res['n_plantillas'] = int(d['msg_template'].nunique())
        if res['n_plantillas'] <= max_cardinality:
            res['burst_plantillas'] = category_burst_report(d, cols=['msg_template'], bucket=bucket, top=10)
        else:
            res['plantillas_mas_raras'] = d['msg_template'].value_counts().tail(15)
    return res


# --------------------------------------------------------------------------
# 6) Familias de mensajes (sap_function_message)
# --------------------------------------------------------------------------
def message_families(df, msg_col='sap_function_message',
                     app_col='sap_function_application', src_col='sap_source_type'):
    """
    Reduce cada mensaje a su 'familia': sustituye la aplicación y el tipo de fuente
    (tomados de sus propias columnas) por <APP> y <SRC>, y los números por <N>.
    Ej.: 'Token validity period is longer than recommended on sap-btp-cf via BAPI.'
      -> 'Token validity period is longer than recommended on <APP> via <SRC>.'
    """
    msg = df[msg_col].astype('string')
    for col, token in ((app_col, '<APP>'), (src_col, '<SRC>')):
        if col not in df.columns:
            continue
        vals = sorted(df[col].dropna().astype(str).unique(), key=len, reverse=True)
        if vals:
            pat = r'(?<![\w-])(?:' + '|'.join(map(re.escape, vals)) + r')(?![\w-])'
            msg = msg.str.replace(pat, token, regex=True)
    # Mensajes de LLM con sufijo '(model=<modelo>, app=<app>)'
    msg = msg.str.replace(r'(?<=model=)[^,)\s]+', '<MODEL>', regex=True)
    msg = msg.str.replace(r'(?<=app=)[^,)\s]+', '<APP>', regex=True)
    return msg.str.replace(r'\d+', '<N>', regex=True)


def message_family_report(df_system, bucket='5min', top=25, max_families=5000):
    """
    Resumen de familias: frecuencia, tipo de log en que aparecen y ráfagas (test de Poisson).
    Si no hay mensajes o la normalización no los colapsa (demasiadas familias), lo avisa en
    res['aviso'] y devuelve una muestra en lugar de fallar.
    """
    d = df_system.copy()
    d['timestamp'] = pd.to_datetime(d['@timestamp'], utc=True, errors='coerce')
    res = {'mensajes_no_nulos': float(d['sap_function_message'].notna().mean())}
    if res['mensajes_no_nulos'] == 0:
        res['aviso'] = 'sap_function_message viene vacío en estas filas.'
        return res
    d['msg_family'] = message_families(d)
    vc = d['msg_family'].value_counts()
    res['n_familias'] = int(len(vc))
    res['frecuencia_familias'] = pd.DataFrame({'logs': vc, 'share': vc / len(d)})
    if len(vc) > max_families:
        res['aviso'] = (f'{len(vc)} familias (> {max_families}): la normalización no está colapsando '
                        'los mensajes. Revisa res["muestra_mensajes"] para ver qué parte varía.')
        res['muestra_mensajes'] = d['sap_function_message'].dropna().sample(10, random_state=0).tolist()
        return res
    if 'sap_function_log_type' in d:
        res['familias_raras_x_log_type'] = pd.crosstab(
            d['msg_family'], d['sap_function_log_type']).loc[vc.index[-top:]]
    res['burst_familias'] = poisson_burst_test(d, 'msg_family', bucket=bucket,
                                               max_cardinality=max_families, top=top)
    return res


# --------------------------------------------------------------------------
# 7) Test de ráfagas con cola de Poisson (reemplaza a la heurística de category_burst_report)
# --------------------------------------------------------------------------
def poisson_burst_test(logs, col, bucket='5min', max_cardinality=5000, top=10, alpha=0.01):
    """
    Para cada (bucket, categoría) compara el conteo observado con el esperado si la
    categoría se repartiera sin cambios en el tiempo (esperado = total del bucket x
    participación global de la categoría) y calcula la probabilidad exacta de ver
    ese conteo o más (cola de Poisson). Reporta, por categoría, su bucket más
    sorprendente como -log10(p).
      - Si no hay ráfagas, el máximo de TODAS las celdas ronda log10(n_celdas).
      - 'significativo' = supera el umbral de Bonferroni (alpha / n_celdas).
    A diferencia de category_burst_report, no favorece a las categorías raras: una
    ráfaga en una categoría frecuente también destaca.
    """
    from scipy.stats import poisson
    b = logs['timestamp'].dt.floor(bucket)
    cat = logs[col].astype('string').fillna('__NA__')
    ct = pd.crosstab(b, cat)
    if ct.shape[1] > max_cardinality:
        raise ValueError(f'{col}: {ct.shape[1]} categorías (> max_cardinality)')
    n_b = ct.sum(axis=1)
    ct = ct[n_b >= 0.5 * n_b.median()]                      # descarta buckets casi vacíos
    n_b = ct.sum(axis=1)
    share = ct.sum(axis=0) / ct.to_numpy().sum()
    exp = np.outer(n_b.to_numpy(float), share.to_numpy(float))
    obs = ct.to_numpy()
    nlp = -poisson.logsf(obs - 1, exp) / np.log(10)         # -log10 P(X >= obs)
    nlp = np.where(obs == 0, 0.0, nlp)
    n_cells = nlp.size
    umbral = np.log10(n_cells / alpha)
    j = nlp.argmax(axis=0)                                  # bucket más sorprendente por categoría
    k = np.arange(nlp.shape[1])
    rep = pd.DataFrame({
        'columna': col, 'categoria': ct.columns, 'bucket': ct.index[j],
        'observado': obs[j, k], 'esperado': exp[j, k].round(2),
        'neg_log10_p': nlp[j, k].round(2), 'umbral_bonferroni': round(umbral, 2),
        'esperado_sin_rafagas': round(np.log10(n_cells), 2),
    })
    rep['significativo'] = rep['neg_log10_p'] > umbral
    return rep.sort_values('neg_log10_p', ascending=False).head(top).reset_index(drop=True)


# --------------------------------------------------------------------------
# 8) Familias de prompts (llm_prompt)
# --------------------------------------------------------------------------
_PROMPT_SLOTS = [
    (r'sap-[a-z0-9]+(?:-[a-z0-9]+)*', '<APP>'),                                   # aplicación
    (r"[A-Z][\w.'-]*(?: [A-Z][\w.'-]*)* \| [A-Z0-9][\w.'-]*(?<!\.)(?: [A-Z0-9][\w.'-]*(?<!\.))*(?: \(\d+\))?", '<REGION>'),  # País | Ciudad (n)
    (r'\b[A-Z]{2,}(?:_[A-Z0-9]+)+\b', '<ERR>'),                                   # HTTP_401_UNAUTHORIZED, RFC_NO_AUTHORITY
    (r'(?<=vendor )[A-Z]\w+', '<VENDOR>'),                                        # vendor TitanSupplies
    (r'\bQ[1-4]\b', '<Q>'),                                                       # trimestre
]


def prompt_families(prompts):
    """
    Reduce cada prompt a su plantilla sustituyendo los datos variables (aplicación, región,
    código de error, proveedor, trimestre, números) por marcadores. Así los ~179 mil prompts
    distintos se colapsan en un catálogo pequeño de plantillas comparables.
    """
    s = prompts.astype('string')
    for pat, token in _PROMPT_SLOTS:
        s = s.str.replace(pat, token, regex=True)
    return s.str.replace(r'\d+', '<N>', regex=True)


def prompt_family_report(df_llm, time_col='@timestamp', bucket='5min', top=25):
    """
    Frecuencia de plantillas de prompt, su categoría (llm_prompt_category) y la prueba de
    ráfagas de Poisson por plantilla. Recibe el DataFrame crudo de LLM.
    """
    d = df_llm.copy()
    d['timestamp'] = pd.to_datetime(d[time_col], utc=True, errors='coerce')
    d['prompt_family'] = prompt_families(d['llm_prompt'])
    vc = d['prompt_family'].value_counts()
    res = {'n_familias': int(len(vc)),
           'frecuencia_familias': pd.DataFrame({'logs': vc, 'share': vc / len(d)})}
    if 'llm_prompt_category' in d:
        ct = pd.crosstab(d['prompt_family'], d['llm_prompt_category'])
        res['familias_x_categoria'] = ct.loc[vc.index]
        res['familias_con_varias_categorias'] = int((ct > 0).sum(axis=1).gt(1).sum())
    if len(vc) <= 5000:
        res['burst_familias'] = poisson_burst_test(d, 'prompt_family', bucket=bucket, max_cardinality=5000, top=top)
    return res