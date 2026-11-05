"""
explicacion.py - Explica por qué el Isolation Forest marcó una ventana como anómala.

El explicador nativo de hana_ml (show_explainer=True) requiere el servicio PAL del servidor,
que no está habilitado en esta instancia. Se usan dos alternativas que solo necesitan
fit/predict normales:

  * Contrafactual: se reemplaza cada feature, una a la vez, por la mediana de su semana y se
    vuelve a puntuar con el MISMO modelo. DELTA = SCORE_ORIG - SCORE_NUEVO; la feature con
    mayor DELTA es la que más contribuye a que la ventana parezca anómala. Es una
    sensibilidad del modelo, no una causa del ataque.
  * Z-score: desviación de cada feature contra la media y desviación de su semana.

Las 24 (o 12) versiones editadas de cada ventana se apilan en una sola consulta, de modo que
cada flujo y semana cuesta una sola llamada de predicción.
"""
import numpy as np
import pandas as pd

_KEY = 'CF_ID'   # clave compuesta: WINDOW_INDEX * 100 + posición de la feature


def contrafactual(conn, model, win, features, score_orig, anomalous_idx,
                  contamination=0.01, key='WINDOW_INDEX'):
    """
    model:         IsolationForest ya entrenado sobre `win`.
    win:           DataFrame de hana_ml con las ventanas del batch (incluye `key` y las features).
    score_orig:    Series indexada por `key` con el SCORE original de las ventanas anómalas.
    anomalous_idx: lista de valores de `key` de las ventanas a explicar.
    Devuelve un DataFrame largo: WINDOW_INDEX, FEATURE, SCORE_ORIG, SCORE_NUEVO, DELTA.
    """
    cols_out = ['WINDOW_INDEX', 'FEATURE', 'SCORE_ORIG', 'SCORE_NUEVO', 'DELTA']
    if len(anomalous_idx) == 0:
        return pd.DataFrame(columns=cols_out)
    if len(features) > 99:
        raise ValueError('contrafactual soporta hasta 99 features.')

    q = lambda c: '"%s"' % c
    src = win.select_statement
    med = conn.sql('SELECT %s FROM (%s) m' % (
        ', '.join('MEDIAN(%s) AS %s' % (q(f), q(f)) for f in features), src)).collect().iloc[0]
    idx = ', '.join(str(int(i)) for i in anomalous_idx)

    parts = []
    for k, f in enumerate(features):
        if pd.isna(med[f]):
            continue                                   # sin mediana no hay reemplazo posible
        cols = ', '.join(('CAST(%.17g AS DOUBLE)' % med[f] if g == f else q(g)) + ' AS ' + q(g)
                         for g in features)
        parts.append('SELECT %s * 100 + %d AS %s, %s FROM (%s) w WHERE %s IN (%s)'
                     % (q(key), k, q(_KEY), cols, src, q(key), idx))
    edited = conn.sql(' UNION ALL '.join(parts))

    pred = model.predict(data=edited, key=_KEY, features=features,
                         contamination=contamination, show_explainer=False).collect()
    pred['WINDOW_INDEX'] = pred[_KEY] // 100
    pred['FEATURE'] = [features[i] for i in (pred[_KEY] % 100)]
    pred['SCORE_ORIG'] = pred['WINDOW_INDEX'].map(score_orig)
    pred = pred.rename(columns={'SCORE': 'SCORE_NUEVO'})
    pred['DELTA'] = pred['SCORE_ORIG'] - pred['SCORE_NUEVO']
    return pred[cols_out].reset_index(drop=True)


def zscores(win_pdf, anomalous_idx, features, key='WINDOW_INDEX'):
    """Z-score de cada feature en las ventanas anómalas, contra todas las ventanas del batch."""
    mu = win_pdf[features].mean()
    sd = win_pdf[features].std(ddof=0).replace(0, np.nan)
    a = win_pdf[win_pdf[key].isin(anomalous_idx)]
    z = ((a[features] - mu) / sd).fillna(0)
    z[key] = a[key].values
    out = z.melt(id_vars=key, var_name='FEATURE', value_name='Z')
    return out.rename(columns={key: 'WINDOW_INDEX'}).reset_index(drop=True)


def _top(df, val, name_f, name_v, keys, absolute=False):
    if df.empty:
        return pd.DataFrame(columns=list(keys) + [name_f, name_v])
    df = df.reset_index(drop=True)
    s = df[val].abs() if absolute else df[val]
    i = s.groupby([df[k] for k in keys]).idxmax()
    out = df.loc[i, list(keys) + ['FEATURE', val]].rename(columns={'FEATURE': name_f, val: name_v})
    return out.reset_index(drop=True)


def dominantes_por_ventana(cf, z, keys=('WEEK', 'WINDOW_INDEX')):
    """Una fila por ventana anómala: feature dominante por contrafactual y por z-score."""
    a = _top(cf, 'DELTA', 'CF_FEATURE', 'CF_DELTA', keys)
    b = _top(z, 'Z', 'Z_FEATURE', 'Z_VALOR', keys, absolute=True)
    return a.merge(b, on=list(keys), how='outer')


def explicar_eventos(if_df, cf, z):
    """
    Una fila por evento (requiere EVENTO_ID en if_df y WEEK en cf y z). Promedia sobre las
    ventanas del evento: DELTA para el contrafactual y |Z| para el z-score.
    """
    keys = ['WEEK', 'WINDOW_INDEX']
    m = if_df.loc[if_df['LABEL'] == -1, keys + ['EVENTO_ID']]
    if cf.empty or z.empty or m.empty:
        return pd.DataFrame()

    c = (cf.merge(m, on=keys).groupby(['EVENTO_ID', 'FEATURE'])['DELTA'].mean().reset_index()
         .sort_values(['EVENTO_ID', 'DELTA'], ascending=[True, False]))
    c['_txt'] = c['FEATURE'] + ' (' + c['DELTA'].map('{:.3f}'.format) + ')'
    cf_ev = pd.DataFrame({
        'CF_FEATURE': c.groupby('EVENTO_ID')['FEATURE'].first(),
        'CF_DELTA_MEDIO': c.groupby('EVENTO_ID')['DELTA'].first(),
        'CF_TOP3': c.groupby('EVENTO_ID')['_txt'].apply(lambda s: ', '.join(s.head(3))),
    })

    ze = z.merge(m, on=keys)
    ze['ABSZ'] = ze['Z'].abs()
    zg = (ze.groupby(['EVENTO_ID', 'FEATURE']).agg(ABSZ=('ABSZ', 'mean'), Z=('Z', 'mean'))
          .reset_index().sort_values(['EVENTO_ID', 'ABSZ'], ascending=[True, False]))
    z_ev = pd.DataFrame({
        'Z_FEATURE': zg.groupby('EVENTO_ID')['FEATURE'].first(),
        'Z_VALOR_MEDIO': zg.groupby('EVENTO_ID')['Z'].first(),
    })
    return cf_ev.join(z_ev, how='outer').reset_index()
