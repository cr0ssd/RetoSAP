# explicacion.py
"""
Z-scores de las ventanas anómalas contra todas las ventanas de su semana.

Es lo único de la explicación que usa producción (relevancia de eventos y clasificación).
El contrafactual, las features dominantes, los perfiles por familia y las familias son
diagnóstico y siguen en sandbox/explicacion.py.
"""

from helpers.imports import pd, np


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
