# normalizacion.py

from helpers.imports import pd, np

def normalizar_por_semana(ds, if_df, features, clip=None):
    """
    Z-score con signo de las `features` de las ventanas de `ds`, con media y desviación de TODAS
    las ventanas de su semana (if_df), no solo de las anómalas. Devuelve solo esas variables,
    en el mismo orden de filas que `ds`. sd = 0 o valor faltante -> z = 0.
    """
    g = if_df.groupby('WEEK')[features]
    mu = g.mean()
    sd = g.std(ddof=0).replace(0, np.nan)
    sem = ds['WEEK'].values
    z = (ds[features].astype(float).values - mu.loc[sem].values) / sd.loc[sem].values
    z = pd.DataFrame(z, columns=features).fillna(0)
    return z.clip(-clip, clip) if clip else z


