# relevantes.py

def marcar_relevantes(if_df, z, eventos, umbral):
    """Añade a la tabla de eventos Z_MAX, Z_MAX_FEATURE y RELEVANTE (Z_MAX >= umbral)."""
    zz = z.merge(if_df.loc[if_df['LABEL'] == -1, ['WEEK', 'WINDOW_INDEX', 'EVENTO_ID']],
                 on=['WEEK', 'WINDOW_INDEX'])
    zz['ZABS'] = zz['Z'].abs()
    pico = zz.loc[zz.groupby('EVENTO_ID')['ZABS'].idxmax(), ['EVENTO_ID', 'FEATURE', 'ZABS']]
    pico = pico.rename(columns={'FEATURE': 'Z_MAX_FEATURE', 'ZABS': 'Z_MAX'})
    out = eventos.drop(columns=['Z_MAX', 'Z_MAX_FEATURE', 'RELEVANTE'], errors='ignore') \
                 .merge(pico, on='EVENTO_ID', how='left')
    out['RELEVANTE'] = out['Z_MAX'] >= umbral
    return out


# Dataset para el clustering multiclase: ventanas anómalas de eventos relevantes,
# solo con las columnas de la ventana (sin SCORE, LABEL, EVENTO_ID ni explicaciones).
POST_IF = {'SCORE', 'LABEL', 'EVENTO_ID', 'WINDOW_INDEX'}

def dataset_multiclase(if_df, eventos):
    post = POST_IF | {c for c in if_df.columns if c.startswith(('CF_', 'Z_'))}
    rel = eventos.loc[eventos['RELEVANTE'], 'EVENTO_ID']
    a = if_df[(if_df['LABEL'] == -1) & if_df['EVENTO_ID'].isin(rel)]
    ds = a[[c for c in if_df.columns if c not in post]].reset_index(drop=True)
    mapa = a[['WEEK', 'WINDOW_START', 'EVENTO_ID']].reset_index(drop=True)   # para volver al evento
    return ds, mapa


