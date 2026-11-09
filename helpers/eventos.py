# eventos.py

from helpers.imports import pd

MAX_GAP = {'system': 3, 'llm': 3}   # en ventanas consecutivas, como en tu baseline

def agrupar_eventos(res, max_gap):
    """Marca EVENTO_ID: anomalías separadas por <= max_gap ventanas forman un evento. No cruza semanas."""
    res = res.sort_values(['WEEK', 'WINDOW_INDEX']).reset_index(drop=True)
    a = res[res['LABEL'] == -1]
    gap = a.groupby('WEEK')['WINDOW_INDEX'].diff()
    res['EVENTO_ID'] = pd.NA
    res.loc[a.index, 'EVENTO_ID'] = ((gap.isna()) | (gap > max_gap)).cumsum()
    return res

def resumen_eventos(res):
    """Una fila por evento: inicio, fin, nº de ventanas y ventana pico (mayor SCORE de anomalía)."""
    a = res[res['LABEL'] == -1]
    if a.empty:
        return pd.DataFrame()
    # En PAL el score más alto suele ser más anómalo; se confirma comparando medias.
    alto_es_anomalo = a['SCORE'].mean() >= res.loc[res['LABEL'] != -1, 'SCORE'].mean()
    pick = a.groupby('EVENTO_ID')['SCORE'].idxmax() if alto_es_anomalo else a.groupby('EVENTO_ID')['SCORE'].idxmin()
    pico = (a.loc[pick, ['EVENTO_ID', 'WEEK', 'WINDOW_START', 'SCORE']]
            .rename(columns={'WINDOW_START': 'VENTANA_PICO', 'SCORE': 'SCORE_PICO'})
            .set_index('EVENTO_ID'))
    g = a.groupby('EVENTO_ID').agg(INICIO=('WINDOW_START', 'min'), FIN=('WINDOW_START', 'max'),
                                   N_VENTANAS=('WINDOW_INDEX', 'size'))
    return g.join(pico).reset_index()
