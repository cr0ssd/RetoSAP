# IsolationForest.py
"""
Isolation Forest (PAL) por semana x tamaño de ventana x stream.

Se entrena por semana para no contaminar el modelo con meses muy anómalos o muy normales.
No se calcula el contrafactual (es diagnóstico y es lo más caro del paso); solo los z-scores de
las ventanas anómalas, que se necesitan para relevancia y clasificación.
"""

import logging
import time

from helpers.imports import pd, IsolationForest
from helpers.explicacion import zscores
from helpers import ventanas

log = logging.getLogger(__name__)

STREAMS = ('system', 'llm')

N_ESTIMATORS = 250
MAX_SAMPLES = 256
CONTAMINATION = 0.01
RANDOM_STATE = 42


def run_if(win, features):
    """
    Entrena y predice sobre las ventanas `win` (DataFrame de hana_ml con WINDOW_INDEX).
    Devuelve (ventanas + SCORE + LABEL, z-scores largos de las anómalas).
    """
    n = win.count()
    # con pocas ventanas (3 h) max_samples no puede pasar del total
    model = IsolationForest(n_estimators=N_ESTIMATORS, max_samples=min(MAX_SAMPLES, n),
                            random_state=RANDOM_STATE)
    model.fit(data=win, key='WINDOW_INDEX', features=features)
    pred = model.predict(data=win, key='WINDOW_INDEX', features=features,
                         contamination=CONTAMINATION, show_explainer=False)
    win_pdf = win.collect()
    r = win_pdf.merge(pred.collect(), on='WINDOW_INDEX')           # ventana + SCORE + LABEL
    idx = r.loc[r['LABEL'] == -1, 'WINDOW_INDEX'].tolist()
    z = zscores(win_pdf, idx, features)
    return r, z


def ModelIForest(conn, n_weeks, system_weeks, llm_weeks, features, sizes):
    """
    Corre el IF para todas las semanas, tamaños y streams sin releer la BD por cada tamaño:
    los eventos base se preparan una vez por semana y las ventanas se generan por tamaño.

    features: {'system': [...], 'llm': [...]}
    Devuelve R[size_s][stream] = {'if': ventanas con SCORE/LABEL, 'z': z-scores largos}.
    """
    crudo = {s: {st: {'if': [], 'z': []} for st in STREAMS} for s in sizes}

    for w in range(n_weeks):
        ventanas.prepare_week(conn, system_weeks[w], llm_weeks[w])   # una vez por semana
        for size_s in sizes:
            t = time.perf_counter()
            win_sys, win_llm = ventanas.make_windows(conn, size_s)
            for stream, win in (('system', win_sys), ('llm', win_llm)):
                r, z = run_if(win, features[stream])
                for df, name in ((r, 'if'), (z, 'z')):
                    df.insert(0, 'WEEK', w)
                    crudo[size_s][stream][name].append(df)
            log.info('IF semana %d | %s | %.0f s', w, ventanas.size_label(size_s),
                     time.perf_counter() - t)
        ventanas.drop_week(conn)

    return {s: {st: {k: pd.concat(v, ignore_index=True) for k, v in crudo[s][st].items()}
                for st in STREAMS} for s in sizes}
