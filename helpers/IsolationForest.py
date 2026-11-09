# IsolationForest.py

from helpers.imports import IsolationForest
from helpers.explicacion import contrafactual, zscores

from helpers.ventanas import make_windows

# Modelo isolation forest, fit & predict
def run_if(conn, win, features):
    model = IsolationForest(n_estimators=250, max_samples=256, random_state=42)
    model.fit(data=win, key='WINDOW_INDEX', features=features)
    pred = model.predict(data=win, key='WINDOW_INDEX', features=features,
                         contamination=0.01, show_explainer=False)
    win_pdf = win.collect()
    r = win_pdf.merge(pred.collect(), on='WINDOW_INDEX')              # ventana + SCORE + LABEL

    an = r[r['LABEL'] == -1]
    idx = an['WINDOW_INDEX'].tolist()
    cf = contrafactual(conn, model, win, features,
                                   an.set_index('WINDOW_INDEX')['SCORE'], idx)
    z = zscores(win_pdf, idx, features)
    return r, cf, z

res = {s: {'r': [], 'cf': [], 'z': []} for s in ('system', 'llm')}

def ModelIForest(conn, n_weeks, system_weeks, llm_weeks, df_system,
                 FEATURES_SYSTEM, FEATURES_LLM):
    res = {s: {'r': [], 'cf': [], 'z': []} for s in ('system', 'llm')}
    for w in range(n_weeks):
        win_sys, win_llm = make_windows(w, conn, system_weeks, llm_weeks, df_system)
        for stream, win, feats in (('system', win_sys, FEATURES_SYSTEM),
                                   ('llm', win_llm, FEATURES_LLM)):
            r, cf, z = run_if(conn, win, feats)
            for df, name in ((r, 'r'), (cf, 'cf'), (z, 'z')):
                df.insert(0, 'WEEK', w)
                res[stream][name].append(df)
    return res

