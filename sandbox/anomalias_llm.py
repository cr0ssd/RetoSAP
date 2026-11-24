"""
Herramientas de análisis para demo_pipeline_maria.ipynb (ventanas LLM).

El Isolation Forest corre en HANA (PAL). Aquí solo hay pandas/numpy para todo lo que
rodea al modelo y que sustituye a la "verdad conocida" que no tenemos:

  1) Línea base robusta por feature (mediana/MAD), con ajuste por hora del día cuando
     la feature tiene estacionalidad diaria.
  2) Detectores independientes del Isolation Forest (Mahalanobis robusta y z univariado)
     para pedir consenso en vez de confiar en un solo modelo.
  3) Estabilidad del ranking entre semillas.
  4) Inyección sintética de anomalías de magnitud conocida: mide qué tan grande debe ser
     un cambio para que el modelo lo detecte (potencia de detección).
  5) Explicación de cada ventana (qué features se desvían y en qué dirección) y
     agrupación de ventanas anómalas contiguas en eventos.

Las columnas siguen los nombres de ventanas.build_llm_windows (MAYÚSCULAS).
"""
import numpy as np
import pandas as pd

TS = 'WINDOW_START'
WINDOW_MIN = 5

# Features que son proporciones: al inyectar anomalías se acotan a [0, 1].
ACOTADAS_01 = {'RATIO_INJECTION', 'LLM_ERROR_RATE', 'LLM_TIMEOUT_RATE',
               'STREAM_RATIO', 'PROMPT_CATEGORY_ENTROPY'}

# A qué "tema" pertenece cada feature, para leer una anomalía en lenguaje de negocio.
FAMILIA = {
    'TOTAL_LLM_REQUESTS': 'volumen',
    'AVG_LLM_COST_USD': 'consumo',
    'AVG_TOKEN_RATIO': 'consumo',
    'AVG_PROMPT_LENGTH': 'prompts',
    'MAX_PROMPT_LENGTH': 'prompts',
    'RATIO_INJECTION': 'seguridad',
    'AVG_RESPONSE_TIME_MS': 'servicio',
    'LLM_ERROR_RATE': 'servicio',
    'LLM_TIMEOUT_RATE': 'servicio',
    'TEMPERATURE_VARIANCE': 'configuración',
    'STREAM_RATIO': 'configuración',
    'PROMPT_CATEGORY_ENTROPY': 'mezcla de temas',
}
LECTURA = {
    'seguridad': 'posible oleada de intentos de prompt injection (mensajes bloqueados)',
    'servicio': 'degradación del servicio (latencia, errores o timeouts)',
    'consumo': 'consumo atípico (costo o relación de tokens)',
    'prompts': 'prompts de longitud atípica',
    'volumen': 'cambio de volumen de tráfico',
    'configuración': 'cambio en los parámetros de las llamadas (temperatura o streaming)',
    'mezcla de temas': 'cambio en la mezcla de categorías de prompt',
    'multivariada': 'combinación inusual de varias métricas (ninguna es extrema por sí sola)',
}


# --------------------------------------------------------------------------
# 1) Calidad de datos y línea base robusta
# --------------------------------------------------------------------------
def limpiar_features(df, feats):
    """Pasa las features a float, reemplaza inf por nulo y rellena nulos con la mediana.
    Devuelve (df_limpio, nulos_por_feature) para poder reportar qué se rellenó."""
    out = df.copy()
    out[feats] = out[feats].astype(float).replace([np.inf, -np.inf], np.nan)
    nulos = out[feats].isna().sum()
    out[feats] = out[feats].fillna(out[feats].median())
    return out, nulos[nulos > 0]


def pares_correlacionados(df, feats, umbral=0.9):
    """Pares de features con |correlación| >= umbral (cuentan dos veces la misma señal)."""
    c = df[feats].corr().abs()
    filas = [(a, b, float(c.loc[a, b])) for i, a in enumerate(feats) for b in feats[i + 1:]
             if c.loc[a, b] >= umbral]
    return pd.DataFrame(filas, columns=['feature_a', 'feature_b', 'abs_corr']).sort_values(
        'abs_corr', ascending=False, ignore_index=True)


def _hora(ts):
    return pd.to_datetime(ts).dt.hour


def eta2_hora(df, feats, ts=TS):
    """Fracción de la varianza de cada feature explicada por la hora del día (0 a 1).
    Valores altos = ciclo diario fuerte: conviene comparar cada ventana con su hora."""
    h = _hora(df[ts])
    res = {}
    for f in feats:
        x = df[f].astype(float)
        total = ((x - x.mean()) ** 2).sum()
        res[f] = 0.0 if total == 0 else float(((x.groupby(h).transform('mean') - x.mean()) ** 2).sum() / total)
    return pd.Series(res).sort_values(ascending=False)


def _centro_escala(x):
    """Mediana y escala robusta. Si MAD = 0 (feature casi constante) usa la desviación
    media absoluta y, en último caso, la desviación estándar."""
    x = np.asarray(x, float)
    med = float(np.median(x))
    mad = 1.4826 * float(np.median(np.abs(x - med)))
    if mad > 1e-12:
        return med, mad
    mad = 1.2533 * float(np.mean(np.abs(x - med)))
    if mad > 1e-12:
        return med, mad
    sd = float(x.std())
    return med, (sd if sd > 1e-12 else 1.0)


def ajustar_linea_base(df, feats, estacionales=(), ts=TS):
    """Parámetros de la línea base: (mediana, escala) global por feature y, para las
    `estacionales`, uno por hora del día. Se ajusta una vez con las ventanas reales y se
    reutiliza tal cual para las ventanas inyectadas."""
    h = _hora(df[ts])
    p = {'feats': list(feats), 'global': {}, 'hora': {}}
    for f in feats:
        p['global'][f] = _centro_escala(df[f])
        if f in estacionales:
            p['hora'][f] = {int(k): _centro_escala(g.to_numpy()) for k, g in df[f].groupby(h)}
    return p


def centro_escala(df, p, ts=TS):
    """Mediana y escala de la línea base aplicables a cada fila (por hora si corresponde)."""
    h = _hora(df[ts]).to_numpy()
    C = pd.DataFrame(index=df.index, columns=p['feats'], dtype=float)
    S = C.copy()
    for f in p['feats']:
        med = np.full(len(df), p['global'][f][0], float)
        esc = np.full(len(df), p['global'][f][1], float)
        for k, (m, e) in p['hora'].get(f, {}).items():
            sel = h == k
            med[sel], esc[sel] = m, e
        C[f], S[f] = med, esc
    return C, S


def aplicar_linea_base(df, p, ts=TS):
    """Devuelve (Z, C): z robusto de cada feature y el valor típico (mediana) con que se comparó."""
    C, S = centro_escala(df, p, ts)
    Z = (df[p['feats']].astype(float) - C) / S
    return Z, C


# --------------------------------------------------------------------------
# 2) Detectores independientes y consenso
# --------------------------------------------------------------------------
def ajustar_mahalanobis(Z, ridge=1e-2, clip=10.0):
    """Distancia de Mahalanobis sobre z robustos (con z acotado para que una anomalía no
    distorsione la covarianza y una pequeña regularización por colinealidad)."""
    A = Z.clip(-clip, clip).to_numpy(float)
    inv = np.linalg.inv(np.cov(A, rowvar=False) + ridge * np.eye(A.shape[1]))
    return {'inv': inv, 'clip': clip}


def distancia_mahalanobis(Z, modelo):
    A = Z.clip(-modelo['clip'], modelo['clip']).to_numpy(float)
    return pd.Series(np.einsum('ij,jk,ik->i', A, modelo['inv'], A), index=Z.index)


def umbral_mad(x, k):
    """Regla robusta: mediana + k·MAD. El número de ventanas marcadas lo decide la
    distribución de los datos, no una cuota fija (como contamination)."""
    x = np.asarray(x, float)
    med = np.median(x)
    return float(med + k * 1.4826 * np.median(np.abs(x - med)))


def sensibilidad_umbral(score, ks=(2, 2.5, 3, 3.5, 4, 5, 6)):
    """Cuántas ventanas se marcan según k: para ver si hay un corte natural o no."""
    s = np.asarray(score, float)
    filas = []
    for k in ks:
        u = umbral_mad(s, k)
        filas.append({'k': k, 'umbral_score': round(u, 4), 'ventanas': int((s >= u).sum()),
                      'pct': round(100 * float((s >= u).mean()), 2)})
    return pd.DataFrame(filas)


def fijar_umbrales(score_if, d2, k=3.5, z_uni=4.5):
    """Umbrales de los tres detectores, calculados con las ventanas reales."""
    return {'k': k, 'if': umbral_mad(score_if, k),
            'mah_log': umbral_mad(np.log(np.clip(d2, 1e-12, None)), k), 'z_uni': z_uni}


def detectar(score_if, d2, Z, u):
    """Aplica los umbrales `u` a cualquier conjunto de ventanas (reales o inyectadas).
    CONSENSO = cuántos de los 3 detectores independientes la marcan."""
    out = pd.DataFrame(index=Z.index)
    out['SCORE_IF'] = np.asarray(score_if, float)
    out['D_IF'] = out['SCORE_IF'] >= u['if']
    out['D_MAH'] = np.log(np.clip(np.asarray(d2, float), 1e-12, None)) >= u['mah_log']
    out['Z_MAX'] = Z.abs().max(axis=1)
    out['D_UNI'] = out['Z_MAX'] >= u['z_uni']
    out['CONSENSO'] = out[['D_IF', 'D_MAH', 'D_UNI']].sum(axis=1)
    return out


# --------------------------------------------------------------------------
# 3) Estabilidad entre semillas
# --------------------------------------------------------------------------
def estabilidad(scores, k_top):
    """`scores`: DataFrame ventanas x semillas. Mide si el ranking depende de la semilla:
    Jaccard de los top-k y correlación de Spearman entre semillas."""
    tops = {c: set(scores[c].nlargest(k_top).index) for c in scores}
    cols = list(scores)
    jac = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            jac.loc[a, b] = len(tops[a] & tops[b]) / len(tops[a] | tops[b])
    off = ~np.eye(len(cols), dtype=bool)
    spe = scores.corr(method='spearman')
    return {
        'k_top': k_top,
        'jaccard': jac,
        'jaccard_medio': float(jac.to_numpy()[off].mean()),
        'spearman_min': float(spe.to_numpy()[off].min()),
        'pct_top_en_todas': len(set.intersection(*tops.values())) / k_top,
        'sd_score': scores.std(axis=1),
    }


# --------------------------------------------------------------------------
# 4) Inyección sintética: potencia de detección sin verdad conocida
# --------------------------------------------------------------------------
def escenarios_por_defecto(feats):
    """Un escenario por feature (sube) y unos pocos compuestos, que imitan incidentes reales."""
    # La entropía ya está casi en su tope (1): lo anómalo es que BAJE (un tema domina).
    esc = {f'{f} ↓' if f == 'PROMPT_CATEGORY_ENTROPY' else f'{f} ↑': [(f, -1 if f == 'PROMPT_CATEGORY_ENTROPY' else +1)]
           for f in feats}
    if 'TOTAL_LLM_REQUESTS' in feats:
        esc['TOTAL_LLM_REQUESTS ↓'] = [('TOTAL_LLM_REQUESTS', -1)]
    combos = {
        'Degradación combinada': ['AVG_RESPONSE_TIME_MS', 'LLM_TIMEOUT_RATE', 'LLM_ERROR_RATE'],
        'Consumo combinado': ['AVG_LLM_COST_USD', 'AVG_TOKEN_RATIO', 'AVG_PROMPT_LENGTH'],
    }
    for nombre, fs in combos.items():
        if all(f in feats for f in fs):
            esc[nombre] = [(f, +1) for f in fs]
    return esc


def inyectar(df, p, escenarios, magnitudes=(2, 3, 4, 6, 8), n_por_celda=30, semilla=0,
             key0=10_000_000, ts=TS):
    """Copia `n_por_celda` ventanas reales al azar por (escenario, magnitud) y desplaza las
    features del escenario `magnitud` desviaciones robustas (σ de su línea base). Como la
    magnitud es conocida, se puede medir qué fracción detecta el modelo. Devuelve las
    ventanas modificadas con _KEY (clave entera única), _ESCENARIO y _MAG."""
    rng = np.random.default_rng(semilla)
    _, S = centro_escala(df, p, ts)
    partes, key = [], key0
    for nombre, cambios in escenarios.items():
        for m in magnitudes:
            idx = rng.choice(len(df), size=min(n_por_celda, len(df)), replace=False)
            nuevo = df.iloc[idx].copy()
            for f, signo in cambios:
                v = nuevo[f] + signo * m * S.iloc[idx][f]
                nuevo[f] = v.clip(0, 1) if f in ACOTADAS_01 else v.clip(lower=0)
            nuevo['_KEY'] = np.arange(key, key + len(nuevo))
            key += len(nuevo)
            nuevo['_ESCENARIO'], nuevo['_MAG'] = nombre, m
            partes.append(nuevo)
    return pd.concat(partes, ignore_index=True)


def potencia(inj, detectada, orden=None):
    """Fracción detectada por escenario (filas) y magnitud en σ (columnas)."""
    t = inj[['_ESCENARIO', '_MAG']].assign(det=np.asarray(detectada, bool)).pivot_table(
        index='_ESCENARIO', columns='_MAG', values='det', aggfunc='mean')
    return t.reindex(orden) if orden is not None else t


def magnitud_minima(pot, objetivo=0.8):
    """Menor magnitud (en σ) con detección >= objetivo. NaN = no la alcanza con las magnitudes probadas."""
    def minimo(fila):
        ok = fila[fila >= objetivo]
        return float(ok.index.min()) if len(ok) else np.nan
    return pot.apply(minimo, axis=1).rename('sigma_minimo')


# --------------------------------------------------------------------------
# 5) Explicación e eventos
# --------------------------------------------------------------------------
def explicar(df, Z, C, filas, top=3, min_z=3.0):
    """Para cada fila de `filas` (índices de df): features más desviadas, su dirección, el
    valor observado vs el típico, y el tema dominante con su lectura de negocio."""
    out = []
    for i in filas:
        z = Z.loc[i]
        orden = z.abs().sort_values(ascending=False)
        sel = [f for f in orden.index[:top] if abs(z[f]) >= min_z] or []
        if sel:
            txt = '; '.join(f"{f} {'↑' if z[f] > 0 else '↓'} {z[f]:+.1f}σ "
                            f"({df.loc[i, f]:.4g} vs {C.loc[i, f]:.4g})" for f in sel)
            peso = {}
            for f in sel:
                peso[FAMILIA.get(f, 'otra')] = peso.get(FAMILIA.get(f, 'otra'), 0) + abs(z[f])
            fam = max(peso, key=peso.get)
        else:
            txt = f"ninguna feature supera {min_z}σ; la más alta: {orden.index[0]} ({z[orden.index[0]]:+.1f}σ)"
            fam = 'multivariada'
        out.append({'idx': i, 'EXPLICACION': txt, 'FAMILIA': fam, 'LECTURA': LECTURA.get(fam, fam)})
    return pd.DataFrame(out).set_index('idx')


def agrupar_eventos(w, marcada, max_gap_min=7.5, ts=TS, ventana_min=WINDOW_MIN):
    """Une ventanas marcadas separadas por <= max_gap_min en un evento y lo resume.
    Nivel: alta  = >=2 detectores coinciden y (>=2 ventanas o z >= 6);
           media = >=2 detectores coinciden, o una feature se desvía z >= 6;
           baja  = lo marca un solo detector con desviación moderada.
    `w` debe traer SCORE_IF, CONSENSO, Z_MAX, FAMILIA, LECTURA y EXPLICACION."""
    a = w[np.asarray(marcada, bool)].sort_values(ts)
    if a.empty:
        return pd.DataFrame(), a.assign(EVENTO_ID=pd.Series(dtype=int))
    nuevo = a[ts].diff() > pd.Timedelta(minutes=max_gap_min)
    a = a.assign(EVENTO_ID=nuevo.cumsum() + 1)
    g = a.groupby('EVENTO_ID')
    ev = g.agg(INICIO=(ts, 'min'), ULTIMA_VENTANA=(ts, 'max'), N_VENTANAS=(ts, 'size'),
               SCORE_PICO=('SCORE_IF', 'max'), Z_MAX=('Z_MAX', 'max'), CONSENSO_MAX=('CONSENSO', 'max'),
               FAMILIA=('FAMILIA', lambda s: s.value_counts().index[0]))
    pico = a.loc[g['SCORE_IF'].idxmax()].set_index('EVENTO_ID')
    ev['VENTANA_PICO'] = pico[ts]
    ev['EXPLICACION_PICO'] = pico['EXPLICACION']
    ev['LECTURA'] = ev['FAMILIA'].map(LECTURA)
    ev['FIN'] = ev['ULTIMA_VENTANA'] + pd.Timedelta(minutes=ventana_min)
    ev['DURACION_MIN'] = (ev['FIN'] - ev['INICIO']).dt.total_seconds() / 60
    ev['NIVEL'] = np.where((ev['CONSENSO_MAX'] >= 2) & ((ev['N_VENTANAS'] >= 2) | (ev['Z_MAX'] >= 6)), 'alta',
                           np.where((ev['CONSENSO_MAX'] >= 2) | (ev['Z_MAX'] >= 6), 'media', 'baja'))
    cols = ['NIVEL', 'INICIO', 'FIN', 'DURACION_MIN', 'N_VENTANAS', 'SCORE_PICO', 'Z_MAX', 'CONSENSO_MAX',
            'FAMILIA', 'LECTURA', 'VENTANA_PICO', 'EXPLICACION_PICO']
    orden = {'alta': 0, 'media': 1, 'baja': 2}
    ev = ev[cols].sort_values(['NIVEL', 'SCORE_PICO'], key=lambda s: s.map(orden) if s.name == 'NIVEL' else -s)
    return ev.reset_index(), a
