"""
clustering.py - Comprobación de consistencia entre los tipos de ataque (clasificación por reglas)
y un k-means sobre las ventanas anómalas de los eventos relevantes.

No es una validación independiente: la clasificación sale de los mismos z-scores que ve el k-means.
Sirve para reforzar que los tipos se distinguen con las variables de la ventana.

Flujo, por tamaño de ventana y stream (X, mapa vienen de R[tamaño][stream]):

    cluster, estados = clustering.clusterizar_semanas(conn, X, mapa)   # una corrida por semana
    mapa['CLUSTER_ID'] = cluster
    tabla = clustering.validar_semanas(mapa, estados)                  # métricas por semana

Decisiones de diseño:
  * K de cada semana = número de tipos de ataque presentes en esa semana.
  * Una semana es "no evaluable" si K < 2, si hay menos ventanas que K o menos puntos distintos que K.
  * El k-means corre en HANA (PAL) sobre una tabla temporal local, sin normalizar: X ya viene
    en z-score por semana contra todas las ventanas, y volver a estandarizar con solo las
    anómalas amplificaría las variables que casi no varían entre ellas.
  * Las etiquetas de cluster no tienen orden propio; por eso se comparan con métricas que no
    dependen de la numeración (ARI, NMI, homogeneidad, completitud) y con una tabla cruzada.
"""
import numpy as np
import pandas as pd

import ventanas

KM_TABLE = '#KM_ENTRADA'


# --------------------------------------------------------------------------
# 1) K-means en HANA (PAL) sobre una tabla temporal local
# --------------------------------------------------------------------------
def kmeans_pal(conn, X, k, table=KM_TABLE):
    """
    K-means de PAL sobre las filas de X (DataFrame de pandas, solo variables numéricas).
    Devuelve un arreglo de CLUSTER_ID (1..k) en el mismo orden de filas que X.
    """
    from hana_ml.algorithms.pal.clustering import KMeans as HanaKMeans

    feats = list(X.columns)
    q = ventanas._q
    ventanas.drop_temp(conn, table)
    columnas = ', '.join('%s DOUBLE' % q(c) for c in feats)
    ventanas._run(conn, 'CREATE LOCAL TEMPORARY COLUMN TABLE %s ("PAL_ID" INTEGER, %s)' % (q(table), columnas))
    filas = [(i, *[float(v) for v in fila]) for i, fila in enumerate(X.to_numpy())]
    marcas = ', '.join(['?'] * (len(feats) + 1))
    with conn.connection.cursor() as cur:
        cur.executemany('INSERT INTO %s VALUES (%s)' % (q(table), marcas), filas)
    if not conn.connection.getautocommit():
        conn.connection.commit()

    datos = conn.sql('SELECT * FROM %s' % q(table))
    modelo = HanaKMeans(n_clusters=k, init='patent', max_iter=300, tol=1e-6, normalization='no')
    asig = modelo.fit_predict(datos, key='PAL_ID', features=feats).collect().sort_values('PAL_ID')
    if (len(asig) != len(X) or asig['PAL_ID'].duplicated().any()
            or asig['CLUSTER_ID'].isna().any() or asig['CLUSTER_ID'].nunique() != k):
        raise RuntimeError('PAL no devolvió una asignación válida de %d clusters para %d filas.' % (k, len(X)))
    return asig['CLUSTER_ID'].astype(int).to_numpy() + 1


# --------------------------------------------------------------------------
# 2) Una corrida por semana
# --------------------------------------------------------------------------
def motivo_no_evaluable(Xw, k):
    """None si se puede agrupar; si no, el motivo."""
    if k < 2:
        return 'solo %d tipo(s) en la semana' % k
    if len(Xw) < k:
        return '%d ventanas < K=%d' % (len(Xw), k)
    if len(np.unique(Xw.to_numpy(), axis=0)) < k:
        return 'menos puntos distintos que K=%d' % k
    return None


def clusterizar_semanas(conn, X, mapa, clusterer=kmeans_pal):
    """
    Agrupa las ventanas de cada semana con K = tipos presentes en esa semana.
    X y mapa deben tener las mismas filas en el mismo orden (índice 0..n-1).
    Devuelve (CLUSTER_ID por fila, NaN donde la semana no es evaluable; tabla de estados por semana).
    """
    cols = ['WEEK', 'ventanas', 'eventos', 'K', 'estado', 'motivo']
    cluster = pd.Series(np.nan, index=mapa.index, dtype=float)
    filas = []
    for w, g in mapa.groupby('WEEK'):
        k = int(g['TIPO_ATAQUE'].nunique())
        Xw = X.loc[g.index].reset_index(drop=True)
        motivo = motivo_no_evaluable(Xw, k)
        if motivo is None:
            cluster.loc[g.index] = clusterer(conn, Xw, k)
        filas.append({'WEEK': w, 'ventanas': len(g), 'eventos': int(g['EVENTO_ID'].nunique()), 'K': k,
                      'estado': 'evaluable' if motivo is None else 'no evaluable', 'motivo': motivo or ''})
    return cluster, pd.DataFrame(filas, columns=cols)


# --------------------------------------------------------------------------
# 3) Comparación contra los tipos
# --------------------------------------------------------------------------
def _metricas(tipos, clusters):
    from sklearn.metrics import (adjusted_rand_score, completeness_score,
                                 homogeneity_score, normalized_mutual_info_score)
    return {'ARI': adjusted_rand_score(tipos, clusters),
            'NMI': normalized_mutual_info_score(tipos, clusters),
            'homogeneidad': homogeneity_score(tipos, clusters),
            'completitud': completeness_score(tipos, clusters)}


def validar_semanas(mapa, estados):
    """
    Una fila por semana: estado, K y métricas de acuerdo entre TIPO_ATAQUE y CLUSTER_ID,
    a nivel ventana y a nivel evento (cluster mayoritario de cada evento; ARI solo con >= 3 eventos).
    """
    metricas = ['ARI', 'NMI', 'homogeneidad', 'completitud', 'ARI_eventos']
    filas = []
    for _, e in estados.iterrows():
        fila = e.to_dict()
        fila.update({m: np.nan for m in metricas})
        if e['estado'] == 'evaluable':
            g = mapa[(mapa['WEEK'] == e['WEEK'])]
            fila.update(_metricas(g['TIPO_ATAQUE'], g['CLUSTER_ID'].astype(int)))
            ev = g.groupby('EVENTO_ID').agg(tipo=('TIPO_ATAQUE', 'first'),
                                            cluster=('CLUSTER_ID', lambda s: s.mode().iloc[0]))
            if len(ev) >= 3:
                fila['ARI_eventos'] = _metricas(ev['tipo'], ev['cluster'].astype(int))['ARI']
        filas.append(fila)
    return pd.DataFrame(filas, columns=list(estados.columns) + metricas)


def tabla_cruzada(mapa, semana):
    """Ventanas de la semana por tipo (filas) y cluster (columnas)."""
    g = mapa[(mapa['WEEK'] == semana) & mapa['CLUSTER_ID'].notna()]
    return pd.crosstab(g['TIPO_ATAQUE'], g['CLUSTER_ID'].astype(int), margins=True, margins_name='total')


# --------------------------------------------------------------------------
# 4) Diagnóstico visual (PCA local)
# --------------------------------------------------------------------------
def graficar_pca(X, mapa, semana, titulo):
    """Ventanas de una semana evaluable en 2D: color = cluster, forma = tipo de ataque."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA

    g = mapa[(mapa['WEEK'] == semana) & mapa['CLUSTER_ID'].notna()]
    if len(g) < 3:
        print('PCA: se necesitan al menos 3 ventanas.')
        return
    pca = PCA(n_components=2)
    p = pca.fit_transform(X.loc[g.index].to_numpy())
    d = pd.DataFrame({'PC1': p[:, 0], 'PC2': p[:, 1],
                      'Cluster': g['CLUSTER_ID'].astype(int).astype(str).to_numpy(),
                      'Tipo': g['TIPO_ATAQUE'].to_numpy()})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(data=d, x='PC1', y='PC2', hue='Cluster', style='Tipo', palette='tab10', s=80, alpha=0.85, ax=ax)
    ax.set(title=titulo,
           xlabel='PC1 (%.1f%% de varianza)' % (100 * pca.explained_variance_ratio_[0]),
           ylabel='PC2 (%.1f%% de varianza)' % (100 * pca.explained_variance_ratio_[1]))
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
