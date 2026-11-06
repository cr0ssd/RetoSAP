"""
clasificacion.py: tipo de ataque más probable de cada evento relevante, por reglas.
Solo clasificación: no genera ventanas ni entrena modelos.

Método (por evento):
  1. Top 3 features por |z| promedio sobre las ventanas del evento (z con signo, contra su semana).
  2. Cada una vota por los tipos de su firma cuyo signo coincide con el del z. Su voto (1) se reparte
     en partes iguales entre esos tipos. Las firmas con '~' aceptan cualquier signo.
  3. Gana el tipo con más votos. Empate: mayor suma ponderada por posición (3, 2, 1);
     si aún hay empate, el tipo que vota la primera feature.
  4. Si ninguna feature vota con el signo exigido, se repite ignorando el signo (SIN_SIGNO = True).
  5. Confianza: MARGEN = votos del 1.º - votos del 2.º. >= 1 alta, [0.5, 1) media, < 0.5 baja;
     siempre baja si hubo que ignorar el signo.
"""
import pandas as pd

PESO_POS = (3, 2, 1)
MARGEN_ALTA, MARGEN_MEDIA = 1.0, 0.5
TIPOS_NO_ATAQUE = {'Falla operativa'}     # se cuentan como tipo, pero no como ataque


def _parse(spec):
    """'DoS+;Falla operativa-' -> {'DoS': '+', 'Falla operativa': '-'}  (+ sube, - baja, ~ cualquiera)."""
    out = {}
    for item in spec.split(';'):
        item = item.strip()
        tipo, signo = item[:-1].strip(), item[-1]
        assert signo in '+-~', f'Signo inválido en {item!r}'
        out[tipo] = signo
    return out


FIRMAS_SYSTEM = {k: _parse(v) for k, v in {
    'TOTAL_REQUESTS': 'DoS+;DDoS+;Falla operativa-',
    'REQUEST_RATE_STD': 'DoS+;DDoS+',
    'INTERVAL_CV': 'DoS+;Fuerza bruta-',
    'UNIQUE_IPS_COUNT': 'DDoS+;DoS-',
    'IP_ENTROPY': 'DDoS+;DoS-',
    'IP_TOP_SHARE': 'DoS+;Fuerza bruta+;DDoS-',
    'ERROR_RATE_400': 'Fuerza bruta+;Reconocimiento+',
    'ERROR_RATE_500': 'Explotación+;Falla operativa+',
    'NON_200_RATE': 'Fuerza bruta+;Reconocimiento+;Falla operativa+',
    'RATIO_ERROR': 'Explotación+;Falla operativa+',
    'RATIO_WARN': 'Falla operativa+',
    'RATIO_SECURITY': 'Fuerza bruta+;Explotación+',
    'RATIO_INFO': 'Falla operativa-',
    'RATIO_AUDIT': 'Fuerza bruta+;Contexto SAP+',
    'GET_VS_POST_RATIO': 'Fuerza bruta-;Explotación-;Reconocimiento+',
    'JS_HTTP_STATUS_CODE': 'Fuerza bruta~;Reconocimiento~;Explotación~;Falla operativa~',
    'JS_HEADERS_HTTP_REQUEST_METHOD': 'Explotación~;Fuerza bruta~',
    'JS_SAP_FUNCTION_LOG_TYPE': 'Contexto SAP~;Falla operativa~',
    'JS_HEATHERS_REQUEST_PATH': 'Reconocimiento~;Explotación~',
    'JS_SAP_FUNCTION_APPLICATION': 'Contexto SAP~',
    'JS_SAP_SOURCE_TYPE': 'Contexto SAP~',
    'JS_SAP_APP_ENV': 'Contexto SAP~',
    'JS_REGION_CODE': 'Contexto SAP~;DDoS~',
    'JS_HEADERS_CONTENT_TYPE': 'Explotación~',
}.items()}

FIRMAS_LLM = {k: _parse(v) for k, v in {
    'TOTAL_LLM_REQUESTS': 'Abuso de recursos+;Automatización+',
    'AVG_LLM_COST_USD': 'Abuso de recursos+',
    'INJECTION_COUNT': 'Prompt injection+',
    'SAFETY_POLICY_COUNT': 'Prompt injection+',
    'AUTH_FAILURE_COUNT': 'Acceso no autorizado+',
    'ACCESS_DENIED_COUNT': 'Acceso no autorizado+',
    'RATE_ABUSE_COUNT': 'Abuso de recursos+;Degradación del servicio+',
    'DUPLICATE_REQUEST_COUNT': 'Automatización+',
    'MALFORMED_INPUT_COUNT': 'Automatización+',
    'AVG_TOKEN_RATIO': 'Abuso de recursos+',
    'AVG_RESPONSE_TIME_MS': 'Degradación del servicio+;Abuso de recursos+',
    'LLM_ERROR_RATE': 'Degradación del servicio+;Acceso no autorizado+;Automatización+',
    'LLM_TIMEOUT_RATE': 'Degradación del servicio+',
    'TEMPERATURE_VARIANCE': 'Automatización+',
    'PROMPT_CATEGORY_ENTROPY': 'Prompt injection+;Automatización+',
    'STREAM_RATIO': 'Automatización~;Abuso de recursos~',
    'AVG_PROMPT_LENGTH': 'Abuso de recursos+;Prompt injection+',
    'MAX_PROMPT_LENGTH': 'Prompt injection+;Abuso de recursos+',
}.items()}


def validar_firmas(features, firmas):
    """Cada feature del IF debe tener firma, y no debe haber firmas de features que el IF no usa."""
    faltan = [f for f in features if f not in firmas]
    sobran = [f for f in firmas if f not in features]
    assert not faltan, f'Features del IF sin firma: {faltan}'
    assert not sobran, f'Firmas de features que el IF no usa: {sobran}'


def _votar(top, firmas, usar_signo=True):
    """Votos y suma ponderada por posición de cada tipo, y tipos que vota la primera feature."""
    votos, pos, primera = {}, {}, set()
    for p, (f, z) in enumerate(top):
        tipos = [t for t, s in firmas[f].items()
                 if (not usar_signo) or s == '~' or (s == '+' and z > 0) or (s == '-' and z < 0)]
        for t in tipos:
            votos[t] = votos.get(t, 0.0) + 1 / len(tipos)
            pos[t] = pos.get(t, 0.0) + PESO_POS[p] / len(tipos)
            if p == 0:
                primera.add(t)
    return votos, pos, primera


def _confianza(margen, sin_signo):
    if sin_signo or margen < MARGEN_MEDIA:
        return 'baja'
    return 'media' if margen < MARGEN_ALTA else 'alta'


def clasificar_eventos(if_df, z, eventos, firmas, top=3):
    """
    Una fila por evento RELEVANTE: TIPO_ATAQUE, MARGEN, CONFIANZA, SIN_SIGNO, TOP3 y VOTOS.
    `z` es la tabla larga (WEEK, WINDOW_INDEX, FEATURE, Z) de las ventanas anómalas.
    """
    cols = ['EVENTO_ID', 'TIPO_ATAQUE', 'MARGEN', 'CONFIANZA', 'SIN_SIGNO', 'TOP3', 'VOTOS']
    ids = eventos.loc[eventos['RELEVANTE'], 'EVENTO_ID'].astype(int)
    if ids.empty:
        return pd.DataFrame(columns=cols)

    a = if_df.loc[if_df['LABEL'] == -1, ['WEEK', 'WINDOW_INDEX', 'EVENTO_ID']].copy()
    a['EVENTO_ID'] = a['EVENTO_ID'].astype(int)
    zz = z.merge(a, on=['WEEK', 'WINDOW_INDEX'])
    zz = zz[zz['EVENTO_ID'].isin(ids)].assign(ZABS=lambda d: d['Z'].abs())
    perfil = (zz.groupby(['EVENTO_ID', 'FEATURE'])
                .agg(Z=('Z', 'mean'), ZABS=('ZABS', 'mean')).reset_index())

    filas = []
    for ev, g in perfil.groupby('EVENTO_ID'):
        t = g.sort_values('ZABS', ascending=False).head(top)
        feats = list(zip(t['FEATURE'], t['Z']))
        votos, pos, prim = _votar(feats, firmas, True)
        sin_signo = not votos
        if sin_signo:
            votos, pos, prim = _votar(feats, firmas, False)
        orden = sorted(votos, key=lambda k: (-round(votos[k], 9), -round(pos[k], 9), k not in prim, k))
        v2 = votos[orden[1]] if len(orden) > 1 else 0.0
        margen = votos[orden[0]] - v2
        filas.append({
            'EVENTO_ID': int(ev), 'TIPO_ATAQUE': orden[0], 'MARGEN': round(margen, 2),
            'CONFIANZA': _confianza(margen, sin_signo), 'SIN_SIGNO': sin_signo,
            'TOP3': ', '.join(f'{f} ({v:+.1f})' for f, v in feats),
            'VOTOS': ', '.join(f'{k} {votos[k]:.2f}' for k in orden),
        })
    return pd.DataFrame(filas, columns=cols)


def resumen_tipos(tipos):
    """Cantidad de tipos distintos (y de ataques, sin contar fallas operativas) con su conteo de eventos."""
    if tipos.empty:
        return {'tipos': 0, 'ataques': 0, 'lista': [], 'eventos_por_tipo': {}}
    c = tipos['TIPO_ATAQUE'].value_counts()
    return {'tipos': int(len(c)),
            'ataques': int(sum(t not in TIPOS_NO_ATAQUE for t in c.index)),
            'lista': list(c.index),
            'eventos_por_tipo': {k: int(v) for k, v in c.items()}}
