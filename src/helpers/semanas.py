# semanas.py
"""
División del dataset en semanas de 7 días contadas desde t0 (no ancladas al calendario).

  * t0 y t_max se calculan sobre la tabla COMPLETA (antes de separar system y LLM), para que
    ambos streams compartan las mismas semanas.
  * Sobrante final: si dura >= min_last_days es una semana corta; si dura menos se pega a la
    semana anterior (no se descarta: con entregas semanales esos días no vuelven a llegar).
"""

from helpers.imports import pd

TS_EXPR = ("TO_TIMESTAMP(REPLACE(REPLACE(\"TIMESTAMP\",'T',' '),'Z',''), "
           "'YYYY-MM-DD HH24:MI:SS.FF3')")


def calculate_week_range(conn, tabla, week_days=7, min_last_days=3):
    """t0, t_max y número de semanas (al menos 1) de la tabla completa."""
    r = conn.sql(f'SELECT MIN({TS_EXPR}) AS T0, MAX({TS_EXPR}) AS TMAX FROM {tabla}').collect()
    t0, t_max = pd.to_datetime(r.loc[0, 'T0']), pd.to_datetime(r.loc[0, 'TMAX'])

    span = t_max - t0
    n_full = span // pd.Timedelta(days=week_days)
    rest = span - n_full * pd.Timedelta(days=week_days)
    n_weeks = max(1, n_full + (1 if rest >= pd.Timedelta(days=min_last_days) else 0))
    return t0, t_max, int(n_weeks)


def week_expr(t0, n_weeks, week_days=7):
    """Índice de semana en SQL. LEAST manda el sobrante corto (y el último log) a la última semana."""
    t0_sql = t0.strftime('%Y-%m-%d %H:%M:%S')
    return (f"LEAST(FLOOR(SECONDS_BETWEEN(TO_TIMESTAMP('{t0_sql}','YYYY-MM-DD HH24:MI:SS'), "
            f"{TS_EXPR}) / {week_days * 86400}), {n_weeks - 1})")


def add_week_column(df, t0, n_weeks, week_days=7):
    """Agrega WEEK_IDX a un DataFrame de hana_ml (perezoso)."""
    return df.select('*', (week_expr(t0, n_weeks, week_days), 'WEEK_IDX'))


def tabla_semanas(t0, t_max, n_weeks, week_days=7):
    """WEEK (0..n-1), WEEK_START absoluto, WEEK_END (inicio de la siguiente o t_max), DIAS_CUBIERTOS."""
    inicios = [t0 + w * pd.Timedelta(days=week_days) for w in range(n_weeks)]
    semanas = pd.DataFrame({'WEEK': range(n_weeks), 'WEEK_START': inicios,
                            'WEEK_END': inicios[1:] + [t_max]})
    semanas['DIAS_CUBIERTOS'] = ((semanas['WEEK_END'] - semanas['WEEK_START'])
                                 .dt.total_seconds() / 86400).round(2)
    return semanas


def split_into_weeks(df_system, df_llm, conn, tabla, week_days=7, min_last_days=3):
    """
    Divide system y LLM en semanas con el mismo rango temporal (el de la tabla completa).

    Returns:
        system_weeks, llm_weeks: {semana: DataFrame de hana_ml perezoso}
        semanas: DataFrame WEEK, WEEK_START, WEEK_END, DIAS_CUBIERTOS
    """
    t0, t_max, n_weeks = calculate_week_range(conn, tabla, week_days, min_last_days)

    df_system = add_week_column(df_system, t0, n_weeks, week_days)
    df_llm = add_week_column(df_llm, t0, n_weeks, week_days)

    system_weeks = {w: df_system.filter(f'WEEK_IDX = {w}') for w in range(n_weeks)}
    llm_weeks = {w: df_llm.filter(f'WEEK_IDX = {w}') for w in range(n_weeks)}

    return system_weeks, llm_weeks, tabla_semanas(t0, t_max, n_weeks, week_days)
