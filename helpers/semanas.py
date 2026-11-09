# semanas.py

from helpers.imports import pd


def calculate_week_range(
    conn,
    tabla,
    week_days=7,
    min_last_days=3
):
    """
    Calcula el rango temporal de semanas usando la tabla completa.

    La última semana solo se incluye si contiene al menos
    min_last_days días.

    Returns:
        t0: Fecha inicial.
        t_max: Fecha máxima.
        n_weeks: Número de semanas consideradas.
    """

    ts_expr = (
        "TO_TIMESTAMP(REPLACE(REPLACE(\"TIMESTAMP\",'T',' '),'Z',''), "
        "'YYYY-MM-DD HH24:MI:SS.FF3')"
    )

    query = f"""
        SELECT
            MIN({ts_expr}) AS T0,
            MAX({ts_expr}) AS TMAX
        FROM {tabla}
    """

    r = conn.sql(query).collect()

    t0 = pd.to_datetime(r.loc[0, "T0"])
    t_max = pd.to_datetime(r.loc[0, "TMAX"])

    span = t_max - t0

    n_full = span // pd.Timedelta(days=week_days)
    rest = span - n_full * pd.Timedelta(days=week_days)

    n_weeks = n_full + (
        1 if rest >= pd.Timedelta(days=min_last_days) else 0
    )

    return t0, t_max, int(n_weeks)


def add_week_column(df, t0, week_days=7):
    """
    Agrega la columna WEEK_IDX a un DataFrame.

    El cálculo utiliza el mismo t0 para todos los DataFrames,
    garantizando que compartan la misma división temporal.
    """

    ts_expr = (
        "TO_TIMESTAMP(REPLACE(REPLACE(\"TIMESTAMP\",'T',' '),'Z',''), "
        "'YYYY-MM-DD HH24:MI:SS.FF3')"
    )

    t0_sql = t0.strftime("%Y-%m-%d %H:%M:%S")

    week_expr = (
        f"FLOOR(SECONDS_BETWEEN("
        f"TO_TIMESTAMP('{t0_sql}','YYYY-MM-DD HH24:MI:SS'), "
        f"{ts_expr}) / {week_days * 86400})"
    )

    return df.select("*", (week_expr, "WEEK_IDX"))


def split_into_weeks(
    df_system,
    df_llm,
    conn,
    tabla,
    week_days=7,
    min_last_days=3
):
    """
    Divide los DataFrames system y llm en semanas.

    Ambos DataFrames utilizan el mismo rango temporal,
    calculado sobre la tabla completa.

    Returns:
        system_weeks: Diccionario de DataFrames por semana.
        llm_weeks: Diccionario de DataFrames por semana.
        metadata: Información del rango temporal.
    """

    # Calcular rango sobre la tabla completa
    t0, t_max, n_weeks = calculate_week_range(
        conn,
        tabla,
        week_days,
        min_last_days
    )

    # Agregar índice semanal
    df_system = add_week_column(df_system, t0, week_days)
    df_llm = add_week_column(df_llm, t0, week_days)

    # Separar por semanas (DataFrames lazy)
    system_weeks = {
        w: df_system.filter(f"WEEK_IDX = {w}")
        for w in range(n_weeks)
    }

    llm_weeks = {
        w: df_llm.filter(f"WEEK_IDX = {w}")
        for w in range(n_weeks)
    }

    metadata = {
        "t0": t0,
        "t_max": t_max,
        "n_weeks": n_weeks
    }

    return system_weeks, llm_weeks, metadata

