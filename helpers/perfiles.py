# perfiles.py

from helpers.explicacion import perfil_familias

base_cols = ['EVENTO_ID', 'WEEK', 'INICIO', 'FIN', 'N_VENTANAS', 'SCORE_PICO', 'Z_MAX']

def perfil_relevantes(eventos, if_df, z, familias):
    """Perfil por familia solo de los eventos relevantes (RELEVANTE = True)."""
    rel = eventos[eventos['RELEVANTE']]
    return rel[base_cols].merge(perfil_familias(if_df, z, familias), on='EVENTO_ID')



   