"""Colores y constantes compartidas por el dashboard y sus módulos (sin dependencias de Streamlit)."""

BRAND, NAVY, GOLD = "#0070F2", "#1D2D3E", "#F0AB00"
# Paleta de gráficas de SAP Fiori, en un orden validado para daltonismo.
SERIES = ["#5899DA", "#19A979", "#945ECF", "#ED4A7B", "#13A4B4", "#525DF4"]
# El dashboard usa siempre el tema claro (.streamlit/config.toml no define tema oscuro).
INK, INK2, MUTED, GRID, AXIS = "#1D2D3E", "#3d5166", "#556B82", "#eaedf0", "#d5dadd"
NEUTRO, SURFACE, CARD_BORDER, PAGE = "#c4cbd3", "#ffffff", "#e1e5ea", "#f5f6f7"
DIV_MID = "#f0f2f4"
CRITICO = "#D20A0A"
# Escala de azules de SAP Horizon (de claro a oscuro) para los mapas de calor.
AZUL_SEQ = ["#d1efff", "#a6e0ff", "#7bcfff", "#4db1ff", "#1b90ff", "#0070f2", "#0057d2", "#0040b0"]
CONFIANZA_COLOR = {"alta": "#0040b0", "media": "#0070f2", "baja": "#4db1ff"}

STREAMS = {"system": "Sistema SAP", "llm": "Servicios de IA"}
