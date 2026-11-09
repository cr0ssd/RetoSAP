# conexion.py

from helpers.imports import os, load_dotenv, dataframe

load_dotenv()

def _params():
    return dict(
        address=os.getenv("HANA_ADDRESS"),
        port=int(os.getenv("HANA_PORT")),
        user=os.getenv("HANA_USER"),
        password=os.getenv("HANA_PASSWORD"),
    )

def conexion():
    return dataframe.ConnectionContext(**_params())