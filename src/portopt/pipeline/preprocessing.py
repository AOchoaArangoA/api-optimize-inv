# Funciones de preprocesamiento de datos
import pandas as pd


def pivot_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pivota los datos de manera que cada columna sea un activo y cada fila sea una fecha.

    Espera un DataFrame en formato largo con columnas 'ticker', 'close' y la fecha
    en el índice (timestamp) o en una columna. Retorna un DataFrame con fechas en
    filas, tickers en columnas y precios de cierre como valores.
    """
    df = data.reset_index()
    # pivot() requiere nombres de columnas: la fecha debe estar como columna
    date_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    return df.pivot(index=date_col, columns="ticker", values="close")

def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza los datos de manera que cada columna tenga un valor medio de 0 y una desviación estándar de 1.
    """
    return (data - data.mean()) / data.std()

def scale_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Escala los datos de manera que cada columna tenga un valor mínimo de 0 y un valor máximo de 1.
    """
    return (data - data.min()) / (data.max() - data.min())