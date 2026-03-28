import os
import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

# Importaciones condicionales para evitar errores si las librerías no están instaladas
try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
except ImportError:
    flag_alpaca_missing = True
    print("Advertencia: Librería 'alpaca-py' no encontrada. Las funciones de Alpaca no estarán disponibles.")

try:
    from massive import RESTClient
except ImportError:
    flag_alpaca_missing = True
    print("Advertencia: Librería 'massive' no encontrada. Las funciones de Massive no estarán disponibles.")

class AlpacaDataDownloader:
    """
    Clase para descargar datos históricos usando la API de Alpaca.
    Encapsula la lógica para Crypto y Stocks.
    """
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.crypto_client = None
        self.stock_client = None
        
        # Inicializar clientes si las librerías están disponibles
        if 'CryptoHistoricalDataClient' in globals():
            # Crypto no requiere claves para datos básicos
            self.crypto_client = CryptoHistoricalDataClient()
            
            if self.api_key and self.secret_key:
                self.stock_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_crypto_data(self, symbols: List[str], start_date: str, timeframe=None) -> pd.DataFrame:
        """
        Descarga datos de criptomonedas.
        """
        if not self.crypto_client:
            raise RuntimeError("Cliente Alpaca Crypto no disponible.")
        
        if timeframe is None:
            timeframe = TimeFrame.Day

        request_params = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=timeframe,
            start=start_date
        )
        bars = self.crypto_client.get_crypto_bars(request_params)
        return bars.df

    def fetch_stock_data(self, symbols: List[str], years: int = 5) -> pd.DataFrame:
        """
        Descarga datos de acciones para los últimos 'years' años.
        Retorna un diccionario {symbol: DataFrame}.
        """
        if not self.stock_client:
            raise ValueError("API Key y Secret Key son necesarios para datos de acciones o el cliente no se pudo inicializar.")

        end = datetime.now()
        start = end - timedelta(days=365 * years)
        
        req = StockBarsRequest(
            symbol_or_symbols=symbols, 
            timeframe=TimeFrame.Day, 
            start=start, 
            end=end
        )
        bars = self.stock_client.get_stock_bars(req)

        dfs = []
        for sym, records in bars.data.items():
            # Convertir registros a diccionarios usando model_dump (Alpaca SDK v2)
            data = []
            for r in records:
                if hasattr(r, 'model_dump'):
                    data.append(r.model_dump())
                elif hasattr(r, '__dict__'):
                    data.append(r.__dict__)
                else:
                    data.append(r) # Fallback

            df = pd.DataFrame(data)
            # Asegurar formato de timestamp y establecer índice
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp').sort_index()
                df['ticker'] = sym
            
            dfs.append(df)
            
        return pd.concat(dfs)

class MassiveDataDownloader:
    """
    Clase para descargar datos usando la API de Massive (Empresa X).
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = None
        if 'RESTClient' in globals():
            self.client = RESTClient(self.api_key)

    def fetch_data(self, tickers: List[str], days: int = 730) -> pd.DataFrame:
        """
        Descarga datos agregados para una lista de tickers.
        """
        if not self.client:
            raise RuntimeError("Cliente Massive no disponible.")

        end = datetime.now()
        start = end - timedelta(days=days)
        
        all_data = []
        
        for ticker in tickers:
            # Llamada a la API según el notebook
            response = self.client.list_aggs(
                ticker, 
                1, 
                "day", 
                str(start.date()), 
                str(end.date()), 
                limit=50000
            )
            
            aggs = []
            for agg in response:
                aggs.append({
                    "ticker": ticker,
                    "timestamp": agg.timestamp,
                    "close": agg.close,
                    "volume": agg.volume
                })
            
            all_data.extend(aggs)
            time.sleep(0.2) # Rate limit
            
        df = pd.DataFrame(all_data)
        if not df.empty and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.index = df['timestamp']
            
        return df

class RiskFreeRateDownloader:
    """
    Downloader for risk‑free rate series from FRED.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        if Fred is None:
            raise ImportError("fredapi is required for RiskFreeRateDownloader. Install with: pip install fredapi")
        self.client = Fred(api_key=self.api_key)

    def fetch_rate(
        self,
        series_id: str = "DGS10",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with a Date index and a column ``Rate`` containing the series values.

        Parameters
        ----------
        series_id: str
            FRED series identifier (default ``DGS10`` – 10‑year Treasury constant maturity rate).
        start, end: str, optional
            ISO‑format dates (e.g. '2023-01-01'). If omitted the full available range is returned.
        """
        data = self.client.get_series(series_id, observation_start=start, observation_end=end)
        df = pd.DataFrame(data, columns=["Rate"])
        df.index.name = "Date"
        df = df.dropna()
        return df
