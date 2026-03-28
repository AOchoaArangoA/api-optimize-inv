import os
import io
import logging
from typing import Optional, Union

import pandas as pd

try:
    from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
except ImportError:
    BlobServiceClient = None
    print("Advertencia: Librería 'azure-storage-blob' no encontrada. Instalar con: pip install azure-storage-blob")

logger = logging.getLogger(__name__)


class AzureBlobConnector:
    """
    Conector para Azure Blob Storage.

    Soporta autenticación por connection string o account key.
    Permite subir, descargar y listar blobs, así como leer/escribir
    DataFrames directamente en formato CSV o Parquet.

    Parameters
    ----------
    connection_string : str, optional
        Connection string completa de la cuenta de almacenamiento.
        Si no se provee, se busca en la variable de entorno
        ``AZURE_STORAGE_CONNECTION_STRING``.
    account_name : str, optional
        Nombre de la cuenta de almacenamiento (alternativa a connection_string).
    account_key : str, optional
        Clave de la cuenta (requerida si se usa account_name).
    container_name : str, optional
        Contenedor por defecto para las operaciones.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        if BlobServiceClient is None:
            raise ImportError(
                "azure-storage-blob es requerido. Instalar con: pip install azure-storage-blob"
            )

        self.container_name = container_name
        self._client: BlobServiceClient = self._build_client(
            connection_string, account_name, account_key
        )
        logger.info("AzureBlobConnector inicializado correctamente.")

    # ------------------------------------------------------------------
    # Construcción del cliente
    # ------------------------------------------------------------------

    def _build_client(
        self,
        connection_string: Optional[str],
        account_name: Optional[str],
        account_key: Optional[str],
    ) -> "BlobServiceClient":
        conn_str = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")

        if conn_str:
            return BlobServiceClient.from_connection_string(conn_str)

        if account_name and account_key:
            if "AccountName=" in account_name or "DefaultEndpointsProtocol" in account_name:
                raise ValueError(
                    "Parece que pasaste una connection string completa como 'account_name'. "
                    "Usa el parámetro 'connection_string' en su lugar."
                )
            account_url = f"https://{account_name}.blob.core.windows.net"
            from azure.storage.blob import BlobServiceClient as _BSC
            return _BSC(account_url=account_url, credential=account_key)

        raise ValueError(
            "Se requiere 'connection_string', la variable de entorno "
            "'AZURE_STORAGE_CONNECTION_STRING', o bien 'account_name' + 'account_key'."
        )

    # ------------------------------------------------------------------
    # Gestión de contenedores
    # ------------------------------------------------------------------

    def create_container(self, container_name: Optional[str] = None) -> None:
        """Crea el contenedor si no existe."""
        name = container_name or self.container_name
        try:
            self._client.create_container(name)
            logger.info("Contenedor '%s' creado.", name)
        except ResourceExistsError:
            logger.info("Contenedor '%s' ya existe.", name)

    def list_containers(self) -> list[str]:
        """Retorna los nombres de todos los contenedores de la cuenta."""
        return [c["name"] for c in self._client.list_containers()]

    # ------------------------------------------------------------------
    # Operaciones con blobs
    # ------------------------------------------------------------------

    def list_blobs(self, container_name: Optional[str] = None, prefix: str = "") -> list[str]:
        """Lista los blobs dentro de un contenedor, opcionalmente filtrados por prefijo."""
        name = container_name or self.container_name
        container: ContainerClient = self._client.get_container_client(name)
        return [b.name for b in container.list_blobs(name_starts_with=prefix)]

    def upload_bytes(
        self,
        data: bytes,
        blob_name: str,
        container_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """Sube bytes crudos a un blob."""
        name = container_name or self.container_name
        blob: BlobClient = self._client.get_blob_client(container=name, blob=blob_name)
        blob.upload_blob(data, overwrite=overwrite)
        logger.info("Blob '%s' subido al contenedor '%s'.", blob_name, name)

    def download_bytes(self, blob_name: str, container_name: Optional[str] = None) -> bytes:
        """Descarga el contenido de un blob como bytes."""
        name = container_name or self.container_name
        blob: BlobClient = self._client.get_blob_client(container=name, blob=blob_name)
        try:
            return blob.download_blob().readall()
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob '{blob_name}' no encontrado en '{name}'.")

    def upload_file(
        self,
        local_path: str,
        blob_name: Optional[str] = None,
        container_name: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        """Sube un archivo local a Azure Blob Storage."""
        blob_name = blob_name or os.path.basename(local_path)
        with open(local_path, "rb") as f:
            self.upload_bytes(f.read(), blob_name, container_name, overwrite)

    def download_file(
        self,
        blob_name: str,
        local_path: str,
        container_name: Optional[str] = None,
    ) -> None:
        """Descarga un blob y lo guarda en disco."""
        data = self.download_bytes(blob_name, container_name)
        with open(local_path, "wb") as f:
            f.write(data)
        logger.info("Blob '%s' guardado en '%s'.", blob_name, local_path)

    def delete_blob(self, blob_name: str, container_name: Optional[str] = None) -> None:
        """Elimina un blob del contenedor."""
        name = container_name or self.container_name
        blob: BlobClient = self._client.get_blob_client(container=name, blob=blob_name)
        blob.delete_blob()
        logger.info("Blob '%s' eliminado de '%s'.", blob_name, name)

    # ------------------------------------------------------------------
    # Integración con DataFrames
    # ------------------------------------------------------------------

    def upload_dataframe(
        self,
        df: pd.DataFrame,
        blob_name: str,
        container_name: Optional[str] = None,
        fmt: str = "parquet",
        overwrite: bool = True,
    ) -> None:
        """
        Serializa un DataFrame y lo sube como blob.

        Parameters
        ----------
        df : pd.DataFrame
        blob_name : str
            Nombre del blob destino (ej. 'data/prices.parquet').
        fmt : str
            Formato de serialización: ``'parquet'`` (default) o ``'csv'``.
        overwrite : bool
        """
        buf = io.BytesIO()
        if fmt == "parquet":
            df.to_parquet(buf, index=True)
        elif fmt == "csv":
            buf.write(df.to_csv(index=True).encode("utf-8"))
        else:
            raise ValueError(f"Formato no soportado: '{fmt}'. Usar 'parquet' o 'csv'.")
        self.upload_bytes(buf.getvalue(), blob_name, container_name, overwrite)

    def download_dataframe(
        self,
        blob_name: str,
        container_name: Optional[str] = None,
        fmt: str = "parquet",
    ) -> pd.DataFrame:
        """
        Descarga un blob y lo deserializa como DataFrame.

        Parameters
        ----------
        blob_name : str
        fmt : str
            ``'parquet'`` (default) o ``'csv'``.
        """
        data = self.download_bytes(blob_name, container_name)
        buf = io.BytesIO(data)
        if fmt == "parquet":
            return pd.read_parquet(buf)
        elif fmt == "csv":
            return pd.read_csv(buf, index_col=0)
        else:
            raise ValueError(f"Formato no soportado: '{fmt}'. Usar 'parquet' o 'csv'.")
