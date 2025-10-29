import os
import zipfile
from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Skips download since data is already available locally
        '''
        try:
            zip_download_dir = self.config.local_data_file
            logger.info(f"Using existing local data file at {zip_download_dir}")
            
            if not os.path.exists(zip_download_dir):
                raise FileNotFoundError(f"Local data file not found at {zip_download_dir}")
            
            logger.info("Local data file found successfully.")
            return zip_download_dir

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        """
        try:
            unzip_path = self.config.unzip_dir
            os.makedirs(unzip_path, exist_ok=True)

            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Data extracted successfully to {unzip_path}")

        except Exception as e:
            raise e
