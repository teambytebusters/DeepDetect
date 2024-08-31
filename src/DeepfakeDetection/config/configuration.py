from DeepfakeDetection.constants import PARAMS_FILE_PATH, CONFIG_FILE_PATH
from DeepfakeDetection.utils.common import read_yaml, create_directories
from DeepfakeDetection.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            output_dir=config.output_dir,
            max_frames=self.params.max_frames,
            target_size=self.params.target_size,
            min_size=self.params.min_size,
            min_neighbors=self.params.min_neighbors,
            scale_factor=self.params.scale_factor
        )

        return data_preprocessing_config