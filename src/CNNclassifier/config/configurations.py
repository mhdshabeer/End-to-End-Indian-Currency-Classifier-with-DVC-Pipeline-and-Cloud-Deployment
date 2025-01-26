from CNNclassifier.constants import __init
from pathlib import Path
from CNNclassifier.utils.common import read_yaml, create_directories
from CNNclassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
)

cfp = __init.CONFIG_FILE_PATH
pfp = __init.PARAMS_FILE_PATH


class ConfigurationManager:
    def __init__(self, config_filepath, params_filepath):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_image_size=tuple(self.params.IMAGE_SIZE),
            params_classes=self.params.CLASSES,
        )

        return prepare_base_model_config
