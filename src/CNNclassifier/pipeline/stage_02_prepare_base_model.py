from CNNclassifier.config.configurations import ConfigurationManager
from CNNclassifier.components.prepare_base_model import PrepareBaseModel
from CNNclassifier.__init import logger
from CNNclassifier.constants import __init

cfp = __init.CONFIG_FILE_PATH
pfp = __init.PARAMS_FILE_PATH
STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager(cfp, pfp)
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()


if __name__ == "__main__":
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
