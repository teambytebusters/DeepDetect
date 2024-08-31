from DeepfakeDetection import logger
from DeepfakeDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from DeepfakeDetection.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline

# STAGE_NAME = "Data Ingestion stage"
# try:
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n") 
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Data Preprocessing"

try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n") 
    data_preprocessing = DataPreprocessingPipeline()
    data_preprocessing.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e