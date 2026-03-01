from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.Model_Training import ModelTraining
from src.feature_store import RedisFeatureStore
from config.paths_config import *


if __name__=="__main__":
    data_ingestion=DataIngestion(RAW_DIR)
    data_ingestion.run()

    feature_store=RedisFeatureStore()
    data_processor=DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()

    model_trainer=ModelTraining(feature_store)
    model_trainer.run(label_path=TRAIN_PATH)