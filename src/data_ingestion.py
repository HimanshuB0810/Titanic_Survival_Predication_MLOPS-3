import pandas as pd
from sqlalchemy import create_engine
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.model_selection import train_test_split
import os
import sys
# from config.database_config import DB_CONFIG
from config.database_config import (
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
)
from config.paths_config import *

logger=get_logger(__name__)

class DataIngestion:
    
    def __init__(self,output_dir):

        self.output_dir=output_dir

        os.makedirs(self.output_dir,exist_ok=True)

    def connect_to_db(self):
        try:
            # engine = create_engine(
            #     f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@"
            #     f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}"
            # )
            engine = create_engine(
                f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )

            with engine.connect() as conn:
                pass   # forces actual DB connection

            logger.info("Database Connection Established")
            return engine

        except Exception as e:
            logger.error(f"Error while Establishing Database Connection {e}")
            raise CustomException(str(e), sys)
        
    def extract_data(self):
        try:
            engine = self.connect_to_db()
            query = "SELECT * FROM public.titanic"
            df = pd.read_sql_query(query,engine)

            logger.info("Data Extracted from DB")
            return df
        
        except Exception as e:
            logger.error(f"Error while Extracting Data {e}")
            raise CustomException(str(e),sys)
        
    def save_data(self,df):
        try:
            train_df,test_df=train_test_split(df,test_size=0.2,random_state=42)
            train_df.to_csv(TRAIN_PATH,index=False)
            test_df.to_csv(TEST_PATH,index=False)

            logger.info("Data Splitting And Saving Done")
        
        except Exception as e:
            logger.error(f"Error while Saving Data {e}")
            raise CustomException(str(e),sys)
        
    def run(self):
        try:
            logger.info("Data Ingestion Pipeline Started")

            df=self.extract_data()
            self.save_data(df)

            logger.info("End of Data Ingestion Pipeline")
        
        except Exception as e:
            logger.error(f"Error while Data Ingestion Pipeline {e}")
            raise CustomException(str(e),sys)
        
if __name__=="__main__":
    data_ingestion=DataIngestion(RAW_DIR)
    data_ingestion.run()