from src.logger import get_logger
from src.custom_exception import CustomException
import pandas as pd
from src.feature_store import RedisFeatureStore
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from config.paths_config import *
import pickle
import os
import sys


logger=get_logger(__name__)

class ModelTraining:

    def __init__(self,feature_store:RedisFeatureStore, model_save_path=MODEL_SAVE_PATH):
        self.feature_store=feature_store
        self.model_save_path=model_save_path
        self.model=None

        os.makedirs(self.model_save_path,exist_ok=True)
        logger.info("Model Training Initialized")
    
    def load_features_from_redis(self):
        try:

            entity_ids=self.feature_store.get_all_entity_ids()

            records=[]
            for entity_id in entity_ids:
                features=self.feature_store.get_features(entity_id)
                if features:
                    features["PassengerId"] = int(entity_id)
                    records.append(features)
            
            feature_df=pd.DataFrame(records)

            logger.info("Loading Features from redis")
            return feature_df
        
        except Exception as e:
            logger.error(f"Error While Loading data from Redis {e}")
            raise CustomException(str(e),sys)
        
    def load_labels(self,label_path):
        try:
            label_df=pd.read_csv(label_path, usecols=["PassengerId","Survived"])
            logger.info("Successfully loaded the Label (Survived) Column")
            return label_df

        except Exception as e:
            logger.error(f"Error While Loading Label {e}")
            raise CustomException(str(e),sys)
        
    def build_training_dataframe(self,feature_df,label_df):
        try:
            df = feature_df.merge(label_df, on="PassengerId", how="inner")
            logger.info("Successfully Merged the feature df with label df")
            return df
        
        except Exception as e:
            logger.error(f"Error While Building Training Dataframe {e}")
            raise CustomException(str(e),sys)
    
    def split_data(self,df):
        try:
            X=df.drop(columns=["PassengerId","Survived"])
            y=df["Survived"]

            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

            logger.info("Train Test Split Done")
            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            logger.error(f"Error While Train Test Split {e}")
            raise CustomException(str(e),sys)
        
    def handle_imbalance_data(self,X_train, y_train):
        try:
            smote=SMOTE(random_state=42)
            X_resampled,y_resampled=smote.fit_resample(X_train,y_train)
            logger.info("Handled Imbalanced data Sucessfully")
            return X_resampled,y_resampled

        except Exception as e:
                logger.error(f"Error while handling imbalanced data {e}")
                raise CustomException(str(e),sys)
        
    def hyperParameter_tuning(self,X_train,y_train):
        try:
            param_distributions = {
            'n_estimators': [200, 300, 500, 800],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.5]
        }

            rf=RandomForestClassifier(random_state=42)
            random_search=RandomizedSearchCV(estimator=rf,param_distributions=param_distributions,n_iter=10,cv=3,scoring='accuracy',random_state=42)
            random_search.fit(X_train,y_train)

            logger.info(f"Best Parameters: {random_search.best_params_}")
            return random_search.best_estimator_
        
        except Exception as e:
            logger.error(f"Error While HyperParameter Tuning data {e}")
            raise CustomException(str(e),sys)
    
    def evaluate_model(self,X_test,y_test,model):
        try:
            y_pred=model.predict(X_test)
            accuracy=accuracy_score(y_test,y_pred)
            logger.info(f"Accuracy is {accuracy}")

            return accuracy
        
        except Exception as e:
            logger.error(f"Error While Model Training {e}")
            raise CustomException(str(e),sys)
        
    def save_model(self,model):
        try:
            model_filename=f"{self.model_save_path}/random_forest_model.pkl"

            with open(model_filename,'wb') as model_file:
                pickle.dump(model,model_file)

            logger.info(f"Model Saved at {model_filename}")
        except Exception as e:
            logger.error(f"Error While Model Saving {e}")
            raise CustomException(str(e),sys)
        
    def run(self,label_path):
        try:
            logger.info("Starting Model Training Pipeline")
            
            feature_df=self.load_features_from_redis()
            label_df=self.load_labels(label_path)

            df=self.build_training_dataframe(feature_df,label_df)
            X_train,X_test,y_train,y_test=self.split_data(df)

            X_resampled,y_resampled=self.handle_imbalance_data(X_train,y_train)

            model=self.hyperParameter_tuning(X_resampled,y_resampled)
            accuracy=self.evaluate_model(X_test,y_test,model)
            self.save_model(model)

            print(f"Model Training Successful with Accuracy: {accuracy}")

            logger.info(f"Model Training Successful with Accuracy: {accuracy}")
            logger.info("End of Model Training Pipeline")

        except Exception as e:
            logger.error(f"Error While Model Training Pipeline {e}")
            raise CustomException(str(e),sys)
        
if __name__=="__main__":
    feature_store=RedisFeatureStore()
    model_trainer=ModelTraining(feature_store)
    model_trainer.run(label_path=TRAIN_PATH)