import os
import sys 
from src.exception import  CustomException 
from src.logger import logging 
import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

from tqdm import tqdm
import time

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Enter the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")

            # TODO : What does exit_ok do? 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                # Ready to be passed to the Data Transformation module for next step
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__=="__main__":
    # Create an overall progress bar
    with tqdm(total=4, desc="Overall Progress") as pbar:
        # Data Ingestion
        pbar.set_description("Data Ingestion")
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        pbar.update(1)
        time.sleep(0.5)  # Add short delay for demonstration

        # Data Transformation
        pbar.set_description("Data Transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)
        pbar.update(1)
        time.sleep(0.5)

        # Model Training
        pbar.set_description("Model Training")
        modeltrain = ModelTrainer()
        results = modeltrain.initiate_model_trainer(train_arr, test_arr)
        pbar.update(1)
        time.sleep(0.5)

        # Printing Results
        pbar.set_description("Printing Results")

        print("\nAll Models Report:")
        for model, details in results['all_models_report'].items():
            print(f"{model}:")
            print(f"  Test R2 Score: {details['test_r2_score']}")
            print(f"  Train R2 Score: {details['train_r2_score']}")
            print(f"  Best Parameters: {details['best_params']}")
            print()

        print(f"\nBest Model: {results['best_model_name']}")
        print(f"Best Model Score: {results['best_model_score']}")
        print(f"Best Model Parameters: {results['best_model_params']}")
        print(f"Model saved at: {results['trained_model_file_path']}")
        pbar.update(1)

print("Process completed!")