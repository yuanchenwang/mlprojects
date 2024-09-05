import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging 
import os 
from src.utils import save_object

@dataclass # TODO: Explain what the dataclass is, why we use it, and how it works
class DataTransformationConfig:
    preproceessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''

        This function is responsible for transforming the data
        
        '''
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            '''
            Why with_maen=False?

            ValueError: Cannot center sparse matrices: pass `with_mean=False` instead. See docstring for motivation and alternatives.

            这个错误发生在 StandardScaler 尝试对稀疏矩阵进行中心化（即减去平均值）时。StandardScaler 默认会进行中心化和缩放，但对于稀疏矩阵，中心化操作会将稀疏矩阵转换为密集矩阵，这可能会导致内存问题。

            对于包含稀疏特征（如经过 OneHotEncoder 处理的分类特征）的数据，我们需要在 StandardScaler 中设置 with_mean=False。这样可以避免对稀疏矩阵进行中心化操作。
            '''
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            

            logging.info("Numerical columns standard scaling completed")
            logging.info(f"Numerical columns: {numerical_columns}")

            logging.info("Categorical columns encoding completed")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            # TODO: figure out what np.c_ does, what train_arr and test_arr are, and why we need it.
            '''
            现在，让我解释这段代码和 np.c_ 的作用：

            np.c_ 的作用：

            np.c_ 是 NumPy 中用于数组连接的便捷方法。
            它沿着第二个轴（列）连接两个或多个数组。


            train_arr 和 test_arr 的结构：

            这些数组是通过将特征数组（input_feature_train_arr 或 input_feature_test_arr）与目标变量数组（target_feature_train_df 或 target_feature_test_df）连接而成的。
            结果是一个新的 2D 数组，其中最后一列是目标变量（在这个例子中是 "math_score"）。


            为什么需要这样做：

            这种结构将特征和目标变量保存在同一个数组中，便于后续的处理和分析。
            在机器学习中，通常需要将数据分为特征（X）和目标变量（y）。这种结构使得这种分离变得容易。


            数组的形状：

            假设 input_feature_train_arr 的形状是 (n_samples, n_features)。
            target_feature_train_df 的形状是 (n_samples,)。
            结果 train_arr 的形状将是 (n_samples, n_features + 1)。


            具体到你的例子：

            你的 input_feature_arr 的切片显示每个样本有 19 个特征。
            连接后，train_arr 的每个样本将有 20 列：19 个特征列 + 1 个目标变量列。


            这个步骤返回的内容：

            train_arr：训练数据的特征和目标变量。
            test_arr：测试数据的特征和目标变量。
            预处理器对象的文件路径。


            这些返回值的用途：

            train_arr 和 test_arr 可以直接用于模型训练和评估。
            预处理器对象的文件路径可用于后续加载预处理器，以便对新数据进行转换。



            总结：这个步骤创建了一个结构化的数据集，其中特征和目标变量被组织在同一个数组中。这种格式便于后续的机器学习任务，如模型训练和评估。np.c_ 的使用使得这种数组连接变得简单和高效。
            '''
            train_arr=np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preproceessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preproceessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)