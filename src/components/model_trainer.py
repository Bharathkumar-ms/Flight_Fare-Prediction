import os
import sys
from dataclasses import dataclass
import mlflow
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                                 models=models, param=params)

            print(model_report)

            # To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            #print(best_model_name)
            #print(best_model)


            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

             # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            # Calculate R-squared (R2) score
            r2 = r2_score(y_test, predicted)

            # Calculate Mean Squared Error (MSE)
            mse = mean_squared_error(y_test, predicted)

            # Calculate Root Mean Squared Error (RMSE)
            rmse = mse ** 0.5

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, predicted)
            
            test_r2 = r2_score(y_test, y_test_pred)
            test_adj_r2 = 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)



            logging.info("mlflow starts")
            print('mlflow starts')
            # Example parameters, modify as needed
            model_name = best_model_name
            model_params = params[best_model_name]

            mlflow.set_experiment("Flight Fare Prediction")
            
            with mlflow.start_run(run_name=model_name) as run:  # Initialize an MLflow run

                mlflow.set_tag("dev", "Bharathkumar M S")
                mlflow.set_tag("algo", "XGB")
                mlflow.log_params(model_params)  # Log model parameters

                # Train your model here
                best_model.fit(X_train, y_train)
    
                # Log metrics
                predicted = best_model.predict(X_test)
                r2 = r2_score(y_test, predicted)
                mse = mean_squared_error(y_test, predicted)
                rmse = mse ** 0.5
                mae = mean_absolute_error(y_test, predicted)
    
                mlflow.log_metrics({"r2": r2, "mse": mse, "rmse": rmse, "mae": mae})
    
                # Log the best model as an artifact
                #mlflow.sklearn.log_model(best_model, "best_model")

                return f"Best Model found is {best_model_name}, R-Squared is {r2}, Adjusted R-Squared is {test_adj_r2}, MSE is {mse}, RMSE is {rmse}, MAE is {mae}"

        except Exception as e:
            raise CustomException(e, sys)

