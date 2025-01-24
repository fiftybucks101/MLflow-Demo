import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import dagshub
import logging

dagshub.init(repo_owner='fiftybucks001', repo_name='MLflow-Demo', mlflow=True)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #Read the dataset red wine quality

    try:
        data = pd.read_csv('winequality-red.csv')
    except Exception as e:
        logger.exception('Unable to read dataset. Error: %s', e)

    # train test split
    train, test = train_test_split(data,test_size=0.25,random_state=42)

    train_x = train.drop(['quality'], axis=1)
    test_x = test.drop(['quality'], axis=1)
    train_y = train[['quality']]
    test_y = test[['quality']]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_leaf_nodes = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    min_samples_split = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    criterion = str(sys.argv[5]) if len(sys.argv) > 5 else "squared_error"

    with mlflow.start_run():
        rf_reg = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=max_depth,
                                    max_leaf_nodes=max_leaf_nodes, 
                                    min_samples_split=min_samples_split,
                                    criterion=criterion, 
                                    random_state=42)
        rf_reg.fit(train_x,train_y)

        predicted_qualities = rf_reg.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y,predicted_qualities)

        print("RMSE: %s" %rmse)
        print("MAE: %s" %mae)
        print("R2: %s" %r2)

        mlflow.log_param("n_estimators",n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_leaf_nodes", max_leaf_nodes)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("criterion", criterion)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        remote_server_uri = "https://dagshub.com/fiftybucks001/MLflow-Demo.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                rf_reg, "model", registered_model_name="RandomForestRegressionWineModel"
            )
        else:
            mlflow.sklearn.log_model(rf_reg,"model")