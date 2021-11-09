from prefect import task, Flow, Parameter, Client
from prefect.run_configs import KubernetesRun
from prefect.schedules import IntervalSchedule
from prefect.storage import GitHub


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

from datetime import timedelta

import numpy as np
import pandas as pd

import mlflow
import requests

@task
def fetch_data():
    csv_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(csv_url, sep=";")
    return data

@task
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
    
@task
def train_model(data, mlflow_experiment_id, alpha=0.5, l1_ratio=0.5):
    mlflow.set_tracking_uri("http://10.105.245.172:5000")

    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    
    with mlflow.start_run(experiment_id=mlflow_experiment_id):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")
        

prefect_project_name = "wine-quality-project"         # you can use what you want here
docker_image = "drtools/prefect:wine-classifier-3"    # any docker image that has the required Python dependencies
prefect_url = f"http://10.0.0.10:4200/graphql"
schedule = IntervalSchedule(interval=timedelta(minutes=2))


with Flow("train-wine-quality-model", schedule) as flow:
        data = fetch_data()
        train_model(data=data, mlflow_experiment_id=1, alpha=0.3, l1_ratio=0.3)
        
flow.run_config = KubernetesRun(
        labels=["dev"],
        service_account_name="prefect-server-serviceaccount",
        image=docker_image
    )
flow.storage = GitHub(repo="ksvcu/ml", path="flows/winepredict.py")
