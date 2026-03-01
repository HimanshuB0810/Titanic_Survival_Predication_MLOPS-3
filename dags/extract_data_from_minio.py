from airflow import DAG
from airflow.providers.amazon.aws.operators.s3 import S3ListOperator
from airflow.providers.standard.operators.python import PythonOperator
from airflow.sdk.bases.hook import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime
import pandas as pd
import sqlalchemy
import os

LOCAL_FILE = "/tmp/Titanic-Dataset.csv"

def extract_and_load():
    # EXTRACT
    s3 = S3Hook(aws_conn_id="minio_conn")
    local_path=s3.download_file(
        key="Titanic-Dataset.csv",
        bucket_name="mlops-3",
        local_path="/tmp"
    )

    conn = BaseHook.get_connection("postgres_conn")
    engine = sqlalchemy.create_engine(
        f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}"
        
    )

    df = pd.read_csv(local_path)

    df.to_sql(
    "titanic",
    engine, 
    schema="public",
    if_exists="replace",
    index=False
)

    engine.dispose()

with DAG(
    dag_id="minio_to_postgres_etl",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    list_files = S3ListOperator(
        task_id="list_minio_files",
        bucket="mlops-3",
        aws_conn_id="minio_conn",
    )

    etl_task=PythonOperator(
        task_id="extract_and_load",
        python_callable=extract_and_load,
    )

    list_files >> etl_task