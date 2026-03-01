FROM astrocrpublic.azurecr.io/runtime:3.1-13

RUN pip install \
    apache-airflow-providers-amazon \
    apache-airflow-providers-postgres

