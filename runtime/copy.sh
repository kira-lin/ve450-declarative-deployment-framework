#!/bin/bash
airflow=$(kubectl get pods --selector name=airflow | tail -n1 | cut -d" " -f1)
kubectl cp ./media_template.py $airflow:/root/airflow/runtime/
kubectl cp ./media_dag_template.py $airflow:/root/airflow/runtime/
kubectl cp ./tf_dag_template.py $airflow:/root/airflow/runtime/
kubectl cp ./run_job.py $airflow:/root/airflow/runtime/
kubectl cp ./dl_template.py $airflow:/root/airflow/runtime/
