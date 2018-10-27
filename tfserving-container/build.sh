#!/bin/bash
cp ../airflow-container/airflow.tar.gz .
docker build . -t tfserving:airflow
rm ./airflow.tar.gz
