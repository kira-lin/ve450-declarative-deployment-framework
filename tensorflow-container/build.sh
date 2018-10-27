#!/bin/bash
cp ../airflow-container/airflow.tar.gz .
docker build . -t tensorflow:airflow
rm ./airflow.tar.gz
