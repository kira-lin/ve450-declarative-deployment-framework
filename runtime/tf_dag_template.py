# -*- coding: utf-8 -*-
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from __future__ import print_function
import airflow
from airflow.operators.python_operator import BranchPythonOperator
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.contrib.kubernetes.volume_mount import VolumeMount
from airflow.contrib.kubernetes.volume import Volume
from airflow.models import DAG
import os

args = {
    'owner': '{{ owner }}',
    'start_date': airflow.utils.dates.days_ago(2)
}

dag = DAG(
    dag_id='{{ ID }}', default_args=args,
    schedule_interval=None
)

volume_mount = VolumeMount(name='test-volume', mount_path='/root/runtime', sub_path='{{ subpath }}', read_only=False)
volume_config= {
    'persistentVolumeClaim':
      {
        'claimName': 'test-volume'
      }
    }
volume = Volume(name='test-volume', configs=volume_config)

model_name = '{{ model_name }}'
t1 = KubernetesPodOperator(task_id="train_and_save", dag=dag, in_cluster=True, volume_mounts=[volume_mount],
                           namespace='default', name="{}-trainer".format(model_name.lower()), volumes=[volume],
                           image='tensorflow:latest', arguments=['python', '/root/runtime/run_job.py'])

def model_exist():
    if {{ not_serve }} or os.path.isdir('/root/airflow/runtime/models/{}'.format(model_name)):
        return 'update_version_or_not_serve'
    else:
        return 'serve_model'


branch = BranchPythonOperator(
    task_id="serve_or_not", python_callable=model_exist, dag=dag
)

t2 = KubernetesPodOperator(namespace="default", name="{}-restapi".format(model_name.lower()), image="tensorflow/serving:latest",
                           env_vars={'MODEL_NAME':'saved_model', 'MODEL_BASE_PATH':'/root/runtime'},
                           task_id="serve_model", port=8501, dag=dag, async=True, in_cluster=True,
                           labels={'name':'{}-restapi'.format(model_name.lower())},
                           volume_mounts=[volume_mount], volumes=[volume])

t3 = DummyOperator(
    task_id="update_version_or_not_serve", dag=dag
)

t1.set_downstream(branch)
branch.set_downstream(t2)
branch.set_downstream(t3)
