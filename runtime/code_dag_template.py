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
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
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

t1 = KubernetesPodOperator(task_id="run_script", dag=dag, in_cluster=True, volume_mounts=[volume_mount],
                           namespace='default', name="{}-runner".format(ID), volumes=[volume],
                           image='{{ image }}', arguments=['python', '/root/runtime/run_job.py'])