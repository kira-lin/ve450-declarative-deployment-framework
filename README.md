# ve450-airflow-on-k8s
#### JI capstone project
#### Edge Computing Declarative Deployment Framework

### Prerequisites
* Docker
* Minikube(k8s)

### Airflow
* Version 1.10
* changed flask-appbuilder from <2.0.0 to <1.12.0 due to compatibility
* modified Kubernetes executor a little to mount runtime directory in worker

### Guides
The project is still in development
1. run `build.sh` to build the image
2. run `deploy.sh` to deploy airflow
3. You can access airflow web ui on localhost:30809
4. Use `kubectl cp` command to copy files into `/root/airflow/dags` or `/root/airflow/runtime`

###To-do lists
* Write a parser to generate dags file
* Integrate functions into our UI
* Design a more complex demo
* Create more containers to satisfy various workloads
