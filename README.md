# ve450-airflow-on-k8s
#### JI capstone project
#### Edge Computing Declarative Deployment Framework

### Prerequisites
* Docker
    * Chinese users can use registry.docker-cn.com to have faster speed
* Minikube(k8s)
    * Chinese user can find a minikube compiled by Chinese which replaces google sources.
    * Refer to https://kubernetes.io/docs/setup/minikube/ for more instructions.
    * When you start it, remember to give it more cpus and memory through `--cpus` and `--memory`

### Airflow
* Version 1.10
* changed flask-appbuilder from <2.0.0 to <1.12.0 due to compatibility
* modified Kubernetes executor a little to mount runtime directory in worker
* airflow-www directory contains the UI of airflow, a file upload field is added to the homepage. To use/develop it, please clone airflow-1-10.0 and replace the www folder with airflow-www, then run `python setup.py sdist -q` to make the archive. Finally run `pip install ` that tarball in your venv to install airflow.

### Guides
The project is still in development
1. In `airflow-container/`, run `build.sh` to build the image
2. In `kube/`, run `deploy.sh` to deploy airflow on minikube
3. You can access airflow web ui on localhost:30809
4. Use `kubectl cp` command to copy files into `/root/airflow/dags` or `/root/airflow/runtime`
5. To run the demo, you need to copy `mnist_keras.py` to `/root/airflow/runtime`, and `demo.py` to `/root/airflow/dags`. Then you can trigger it on webUI.
6. It's normal that the dag remains running, because there is a sever running. You can mark it to succeed.

### To-do lists
* Write a parser to generate dags file(in progress)
* Integrate functions into our UI
* Design a more complex demo
* Create more containers to satisfy various workloads
