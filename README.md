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
* Add an file upload field in homepage.

### Guides
The project is still in development
1. In `airflow-container/`, `media-container` and `tf-container/`, run `build.sh` to build the image
2. In `kube/`, run `deploy.sh` to deploy airflow on minikube
3. In `runtime/`, run `copy.sh` to copy files into PV
4. You can access airflow web ui on localhost:30800
5. To run demos, compress files in `demo/deep-learning` or `demo/media` to a zip, and upload it on the UI
6. The web UI pick up dags in a period, so please wait some time and refresh to see your dag.
7. To run it, first unpause it (switch to on), and then trigger it by clicking the left first buttuon among icons on the right.
8. Refresh the webpage you can see the status of subtasks in your dag. Go to task instance via graph view or tree view, etc, 
to view the log.
### To-do lists
* Clearly define roles and permissions
* Create more containers/templates to satisfy various workloads
