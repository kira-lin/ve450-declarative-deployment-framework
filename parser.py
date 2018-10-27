import yaml
from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator


def run(script):
    execFile('/root/airflow/runtime/{}'.format(script))


def create_python_task(task, dag):
    if 'executor_config' in task:
        t = PythonOperator(task, python_callable=run, op_args=[task['script']],dag=dag,
        executor_config={'KubernetesExecutor': task['executor_config']})
    else:
        t = PythonOperator(task, python_callable=run, op_args=[task['script']],dag=dag)

def create_bash_task(task, dag):
    if 'executor_config' in task:
        t = BashOperator(task, bash_command=task['bash_command'],dag=dag,
        executor_config={'KubernetesExecutor': task['executor_config']})
    else:
        t = BashOperator(task, bash_command=task['bash_command'],dag=dag)

def parse(stream):
    content = yaml.load(stream)
    args = {'owner': content['owner'], 'start_date': content['start_date']}
    dag_id = content['dag_id']
    dag = DAG(dag_id, schedule_interval=content['schedule_interval'], default_args = args)
    tasks = content['tasks']
    t = {}
    operator_map = {"python": create_python_task, "bash": create_bash_task}
    for task in tasks:
        t[task] = operator_map[task['operator']](task, dag)
        if 'upstream' in task:
            t[task].set_upstream(t[task['upstream']])
    globals()[dag_id] = dag



if __name__ == '__main__':
    from sys import argv
    input_file = argv[1]
    with open(input_file, 'r') as fin:
        parse(fin)
