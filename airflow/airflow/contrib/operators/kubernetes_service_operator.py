from airflow.exceptions import AirflowException
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.contrib.kubernetes import kube_client
import yaml

template_fields = ('templates_dict',)
template_ext = tuple()
ui_color = '#ffefeb'

class KubernetesServiceOperator(BaseOperator):
    _yaml = """
    apiVersion: v1
    kind: Service
    metadata:
      name: name
    spec:
      type: NodePort
      ports:
        - port: 8051
      selector:
        name: airflow
        """
    def execute(self, context):
        try:
            client = kube_client.get_kube_client(in_cluster=self.in_cluster,
                                                 cluster_context=self.cluster_context,
                                                 config_file=self.config_file)
            # req = {'apiVersion': 'v1', 'kind': 'Service', 'metadata': {'name': 'name'}, 'spec': {'type': 'NodePort', 'ports': [{'port': 8051, 'node_port': 30851}], 'selector': {'name': 'airflow'}}}
            req = yaml.load(self._yaml)
            req['metadata']['name'] = self.name
            req['spec']['ports'][0]['port'] = self.port
            # req['spec']['ports'][0]['node_port'] = self.node_port
            req['spec']['selector']['name'] = self.selector
            self.log.debug(req)
            client.create_namespaced_service(body=req, namespace=self.namespace)
            self.log.debug("Creating service")
        except AirflowException as ex:
            raise AirflowException('Pod Launching failed: {error}'.format(error=ex))

    @apply_defaults
    def __init__(self,
                 namespace,
                 name,
                 selector,
                 node_port,
                 port,
                 in_cluster=False,
                 cluster_context=None,
                 labels=None,
                 startup_timeout_seconds=120,
                 annotations=None,
                 affinity=None,
                 config_file=None,
                 *args,
                 **kwargs):
        super(KubernetesServiceOperator, self).__init__(*args, **kwargs)
        self.namespace = namespace
        self.labels = labels or {}
        self.startup_timeout_seconds = startup_timeout_seconds
        self.name = name
        self.selector = selector
        self.node_port = node_port
        self.port = port,
        self.in_cluster = in_cluster
        self.cluster_context = cluster_context
        self.annotations = annotations or {}
        self.affinity = affinity or {}
        self.config_file = config_file
