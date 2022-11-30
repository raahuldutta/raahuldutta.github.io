---
title: Model Deployment with KServe and KServe Features 
author: raahul
date: 2022-11-19 18:32:00 -0500
---

On a Friday Evening, Mid-September 2019 - Coworkers were leaving the office. The evening was welcoming a bright night. And I was working to deploy the LSTM models on production. I knew - “don't deploy anything on Friday”. But I rejected the guidance. I applied a small change on production Friday morning. Unfortunately, the model was not serving any requests due to poor infrastructure. At midnight - I was able to deploy the models perfectly. While returning home - I was sharing the bus with the returned party hoppers. You feel the pain right?

Now, It's 2022. The little hummingbird has fluttered her wings 7.5 billion times in the last three years. And We have now a simple pluggable solution for many burning machine learning problems like:

- Cost: Is the model over or under-scaled? Are resources being used efficiently?
- Monitoring: Are all the endpoints healthy? What is the performance profile and request trace?
- Rollouts: Is this rollout safe? How do I roll back? Can I test a change without swapping traffic?
- Protocol Standards: How do I make a prediction? GRPC? HTTP? Kafka
- How do I handle batch predictions?
- How do I leverage standardized Data Plane protocol so that I can move the model across MLServing Platforms?
- How do I serve Tensorflow, Xgboost, and Pytorch on the same infrastructure?
- How do I explain predictions?
- How do I wire up custom pre and post-processing?

And the Answer is KServe.

- KServe is a Model Inferencing Platform on Kubernetes.
- Run Anywhere Kubernetes runs. Provides Performant, Standardized inference protocol across ML Frameworks.
- Support modern serverless inference workload with Autoscaling including a scale to zero on GPU.
- Simple and Pluggable Production Serving including Prediction, pre/post-processing, monitoring, and explainable.
- Advanced deployments with Canary rollout, transformers, experiments, ensembles, and Model-Mesh.

![kserve](/posts/202201119/kserve1.png){: width="616" height="557"}

This image is not related with Kserve - Got the picture from reddit, its not an actual view.The picture was generated with the diffusion model.

## Starting KServe on EKS

We need a cluster with K8 version `1.22(minimum)`. We created a cluster called `dataplatform-sandbox_ml` to execute the exploration of KServe.

### Local Installation

If you are exploring on the local machine - you can install Minikube.

```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
minikube start --memory=max --cpus=max
minikube tunnel
#open a new terminal
```

### Kubernetes Installation

```bash
curl -LO "https://storage.googleapis.com/kubernetes-release/release/$(curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin/kubectl
kubectl cluster-info
```

### KServe Installation

Execute the Hack File:

```bash
curl -s "https://raw.githubusercontent.com/kserve/kserve/master/hack/quick_install.sh" | bash
```

Expected Output:

```bash
"😀 Successfully installed Istio”
"😀 Successfully installed Knative" 
"😀 Successfully installed Cert Manager”
"😀 Successfully installed KServe”
```

The Version Matrix

| Component                      | Version          |
|:-----------------------------|:-----------------|
| Knative          | 1.4.0     |
| Cert Manager Version              | 1.14.0   |
| Istio | Giovanni Rovelli |
| Kserve | 0.9 |

## Prerequisite: Model Training

In this example we will show how to serve [Huggingface Transformers with TorchServe](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers)
on KServe.

### Model archive file creation

Clone [pytorch/serve](https://github.com/pytorch/serve) repository,
navigate to `examples/Huggingface_Transformers` and follow the steps for creating the MAR file including serialized model and other dependent files.
TorchServe supports both eager model and torchscript and here we save as the pretrained model. 
Download the preprocess script from [here](sequence_classification/Transformer_kserve_handler.py)

```bash
torch-model-archiver --model-name BERTSeqClassification --version 1.0 \
--serialized-file Transformer_model/pytorch_model.bin \
--handler ./Transformer_kserve_handler.py \
--extra-files "Transformer_model/config.json,./setup_config.json,./Seq_classification_artifacts/index_to_name.json,./Transformer_handler_generalized.py"
```

The `BERTSeqClassification.mar` file will be generated.

We can use the below `config.properties` as a template.

```bash
inference_address=http://0.0.0.0:8085
management_address=http://0.0.0.0:8085
metrics_address=http://0.0.0.0:8082
grpc_inference_port=7070
grpc_management_port=7071
enable_metrics_api=true
metrics_format=prometheus
number_of_netty_threads=4
job_queue_size=10
enable_envvars_config=true
install_py_dep_per_model=false
service_envelope=body
model_store=/mnt/models/model-store
model_snapshot={"name":"startup.cfg","modelCount":1,"models":{"BERTSeqClassification":{"1.0":{"defaultVersion":true,"marName":"BERTSeqClassification.mar","minWorkers":1,"maxWorkers":2,"batchSize":4,"maxBatchDelay":1000,"responseTimeout":120}}}}
```

### Creating model storage with model archive and config file

The KServe/TorchServe integration expects following model store layout on the storage with TorchServe Model Archive and Model Configuration.

├── config
│   ├── config.properties
├── model-store
│   ├── BERTSeqClassification.mar

### Let's Run Our First Inference

For deploying the model on the CPU, apply the following `seqbert.yaml` to create the InferenceService.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "seqbert"
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      protocolVersion: v2
      storageUri: pvc://task-pv-claim/models
```

Kubectl

```bash
kubectl apply -f seqbert.yaml
```

Expected Output

```bash
$inferenceservice.serving.kserve.io/seqbert created
```

First, let's find out the Ingress host and port (Minikube) or Cluster IP (for AWS).

```bash
SERVICE_HOSTNAME=$(kubectl get inferenceservice seqbert -o jsonpath='{.status.url}' | cut -d "/" -f 3)
MODEL_NAME = 'BERTSeqClassification'
export INGRESS_HOST=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
export INGRESS_PORT=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.spec.ports[?(@.name=="http2")].port}')
CLUSTER_IP=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
```

Input

```bash
curl --location --request POST 'http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/infer' 
--header 'Host: ${SERVICE_HOSTNAME}' 
--header 'Content-Type: application/json'
--data-raw '{
"id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298",
"inputs": [{
"name": "4b7c7d4a-51e4-43c8-af61-04639f6ef4bc",
"shape": -1,
"datatype": "BYTES",
"data": "{"text":"Risk assessment implications of site-specific oral relative bioavailability."}"
}
]
}'
```

Expected Output

```bash
{"id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298", "model_name": "BERTSeqClassification", "model_version": "1.0", "outputs": [{"name": "predict", "shape": [], "datatype": "BYTES", "data": ["Pharma-Toxico"]}]}
```

## Model Storage

KServe supports Azure, hdfs, PVC, S3, URI, storageSpec.
We explored PVC ( Persistent Volume Claim) and S3 to store the models. The creation of PV and PVC and the Transfer of the model to PV have been covered here.

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: task-pv-volume
  labels:
    type: local
spec:
  storageClassName: manual
  capacity:
    storage: 2Gi
  accessModes:
    - ReadWriteOnce
  hostPath:
    path: "/home/ubuntu/mnt/data"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: task-pv-claim
spec:
  storageClassName: manual
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Gi
```

Please save the file as `pv-and-pvc.yaml` then deploy it.

```bash
kubectl apply -f pv-and-pvc.yaml
```

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: model-store-pod
spec:
  volumes:
    - name: model-store
      persistentVolumeClaim:
        claimName: task-pv-claim
  containers:
    - name: model-store
      image: ubuntu
      command: [ "sleep" ]
      args: [ "infinity" ]
      volumeMounts:
        - mountPath: "/pv"
          name: model-store
      resources:
        limits:
          memory: "1Gi"
          cpu: "1"
```

Please save the file as `pv-model-store.yaml` then deploy it.

```bash
kubectl apply -f pv-model-store.yaml
```

You can use the model-store pod executin the command :

```bash
kubectl exec -it model-store-pod -- bash
```

In different terminal, copy the model from local into PV.

```bash
kubectl cp models model-store-pod:/pv/ -c model-store
```

## Captum Explanations

In order to understand the word importances and attributions when we make an explanation Request, we use Captum Insights for the HuggingFace Transformers pre-trained model.

```bash
curl --location --request POST 'http://${INGRESS_HOST}:${INGRESS_PORT}/v2/models/${MODEL_NAME}/explain' 
--header 'Host: ${SERVICE_HOSTNAME}' 
--header 'Content-Type: application/json'
--data-raw '{
"id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298",
"inputs": [{
"name": "4b7c7d4a-51e4-43c8-af61-04639f6ef4bc",
"shape": -1,
"datatype": "BYTES",
"data": "{"text":"Risk assessment implications."}"
}
]
}'
```

Expected Output

```bash
{"id": "d3b15cad-50a2-4eaf-80ce-8b0a428bd298", "model_name": "BERTSeqClassification", "model_version": "1.0", "outputs": [{"name": "explain", "shape": [], "datatype": "BYTES", "data": [{"words": ["[CLS]", "Risk", "assessment", "implications", "of", "site-specific", "oral", "relative", "bioavailability", ".", "[SEP]"], "importances": [0.0, -0.43571255624310423, -0.11062097534384648, 0.11323803203829622, 0.05438679692935377, -0.11364841625009202, 0.15214504085858935, -0.0013061684457894148, 0.05712844103997178, -0.02296408323390218, 0.1937543236757826, -0.12138265438655091, 0.20713335609474381, -0.8044260616647264, 0.0], "delta": -0.019047775223331675}]}]}
```

> The first BERT model has been deployed. Let's take a break. 
Afterward, we will learn about some of the important features of KServe.
{: .prompt-tip }
> I hope you have enjoyed the Coffee. I read somewhere a nice coffee joke - 
_What’s the best Beatles’ song to play at a coffee shop? Latte Be!_
Let's try some important features of KServe.
{: .prompt-tip }

## Inference Batching

KServe supports batch prediction for any ML framework (TensorFlow, PyTorch, ...) without decreasing the performance.

![kserve](/posts/202201119/ib.png){: width="616" height="557"}

This batcher is implemented in the KServe model agent sidecar, so the requests first hit the agent sidecar, when a batch prediction is triggered the request is then sent to the model server container for inference.

- `maxBatchSize`: the max batch size for triggering a prediction.
- `maxLatency`: the max latency for triggering a prediction (In milliseconds).
- `timeout`: timeout of calling predictor service (In seconds).

All of the fields have default values in the code. You can config them or not as you wish.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "scibert"
spec:
  predictor:
    timeout: 60
    batcher:
      maxBatchSize: 32
      maxLatency: 5000
    model:
      modelFormat:
        name: pytorch
      protocolVersion: v2
      storageUri: pvc://task-pv-claim/models
```

![kserve](/posts/202201119/kserve2.png){: width="616" height="557"}

This image is not related with Kserve - Got the picture from reddit, its not an actual view.The picture was generated with the diffusion model.

## Canary Rollout

Canary rollout is a deployment strategy when you can release a new version of model to a small percent of the production traffic.
Lets deploy the below yaml files on your cluster with the new version.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "torchserve"
spec:
  predictor:
    model:
      modelFormat:
        name: pytorch
      protocolVersion: v2  
      storageUri: gs://kfserving-examples/models/torchserve/image_classifier/v2
```

apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "torchserve"
spec:
  predictor:
    canaryTrafficPercent: 20
    model:
      modelFormat:
        name: pytorch
      protocolVersion: v2  
      storageUri: gs://kfserving-examples/models/torchserve/image_classifier/v2



kubectl get revisions -l serving.kserve.io/inferenceservice=torchserve
NAME                                 CONFIG NAME                    K8S SERVICE NAME   GENERATION   READY   REASON   ACTUAL REPLICAS   DESIRED REPLICAS
torchserve-predictor-default-00001   torchserve-predictor-default                      1            True             1                 1
torchserve-predictor-default-00002   torchserve-predictor-default                      2            True             1                 1

kubectl get pods -l serving.kserve.io/inferenceservice=torchserve
NAME                                                             READY   STATUS    RESTARTS   AGE
torchserve-predictor-default-00001-deployment-7d99979c99-p49gk   2/2     Running   0          28m
torchserve-predictor-default-00002-deployment-c6fcc65dd-rjknq    2/2     Running   0          3m37s


SERVICE_HOSTNAME=$(kubectl get inferenceservice torchserve -o jsonpath='{.status.url}' | cut -d "/" -f 3)
CLUSTER_IP=$(kubectl -n istio-system get service istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
for i in {1..10}; do curl -H "Host: ${SERVICE_HOSTNAME}" http://${CLUSTER_IP}/v2/models/mnist/infer -d @./canary_input.json; done

{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
<html><title>500: Internal Server Error</title><body>500: Internal Server Error</body></html>Handling connection for 8080
<html><title>500: Internal Server Error</title><body>500: Internal Server Error</body></html>Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080
{"predictions": [2]}Handling connection for 8080

## Logging

We can add Prometheus and Grafana to the cluster to capture the model health. We need to add the prometheus.io/scrape and  `prometheus.io/port` under `annotations`

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "scibert"
  annotations:
    prometheus.io/scrape: 'true'
    prometheus.io/port: '8082'
spec:
  predictor:
    timeout: 60
    batcher:
      maxBatchSize: 32
      maxLatency: 5000
    model:
      modelFormat:
        name: pytorch
      protocolVersion: v2
      storageUri: pvc://task-pv-claim/models
```

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/prometheus
kubectl expose service prometheus-server --type=NodePort --target-port=9090 --name=prometheus-server-npkubectl expose service prometheus-server --type=NodePort --target-port=9090 --name=prometheus-server-np
```

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm install my-release grafana/grafana
kubectl get secret --namespace default my-release-grafana -o jsonpath="{.data.admin-password}" | base64 --decode ; echo
export POD_NAME=$(kubectl get pods --namespace default -l "app.kubernetes.io/name=grafana,app.kubernetes.io/instance=my-release" -o jsonpath="{.items[0].metadata.name}")
kubectl --namespace default port-forward $POD_NAME 3000
```

## Auto-Scaling

KServe supports the implementation of Knative Pod Autoscaler (KPA) and Kubernetes’ Horizontal Pod Autoscaler (HPA).

We can configure InferenceService with field containerConcurrency for a hard limit. The hard limit is an enforced upper bound. If concurrency reaches the hard limit, surplus requests will be buffered and must wait until enough capacity is free to execute the requests.

We configure InferenceService with annotation autoscaling.knative.dev/target for a soft limit. The soft limit is a targeted limit rather than a strictly enforced bound, particularly if there is a sudden burst of requests, this value can be exceeded.

I deployed `triton.yaml` on EKS cluster to explore the Autoscaling and I used `hey` to send loads.

```yaml
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: triton-gpu
  annotations:
    prometheus.io/scrape: 'true'
    prometheus.io/port: '8082'
spec:
  predictor:
    timeout: 1
    batcher:
      maxBatchSize: 32
      maxLatency: 5000
    containers:
    - image: {username}/triton-tensorrt-gpu-predictor:latest
      name: triton-container
      env:
      - name: OMP_NUM_THREADS
        value: "1"
      resources:
        limits:
          nvidia.com/gpu: 1
        requests:
          nvidia.com/gpu: 1
  transformer:
    containers:
    - image: {username}/triton-tensorrt-transformer:latest
      name: kfserving-container
      command:
      - "python"
      - "-m"
      - "bert_tokenizer"
```

Sample Json file :

```bash
{
    "instances":[
        {
            "text" : "Risk assessment implications of site-specific oral relative bioavailability."
        }
    ]
}
```

### Hey Installation

```bash
go install github.com/rakyll/hey@latest
hey -m POST -z 60s -D ./triton_sample.json -host triton-gpu.default.example.com http://a6907a1015c574abc96a2e47036d54c5-903913666.us-east-2.elb.amazonaws.com/v1/models/transformer_tensorrt_inference:predict
```

### Execution Results

- Before the “Hey” Execution - Single pod of transformer and predictor is live.
![kserve](/posts/202201119/as1.png){: width="616" height="557"}

- After 2 Seconds - pods are warming up
![kserve](/posts/202201119/as2.png){: width="616" height="557"}

- After 30 Seconds - pods are running
![kserve](/posts/202201119/as3.png){: width="616" height="557"}

- After 70 Seconds - bomberding done. pods are dying
![kserve](/posts/202201119/as4.png){: width="616" height="557"}

- After 90 Seconds - back square 1
![kserve](/posts/202201119/as5.png){: width="616" height="557"}
