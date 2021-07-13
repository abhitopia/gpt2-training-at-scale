# gpt2-training-at-scale
This repository contains code to train GPT2 (and its distilled version) at scale on your own data using Pytorch Elastic.


# AWS 
## Setup
Note the following commands only work for us-west-2 region with cheap g4dn.xlarge spot instances.
For other regions, modify commands accordingly.
### Create cluster
Below command can take approx. 15 mins to finish.
```bash
eksctl create cluster \
    --name=torchelastic \
    --node-type=g4dn.xlarge \
    --region=us-west-2 \
    --version=1.17 \
    --ssh-access \
    --ssh-public-key=~/.ssh/id_rsa.pub \
    --nodes=2 \
    --managed \
    --spot \
    --zones=us-west-2b,us-west-2c,us-west-2d,us-west-2a
```
### Install Nvidia device plugin to enable GPU support on your cluster.
Deploy the following Daemonset (optional, if it doesn't already exist): 
```
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta4/nvidia-device-plugin.yml
```


### Install `ElasticJob` controller and CRD


```shell

git clone https://github.com/pytorch/elastic.git
cd elastic/kubernetes

kubectl apply -k config/default
# or
# kustomize build config/default  | kubectl apply -f -
```

You will see logs like following

```shell
namespace/elastic-job created
customresourcedefinition.apiextensions.k8s.io/elasticjobs.elastic.pytorch.org created
role.rbac.authorization.k8s.io/leader-election-role created
clusterrole.rbac.authorization.k8s.io/manager-role created
rolebinding.rbac.authorization.k8s.io/leader-election-rolebinding created
clusterrolebinding.rbac.authorization.k8s.io/elastic-job-k8s-controller-rolebinding created
deployment.apps/elastic-job-k8s-controller created
```

Verify that the `ElasticJob` custom resource is installed

```shell
kubectl get crd
```

The output should include `elasticjobs.elastic.pytorch.org`

```
NAME                                              CREATED AT
...
elasticjobs.elastic.pytorch.org                   2020-03-18T07:40:53Z
...
```

Verify controller is ready

```shell
kubectl get pods -n elastic-job

NAME                                          READY   STATUS    RESTARTS   AGE
elastic-job-k8s-controller-6d4884c75b-z22cm   1/1     Running   0          15s
```

### Check logs of controller

```shell
kubectl logs -f elastic-job-k8s-controller-6d4884c75b-z22cm -n elastic-job
2020-03-19T10:13:43.532Z	INFO	controller-runtime.metrics	metrics server is starting to listen	{"addr": ":8080"}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	controller-runtime.controller	Starting EventSource	{"controller": "elasticjob", "source": "kind source: /, Kind="}
2020-03-19T10:13:43.534Z	INFO	setup	starting manager
2020-03-19T10:13:43.534Z	INFO	controller-runtime.manager	starting metrics server	{"path": "/metrics"}
2020-03-19T10:13:43.822Z	DEBUG	controller-runtime.manager.events	Normal	{"object": {"kind":"ConfigMap","namespace":"elastic-job","name":"controller-leader-election-helper","uid":"50269b8b-69ca-11ea-b995-0653198c16be","apiVersion":"v1","resourceVersion":"2107564"}, "reason": "LeaderElection", "message": "elastic-job-k8s-controller-6d4884c75b-z22cm_4cf549b7-3289-4285-8e64-647d067178bf became leader"}
2020-03-19T10:13:44.021Z	INFO	controller-runtime.controller	Starting Controller	{"controller": "elasticjob"}
2020-03-19T10:13:44.121Z	INFO	controller-runtime.controller	Starting workers	{"controller": "elasticjob", "worker count": 1}
```

### Create and attach an EFS (Elastic File System)
Reference: https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html

#### Create an IAM OIDC provider for your cluster
Follow instructions at: https://docs.aws.amazon.com/eks/latest/userguide/enable-iam-roles-for-service-accounts.html

```bash
eksctl utils associate-iam-oidc-provider --cluster torchelastic --approve
```

Verify that you ave OIDC provider
```
aws eks describe-cluster --name torchelastic --query "cluster.identity.oidc.issuer" --output text
```

Example output:
```bash
https://oidc.eks.us-west-2.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E
```

```bash
aws iam list-open-id-connect-providers | grep <EXAMPLED539D4633E53DE1B716D3041E>
```

You should see exampl output
```bash
"Arn": "arn:aws-cn:iam::111122223333:oidc-provider/oidc.eks.us-west-2.amazonaws.com/id/EXAMPLED539D4633E53DE1B716D3041E"
```
#### Create and apple service account
```
curl -o iam-policy-example.json https://raw.githubusercontent.com/kubernetes-sigs/aws-efs-csi-driver/v1.3.2/docs/iam-policy-example.json
```

```bash
aws iam create-policy \
    --policy-name AmazonEKS_EFS_CSI_Driver_Policy \
    --policy-document file://iam-policy-example.json
```
```bash
ACCOUNT_ID=<ACCOUNT_ID> # obtain it using aws sts get-caller-identity
eksctl create iamserviceaccount \
    --name efs-csi-controller-sa \
    --namespace kube-system \
    --cluster torchelastic \
    --attach-policy-arn arn:aws:iam::${ACCOUNT_ID}:policy/AmazonEKS_EFS_CSI_Driver_Policy \
    --approve \
    --override-existing-serviceaccounts \
    --region us-west-2
```


#### Install the Amazon EFS driver
```bash
kubectl kustomize "github.com/kubernetes-sigs/aws-efs-csi-driver/deploy/kubernetes/overlays/stable/ecr?ref=release-1.3" > driver.yaml
kubectl apply -f driver.yaml
```

#### Create Amazon EFS System
```bash
vpc_id=$(aws eks describe-cluster \
    --name torchelastic \
    --query "cluster.resourcesVpcConfig.vpcId" \
    --output text)
```

```bash
cidr_range=$(aws ec2 describe-vpcs \
    --vpc-ids $vpc_id \
    --query "Vpcs[].CidrBlock" \
    --output text)
```

```bash
security_group_id=$(aws ec2 create-security-group \
    --group-name MyTorchElasticEfsSecurityGroup \
    --description "MyTorchElasticEfsSecurityGroup" \
    --vpc-id $vpc_id \
    --output text)
```

```bash
aws ec2 authorize-security-group-ingress \
    --group-id $security_group_id \
    --protocol tcp \
    --port 2049 \
    --cidr $cidr_range
```

```bash
file_system_id=$(aws efs create-file-system \
    --region us-west-2 \
    --performance-mode generalPurpose \
    --query 'FileSystemId' \
    --output text)
```

```bash
aws ec2 describe-subnets \
    --filters "Name=vpc-id,Values=$vpc_id" \
    --query 'Subnets[*].{SubnetId: SubnetId,AvailabilityZone: AvailabilityZone,CidrBlock: CidrBlock}' \
    --output table
```

Example output
```bash
|                           DescribeSubnets                          |
+------------------+--------------------+----------------------------+
| AvailabilityZone |     CidrBlock      |         SubnetId           |
+------------------+--------------------+----------------------------+
|  us-west-2c      |  192.168.128.0/19  |  subnet-EXAMPLE6e421a0e97  |
|  us-west-2b      |  192.168.96.0/19   |  subnet-EXAMPLEd0503db0ec  |
|  us-west-2c      |  192.168.32.0/19   |  subnet-EXAMPLEe2ba886490  |
|  us-west-2b      |  192.168.0.0/19    |  subnet-EXAMPLE123c7c5182  |
|  us-west-2a      |  192.168.160.0/19  |  subnet-EXAMPLE0416ce588p  |
|  us-west-2a      |  192.168.64.0/19   |  subnet-EXAMPLE12c68ea7fb  |
+------------------+--------------------+----------------------------+
```

Run below command for each subnetId
```bash
aws efs create-mount-target \
    --file-system-id $file_system_id \
    --subnet-id subnet-EXAMPLEe2ba886490 \
    --security-groups $security_group_id
```

#### Apply the EFS storage class to the cluster
- Download
```bash
curl -o storageclass.yaml https://raw.githubusercontent.com/kubernetes-sigs/aws-efs-csi-driver/master/examples/kubernetes/dynamic_provisioning/specs/storageclass.yaml
```

```bash
kubectl apply -f storageclass.yaml
```
### Deploy a ElasticJob

1. Deploy an etcd server. This will expose a Kubernetes service `etcd-service` with port `2379`.
    ```
    kubectl apply -f config/samples/etcd.yaml
    ```
1. Get the etcd server endpoint
   ```
   $ kubectl get svc -n elastic-job

   NAME           TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)    AGE
   etcd-service   ClusterIP   10.100.104.168   <none>        2379/TCP   5m5s
   ```

1. Update `config/samples/imagenet.yaml`:
    1. set `rdzvEndpoint` (e.g. `10.100.104.168:2379`) to the etcd server you just provisioned.
    1. set `minReplicas` and `maxReplicas` to the desired min and max num nodes
       (max should not exceed your cluster capacity)
    1. set `Worker.replicas` to the number of nodes to start with (you may
       modify this later to scale the job in/out)
    1. set the correct `--nproc_per_node` in `container.args` based on the
       instance you are running on.

    > **NOTE** the `ENTRYPOINT` to `torchelastic/examples` is
      `python -m torchelastic.distributed.launch <args...>`. Notice that you
      do not have to specify certain `launch` options such as `--rdzv_endpoint`,
      and `--rdzv_id`. These are set automatically by the controller.

    > **IMPORTANT** a `Worker` in the context of kubernetes refers to `Node` in
      `torchelastic.distributed.launch`. Each kubernetes `Worker` can run multiple
       trainers processes (a.k.a `worker` in `torchelastic.distributed.launch`).


1. Submit the training job.

    ```
    kubectl apply -f config/samples/imagenet.yaml
    ```

    As you can see, training pod and headless services have been created.
    ```
    $ kubectl get pods -n elastic-job
    NAME                                          READY   STATUS    RESTARTS   AGE
    elastic-job-k8s-controller-6d4884c75b-z22cm   1/1     Running   0          11m
    imagenet-worker-0                             1/1     Running   0          5s
    imagenet-worker-1                             1/1     Running   0          5s
    ```

1. You can scale the number of nodes by adjusting
   `.spec.replicaSpecs[Worker].replicas` and applying the change.
    ```
    kubectl apply -f config/samples/imagenet.yaml
    ```

    > **NOTE** since you are scaling the containers, you will be scaling in
      increments of `nproc_per_node` trainers. In our case ``--nproc_per_node=1``
      For better performance consider using an instance with multiple
      GPUs and setting `--nproc_per_node=$NUM_CUDA_DEVICES`.

    > **WARNING** the name of the job is used as `rdzv_id`, which is used
      to uniquely identify a job run instance. Hence to run multiple parallel
      jobs with the same spec you need to change `.spec.metadata.name` to
      give it a unique run id (e.g. `imagenet_run_0`). Otherwise the new nodes
      will attempt to join the membership of a different run.


### Monitoring jobs

You can describe the job to check job status and job related events.
In following example, `imagenet` job is created in `elastic-job` namespace, change to use your job name and namespace in your command.

```
$ kubectl describe elasticjob imagenet -n elastic-job

Name:         imagenet
Namespace:    elastic-job
<... OMITTED ...>
Status:
  Conditions:
    Last Transition Time:  2020-03-19T10:30:55Z
    Last Update Time:      2020-03-19T10:30:55Z
    Message:               ElasticJob imagenet is running.
    Reason:                ElasticJobRunning
    Status:                True
    Type:                  Running
<... OMITTED ...>
Events:
  Type    Reason                   Age   From                    Message
  ----    ------                   ----  ----                    -------
  Normal  SuccessfulCreatePod      13s   elastic-job-controller  Created pod: imagenet-worker-0
```

Tail the logs of a worker:

```
$ kubectl logs -f -n elastic-job imagenet-worker-0
```



## Teardown
```
eksctl delete cluster --region=us-west-2 --name=torchelastic
```
