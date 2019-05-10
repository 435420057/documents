# Ubuntu 18.04 系统安装Kubernetes + Docker第一次尝试

首先我准备4个虚拟机进行尝试，需要安装kubeadm,kubelet, kubectl 这个应该才能进行下步安装。根据网上教程:

## 1. 先在所有节点上安装kubeadm 
 
 由于官方提供的安装方法需要访问Google要翻墙，所以我只有根据网上的方法通过阿里云的源来的安装。
 
 首先增加Ubuntu源信息 /etc/apt/sources.list 文件中加入如下下信息
 ```shell
 # kubeadm及kubernetes组件安装源
deb https://mirrors.aliyun.com/kubernetes/apt kubernetes-xenial main
 ```

保存后，执行

```shell
$ sudo apt-get update
```
这时可能会出现这样的样的提示:

```shell
Get:1 https://mirrors.aliyun.com/kubernetes/apt kubernetes-xenial InRelease [8,993 B]
Err:1 https://mirrors.aliyun.com/kubernetes/apt kubernetes-xenial InRelease               
  The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 6A030B21BA07F4FB
```

这问题需要注册一下这个Key才能解决所以，你可以执行:

```shell
$ sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 6A030B21BA07F4FB
```

你还会看到这样的错误:

```shell
W: GPG error: https://mirrors.aliyun.com/kubernetes/apt kubernetes-xenial InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY 6A030B21BA07F4FB
E: The repository 'https://mirrors.aliyun.com/kubernetes/apt kubernetes-xenial InRelease' is not signed.
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
N: See apt-secure(8) manpage for repository creation and user configuration details.

```

网上说这个提示可以不用管

接下来执行以下强制安装命令：

```shell
$ sudo apt-get install -y kubelet kubeadm kubectl --allow-unauthenticated
```

接下需要设置主机host, 使用所以主机能通过主名称访问到主机，这样:
```shell
$ sudo vim /etc/hosts
```

设置成这样：

```shell
192.168.18.130  k8s-master
192.168.18.131  docker-node1
192.168.18.132  docker-node2
192.168.18.133  docker-node3
```

主机这个里面的IP是你的节点的主机IP、

## 安装Kubernetes

我们在所有节点上都安装好了kubeadm ，现在我们首先在master节点上面执行:
```shell
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.18.130
```

你会发现执行错误提示:
```shell
[ERROR Swap]: running with swap on is not supported. Please disable swap
```

要求关闭swap,所以接下来我们就关闭Swap,注意所有节点都关闭

关闭swap：如果swap不关闭，可能会导致kubelet组件启动失败

```shell
sudo swapoff -a
```
注销/etc/fstab中的swap，避免开机启动swap

```shell
sudo sed -i "/ swap/ s/^/#/ " /etc/fstab 
```

现在重新执行kubeadm init 上面的命令,发现在这次错误更多，提示我们无法pull images，这是因为k8s.gcr.io 这个域名是Google的被干掉了，所以我们只有通过国内的源来解决这个问题，仔细看了一下错误:
```shell
[ERROR ImagePull]: failed to pull image k8s.gcr.io/kube-controller-manager:v1.14.1: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1
	[ERROR ImagePull]: failed to pull image k8s.gcr.io/kube-scheduler:v1.14.1: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1
	[ERROR ImagePull]: failed to pull image k8s.gcr.io/kube-proxy:v1.14.1: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1
	[ERROR ImagePull]: failed to pull image k8s.gcr.io/pause:3.1: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1
	[ERROR ImagePull]: failed to pull image k8s.gcr.io/etcd:3.3.10: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1
	[ERROR ImagePull]: failed to pull image k8s.gcr.io/coredns:1.3.1: output: Error response from daemon: Get https://k8s.gcr.io/v2/: net/http: request canceled while waiting for connection (Client.Timeout exceeded while awaiting headers)
, error: exit status 1

```

还好告诉了我它需些什么东西，那个我们就用阿里的源把这个些docker镜像弄下来，我的所有节点都安装好了docker-ce 如下:

先下镜像:
```shell
$ docker pull registry.cn-hangzhou.aliyuncs.com/google_containers/kube-controller-manager:v1.14.1
```
等待下载完成，然后我通过tag将他标记为k8s.gcr.io的镜像：
```shell
$  docker tag registry.cn-hangzhou.aliyuncs.com/google_containers/kube-controller-manager:v1.14.1 k8s.gcr.io/kube-controller-manager:v1.14.1
```
重复这样的操作将所有提示出来的所有docker镜像都下载下来。

下载完后，确认都标记完成，继续执行：

```shell
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address=192.168.18.130
```

终于安装成功。注意完提示信息后面有个加入集群的提示最好记下来。这样的：
```shell
kubeadm join 192.168.18.130:6443 --token 13wq39.15xn5oo26op6mp6r \
    --discovery-token-ca-cert-hash sha256:542a3f2a932bf7474ca77a8faf0ffc72370aef490a6129ba30f36a0aff415f8f
```

现在，执行如下命令来配置kubectl。
```shell
$ mkdir -p $HOME/.kube
$ sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
$ sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

这样master的节点就配置好了，并且可以使用kubectl来进行各种操作了，根据上面的提示接着往下做，将slave节点加入到集群。

## Slave节点加入集群

将master节点上init 成功后提示加入命令拿到子节点上面去执行：

在slave节点执行如下的命令,将slave节点加入集群，正常的返回信息如下：

```shell
$ sudo kubeadm join 192.168.18.130:6443 --token 13wq39.15xn5oo26op6mp6r \
    --discovery-token-ca-cert-hash sha256:542a3f2a932bf7474ca77a8faf0ffc72370aef490a6129ba30f36a0aff415f8f
```

返回:
```shell
[preflight] Running pre-flight checks
	[WARNING IsDockerSystemdCheck]: detected "cgroupfs" as the Docker cgroup driver. The recommended driver is "systemd". Please follow the guide at https://kubernetes.io/docs/setup/cri/
[preflight] Reading configuration from the cluster...
[preflight] FYI: You can look at this config file with 'kubectl -n kube-system get cm kubeadm-config -oyaml'
[kubelet-start] Downloading configuration for the kubelet from the "kubelet-config-1.14" ConfigMap in the kube-system namespace
[kubelet-start] Writing kubelet configuration to file "/var/lib/kubelet/config.yaml"
[kubelet-start] Writing kubelet environment file with flags to file "/var/lib/kubelet/kubeadm-flags.env"
[kubelet-start] Activating the kubelet service
[kubelet-start] Waiting for the kubelet to perform the TLS Bootstrap...

This node has joined the cluster:
* Certificate signing request was sent to apiserver and a response was received.
* The Kubelet was informed of the new secure connection details.

Run 'kubectl get nodes' on the control-plane to see this node join the cluster.
```

在master节点查看节点加入情况:
```shell
$ sudo kubectl get nodes
```
返回

```shell
NAME            STATUS     ROLES     AGE       VERSION
ubuntu-1        NotReady   <none>    6m        v1.10.1
ubuntu-2        NotReady   <none>    6m        v1.10.1
ubuntu-3        NotReady   <none>    6m        v1.10.1
ubuntu-master   NotReady   master    10m       v1.10.1
```

查看 节点加入完毕后pod的状态：
```shell
$ sudo kubectl get pod -n kube-system -o wide
```

返回

```shell
NAME                                    READY     STATUS    RESTARTS   AGE       IP              NODE
etcd-ubuntu-master                      1/1       Running   0          21m       192.168.0.200   ubuntu-master
kube-apiserver-ubuntu-master            1/1       Running   0          21m       192.168.0.200   ubuntu-master
kube-controller-manager-ubuntu-master   1/1       Running   0          22m       192.168.0.200   ubuntu-master
kube-dns-86f4d74b45-wkfk2               0/3       Pending   0          22m       <none>          <none>
kube-proxy-6ddb4                        1/1       Running   0          22m       192.168.0.200   ubuntu-master
kube-proxy-7ngb9                        1/1       Running   0          17m       192.168.0.202   ubuntu-2
kube-proxy-fkhhx                        1/1       Running   0          18m       192.168.0.201   ubuntu-1
kube-proxy-rh4lq                        1/1       Running   0          18m       192.168.0.203   ubuntu-3
kube-scheduler-ubuntu-master            1/1       Running   0          21m       192.168.0.200   ubuntu-master
```

kubedns组件需要在网络插件完成安装以后会自动安装完成。


## 安装网络插件canal

从<a href="https://docs.projectcalico.org/v3.0/getting-started/kubernetes/installation/hosted/canal/">canal官方文档</a>参考，如下网址下载2个文件并且安装，其中一个是配置canal的RBAC权限，一个是部署canal的DaemonSet。

```shell
$ kubectl apply -f  https://docs.projectcalico.org/v3.0/getting-started/kubernetes/installation/hosted/canal/rbac.yaml
```

返回

```shell
clusterrole.rbac.authorization.k8s.io "calico" created
clusterrole.rbac.authorization.k8s.io "flannel" created
clusterrolebinding.rbac.authorization.k8s.io "canal-flannel" created
clusterrolebinding.rbac.authorization.k8s.io "canal-calico" created
```

```shell
kubectl apply -f https://docs.projectcalico.org/v3.0/getting-started/kubernetes/installation/hosted/canal/canal.yaml
```

返回

```shell
configmap "canal-config" created
daemonset.extensions "canal" created
customresourcedefinition.apiextensions.k8s.io "felixconfigurations.crd.projectcalico.org" created
customresourcedefinition.apiextensions.k8s.io "bgpconfigurations.crd.projectcalico.org" created
customresourcedefinition.apiextensions.k8s.io "ippools.crd.projectcalico.org" created
customresourcedefinition.apiextensions.k8s.io "clusterinformations.crd.projectcalico.org" created
customresourcedefinition.apiextensions.k8s.io "globalnetworkpolicies.crd.projectcalico.org" created
customresourcedefinition.apiextensions.k8s.io "networkpolicies.crd.projectcalico.org" created
serviceaccount "canal" created
```

查看canal的安装状态。

```shell
kubectl get pod -n kube-system -o wide
```

返回 

```shell
NAME                                    READY     STATUS    RESTARTS   AGE       IP              NODE
canal-fc94k                             3/3       Running   10         4m        192.168.0.201   ubuntu-1
canal-rs2wp                             3/3       Running   10         4m        192.168.0.200   ubuntu-master
canal-tqd4l                             3/3       Running   10         4m        192.168.0.202   ubuntu-2
canal-vmpnr                             3/3       Running   10         4m        192.168.0.203   ubuntu-3
etcd-ubuntu-master                      1/1       Running   0          28m       192.168.0.200   ubuntu-master
kube-apiserver-ubuntu-master            1/1       Running   0          28m       192.168.0.200   ubuntu-master
kube-controller-manager-ubuntu-master   1/1       Running   0          29m       192.168.0.200   ubuntu-master
kube-dns-86f4d74b45-wkfk2               3/3       Running   0          28m       10.244.2.2      ubuntu-3
kube-proxy-6ddb4                        1/1       Running   0          28m       192.168.0.200   ubuntu-master
kube-proxy-7ngb9                        1/1       Running   0          24m       192.168.0.202   ubuntu-2
kube-proxy-fkhhx                        1/1       Running   0          24m       192.168.0.201   ubuntu-1
kube-proxy-rh4lq                        1/1       Running   0          24m       192.168.0.203   ubuntu-3
kube-scheduler-ubuntu-master            1/1       Running   0          28m       192.168.0.200   ubuntu-master
```

可以看到canal和kube-dns都已经运行正常，一个基本功能正常的测试环境就部署完毕了。

让master也运行pod（默认master不运行pod）,这样在测试环境做是可以的，不建议在生产环境如此操作。

```shell
$ kubectl taint nodes --all node-role.kubernetes.io/master-
```

返回 

```shell
node "ubuntu-master" untainted
taint "node-role.kubernetes.io/master:" not found
taint "node-role.kubernetes.io/master:" not found
taint "node-role.kubernetes.io/master:" not found
```