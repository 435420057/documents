# kubernetes1.18+ubuntu20安装与配置

在以前安装k8s1.14的时候感觉有很多不好解决的问题在本次安装中得到了解决，比如：镜像文件摘取问题，由于国内不能访问 k8s.gcr.io 镜像节点，所以以前只有通过阿里的节点将镜像拉取后，然后通过docker tag方法标记，本次不用这么麻烦了，直接修改镜像地址就行了。现在开始安装


# 安装前环境检测与高速

* 1、   禁用交换分区

验证交换分区有没有关闭 
```sh
free -m
```

注释/etc/fstab关于swap的配置

```sh
# /dev/mapper/VoiGroup-lv_swap swap swap defaults 0 0
```

执行如下命令

```sh
echo vm.swappiness=0 >> /etc/sysctl.conf
```

重启
```sh
reboot
```

验证(Swap行均为0)

```sh
free -m
```

```sh
Swap  0   0  0
```

* 2、   防火墙
可以全部禁用
```sh
systemctl stop firewalld
systemctl disable firewalld
```
如果各个主机启用了防火墙，需要开放Kubernetes各个组件所需要的端口。如下图所示，详细信息请看<a href="https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/#check-required-ports">官网</a>。


## 安装docker 

访问<a href="https://docs.docker.com/engine/install/">官网</a>获取其它平台安装方法

* 1、   卸载老版本

```sh
$ sudo apt-get remove docker docker-engine docker.io containerd runc
```

* 2、   设置docker安装信息
```sh
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
```
* 3、   增加docker  GPG key


```sh
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
```

然后

```sh
$ sudo apt-key fingerprint 0EBFCD88
```

* 3、   开始安装
```sh
$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"


$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io
```

等待安装完成

## 安装kubernetes

访问<a href="https://kubernetes.io/zh/docs/setup/production-environment/tools/kubeadm/install-kubeadm/">官网</a>

使用官方提供的方式安装，这个方式需要网络能访问到google

```sh
sudo apt-get update && sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl
```

如果不能访问google那么可以使用阿里的节点来安装

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

主机这个里面的IP是你的节点的主机IP

## 配置kubernetes

导出kubeadm集群部署自定义文件

```sh
kubeadm config print init-defaults > init.default.yaml 
```

修改自定义配置文件

修改如下问题

1、主节点IP——advertiseAddress

2、国内阿里镜像地址imageRepository——registry.aliyuncs.com/google_containers

3、pod网段配置——不同网络插件网段不一样详细见官网

```yaml
apiVersion: kubeadm.k8s.io/v1beta2
bootstrapTokens:
- groups:
  - system:bootstrappers:kubeadm:default-node-token
  token: abcdef.0123456789abcdef
  ttl: 24h0m0s
  usages:
  - signing
  - authentication
kind: InitConfiguration
localAPIEndpoint:
  advertiseAddress: 192.168.31.21  # 主节点IP地址
  bindPort: 6443
nodeRegistration:
  criSocket: /var/run/dockershim.sock
  name: k8s-m   # 节点名称，需要host访问
  taints:
  - effect: NoSchedule
    key: node-role.kubernetes.io/master
---
apiServer:
  timeoutForControlPlane: 4m0s
apiVersion: kubeadm.k8s.io/v1beta2
certificatesDir: /etc/kubernetes/pki
clusterName: kubernetes
controllerManager: {}
dns:
  type: CoreDNS
etcd:
  local:
    dataDir: /var/lib/etcd
imageRepository: registry.aliyuncs.com/google_containers # 将镜像地址替换成阿里的
kind: ClusterConfiguration
kubernetesVersion: v1.18.0
networking:
  dnsDomain: cluster.local
  podSubnet: 172.10.0.0/16 # 子网
  serviceSubnet: 10.96.0.0/12
scheduler: {}

```

拉取阿里云kubernetes容器镜像

```sh
kubeadm config images list --config init.default.yaml

kubeadm config images pull --config init.default.yaml
```

Master安装部署

```sh
kubeadm init --config=init.default.yaml
```

提示成功后，记住里面的加入节点信息。类似于下面这样:
```sh
kubeadm join 192.168.18.130:6443 --token 13wq39.15xn5oo26op6mp6r \
    --discovery-token-ca-cert-hash sha256:542a3f2a932bf7474ca77a8faf0ffc72370aef490a6129ba30f36a0aff415f8
```



配置用户证书

```sh
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config
```

注意如果你的不root帐户，也面home目录下改复制一下:

```sh
sudo cp -i /etc/kubernetes/admin.conf $HOME/admin.conf
sudo chown $(id -u):$(id -g) $HOME/admin.conf
```

查看集群状态

```sh
kubectl get node
```

这时查看节点信息，会显示NotReady原因是没有安装pod网络。

## 安装pod网络——Calico

不同pod网络对pod网段要求不一样见官网。我这边安装的是Calico。

```sh
kubectl apply -f https://docs.projectcalico.org/v3.11/manifests/calico.yaml
```
显示成功

## 添加Node节点

刚才记录下的加入节点链接在工作节点上依次执行
```sh
kubeadm join --token <token> <control-plane-host>:<control-plane-port> --discovery-token-ca-cert-hash sha256:<hash>
```

查看token

```sh
kubeadm token list
```

token24小时过期

```sh
# 重新创建 token
kubeadm token create

# 创建discovery-token-ca-cert-hash
openssl x509 -pubkey -in /etc/kubernetes/pki/ca.crt | openssl rsa -pubin -outform der 2>/dev/null | \
   openssl dgst -sha256 -hex | sed 's/^.* //'
```


## 重置


```sh
kubeadm reset
```
