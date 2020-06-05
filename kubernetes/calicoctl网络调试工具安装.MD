# calicoctl网络调试工具安装
calicoctl allows you to create, read, update, and delete Calico objects from the command line.

## Installing calicoctl as a binary on a single host
> 1. Log into the host, open a terminal prompt, and navigate to the location where you want to install the binary.

*** Tip: Consider navigating to a location that’s in your PATH. For example, /usr/local/bin/. ***

> 2.Use the following command to download the calicoctl binary.

```sh
curl -O -L  https://github.com/projectcalico/calicoctl/releases/download/v3.14.1/calicoctl

```
> 3. Set the file to be executable.
```sh
chmod +x calicoctl

```
*** Note: If the location of calicoctl is not already in your PATH, move the file to one that is or add its location to your PATH. This will allow you to invoke it without having to prepend its location. ***

### Next step:

Configure calicoctl to connect to your datastore.

## Installing calicoctl as a container on a single host
To install calicoctl as a container on a single host, log into the target host and issue the following command.
```sh
docker pull calico/ctl:v3.14.1
```

### Next step:

Configure calicoctl to connect to your datastore.

## Installing calicoctl as a Kubernetes pod
Use the YAML that matches your datastore type to deploy the calicoctl container to your nodes.

* etcd
```sh
 kubectl apply -f https://docs.projectcalico.org/manifests/calicoctl-etcd.yaml
```
*** Note: You can also view the YAML in a new tab.***

* Kubernetes API datastore
```sh
 kubectl apply -f https://docs.projectcalico.org/manifests/calicoctl.yaml
```
*** Note: You can also view the YAML in a new tab.***

You can then run commands using kubectl as shown below.
```sh
kubectl exec -ti -n kube-system calicoctl -- /calicoctl get profiles -o wide
```
An example response follows.
```sh
NAME                 TAGS
kns.default          kns.default
kns.kube-system      kns.kube-system
We recommend setting an alias as follows.
```
```sh
alias calicoctl="kubectl exec -i -n kube-system calicoctl -- /calicoctl"
```
*** Note: In order to use the calicoctl alias when reading manifests, redirect the file into stdin, for example:***
```sh
calicoctl create -f - < my_manifest.yaml
```