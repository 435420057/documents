# 为docker配置固定ip

docker默认使用bridge模式，通过网桥连接到宿主机，而容器内部的ip则从网桥所在的ip段取未用的ip。这样做一个不方便的地方在于容器内部的ip不是固定的，想要连接容器时只能通过映射到宿主机的端口，因而有很多项目使用overlay来为docker提供网络的配置，比如Pipework、Flannel、Kubernetes、Weave、opencontrail等。

想要使用overlay来为docker配置网络，需要首先了解下docker的网络模式：

- --net=bridge — The default action, that connects the container to the Docker bridge as described above.
- --net=host — Tells Docker to skip placing the container inside of a separate network stack. In essence, this choice tells Docker to not containerize the container's networking! While container processes will still be confined to their own filesystem and process list and resource limits, a quick ip addr command will show you that, network-wise, they live “outside” in the main Docker host and have full access to its network interfaces. Note that this doesnot let the container reconfigure the host network stack — that would require --privileged=true — but it does let container processes open low-numbered ports like any other root process. It also allows the container to access local network services like D-bus. This can lead to processes in the container being able to do unexpected things like restart your computer. You should use this option with caution.
- --net=container:NAME_or_ID — Tells Docker to put this container's processes inside of the network stack that has already been created inside of another container. The new container's processes will be confined to their own filesystem and process list and resource limits, but will share the same IP address and port numbers as the first container, and processes on the two containers will be able to connect to each other over the loopback interface.
- --net=none — Tells Docker to put the container inside of its own network stack but not to take any steps to configure its network, leaving you free to build any of the custom configurations explored in the last few sections of this document.

上面这几种方式只有--net=none才可以为docker分配固定ip，来看看如何操作。
首先，配置一个用于创建container interface的网桥，可以使用ovs，也可以使用Linux bridge，以Linux bridge为例：
```bash
br_name=docker
brctl addbr $br_name
ip addr add 192.168.33.2/24 dev $br_name
ip addr del 192.168.33.2/24 dev em1
ip link set $br_name up
brctl addif $br_name eth0
```


接着，可以启动容器了，注意用--net=none方式启动：
```bash
# start new container
hostname='docker.test.com'
cid=$(docker run -d -i -h $hostname --net=none -t centos)
pid=$(docker inspect -f '{{.State.Pid}}' $cid)
```


下面，为该容器配置网络namespace，并设置固定ip：
```bash
# set up netns
mkdir -p /var/run/netns
ln -s /proc/$pid/ns/net /var/run/netns/$pid
# set up bridge
ip link add q$pid type veth peer name r$pid
brctl addif $br_name q$pid
ip link set q$pid up
# set up docker interface
fixed_ip='192.168.33.3/24'
gateway='192.168.33.1'
ip link set r$pid netns $pid
ip netns exec $pid ip link set dev r$pid name eth0
ip netns exec $pid ip link set eth0 up
ip netns exec $pid ip addr add $fixed_ip dev eth0
ip netns exec $pid ip route add default via 192.168.33.1
```


这样，容器的网络就配置好了，如果容器内部开启了sshd服务，通过192.168.33.3就可以直接ssh连接到容器，非常方便。上面的步骤比较长，可以借助pipework来为容器设置固定ip（除了设置IP，还封装了配置网关、macvlan、vlan、dhcp等功能）：
```bash
pipework docker0 be8365e3b2834 10.88.88.8/24
```

那么，当容器需要删除的时候，怎么清理网络呢，其实也很简单：
```bash
# stop and delete container
docker stop $cid
docker rm $cid
# delete docker's net namespace (also delete veth pair)
ip netns delete $pid
```


更多docker网络的配置，可以参考官方手册