# etcd生产环境实践

## 生产环境搭建etcd

以搭建3节点高可用ETCD集群为例，分别在三台主机上初始化ETCD1,ETCD2,ETCD3作为机器IP地址。

```bash
$ ETCD1=http://10.0.0.1
$ ETCD2=http://10.0.0.2
$ ETCD2=http://10.0.0.3

# 节点1为4核4GB的机器
$ etcd --name etcd1 \
--initial-advertise-peer-urls $ETCD1:2380  \
--listen-peer-urls $ETCD1:2380  \
--listen-client-urls $ETCD1:2379,http://127.0.0.1:2379 \
--advertise-client-urls $ETCD1:2379  \
--initial-cluster-token etcd-cluster \
--initial-cluster etcd1=$ETCD1:2380,etcd2=$ETCD2:2380,etcd3=$ETCD3:2380 \
--auto-compaction-retention=1 \
--quota-backend-bytes=$((4*1024*1024*1024)) \
--initial-cluster-state new

# 节点2为4核4GB的机器
$ etcd  --name etcd2 \
--initial-advertise-peer-urls $ETCD2:2380 \
--listen-peer-urls $ETCD2:2380 \
--listen-client-urls $ETCD2:2379,http://127.0.0.1:2379 \
--advertise-client-urls $ETCD2:2379 \
--initial-cluster-token etcd-cluster \
--initial-cluster etcd1=$ETCD1:2380,etcd2=$ETCD2:2380,etcd3=$ETCD3:2380 \
--auto-compaction-retention=1 \
--quota-backend-bytes=$((4*1024*1024*1024)) \
--initial-cluster-state new

# 节点3为4核4GB的机器
$ etcd  --name etcd3 \
--initial-advertise-peer-urls $ETCD3:2380 \
--listen-peer-urls $ETCD3:2380 \
--listen-client-urls $ETCD3:2379,http://127.0.0.1:2379 \
--advertise-client-urls $ETCD3:2379 \
--initial-cluster-token etcd-cluster \
--initial-cluster etcd1=$ETCD1:2380,etcd2=$ETCD2:2380,etcd3=$ETCD3:2380 \
--auto-compaction-retention=1 \
--quota-backend-bytes=$((4*1024*1024*1024)) \
--initial-cluster-state new

# 检查集群是否正确启动
$ etcdctl --endpoints=$ETCD1:2379 member list
400938f1eaf1d0ed: name=etcd3 peerURLs=http://10.0.0.1:2380 clientURLs=http://10.0.0.1:2379 isLeader=false
bd0ff97b33ac1165: name=etcd2 peerURLs=http://10.0.0.2:2380 clientURLs=http://10.0.0.2:2379 isLeader=false
be62429afec4445f: name=etcd1 peerURLs=http://10.0.0.3:2380 clientURLs=http://10.0.0.3:2379 isLeader=true
```

历史记录压缩

如果将etcd用作服务发现，每次服务注册和更新都可以看做一条新数据，日积月累，这些数据的量会导致etcd占用内存越来越大，直到etcd到达空间配额限制的时候，etcd的写入将会被静止，影响线上服务，定期删除历史记录就是避免这种情况。

# 只保留一个小时的历史数据

```bash
$ etcd --auto-compaction-retention=1
```
磁盘去碎片化

内部碎片是指空闲状态的，能被后端使用但是仍然消耗存储空间的磁盘空间。去碎片化实际上是将存储空间还给文件系统。

```bash
$ etcdctl defrag
```
空间配额

空间配额用来保障集群可靠地进行操作。如果没有限制配额，当键空间变大之后，直到用光了磁盘空间，它就会影响etcd集群的表现。当任意节点超出空间配额， 那么它将进入维护状态，只接受读/删操作。只有释放了足够空间、去碎片化了后端数据库并且清理了空间配额之后，集群才能继续正常操作。

默认限制是2GB，可以通过--quota-backend-bytes配置，最高上限为8GB。

```bash
# 显式配置配额为16MB
$ etcd --quota-backend-bytes=$((16*1024*1024))
$ ETCDCTL_API=3 etcdctl --write-out=table endpoint status
```
如果遇到空间配额不足的情况，那么需要对etcd进行操作。

```bash
# 获取当前版本号
$ rev=$(ETCDCTL_API=3 etcdctl --endpoint=:2379 endpoint status --write-out="json" | egrep -o '"revision":[0-9]*' | egrep -o '[0-9]*'）
# 压缩所有旧版本
$ ETCDCTL_API=3 etcdctl compact $rev
# 去碎片化
$ ETCDCTL_API=3 etcdctl defrag
# 取消警报
$ ETCDCTL_API=3 etcdctl alarm disarm
# 测试通过
$ ETCDCTL_API=3 etcdctl put newkey 123
```
快照备份

```bash
$ etcdctl snapshot save backup.db
$ etcdctl --write-out=table snapshot status backup.db
#+----------+----------+------------+------------+
#|   HASH   | REVISION | TOTAL KEYS | TOTAL SIZE |
#+----------+----------+------------+------------+
#| fe01cf57 |       10 |          7 | 2.1 MB     |
#+----------+----------+------------+------------+、
```

调优

跨数据中心时，需要调整心跳间隔和选举超时时间 默认的心跳时间是100ms，建议为round trip时间 默认选举超时是1000ms

> 详细配置 : https://coreos.com/etcd/docs/latest/op-guide/configuration.html
> 运维 ： https://github.com/coreos/etcd/blob/master/Documentation/op-guide/maintenance.md