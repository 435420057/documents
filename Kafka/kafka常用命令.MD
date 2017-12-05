# kafka 常用命令

* 1. 如何linux下让kafka后台运行
```bash
   ./kafka-server-start.sh -daemon /data/kafka/config/server.properties
```

* 2. 如何查看所有通道信息
```bash
    ./kafka-topics.sh --describe --zookeeper 172.16.90.33:2181,172.16.90.34:2181
```