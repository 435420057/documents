# 利用docker和docker-compose部署单机kafka

前提

1、docker

2、docker-compose

其中docker-compose不是必须的,单单使用docker也是可以的,这里主要介绍docker和docker-compose两种方式

## docker部署

docker部署kafka非常简单，只需要两条命令即可完成kafka服务器的部署。

```sh
docker run -d --name zookeeper -p 2181:2181  wurstmeister/zookeeper
docker run -d --name kafka -p 9092:9092 -e KAFKA_BROKER_ID=0 -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 --link zookeeper -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://192.168.1.60(机器IP):9092 -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092 -t wurstmeister/kafka
```

由于kafka是需要和zookeeper共同工作的,所以需要部署一个zookeeper,但有了docker这对部署来说非常轻松.

可以通过docker ps查看到两个容器的状态,这里不再展示.

接下来可以进行生产者和消费者的尝试

## 通过kafka自带工具生产消费消息测试

1.首先,进入到kafka的docker容器中
```sh
docker exec -it kafka sh
```

2.运行消费者,进行消息的监听

```sh
kafka-console-consumer.sh --bootstrap-server 192.168.1.60:9094 --topic kafeidou --from-beginning
```

3.打开一个新的ssh窗口,同样进入kafka的容器中,执行下面这条命令生产消息
```sh
kafka-console-producer.sh --broker-list 192.168.1.60(机器IP):9092 --topic kafeidou
```

输入完这条命令后会进入到控制台，可以输入任何想发送的消息,这里发送一个hello

```sh
>>
>hello
>
>
>
```

可以看到,在生产者的控制台中输入消息后,消费者的控制台立刻看到了消息

到目前为止,一个kafka完整的hello world就完成了.kafka的部署加上生产者消费者测试.

## 通过docker-compose部署kafka

首先创建一个docker-compose.yml文件

```yml
version: '3.7'
services:
  zookeeper:
    image: wurstmeister/zookeeper
    volumes:
      - ./data:/data
    ports:
      - 2182:2181
       
  kafka9094:
    image: wurstmeister/kafka
    ports:
      - 9092:9092
    environment:
      KAFKA_BROKER_ID: 0
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://192.168.1.60:9092
      KAFKA_CREATE_TOPICS: "kafeidou:2:0"   #kafka启动后初始化一个有2个partition(分区)0个副本名叫kafeidou的topic 
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092
    volumes:
      - ./kafka-logs:/kafka
    depends_on:
      - zookeeper
```
部署起来很简单,在docker-compose.yml文件的目录下执行docker-compose up -d就可以了,测试方式和上面的一样。

这个docker-compose做的东西比上面docker方式部署的东西要多一些

1. 数据持久化，在当前目录下挂在了两个目录分别存储zookeeper和kafka的数据,当然在docker run 命令中添加 -v 选项也是可以做到这样的效果的

2. kafka在启动后会初始化一个有分区的topic,同样的,docker run的时候添加 -e KAFKA_CREATE_TOPICS=kafeidou:2:0 也是可以做到的。


## 总结:优先推荐docker-compose方式部署

为什么呢?

因为单纯使用docker方式部署的话，如果有改动(例如:修改对外开放的端口号)的情况下,docker需要把容器停止docker stop 容器ID/容器NAME,然后删除容器docker rm 容器ID/容器NAME,最后启动新效果的容器docker run ...

而如果在docker-compose部署的情况下如果修改内容只需要修改docker-compose.yml文件对应的地方,例如2181:2181改成2182:2182,然后再次在docker-compose.yml文件对应的目录下执行docker-compose up -d就能达到更新后的效果。