# docker参数--restart=always的作用

创建容器时没有添加参数  --restart=always ，导致的后果是：当 Docker 重启时，容器未能自动启动。

现在要添加该参数怎么办呢，方法有二：

## Docker 命令修改

```sh
docker container update --restart=always 容器名字
```

操作实例如下：
```sh
[root@localhost mnt]# docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED              STATUS              PORTS               NAMES
46cdfc60b7a6        nginx               "nginx -g 'daemon ..."   About a minute ago   Up 42 seconds       80/tcp              n3
79d55a734c26        nginx               "nginx -g 'daemon ..."   About a minute ago   Up 42 seconds       80/tcp              n2
f7b2206c019d        nginx               "nginx -g 'daemon ..."   About a minute ago   Up 46 seconds       80/tcp              n1
[root@localhost mnt]# docker container update --restart=always n1
n1
[root@localhost mnt]# systemctl restart docker 
[root@localhost mnt]# docker ps -a
CONTAINER ID        IMAGE               COMMAND                  CREATED             STATUS                     PORTS               NAMES
46cdfc60b7a6        nginx               "nginx -g 'daemon ..."   2 minutes ago       Exited (0) 5 seconds ago                       n3
79d55a734c26        nginx               "nginx -g 'daemon ..."   2 minutes ago       Exited (0) 5 seconds ago                       n2
f7b2206c019d        nginx               "nginx -g 'daemon ..."   2 minutes ago       Up 2 seconds               80/tcp              n1
````


## 直接改配置文件（经测试后无效，修改配置文件后，启动容器后，该参数有自动变成了no，修改不生效）

首先停止容器，不然无法修改配置文件

配置文件路径为：/var/lib/docker/containers/容器ID

在该目录下找到一个文件 hostconfig.json ，找到该文件中关键字 RestartPolicy

修改前配置："RestartPolicy":{"Name":"no","MaximumRetryCount":0}

修改后配置："RestartPolicy":{"Name":"always","MaximumRetryCount":0}

最后启动容器。