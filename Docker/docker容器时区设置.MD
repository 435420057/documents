# docker容器时区设置

## 在Dockerfile中设置镜像时区

   ```sh
   ENV TZ=Asia/Shanghai
   RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
   ```
## 创建容器时设置时区

将在主机与容器的时间进行挂载
```sh
-v /etc/localtime:/etc/localtime
```
## 进入容器进行设置

``sh
1) 进入容器内: docker exec 容器名 /bin/bash
2) 设定时区
rm /etc/localtime
ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```