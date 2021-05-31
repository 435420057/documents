# Docker容器和宿主机之间拷贝文件

从容器里面拷文件到宿主机

在宿主机里面执行以下命令

```sh
docker cp 容器ID:要拷贝的文件在容器里面的路径       要拷贝到宿主机的相应路径 
```

假设容器ID为 608198df85fa（可以通过docker ps查看容器ID），要从容器里面拷贝的文件路为：/app/test.js,  现在要将test.js从容器里面拷到宿主机的/opt路径下面

```sh
docker cp 608198df85fa:/app/test.js /opt
```
从宿主机拷文件到容器里面

在宿主机里面执行如下命令

docker cp 要拷贝的文件路径 容器ID:要拷贝到容器里面对应的路径

```sh
docker cp /opt/test.js 608198df85fa:/app/
```