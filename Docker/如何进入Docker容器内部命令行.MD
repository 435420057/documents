# 如何进入Docker容器内部命令行

需要进入docker容器中进行操作有两种方法：

## docker attach

docker attach命令可以attach到一个已经运行的容器的stdin，然后进行命令执行的动作。但是需要注意的是，如果在这里输入exit，会导致容器的停止。

```sh
Usage:  docker attach [OPTIONS] CONTAINER

Attach local standard input, output, and error streams to a running container

Options:
      --detach-keys string   Override the key sequence for detaching a
                             container
      --no-stdin             Do not attach STDIN
      --sig-proxy            Proxy all received signals to the process
                             (default true)
```


## docker exec

```sh
Usage:  docker exec [OPTIONS] CONTAINER COMMAND [ARG...]

Run a command in a running container

Options:
  -d, --detach               Detached mode: run command in the background
      --detach-keys string   Override the key sequence for detaching a
                             container
  -e, --env list             Set environment variables
  -i, --interactive          Keep STDIN open even if not attached
      --privileged           Give extended privileges to the command
  -t, --tty                  Allocate a pseudo-TTY
  -u, --user string          Username or UID (format:
                             <name|uid>[:<group|gid>])
  -w, --workdir string       Working directory inside the container
```

可以看到，其中参数-i -t -d与docker run有些相同。其实，exec会进入创建一个伪终端，与直接run创建出来的相似。但是不同点在于，不会因为输入exit而终止容器。
这种方式类似于ssh！进入容器内进行操作

通常用法： docker exec -it 容器ID /bin/bash


## 总结

docker run -it IMAGES_NAME会创建前台进程，但是会在输入exit后终止进程。

docker attach DOCKER_ID 会通过连接stdin，连接到容器内输入输出流，会在输入exit后终止进程.

docker exec -it DOCKER_ID /bin/bash 会连接到容器，可以像SSH一样进入容器内部，进行操作，可以通过exit退出容器，不影响容器运行。

以上几种方式均可通过输入Ctrl+P+Q把前台容器放入后台运行，不终止进程。
