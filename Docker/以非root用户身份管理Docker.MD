# 以非root用户身份管理Docker

Docker守护程序绑定到Unix套接字而不是TCP端口。默认情况下，Unix套接字由用户拥有root，其他用户只能使用它sudo。Docker守护程序始终以root用户身份运行。

如果您不想在docker命令前加上sudo，请创建一个名为的Unix组docker并向其添加用户。当Docker守护程序启动时，它会创建一个可由该docker组成员访问的Unix套接字。

### 要创建docker组并添加您的用户：

> 1.创建docker组。

```shell
$ sudo groupadd docker
```

> 2.将您的用户添加到该docker组。
```shell
$ sudo usermod -aG docker $USER
```

> 3. 注销并重新登录，以便重新评估您的组成员身份。

如果在虚拟机上进行测试，则可能需要重新启动虚拟机才能使更改生效。

在桌面Linux环境（如X Windows）上，完全注销会话，然后重新登录。

> 4.验证您是否可以运行docker命令sudo。

```shell
$ docker run hello-world
```

此命令下载测试映像并在容器中运行它。当容器运行时，它会打印一条信息性消息并退出。

如果sudo在将用户添加到docker组之前最初使用Docker CLI命令，则可能会看到以下错误，表明您的~/.docker/目录由于sudo命令而创建的权限不正确。

```shell
WARNING: Error loading config file: /home/user/.docker/config.json -
stat /home/user/.docker/config.json: permission denied
```

要解决此问题，请删除~/.docker/目录（它会自动重新创建，但任何自定义设置都会丢失），或使用以下命令更改其所有权和权限：

```shell
$ sudo chown "$USER":"$USER" /home/"$USER"/.docker -R
$ sudo chmod g+rwx "$HOME/.docker" -R
```