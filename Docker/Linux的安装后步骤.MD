# Linux的安装后步骤

本节包含配置Linux主机以使用Docker更好地工作的可选过程。

## 以非root用户身份管理Docker

Docker守护程序绑定到Unix套接字而不是TCP端口。默认情况下，Unix套接字由用户拥有root，其他用户只能使用它sudo。Docker守护程序始终以root用户身份运行。

如果您不想在docker命令前加上sudo，请创建一个名为的Unix组docker并向其添加用户。当Docker守护程序启动时，它会创建一个可由该docker组成员访问的Unix套接字。

> 警告

该docker组授予与root 用户等效的权限。有关如何影响系统安全性的详细信息，请参阅 Docker Daemon Attack Surface。

### 要创建docker组并添加您的用户：

> 1. 创建docker组。

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
$ sudo chmod g+rwx "$HOME/.docker" -R配置Docker以在启动时启动
```

## 配置Docker以在启动时启动

大多数当前的Linux发行版（RHEL，CentOS，Fedora，Ubuntu 16.04及更高版本）用于systemd管理系统启动时启动的服务。Ubuntu 14.10及以下使用upstart。

### systemd

```shell
$ sudo systemctl enable docker
```

### 要禁用此行为，请disable改用。

```shell
$ sudo systemctl disable docker
```

如果需要添加HTTP代理，为Docker运行时文件设置不同的目录或分区，或进行其他自定义，请参阅 自定义systemd Docker守护程序选项。

### upstart

Docker自动配置为在启动时启动 upstart。要禁用此行为，请使用以下命令：
```shell
$ echo manual | sudo tee /etc/init/docker.override
```

### chkconfig

```shell
$ sudo chkconfig docker on
```

## 使用其他存储引擎

有关不同存储引擎的信息，请参阅 <a href="https://docs.docker.com/storage/storagedriver/">存储驱动程序</a>。默认存储引擎和支持的存储引擎列表取决于主机的Linux发行版和可用的内核驱动程序。

### 配置Docker守护程序侦听连接的位置

默认情况下，Docker守护程序侦听UNIX套接字上的连接以接受来自本地客户端的请求。通过将Docker配置为侦听IP地址和端口以及UNIX套接字，可以允许Docker接受来自远程主机的请求。有关此配置选项的更多详细信息，请参阅<a href="https://docs.docker.com/engine/reference/commandline/dockerd/">Docker CLI</a>参考文章中的“将<a href="https://docs.docker.com/engine/reference/commandline/dockerd/">Docker</a>绑定到另一个主机/端口或unix套接字”部分。

> Docker EE客户

> Docker EE客户可以使用UCP客户端捆绑包获得对UCP的远程CLI访问。UCP客户端捆绑包由UCP生成，并由相互TLS保护。有关更多信息，请参阅有关UCP的CLI访问的文档 。

> 保护您的连接

> 在配置Docker以接受来自远程主机的连接之前，了解打开docker到网络的安全隐患至关重要。如果不采取步骤来保护连接，则远程非root用户可以在主机上获得root访问权限。有关如何使用TLS证书保护此连接的更多信息，请查看有关 <a href="https://docs.docker.com/engine/security/https/">如何保护Docker守护程序套接字的文章</a>。

配置Docker以接受远程连接可以docker.service使用systemd的Linux发行版的systemd单元文件来完成，例如RedHat，CentOS，Ubuntu和SLES的最新版本，或者daemon.json推荐用于不使用systemd的Linux发行版的文件。

> systemd vs daemon.json

> 配置Docker以使用systemd单元文件和daemon.json 文件来侦听连接会导致冲突，从而阻止Docker启动。

### 使用systemd单元文件配置远程访问
1. 使用该命令在文本编辑器中sudo systemctl edit docker.service打开覆盖文件docker.service。

2. 添加或修改以下行，替换您自己的值。

```shell
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// -H tcp://127.0.0.1:2375
```

3. 保存文件。

4. 重新加载systemctl配置。
```shell
 $ sudo systemctl daemon-reload
```

5. 重启Docker。
```shell
$ sudo systemctl restart docker.service
```

6. 通过查看netstat确认的输出是否dockerd正在侦听已配置的端口来检查更改是否得到遵守。
```shell
$ sudo netstat -lntp | grep dockerd
tcp        0      0 127.0.0.1:2375          0.0.0.0:*               LISTEN      3758/dockerd
```
### 使用配置远程访问 daemon.json
1. 设置hosts数组/etc/docker/daemon.json以连接到UNIX套接字和IP地址，如下所示：
```shell
{
"hosts": ["unix:///var/run/docker.sock", "tcp://127.0.0.1:2375"]
}
```

2. 重启Docker。

3. 通过查看netstat确认的输出是否dockerd正在侦听已配置的端口来检查更改是否得到遵守。
```shell
$ sudo netstat -lntp | grep dockerd
tcp        0      0 127.0.0.1:2375          0.0.0.0:*               LISTEN      3758/dockerd
```

### 在Docker守护程序上启用IPv6

要在Docker守护程序上启用IPv6，请参阅 启用IPv6支持。

## 故障排除
### 内核兼容性
如果您的内核早于3.10版本或者缺少某些模块，则Docker无法正常运行。要检查内核兼容性，可以下载并运行该check-config.sh 脚本。
```shell
$ curl https://raw.githubusercontent.com/docker/docker/master/contrib/check-config.sh > check-config.sh

$ bash ./check-config.sh
```

该脚本仅适用于Linux，而不适用于macOS。

>  Cannot connect to the Docker daemon

如果您看到如下所示的错误，则可能将Docker客户端配置为连接到其他主机上的Docker守护程序，并且该主机可能无法访问。


> Cannot connect to the Docker daemon. Is 'docker daemon' running on this host?

要查看客户端配置为连接到哪个主机，请检查DOCKER_HOST环境中变量的值。

```shell
$ env | grep DOCKER_HOST
```

如果此命令返回值，则Docker客户端将设置为连接到在该主机上运行的Docker守护程序。如果未设置，则Docker客户端将设置为连接到本地主机上运行的Docker守护程序。如果设置错误，请使用以下命令取消设置：
```shell
$ unset DOCKER_HOST
```

您可能需要在诸如~/.bashrc或之外的 文件中编辑环境，~/.profile以防止DOCKER_HOST错误地设置变量。

如果DOCKER_HOST按预期设置，请验证Docker守护程序是否正在远程主机上运行，​​并且防火墙或网络中断不会阻止您进行连接。

## IP转发问题
如果使用手动配置你的网络systemd-network有systemd 219或更高版本，Docker容器可能无法访问您的网络。从systemd版本220 开始，给定网络（net.ipv4.conf.<interface>.forwarding）的转发设置默认为关闭。此设置可防止IP转发。它还与Docker net.ipv4.conf.all.forwarding在容器中启用设置的行为相冲突。

要在RHEL，CentOS或Fedora上解决此问题，请<interface>.network 在/usr/lib/systemd/network/Docker主机上编辑该文件（例如：）/usr/lib/systemd/network/80-container-host0.network并在该[Network]部分中添加以下块。
```shell
[Network]
...
IPForward=kernel
# OR
IPForward=true
...
```
此配置允许按预期从容器进行IP转发。

> DNS resolver found in resolv.conf and containers can't use it

使用GUI的Linux系统通常运行网络管理器，该网络管理器使用dnsmasq在环回地址上运行的 实例，例如127.0.0.1或 127.0.1.1缓存DNS请求，并将此条目添加到 /etc/resolv.conf。该dnsmasq服务可加速DNS查询并提供DHCP服务。此配置不拥有自己的网络命名空间的码头工人容器内工作，因为多克尔容器解决回环地址，如127.0.0.1对 自身，这是很不可能的运行在自己的回送地址的DNS服务器。

如果Docker检测到没有引用的DNS服务器/etc/resolv.conf是功能完备的DNS服务器，则会出现以下警告，并且Docker使用Google提供的公共DNS服务器8.8.8.8并8.8.4.4进行DNS解析。
```shell
WARNING: Local (127.0.0.1) DNS resolver found in resolv.conf and containers
can't use it. Using default external servers : [8.8.8.8 8.8.4.4]
```
如果您看到此警告，请先检查您是否使用dnsmasq：
```shell
$ ps aux |grep dnsmasq
```
如果您的容器需要解析网络内部的主机，则公共名称服务器不够用。你有两个选择：

* 您可以为Docker指定要使用的DNS服务器，或
* 您可以dnsmasq在NetworkManager中禁用。如果您这样做，NetworkManager会添加您真正的DNS名称服务器/etc/resolv.conf，但您将失去可能的好处dnsmasq。

您只需要使用这些方法之一。

## 为Docker指定DNS服务器
配置文件的默认位置是/etc/docker/daemon.json。您可以使用--config-file 守护程序标志更改配置文件的位置。以下文档假定配置文件位于/etc/docker/daemon.json。

1. 创建或编辑Docker守护程序配置文件，该/etc/docker/daemon.json文件默认为 file，它控制Docker守护程序配置。
```shell
$ sudo nano /etc/docker/daemon.json
```
2. 添加dns一个或多个IP地址作为值的密钥。如果文件包含现有内容，则只需添加或编辑该dns行。
```shell
{
	"dns": ["8.8.8.8", "8.8.4.4"]
}
```

如果您的内部DNS服务器无法解析公共IP地址，请至少包含一个DNS服务器，以便您可以连接到Docker Hub，以便您的容器可以解析Internet域名。

保存并关闭文件。

3. 重新启动Docker守护程序。
```shell
$ sudo service docker restart
```
4. 通过尝试提取图像来验证Docker是否可以解析外部IP地址：
```shell
$ docker pull hello-world
```

5.如有必要，请验证Docker容器是否可以通过ping它来解析内部主机名。
```shell
$ docker run --rm -it alpine ping -c4 <my_internal_host>

PING google.com (192.168.1.2): 56 data bytes
64 bytes from 192.168.1.2: seq=0 ttl=41 time=7.597 ms
64 bytes from 192.168.1.2: seq=1 ttl=41 time=7.635 ms
64 bytes from 192.168.1.2: seq=2 ttl=41 time=7.660 ms
64 bytes from 192.168.1.2: seq=3 ttl=41 time=7.677 ms
```

### 禁用 DNSMASQ
Ubuntu的

如果您不想更改Docker守护程序的配置以使用特定的IP地址，请按照以下说明dnsmasq在NetworkManager中禁用。

1. 编辑/etc/NetworkManager/NetworkManager.conf文件。

2. 通过dns=dnsmasq在行#的开头添加一个字符来注释掉该行。
```shell
# dns=dnsmasq
```
保存并关闭文件。

3. 重新启动NetworkManager和Docker。作为替代方案，您可以重新启动系统。
```shell
$ sudo restart network-manager
$ sudo restart docker
```
### RHEL，CentOS或Fedora
要dnsmasq在RHEL，CentOS或Fedora上禁用：

1. 禁用该dnsmasq服务：
```shell
$ sudo service dnsmasq stop

$ sudo systemctl disable dnsmasq
```
2. 使用Red Hat文档手动配置DNS服务器 。

## 允许通过防火墙访问远程API
如果您在运行Docker的同一主机上运行防火墙并且想要从另一台主机访问Docker Remote API并启用远程访问，则需要将防火墙配置为允许Docker端口上的传入连接，默认为2376if启用TLS加密传输或2375 以其他方式启用。

两个常见的防火墙守护程序是 UFW（简单防火墙）（通常用于Ubuntu系统）和firewalld（通常用于基于RPM的系统）。请参阅操作系统和防火墙的文档，但以下信息可能有助于您入门。这些选项相当宽松，您可能希望使用不同的配置来更好地锁定系统。

* UFW：DEFAULT_FORWARD_POLICY="ACCEPT"在您的配置中设置。

* firewalld：向策略添加类似于以下内容的规则（一个用于传入请求，另一个用于传出请求）。确保接口名称和链名称正确。
```xml
<direct>
  [ <rule ipv="ipv6" table="filter" chain="FORWARD_direct" priority="0"> -i zt0 -j ACCEPT </rule> ]
  [ <rule ipv="ipv6" table="filter" chain="FORWARD_direct" priority="0"> -o zt0 -j ACCEPT </rule> ]
</direct>
```

> Your kernel does not support cgroup swap limit capabilities

在Ubuntu或Debian主机上，使用图像时，您可能会看到类似于以下内容的消息。
```shell
WARNING: Your kernel does not support swap limit capabilities. Limitation discarded.
```

在基于RPM的系统上不会发生此警告，这些系统默认启用这些功能。

如果您不需要这些功能，则可以忽略该警告。您可以按照这些说明在Ubuntu或Debian上启用这些功能。即使Docker未运行，内存和交换计费也会占总可用内存的1％左右，总体性能降低10％。

1. 以具有sudo权限的用户身份登录Ubuntu或Debian主机。

2. 编辑/etc/default/grub文件。添加或编辑该GRUB_CMDLINE_LINUX行以添加以下两个键值对：
```shell
GRUB_CMDLINE_LINUX="cgroup_enable=memory swapaccount=1"
```
保存并关闭文件。

3. 更新GRUB。

$ sudo update-grub

如果GRUB配置文件的语法不正确，则会发生错误。在这种情况下，请重复步骤2和3。

重新启动系统后，更改将生效。