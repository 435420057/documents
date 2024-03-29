# 查看linux中的TCP连接数

## 一、查看哪些IP连接本机

```shell
netstat -an
```

## 二、查看TCP连接数

> 1)统计80端口连接数

```shell
netstat -nat|grep -i "80"|wc -l
```

> 2）统计httpd协议连接数

```shell
ps -ef|grep httpd|wc -l
```

> 3）、统计已连接上的，状态为“established

```shell
netstat -na|grep ESTABLISHED|wc -l
```

> 4)、查出哪个IP地址连接最多,将其封了.

```shell
netstat -na|grep ESTABLISHED|awk {print $5}|awk -F: {print $1}|sort|uniq -c|sort -r +0n

netstat -na|grep SYN|awk {print $5}|awk -F: {print $1}|sort|uniq -c|sort -r +0n

```

---------------------------------------------------------------------------------------------

> 1、查看apache当前并发访问数：
```shell
netstat -an | grep ESTABLISHED | wc -l
```

对比httpd.conf中MaxClients的数字差距多少。

> 2、查看有多少个进程数：

```shell
ps aux|grep httpd|wc -l
```

> 3、可以使用如下参数查看数据

```shell
server-status?auto

#ps -ef|grep httpd|wc -l

1388
```
统计httpd进程数，连个请求会启动一个进程，使用于Apache服务器。

表示Apache能够处理1388个并发请求，这个值Apache可根据负载情况自动调整。

```shell
#netstat -nat|grep -i "80"|wc -l
4341

```
netstat -an会打印系统当前网络链接状态，而grep -i "80"是用来提取与80端口有关的连接的，wc -l进行连接数统计。

最终返回的数字就是当前所有80端口的请求总数。

```shell
#netstat -na|grep ESTABLISHED|wc -l
376
```

netstat -an会打印系统当前网络链接状态，而grep ESTABLISHED 提取出已建立连接的信息。 然后wc -l统计。

最终返回的数字就是当前所有80端口的已建立连接的总数。

``shell
netstat -nat||grep ESTABLISHED|wc - 可查看所有建立连接的详细记录

```

查看Apache的并发请求数及其TCP连接状态：

　　Linux命令：
```shell
netstat -n | awk '/^tcp/ {++S[$NF]} END {for(a in S) print a, S[a]}'

（

netstat -n | awk ‘/^tcp/ {++S[$NF]} END {for(a in S) print a, S[a]}’
```

TIME_WAIT 8947 等待足够的时间以确保远程TCP接收到连接中断请求的确认

FIN_WAIT1 15 等待远程TCP连接中断请求，或先前的连接中断请求的确认

FIN_WAIT2 1 从远程TCP等待连接中断请求

ESTABLISHED 55 代表一个打开的连接

SYN_RECV 21 再收到和发送一个连接请求后等待对方对连接请求的确认

CLOSING 2 没有任何连接状态

LAST_ACK 4 等待原来的发向远程TCP的连接中断请求的确认


TCP连接状态详解 

LISTEN： 侦听来自远方的TCP端口的连接请求

SYN-SENT： 再发送连接请求后等待匹配的连接请求

SYN-RECEIVED：再收到和发送一个连接请求后等待对方对连接请求的确认

ESTABLISHED： 代表一个打开的连接

FIN-WAIT-1： 等待远程TCP连接中断请求，或先前的连接中断请求的确认

FIN-WAIT-2： 从远程TCP等待连接中断请求

CLOSE-WAIT： 等待从本地用户发来的连接中断请求

CLOSING： 等待远程TCP对连接中断的确认

LAST-ACK： 等待原来的发向远程TCP的连接中断请求的确认

TIME-WAIT： 等待足够的时间以确保远程TCP接收到连接中断请求的确认

CLOSED： 没有任何连接状态

）

　　LAST_ACK 5

　　SYN_RECV 30

　　ESTABLISHED 1597

　　FIN_WAIT1 51

　　FIN_WAIT2 504

　　TIME_WAIT 1057

　　其中的


SYN_RECV表示正在等待处理的请求数；

ESTABLISHED表示正常数据传输状态；

TIME_WAIT表示处理完毕，等待超时结束的请求数。


---------------------------------------------------------------------------------------------

查看Apache并发请求数及其TCP连接状态

查看httpd进程数（即prefork模式下Apache能够处理的并发请求数）：
　
Linux命令：

```shell
ps -ef | grep httpd | wc -l
```

　返回结果示例：

　　1388

　　表示Apache能够处理1388个并发请求，这个值Apache可根据负载情况自动调整，我这组服务器中每台的峰值曾达到过2002。

查看Apache的并发请求数及其TCP连接状态：

　　Linux命令：
```shell
netstat -n | awk '/^tcp/ {++S[$NF]} END {for(a in S) print a, S[a]}'
```
返回结果示例：
　
　LAST_ACK 5

　　SYN_RECV 30

　　ESTABLISHED 1597

　　FIN_WAIT1 51

　　FIN_WAIT2 504

　　TIME_WAIT 1057

　　其中的SYN_RECV表示正在等待处理的请求数；ESTABLISHED表示正常数据传输状态；
TIME_WAIT表示处理完毕，等待超时结束的请求数。

　　状态：描述


　　CLOSED：无连接是活动 的或正在进行

　　LISTEN：服务器在等待进入呼叫

　　SYN_RECV：一个连接请求已经到达，等待确认

　　SYN_SENT：应用已经开始，打开一个连接

　　ESTABLISHED：正常数据传输状态

　　FIN_WAIT1：应用说它已经完成

　　FIN_WAIT2：另一边已同意释放

　　ITMED_WAIT：等待所有分组死掉

　　CLOSING：两边同时尝试关闭

　　TIME_WAIT：另一边已初始化一个释放

　　LAST_ACK：等待所有分组死掉




如发现系统存在大量TIME_WAIT状态的连接，通过调整内核参数解决，

```shell
vim /etc/sysctl.conf
```
编辑文件，加入以下内容：
```shell
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_tw_recycle = 1
net.ipv4.tcp_fin_timeout = 30
```

然后执行 /sbin/sysctl -p 让参数生效。

net.ipv4.tcp_syncookies = 1 表示开启SYN cookies。当出现SYN等待队列溢出时，启用
cookies来处理，可防范少量SYN攻击，默认为0，表示关闭；

net.ipv4.tcp_tw_reuse = 1 表示开启重用。允许将TIME-WAIT sockets重新用于新的TCP连接，默认为0，表示关闭；

net.ipv4.tcp_tw_recycle = 1 表示开启TCP连接中TIME-WAIT sockets的快速回收，默认为0，表示关闭。

net.ipv4.tcp_fin_timeout 修改系統默认的 TIMEOUT 时间

下面附上TIME_WAIT状态的意义：

客户端与服务器端建立TCP/IP连接后关闭SOCKET后，服务器端连接的端口

状态为TIME_WAIT

是不是所有执行主动关闭的socket都会进入TIME_WAIT状态呢？

有没有什么情况使主动关闭的socket直接进入CLOSED状态呢？

主动关闭的一方在发送最后一个 ack 后

就会进入 TIME_WAIT 状态 停留2MSL（max segment lifetime）时间

这个是TCP/IP必不可少的，也就是“解决”不了的。

也就是TCP/IP设计者本来是这么设计的

主要有两个原因

1。防止上一次连接中的包，迷路后重新出现，影响新连接
（经过2MSL，上一次连接中所有的重复包都会消失）

2。可靠的关闭TCP连接

在主动关闭方发送的最后一个 ack(fin) ，有可能丢失，这时被动方会重新发
fin, 如果这时主动方处于 CLOSED 状态 ，就会响应 rst 而不是 ack。所以
主动方要处于 TIME_WAIT 状态，而不能是 CLOSED 。

TIME_WAIT 并不会占用很大资源的，除非受到攻击。

还有，如果一方 send 或 recv 超时，就会直接进入 CLOSED 状态


如何合理设置apache httpd的最大连接数？

手头有一个网站在线人数增多，访问时很慢。初步认为是服务器资源不足了，但经反复测试，一旦连接上，不断点击同一个页面上不同的链接，都能迅速打开，这种现象就是说明apache最大连接数已经满了，新的访客只能排队等待有空闲的链接，而如果一旦连接上，在keeyalive 的存活时间内（KeepAliveTimeout，默认5秒）都不用重新打开连接，因此解决的方法就是加大apache的最大连接数。

1.在哪里设置？

apache 2.24，使用默认配置（FreeBSD 默认不加载自定义MPM配置），默认最大连接数是250

在/usr/local/etc/apache22/httpd.conf中加载MPM配置（去掉前面的注释）：

# Server-pool management (MPM specific)
Include etc/apache22/extra/httpd-mpm.conf

可见的MPM配置在/usr/local/etc/apache22/extra/httpd-mpm.conf，但里面根据httpd的工作模式分了很多块，哪一部才是当前httpd的工作模式呢？可通过执行 apachectl -l 来查看：
```shell
Compiled in modules:
              core.c
              prefork.c
              http_core.c
              mod_so.c
```

看到prefork 字眼，因此可见当前httpd应该是工作在prefork模式，prefork模式的默认配置是：
```shell

<IfModule mpm_prefork_module>
                StartServers                      5
                MinSpareServers                   5
                MaxSpareServers                  10
                MaxClients                      150
                MaxRequestsPerChild               0
</IfModule>
```

2.要加到多少？

连接数理论上当然是支持越大越好，但要在服务器的能力范围内，这跟服务器的CPU、内存、带宽等都有关系。

查看当前的连接数可以用：
```shell
ps aux | grep httpd | wc -l
```
或：
```shell
pgrep httpd|wc -l
```
计算httpd占用内存的平均数:

```shell
ps aux|grep -v grep|awk '/httpd/{sum+=$6;n++};END{print sum/n}'
```

由于基本都是静态页面，CPU消耗很低，每进程占用内存也不算多，大约200K。

服务器内存有2G，除去常规启动的服务大约需要500M（保守估计），还剩1.5G可用，那么理论上可以支持1.5*1024*1024*1024/200000 = 8053.06368

约8K个进程，支持2W人同时访问应该是没有问题的（能保证其中8K的人访问很快，其他的可能需要等待1、2秒才能连上，而一旦连上就会很流畅）

控制最大连接数的MaxClients ，因此可以尝试配置为：
```shell
<IfModule mpm_prefork_module>
                StartServers                      5
                MinSpareServers                   5
                MaxSpareServers                  10
                ServerLimit                    5500
                MaxClients                     5000
                MaxRequestsPerChild               100
</IfModule>
```

注意，MaxClients默认最大为250，若要超过这个值就要显式设置ServerLimit，且ServerLimit要放在MaxClients之前，值要不小于MaxClients，不然重启httpd时会有提示。

重启httpd后，通过反复执行pgrep httpd|wc -l 来观察连接数，可以看到连接数在达到MaxClients的设值后不再增加，但此时访问网站也很流畅，那就不用贪心再设置更高的值了，不然以后如果网站访问突增不小心就会耗光服务器内存，可根据以后访问压力趋势及内存的占用变化再逐渐调整，直到找到一个最优的设置值。

(MaxRequestsPerChild不能设置为0，可能会因内存泄露导致服务器崩溃）

更佳最大值计算的公式：
```shell
apache_max_process_with_good_perfermance < (total_hardware_memory / apache_memory_per_process ) * 2
apache_max_process = apache_max_process_with_good_perfermance * 1.5

```
附：

实时检测HTTPD连接数：
```shell
watch -n 1 -d "pgrep httpd|wc -l"
```

--------------------- 
作者：西部风情 
来源：CSDN 
原文：https://blog.csdn.net/he_jian1/article/details/40787269 
版权声明：本文为博主原创文章，转载请附上博文链接！