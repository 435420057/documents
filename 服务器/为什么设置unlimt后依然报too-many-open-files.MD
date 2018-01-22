# 为什么设置unlimt后依然报too many open files

> 1. 现象

服务的cpu跑满(golang实现), 并大量报too many open files错误.服务使用Supervisor来运行,部署在阿里ecs上.

> 2.分析
从日志来看,cpu的上升主要为到达文件数限制引起的,但之前已经更改过系统的文件数及所有用户的文件数,按道理是不应该出现这个问题的,后来查阅资料发现,文件数可以从三个维度限制分别为操作系统限制,用户态限制,以及进程限制,对于这三个维度选取最小值生效.于是对系统进行分析.

首先查看当前打开文件数, 进程占用的文件数并不多.

```bash
lsof -n|awk '{print $2}'|sort|uniq -c|sort -nr|more
```

然后获取系统级文件数限制

```bash
// 入命令

cat /etc/sysctl.conf

// 得到

// fs.file-max = 1000000
```

查询用户级文件数限制

```bash
cat /etc/security/limits.conf

// 得到

// * soft nofile 655350

// * hard nofile 655350

```

单独获取程序文件数限制(9908为进程id)

```bash
cat /proc/9908/limits 

// 得到

// Max open files            1024                 4096                 files     

```

如上可以发现, 虽然系统及用户的文件数调大了,但是程序的仍然是一个很低的值, 这里进程的文件数一般情况下是默认继承用户级的值的,而这里却没有继承.
修改Supervisor配置：
```bash
[supervisord]
logfile=/tmp/supervisord.log ; (主日志文件的存放位置，默认在程序的工作启动目录)
logfile_maxbytes=50MB        ; (主日志文件的最大值，之后进行切割;默认 50MB)
logfile_backups=10           ; (主日志文件备份的数目;默认 10)
loglevel=info                ; (日志级别;默认是info; 其它: debug,warn,trace)
pidfile=/tmp/supervisord.pid ; (supervisord 运行时的pidfile路径;默认 supervisord.pid)
nodaemon=false               ; (如果为true，程序就以前台运行;默认是 false)

; 增加以下配置
minfds=65535                  ; (min. 启动有效的文件描述符数目;默认 1024)
minprocs=200                 ; (min. 有效进程描述符;默认 200)
;umask=022                   ; (进程文件创建的默认权限;默认 022)
;user=chrism                 ; (默认是当前启动的用户)
;identifier=supervisor       ; (supervisord 标识符, 默认是'supervisor')
;directory=/tmp              ; (默认启动时间不会切换)
;nocleanup=true              ; (在启动时不清理临时文件；默认值为false)
;childlogdir=/tmp            ; ('AUTO' 子进程日志目录, 默认 $TEMP)
;environment=KEY="value"     ; (增加一个环境变量键值对：key=”value“)
;strip_ansi=false            ; (在log日志里去掉ansi转义编码; 默认是 false)
```


单独获取程序(9908为进程id)

```bash
cat /proc/9928/limits 

// 得到

Max open files            65535                 65535                 files  

```

