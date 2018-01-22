# Supervisor: 进程控制配置说明

```bash
]# cat /etc/supervisord.conf
; Sample supervisor config file.
;
; For more information on the config file, please see:
; http://supervisord.org/configuration.html
;
; Notes:
;  - Shell expansion ("~" or "$HOME") is not supported.  Environment
;    variables can be expanded using this syntax: "%(ENV_HOME)s".
;  - Comments must have a leading space: "a=b ;comment" not "a=b;comment".

[unix_http_server]
file=/tmp/supervisor.sock   ; (socket 文件的路径)
;chmod=0700                 ; socket 文件权限 (default 0700)
;chown=nobody:nogroup       ; socket 文件属主:属组
;username=user              ; (启动http的用户 (open server))
;password=123               ; (默认的密码 (open server))

;[inet_http_server]         ; 默认禁用tcp监听的http 服务
;port=127.0.0.1:9001        ; (指定监听在本机ip地址和端口)
;username=user              ; (默认启动http服务的用户)
;password=123               ; (默认的密码)

[supervisord]
logfile=/tmp/supervisord.log ; (主日志文件的存放位置，默认在程序的工作启动目录)
logfile_maxbytes=50MB        ; (主日志文件的最大值，之后进行切割;默认 50MB)
logfile_backups=10           ; (主日志文件备份的数目;默认 10)
loglevel=info                ; (日志级别;默认是info; 其它: debug,warn,trace)
pidfile=/tmp/supervisord.pid ; (supervisord 运行时的pidfile路径;默认 supervisord.pid)
nodaemon=false               ; (如果为true，程序就以前台运行;默认是 false)
minfds=1024                  ; (min. 启动有效的文件描述符数目;默认 1024)
minprocs=200                 ; (min. 有效进程描述符;默认 200)
;umask=022                   ; (进程文件创建的默认权限;默认 022)
;user=chrism                 ; (默认是当前启动的用户)
;identifier=supervisor       ; (supervisord 标识符, 默认是'supervisor')
;directory=/tmp              ; (默认启动时间不会切换)
;nocleanup=true              ; (在启动时不清理临时文件；默认值为false)
;childlogdir=/tmp            ; ('AUTO' 子进程日志目录, 默认 $TEMP)
;environment=KEY="value"     ; (增加一个环境变量键值对：key=”value“)
;strip_ansi=false            ; (在log日志里去掉ansi转义编码; 默认是 false)

; 下面的部分选项必须保留在RPC的配置文件中
; (supervisorctl/web 接口) 使用以下配置来管理
; added by defining them in separate rpcinterface: sections
[rpcinterface:supervisor]
supervisor.rpcinterface_factory = supervisor.rpcinterface:make_main_rpcinterface

[supervisorctl]
serverurl=unix:///tmp/supervisor.sock ; use a unix:// URL  for a unix socket
;serverurl=http://127.0.0.1:9001 ; use an http:// url to specify an inet socket
;username=chris              ; should be same as http_username if set
;password=123                ; should be same as http_password if set
;prompt=mysupervisor         ; cmd line prompt (default "supervisor")
;history_file=~/.sc_history  ; use readline history if available

; 以下是被管理的示例程序显示所有可能用到的配置。
; 创建一个或“多个”程序: 要遵循以下的键值对规则。
; supervisor.

;[program:theprogramname]
;command=/bin/cat              ; 程序的启动命令 (使用绝对路径)
;process_name=%(program_name)s ; process_name 表示 (默认是 %(program_name)s)
;numprocs=1                    ; 启动时的进程数 (默认 1)
;directory=/tmp                ; 执行时切换到的目录 (def no cwd)
;umask=022                     ; umask for process (default None)
;priority=999                  ; 相对启动优先级(default 999)
;autostart=true                ; 是否跟随supervisord程序启动该监控程序 (default: true)
;startsecs=1                   ; # 在设定时间内，程序必须保持运行 (def. 1)
;startretries=3                ; 当启动失败时尝试的最大次数(default 3)
;autorestart=unexpected        ; 如果退出后，什么状态退出的去重启，默认非意外的(def: unexpected)
;exitcodes=0,2                 ; 'expected' 符合退出代码之后去重启 (default 0,2)
;stopsignal=QUIT               ; 用于杀死进程的信号 (default TERM)
;stopwaitsecs=10               ; 最大等待秒数 SIGKILL (default 10)
;stopasgroup=false             ; 发送停止信号到Unix进程组 (default false)
;killasgroup=false             ; SIGKILL UNIX进程组 (def false)
;user=chrism                   ; setuid to this UNIX account to run the program
;redirect_stderr=true          ; 是否开启程序标准错误输出的重定向 (default false)
;stdout_logfile=/a/path        ; 标准输出路径; default AUTO
;stdout_logfile_maxbytes=1MB   ; 文件最大大小 # 日志文件进行切割 (default 50MB)
;stdout_logfile_backups=10     ; # 日志文件备份数目 (default 10)
;stdout_capture_maxbytes=1MB   ; ‘捕获模式’中的字节数 (default 0)
;stdout_events_enabled=false   ; 在标准输出写入文件时发出事件 (default false)
;stderr_logfile=/a/path        ; 标准错误输出, NONE for none; default AUTO
;stderr_logfile_maxbytes=1MB   ; 文件最大大小 # logfile bytes b4 rotation (default 50MB)
;stderr_logfile_backups=10     ; # of stderr logfile backups (default 10)
;stderr_capture_maxbytes=1MB   ; number of bytes in 'capturemode' (default 0)
;stderr_events_enabled=false   ; emit events on stderr writes (default false)
;environment=A="1",B="2"       ; 添加进程环境变量 (def no adds)
;serverurl=AUTO                ; 覆盖serverurl计算 (childutils)

;下面是event事件部分所有可能设置的值，大部分同上面一样。
; eventlistener subsection values, create one or more 'real'
; eventlistener: sections to be able to handle event notifications
; sent by supervisor.

;[eventlistener:theeventlistenername]
;command=/bin/eventlistener    ; the program (relative uses PATH, can take args)
;process_name=%(program_name)s ; process_name expr (default %(program_name)s)
;numprocs=1                    ; number of processes copies to start (def 1)
;events=EVENT                  ; event notif. types to subscribe to (req'd)
;buffer_size=10                ; event buffer queue size (default 10)
;directory=/tmp                ; directory to cwd to before exec (def no cwd)
;umask=022                     ; umask for process (default None)
;priority=-1                   ; the relative start priority (default -1)
;autostart=true                ; start at supervisord start (default: true)
;startsecs=1                   ; # of secs prog must stay up to be running (def. 1)
;startretries=3                ; max # of serial start failures when starting (default 3)
;autorestart=unexpected        ; autorestart if exited after running (def: unexpected)
;exitcodes=0,2                 ; 'expected' exit codes used with autorestart (default 0,2)
;stopsignal=QUIT               ; signal used to kill process (default TERM)
;stopwaitsecs=10               ; max num secs to wait b4 SIGKILL (default 10)
;stopasgroup=false             ; send stop signal to the UNIX process group (default false)
;killasgroup=false             ; SIGKILL the UNIX process group (def false)
;user=chrism                   ; setuid to this UNIX account to run the program
;redirect_stderr=false         ; redirect_stderr=true is not allowed for eventlisteners
;stdout_logfile=/a/path        ; stdout log path, NONE for none; default AUTO
;stdout_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stdout_logfile_backups=10     ; # of stdout logfile backups (default 10)
;stdout_events_enabled=false   ; emit events on stdout writes (default false)
;stderr_logfile=/a/path        ; stderr log path, NONE for none; default AUTO
;stderr_logfile_maxbytes=1MB   ; max # logfile bytes b4 rotation (default 50MB)
;stderr_logfile_backups=10     ; # of stderr logfile backups (default 10)
;stderr_events_enabled=false   ; emit events on stderr writes (default false)
;environment=A="1",B="2"       ; process environment additions
;serverurl=AUTO                ; override serverurl computation (childutils)

; The below sample group section shows all possible group values,
; create one or more 'real' group: sections to create "heterogeneous"
; process groups.

;[group:thegroupname]
;programs=progname1,progname2  ; 这里的progname1，progname2就是定义的监控管理程序的名字，如[program:x]这里就是x
;priority=999                  ; the relative start priority (default 999)

;  下面的 [include] 选项只能包含一个files 设置，功能是定义supervisor管理程序的配置文件，可以单独的移除去，和主配置文件分开，方便。
; setting can list multiple files (separated by whitespace or
; newlines).  It can also contain wildcards.  The filenames are
; interpreted as relative to this file.  Included files *cannot*
; include files themselves.

;[include]
;files = relative/directory/*.ini         ；定义管理监控程序的配置文件的路径
```
