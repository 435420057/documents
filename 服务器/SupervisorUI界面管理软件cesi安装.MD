# SupervisorUI界面管理软件cesi安装

## 第一步，安装相关依赖工具

```bash
$sudo apt-get update
$sudo apt-get install sqlite3 python python-flask
```

## 第二步，下载cesi源码并导入数据库

```bash
$git clone https://github.com/Gamegos/cesi
$cd cesi
$sqlite3 path/to/userinfo.db < userinfo.sql
```

## 第三步，配置

```bash
$cp cesi.conf /etc/cesi.conf
```

参考配置：
```ini
[node:web]
username = admin
password = admin
host = 192.168.18.154
port = 27654



[cesi]
database = /data/cesi/userinfo.db
activity_log = /data/cesi/cesi.log
host = 172.16.90.47
```

# 第4步，安装uwsgi 并配置

```bash
$sudo apt-get install -y uwsgi
$sudo apt-get install uwsgi-plugin-python
```

为cesi配置运行文件 ,在cesi目录创建一个文件 app_config.ini，内容如下:

```ini
[uwsgi]
chdir = /data/cesi/cesi/ 
module = web
callable = app
socket = /data/cesi/uwsgi.sock
master = true
processes = 4
thread = 2
memory-report = true
manage-script-name = true
```

注意目录地址一定要写正确。

现在运行:

```bash
$sudo uwsgi -d /data/cesi/uwsgi.log --ini /data/cesi/app_config.ini --plugins=python
```
# 第5步，nginx安装与端口设置

```bash
$sudo apt-get install -y nginx
```

增加配置如下:

```ini
server {
        listen 27657;
        listen [::]:27657;

        root /data/web;
        # Add index.php to the list if you are using PHP
        index index.html index.htm index.nginx-debian.html;

        server_name _;

        location / {
		include uwsgi_params; 
		uwsgi_pass unix:///data/cesi/uwsgi.sock; 
        }
}
```

重启nginx , 查看端口27657