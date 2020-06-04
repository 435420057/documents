# 在Ubuntu20中在shell下修改系统网络代理

## 1、通过脚本方式临时修改网络代理

可以在系统中增加一下可执行文件，如下

```sh
sudo nano  /etc/profile.d/proxy.sh
```

输入内容:

```sh
# set proxy config via profie.d - should apply for all users
# 
export http_proxy="http://10.10.1.10:8080/"
export https_proxy="http://10.10.1.10:8080/"
export ftp_proxy="http://10.10.1.10:8080/"
export no_proxy="127.0.0.1,localhost"

# For curl
export HTTP_PROXY="http://10.10.1.10:8080/"
export HTTPS_PROXY="http://10.10.1.10:8080/"
export FTP_PROXY="http://10.10.1.10:8080/"
export NO_PROXY="127.0.0.1,localhost"
```

为他增加可执行权限

```sh
sudo chmod +x  /etc/profile.d/proxy.sh
```

使用命令让他生效

```sh
$ source /etc/profile.d/proxy.sh
```

## 2、通过修改系统配置来完成APT代理

需要在apt目录中增加代理设置

```sh
sudo nano /etc/apt/apt.conf.d/80proxy
```

输入内容
```sh
Acquire::http::proxy "http://10.10.1.10:8080/";
Acquire::https::proxy "https://10.10.1.10:8080/";
Acquire::ftp::proxy "ftp://10.10.1.10:8080/";
```

内容里面的http地址是你的网络代理软件的地址

如果有密码，可以这样设置
```sh
Acquire::http::proxy "http://<username>:<password>@<proxy>:<port>/";
Acquire::https::proxy "https://<username>:<password>@<proxy>:<port>/";
Acquire::ftp::proxy "ftp://<username>:<password>@<proxy>:<port>/";
```

## 3、设置代理仅支持wget

```sh
vim ~/.wgetrc      
```
输入内容
```sh
use_proxy = on
http_proxy = http://10.10.1.10:8080/ 
https_proxy = http://10.10.1.10:8080/ 
ftp_proxy = http://10.10.1.10:8080/ 
```