# mongodb阿里云ubuntu16.04安装

## 1. 添加mongodb签名到APT
```bash
$ sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv EA312927
```

## 2.创建/etc/apt/sources.list.d/mongodb-org-3.4.list文件并写入命令
```bash
deb [ arch=amd64,arm64 ] http://mirrors.aliyun.com/mongodb/apt/ubuntu xenial/mongodb-org/3.4 multiverse

```

## 3.更新软件源列表
```bash
$ sudo apt-get update
```

## 4.安装mongodb（默认是安装稳定版）
```bash
$ sudo apt-get install -y mongodb-org
```

或者安装指定版本：

```bash
$ sudo apt-get install -y mongodb-org=3.2.9 mongodb-org-server=3.2.9 mongodb-org-shell=3.2.9 mongodb-org-mongos=3.2.9 mongodb-org-tools=3.2.9
```