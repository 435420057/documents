# Apache + PHP7 + MongoDB + Redis 环境配置笔记

环境

OS：Ubuntu 16.04 LTS / Ubuntu 14.04 Server

准备工作

在安装所有的软件之前，先对apt的源进行更新：

```bash
sudo apt-get update
```

安装Apache2

通过apt来安装Apache2：
```bash
sudo apt-get install apache2
```

安装PHP7

对于Ubuntu 16，同样是通过apt来安装：
```bash
sudo apt-get install php7.0 libapache2-mod-php7.0
```

注意：低版本中的apt源中没有PHP7，所以对于其他版本的使用下面的方法：

- 首先，安装python-software-properties和software-properties-common：

- sudo apt-get install python-software-properties
```bash
sudo apt-get install software-properties-common
```


安装了python-software-properties和software-properties-common之后，我们就可以使用add-apt-repository来添加ppa到当前的库中，并自动导入公钥了。

- 紧接着，就可以通过下面的命令来安装PHP7.0了1：

- sudo add-apt-repository ppa:ondrej/php-7.0
```bash
sudo apt-get update
sudo apt-get install php7.0 libapache2-mod-php7.0
```


- 若出现无法读取，请尝试2：

- sudo apt-get install -y language-pack-en-base
```bash
sudo LC_ALL=en_US.UTF-8 add-apt-repository ppa:ondrej/php

sudo apt-get update
sudo apt-get install php7.0 libapache2-mod-php7.0
```

如果需要安装其他附加包的情况，可以使用apt-get install ***来安装，***替换成你需要安装的包3：
```bash
    $ sudo apt-cache search php7-* 

    php7.0-common - Common files for packages built from the PHP source
    libapache2-mod-php7.0 - server-side, HTML-embedded scripting language (Apache 2 module)
    php7.0-cgi - server-side, HTML-embedded scripting language (CGI binary)
    php7.0-cli - command-line interpreter for the PHP scripting language
    php7.0-phpdbg - server-side, HTML-embedded scripting language (PHPDBG binary)
    php7.0-fpm - server-side, HTML-embedded scripting language (FPM-CGI binary)
    libphp7.0-embed - HTML-embedded scripting language (Embedded SAPI library)
    php7.0-dev - Files for PHP7.0 module development
    php7.0-dbg - Debug symbols for PHP7.0
    php7.0-curl - CURL module for PHP
    php7.0-gd - GD module for PHP
    php7.0-imap - IMAP module for PHP
    php7.0-intl - Internationalisation module for PHP
    php7.0-ldap - LDAP module for PHP
    php7.0-pgsql - PostgreSQL module for PHP
    php7.0-pspell - pspell module for PHP
    php7.0-recode - recode module for PHP
    php7.0-snmp - SNMP module for PHP
    php7.0-tidy - tidy module for PHP
    php7.0-json - JSON module for PHP
    php-all-dev - package depending on all supported PHP development packages
    php7.0-sybase - Sybase module for PHP
    php7.0-modules-source - PHP 7.0 modules source package
    php7.0-sqlite3 - SQLite3 module for PHP
    php7.0-mysql - MySQL module for PHP
    php7.0-opcache - Zend OpCache module for PHP
```

安装MongoDB并向PHP中添加扩展

通过apt来安装MongoDB：
```bash
sudo apt-get install mongodb
```

安装MongoDB驱动

使用apt安装MongoDB之后，还需要安装PHP的MongoDB驱动。首先需要安装pecl：
```bash
sudo apt-get install php-pear

```
然后通过pecl安装MongoDB的扩展：
```bash
sudo pecl install mongodb
```

注意：如果出现了phpize: not found，就需要安装php-dev，然后再执行上面的操作：
```bash
sudo apt-get install php-dev
```

注意：如果出现了configure: error: Cannot find OpenSSL's libraries，请尝试安装以下包4：
```bash
sudo apt-get install libcurl4-openssl-dev
sudo apt-get install -y autoconf g++ make openssl libssl-dev libcurl4-openssl-dev
sudo apt-get install -y libcurl4-openssl-dev pkg-config
sudo apt-get install -y libsasl2-dev
```
然后，向php.ini添加extension=mongodb.so。
重新启动Apache2：
```bash
service restart apache2
```

如果不知道php.ini处于哪个位置，可以使用phpinfo()，搜索页面中的php.ini得到：
这个就是需要修改的php.ini的所在位置。
```bash
Configuration File (php.ini) Path  |  /etc/php/7.0/apache2
```

确认是否安装成功

使用phpinfo()：
```php
$ echo "<?php phpinfo(); ?>" > /var/www/html/phpinfo.php
```

访问http://yourdomain/phpinfo.php，若页面中存在：

mongodb

mongodb support

enabled

mongodb version

1.1.6

mongodb stability

stable

libmongoc version

1.3.5

libbson version

1.3.5

则说明安装成功。

配置MongoDB

MongoDB的配置文件处于/etc/mongodb.conf，我们可以编辑它，具体参数设置可以参考MongoDB Docs > Reference > Configuration File Options5;

安装Redis并向PHP中添加扩展

通过apt安装redis：
```bash
sudo apt-get install redis-server
```

安装Redis驱动

开始在 PHP 中使用 Redis 前， 我们需要确保已经安装了 redis 服务及 PHP redis 驱动，且你的机器上能正常使用 PHP。 接下来让我们安装 PHP redis 驱动：
Redis的驱动没有在pecl中提供，所以我们需要手动编译安装。
下载地址为:https://github.com/phpredis/phpredis/releases。

文章编写时，最新版本是2.2.7，所以我们以2.2.7为例：

在用户文件中，创建temp文件夹，保存我们安装过程中使用的文件：
```bash
mkdir ~/temp
cd ~/temp

wget https://github.com/phpredis/phpredis/archive/php7.zip

unzip php7.zip

cd phpredis-php7
```

紧接着使用编译安装：
```bash
phpize
./configure

make & make install
```

安装完成之后，最后一行显示的是安装路径：
```bash
Installing shared extensions:    /usr/lib/php/20151012/
```

然后，向php.ini添加：
```bash
extension_dir = "/usr/lib/php/20151012"
extension = redis.so
```

这里的extension_dir要与安装之后输出的路径一致。

确认是否安装成功

重启Apache2，进入浏览器访问phpinfo。
如果出现下面的信息则表示安装成功。

redis

Redis Support

enabled

Redis Version

2.2.8-devphp7

配置Redis

Redis的配置文件处于/etc/redis/redis.conf，我们可以编辑它，具体参数设置可以参考<a href="http://www.runoob.com/redis/redis-conf.html">菜鸟教程 - Redis 配置6</a>。