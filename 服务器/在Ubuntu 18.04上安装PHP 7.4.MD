#  在Ubuntu 18.04上安装PHP 7.4

PHP 7.4是PHP的最新版本，如果你想装指定版本的PHP，这篇文章对你有很大的帮助。本文将帮助你在Ubuntu 18.04，16.04上通过PPA的方式安装PHP 7.4。


## 安装PHP 7.4
通过下列命令安装PHP 7.4

```sh
$ sudo apt-get install software-properties-common
$ sudo add-apt-repository ppa:ondrej/php
$ sudo apt-get update
$ sudo apt-get install -y php7.4
```

安装完成后，使用如下命令检查PHP的版本

```sh
$ php -v 

PHP 7.4.15 (cli) (built: Feb 23 2021 15:12:26) ( NTS )
Copyright (c) The PHP Group
Zend Engine v3.4.0, Copyright (c) Zend Technologies
    with Zend OPcache v7.4.15, Copyright (c), by Zend Technologies
    with Xdebug v3.0.3, Copyright (c) 2002-2021, by Derick Rethans
```

可以通过以上方法安装php的其它版本

## 安装PHP 7模块

使用如下命令安装PHP 7模块,先从缓存中查询可以与系统php版本所对应的模拟
```sh
$ sudo apt-cache search php7*
```

```sh
php7.4-gnupg - PHP wrapper around the gpgme library
php7.4-grpc - High performance, open source, general RPC framework for PHP
php7.4-http - PECL HTTP module for PHP Extended HTTP Support
php7.4-igbinary - igbinary PHP serializer
php7.4-imagick - Provides a wrapper to the ImageMagick library
php7.4-inotify - Inotify bindings for PHP
php7.4-lua - PHP Embedded lua interpreter
php7.4-lz4 - LZ4 Extension for PHP
php7.4-mailparse - Email message manipulation for PHP
php7.4-maxminddb - Reader for the MaxMind DB file format for PHP
php7.4-mcrypt - PHP bindings for the libmcrypt library

.....

php7.4-uploadprogress - file upload progress tracking extension for PHP
php7.4-uuid - PHP UUID extension
php7.4-vips - PHP extension for interfacing with libvips
php7.4-xdebug - Xdebug Module for PHP
php7.4-xhprof - Hierarchical Profiler for PHP 5.x
php7.4-yac - YAC (Yet Another Cache) for PHP
php7.4-yaml - YAML-1.1 parser and emitter for PHP
php7.4-zmq - ZeroMQ messaging bindings for PHP
php7.4-zstd - Zstandard extension for PHP
```
用如下命令，按需安装PHP扩展模块，安装之前，确保安装正确的PHP版本。

```sh
 sudo apt-get install php7.4-mysql php7.4-curl php7.4-json php7.4-cgi php7.4-xsl
 ```