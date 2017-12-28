# 关于MYSQL数据目录迁移

## 1. 确定mysql数据库文件存放目录
一般默认是在/var/lib/mysql目录下。先登录自己的mysql数据库，比如我用root账户登录，然后使用下面查询语句查询：

```sql
show variables like '%dir%';  
```

可以看到其中datadir的值为/var/lib/mysql/即为当前数据库文件存放目录。

另外一个basedir参数表示mysql数据库的安装位置，迁移数据库文件位置不需要改动这个参数。

## 2. 迁移数据库文件到新的目录下
先使用下面命令将mysql数据库服务停止：
```bash
$ service mysql stop
```

我新的数据盘挂载在/mnt/data目录下，因此要将数据库迁移到/mnt/data。

> 2.1 可以使用mv命令将原数据库目录文件移动到新的目录，好处是不会简单，不会修改原数据库文件的权限，以及用户和用户组归属：

```bash
$ sudo mv /var/lib/mysql /mnt/data/  
```

> 2.2 也可以使用cp复制命令将原数据库目录文件复制到新的目录，好处是。。万一迁移失败，恢复工作相对简单一点，等确认迁移成功再来删掉原数据库目录文件也不迟。为了不影响复制过来数据库目录文件权限和用户用户组归属问题，使用cp命令时要加上-a参数：

```bash
$ sudo cp -a /var/lib/mysql /mnt/data/  
```

注：由于/var/lib/mysql目录归属于mysql数据库创建的mysql用户和mysql用户组，所以迁移文件的时候需要使用root权限，命令要使用sudo

## 3. 修改配置文件
一共有三个配置文件需要修改：
> 3.1 my.cnf文件

mysql数据库会按顺序优先级从/etc/my.cnf、/etc/mysql/my.cnf、/usr/etc/my.cnf、~/.my.cnf四个位置找my.cnf配置文件，一旦找到就不再继续往下找。Ubuntu默认将my.cnf配置文件放在/etc/mysql/my.cnf位置，所以在/etc/my.cnf位置没有找到这个配置文件。

选择自己使用的文本编辑器编辑my.cnf配置文件，我用vim，所以sudo vim /etc/mysql/my.cnf。一样需要sudo，使用root权限编辑。将其中[mysqld]标签下的datadir属性值改为新数据库目录路径/mnt/data/mysql

修改后保存并退出。

> 3.2 usr.bin.mysqld文件

由于Ubuntu使用了apparmor安全模块，就是类似于沙盒运行的一种机制，它可以限制软件在运行时的一些行为，比如对哪些目录和文件可以读写加锁等等。

由于修改了数据库文件路径，所以要修改mysql数据库的apparmor配置文件，在其中将新数据库文件目录和文件的读写及加锁权限添加上去，同时可以删除或者注释掉原先/var/lib/mysql数据库文件目录的权限。mysql数据库的apparmor配置文件路径在/etc/apparmor.d/usr.sbin.mysqld。使用下面命令编辑这个配置文件：
```bash
$ sudo vim /etc/apparmor.d/usr.sbin.mysqld 
```

找到其中的

  /var/lib/mysql/ r,
  /var/lib/mysql/** rwk,

两行权限声明，可以在前面加上#好注释掉。然后对照格式，加入新路径的权限声明：
  /mnt/data/mysql/ r,
  /mnt/data/mysql/** rwk,

  保存并关闭。

  > 3.3 abstractions/mysql文件
  由于usr.bin.mysqld文件中引用了abstractions/mysql文件，也就是会将abstractions/mysql文件中的权限声明导入进来。因此，也修改下这个文件：

sudo vim /etc/apparmor.d/abstractions/mysql

同样也是将新数据库文件路径中的socket文件权限添加进去，同时可以删除或者注释掉全路径中申请的权限

 保存后退出。

 ## 4.重启数据库
 配置文件修改成功后就可以重启数据库，重启数据库之前需要先重新载入apparmor配置文件，使用下面命令重新载入:
 ```bash
 $ sudo /etc/init.d/apparmor restart  
 ```

 重载成功就可以使用下面命令启动数据库：

 ```bash
 $ service mysql start
 ```

 ## 5.权限问题
 经过上诉步骤之后，你有可能数据库无法启动。忽略继续登录数据库出现下面关于sock的错误：

ERROR 2002 (HY000): Can't connect to local MySQL server through socket '/var/run/mysqld/mysqld.sock' (2)

查看数据库的启动错误日志，sudo vim /var/log/mysql/error.log.

解决方法：在迁移后的mysql目录上执行命令将这个目录的用户与用户组改成mysql
```bash
$ chown -R mysql:mysql /mnt/data/mysql
```

数据库已经正常启动， 并且数据库文件路径也已经替换到/mnt/data/mysql目标路径，数据库文件迁移成功。