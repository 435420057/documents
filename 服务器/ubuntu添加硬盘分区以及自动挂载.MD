# ubuntu下添加硬盘，分区以及自动挂载

## 查看硬盘状况
```shell
> sudo fdisk -l
```

注意要添加sudo命令，否则查看不到硬盘信息。如果有多块硬盘就可以看到/dev/sda、/dev/sdb、/dev/sdc等等

## 使用fdisk命令分区，例如为/dev/sdb分区

```shell
> sudo fdisk /dev/sdb
```


回车进入分区命令

输入m查看fdisk分区工具选项

输入n开始分区

可以选择p（主分区）或e（扩展分区）等创建好扩展分区后就可以看到逻辑分区选项。根据提示输入盘符号（按提示的输入或直接回车），分区大小等信息。

可以多次执行n操作创建多个分区，创建好后输入w保存退出。

执行partprobe重新读取整个分区表。


## 格式化分区

```shell
> sudo mkfs -t ext4 /dev/sdb1
```

注意不能格式化扩展分区

## 挂载

创建挂载点，就是新建一个目录


```shell
> mkdir ~/disk1
> mkdir ~/disk2
```

挂载


```shell
> sudo mount /dev/sdb1 ~/disk1
> sudo mount /dev/sdb5 ~/disk2
```

输入mount命令查看是否挂载成功，如果成功会看到如下信息/dev/sdb1 on /home/lgd/disk1 type ext4 (rw)

也可以使用df命令查看是否挂载成功

## 自动挂载

修改/etc/fstab文件实现自动挂载

添加如下信息

UUID=b543f8f7-579c-45b5-96d6-31de6fa1a55e /home/lgd/disk1 ext4 defaults 1 2

```shell
/dev/vdb1       /data           ext4    defaults                 1       2
```

共6个字段分别代表

分区设备文件名或UUID

挂载点

文件系统名称

挂载参数，挂载权限

指定分区是否被dump备份，0代表不备份，1代表每天备份，2代表不定期备份。

指定分区是否被fsck检测，0代表不检测，其他数字代表检测的优先级，比如1的优先级比2高。根目录所在的分区的优先级为1，其他分区的优先级为大于或等于2

添加完信息保存后，执行 sudo mount -a 命令，如果没有报错就证明配制好了。


修复由/etc/fstab文件配制错误引起的系统不能启动问题

启动后根据提示按 m 进入root命令行页面，更改/etc/fstab文件，然后重新启动。如果不能修改/etc/fstab文件，可能是根分区挂载权限问题，可使用 mount -o remount,rw / 重新挂载根分区，其中rw代表读写权限。修改好后，重启完成修复。
