# 关于清理Linux缓存(drop_caches)

```bash
cat /proc/sys/vm/drop_caches

 // 默认是0；1-清空页缓存；2-清空inode和目录树缓存；3-清空所有缓存
 ```

 大多数GNU Linux都是默认开启了缓存机制的，查看配置文件 cat /proc/sys/vm/drop_caches 若为0的话，则表示开启了缓存机制的，这种机制在一直运行某个服务的时候非常有效，便于服务本身的快速读取，而作为桌面用户来说，打开的程序并不那么单一，所以桌面用户还是建议设置此配置文件为1，及时对系统缓存进行清理，释放空间以供其它程序使用。

 ```bash
root@bak ~]# sync //注意：在清空缓存之前使用sync命令同步数据到磁盘
[root@bak ~]# free -m
             total       used       free     shared    buffers     cached
Mem:         15898       3029      12869          0        191       1064
-/+ buffers/cache:       1772      14125
Swap:        31999          0      31999
[root@bak ~]# echo 3 > /proc/sys/vm/drop_caches
[root@bak ~]# free -m //发现缓存明显减少了
             total       used       free     shared    buffers     cached
Mem:         15898       1770      14128          0          1         30
-/+ buffers/cache:       1738      14160
Swap:        31999          0      31999

 ```

 清理缓存脚本：
 ```bash
#！bin/bash

sync

sync

echo 3 > /proc/sys/vm/drop_caches
```