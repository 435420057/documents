# MongoDB定时将数据库指定表备份脚本
```bash
#!/usr/bin/env bash

set -e

nowtime=$(date +%Y%m%d)
backdir=/data/mongobak

backup() {
  mongodump --host $1 --port $2 -d $3 -c $4 --out ${backdir}/${nowtime}
}

execute() {
  backup $1 $2 $3 $4
  if [ $? -eq 0 ]; then
    echo "backup host $1 port $2 database $3 collection $4 done"
  else
    echo "backup host $1 port $2 database $3 collection $4 failed"
  fi
}


if [ ! -d "$backdir/$nowtime" ]; then
  mkdir -p $backdir/$nowtime
fi

execute 127.0.0.1 27017 hb player
```


# 定时将备份的历史文件删除防止硬盘写满
```bash
#!/usr/bin/env bash

set -e

nowtime=$(date -d '-7 days' "+%Y%m%d")
backdir=/data/mongobak

if [ -d "$backdir/$nowtime" ]; then
  rm -rf "$backdir/$nowtime"
  echo "delete $backdir/$nowtime done"
fi
```

