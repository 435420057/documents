
# 定时删除MYSQL数据脚本
```bash
#!/usr/bin/env bash

#set -e
if [ $# -ne 4 ]; then
  echo "Min ID And Max ID AND SETP needed."
  exit 2
fi

export MYSQL_PWD=123456
         
minLastID=$1
maxLastID=$2
setp=$3
mysqlUser="root"
mysqlDB="k_logs"
tableName=$4
         
for ((i=$minLastID;i <= $maxLastID; i+=$3)); do
mysql -u$mysqlUser $mysqlDB << EOF
  DELETE FROM $tableName WHERE id <= $i;
  
EOF
  echo "DELETE ID $i"
  sleep 0.01
done

```

运行命令,如：

```bash
# 名称  开始删除id 结束删除ID  每次删除行数 表名 
./xx.sh 0 100 10 log_scoreinfo
```