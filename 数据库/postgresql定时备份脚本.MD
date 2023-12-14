# postgresql定时备份脚本

```sh
#!/bin/sh

####################################################################################################
#
# 该脚本用于Linux系统下备份PostgreSQL数据
# 采用pg_dump工具导出SQL文件，并压缩存放
# 备份后删除保留天数以前的历史备份文件
# 
#
# 备份示例：
# 执行
# ./backup-postgresql.sh
#
#
# 添加系统定时任务
# # crontab -e
# 追加如下内容保存并退出（每天凌晨1点开始备份）
# 0 1 * * * sh /workspace/scripts/backup-postgresql.sh
# 
#
# 恢复示例：
# 1. 解压数据库备份文件
# unzip mydb-20211203142616.bak.zip
#
# 2. 创建数据库
# PostgreSQL命令行创建数据库（数据库名需与待恢复的数据库名一致）
# postgres=# create database mydb with owner mydbdata tablespace tbs_mydb;
#
# 3. 导入数据
# 切换到postgres用户
# su - postgres
# 通过psql工具导入数据
# psql -h 192.168.56.100 -p 5432 -d mydb -U postgres < mydb-20211203142616.bak
# 
#
# 注意事项：
# - 保证执行备份脚本的用户具备备份文件存放目录的读写权限
#
####################################################################################################

##################################################
# 环境变量及系统参数
##################################################
# PostgreSQL安装目录
PGHOME=/workspace/postgresql
# Host
PGHOST=192.168.56.100
# 监听端口
PGPORT=5432
# 用户
PGUSER=postgres
# 密码
PGPASSWORD=postgres
# 数据库
DBNAME=mydb
# 编码
ENCODING=UTF8

# 备份文件存放目录
BACKUP_DIR=/workspace/postgresql/backup
# 备份文件名称
BACKUP_FILE_NAME=${DBNAME}-$(date +%Y%m%d%H%M%S).bak
# 备份文件保留天数，默认30天
BACKUP_FILE_RESERVED_DAYS=30

# 脚本名称
SCRIPT_NAME="$(basename "${0}")"
# 备份执行日志存放目录
BACKUP_LOG_PATH=/tmp/postgresql-backup-logs
# 备份执行日志文件名称
BACKUP_LOG_FILENAME=${BACKUP_LOG_PATH}/${SCRIPT_NAME}-$(date +%Y%m%d%H%M%S).log


# 准备
function prepare() {
    if [ ! -d "${BACKUP_DIR}" ]; then
        mkdir -p "${BACKUP_DIR}"
    fi
    if [ ! -d "${BACKUP_LOG_PATH}" ]; then
        mkdir -p "${BACKUP_LOG_PATH}"
    fi
    if [ ! -f "${BACKUP_LOG_FILENAME}" ]; then
        touch "${BACKUP_LOG_FILENAME}"
    fi
}

# 记录INFO日志
function info() {
  echo "$(date "+%Y-%m-%d %H:%M:%S") [ INFO] ${1}" >> "${BACKUP_LOG_FILENAME}"
}

# 记录ERROR日志
function error() {
  echo -e "$(date "+%Y-%m-%d %H:%M:%S") \033[31m[ERROR]\033[0m ${1}" >> "${BACKUP_LOG_FILENAME}"
}

# 备份数据
function backup() {
    info "备份数据库${DBNAME}开始"
    export PGPASSWORD
    ${PGHOME}/bin/pg_dump --file ${BACKUP_DIR}/${BACKUP_FILE_NAME} --host ${PGHOST} --port ${PGPORT} --dbname ${DBNAME} --username ${PGUSER} --encoding ${ENCODING} --verbose >> ${BACKUP_LOG_FILENAME} 2>&1 
 # 获取上一个命令的退出状态(0表示正常退出)
 dump_status=$?
    if [ ! ${dump_status} -eq 0 ]; then
        error "备份数据库${DBNAME}失败"
        exit 1
    fi
    info "数据库${DBNAME}备份文件：${BACKUP_DIR}/${BACKUP_FILE_NAME}"
    info "备份数据库${DBNAME}结束"
}

# 压缩数据库备份文件
function compress() {
    info "压缩数据库${DBNAME}备份文件开始"
    cd ${BACKUP_DIR}
    if [ ! -f "${BACKUP_FILE_NAME}" ]; then
        error "压缩数据库${DBNAME}备份文件失败，文件${BACKUP_DIR}/${BACKUP_FILE_NAME}不存在"
        exit 1
    fi
    zip -qm ${BACKUP_FILE_NAME}.zip ${BACKUP_FILE_NAME}
    info "压缩数据库${DBNAME}备份文件结束"
}

# 删除历史数据库备份文件
function clear() {
    info "删除数据库${DBNAME}历史备份文件开始"
 # 获取更新时间大于保留天数以前的历史备份文件
 history_bak_zip_files=$(find ${BACKUP_DIR} -type f -mtime +${BACKUP_FILE_RESERVED_DAYS} -name "*.zip")
    if [ -z "${history_bak_zip_files}" ]; then
        info "无${BACKUP_FILE_RESERVED_DAYS}天前的历史备份文件"
        exit 0
    fi
 # 逐个删除历史备份文件
    for history_bak_zip_file in ${history_bak_zip_files}
    do
        rm -f ${history_bak_zip_file}
        info "删除数据库${DBNAME}历史备份文件${history_bak_zip_file}"
    done
    info "删除数据库${DBNAME}历史备份文件结束"

 # 也可通过批量删除命令删除数据库历史备份文件
 # find ${BACKUP_DIR} -type f -mtime +${BACKUP_FILE_RESERVED_DAYS} -name "*.zip" -exec rm -f {} \
}

function doBackup() {
 # 准备
    prepare
 # 备份数据库
    backup
 # 压缩备份文件
    compress
 # 删除历史备份文件
    clear
}

doBackup

exit 0
```