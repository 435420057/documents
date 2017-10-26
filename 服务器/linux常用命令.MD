# Linux 服务器常用命令以及注意事项
主要记录在日常工作常用的一些Linux命令，以及服务器项目配置信息

> 1. 检查服务器中某个进程的内存以及CPU使用情况
     
     top -p $(pidof 进程名) 如:top -p $(pidof mongod) 

> 2. 