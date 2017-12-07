# mongodb配置事项

> 1、官方建议mongodb配置信息
```bash
processManagement:
   fork: true
net:
   bindIp: 127.0.0.1
   port: 27017
storage:
   dbPath: /srv/mongodb
systemLog:
   destination: file
   path: "/var/log/mongodb/mongod.log"
   logAppend: true
storage:
   journal:
      enabled: true
```

> * 最近遇到一个奇怪的问题， mongodb无原无故退出问题。并且查看日志发现mongo在收信号量15后退出（ERROR: got signal 15 (Terminated), will terminate after current cmd ends）。最后发现fork应该为true,由于默认安装mongo后通过systemd启动服务后fork默认为false.所以这点值得注意。