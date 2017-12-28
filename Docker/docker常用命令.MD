# Docker 日常维护中常用的命令以及注意事项
主要记录Docker在日常工作常用的命令以及注意事项

> 1. 关于Docker如果获取容器IP
```bash
   $ docker inspect redis | grep IPA
```

> 2. docker 查找 PID
  ```bash
   $ docker inspect -f '{{.State.Pid}}' 74e320677d80
  ```

   pid=3182

> 3. 如何将本地目录加载docker容器中
  ```bash
  $ docker run -d -p 8080:8080 -p 50000:50000 --name myjenkins -v E:/jenkins:/var/jenkins_home -t jenkins
  ```