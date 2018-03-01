# Jenkins docker部署

<a href="https://hub.docker.com/_/jenkins/">https://hub.docker.com/_/jenkins/</a>


# jenkins 安装是无法加载本地目录报错无权限问题。

  ```shell
  sudo docker run --restart=always --name jenkins -it -p 8088:8080 -p 50000:50000 -v /data/docker/volumes/jenkins:/var/jenkins_home jenkins
  ```

  报错: touch: cannot touch '/var/jenkins_home/copy_reference_file.log': Permission denied
  
  这会将jenkins数据存储在主机上的/ your / home中。确保容器中的jenkins用户（jenkins user - uid 1000）可以访问/ your / home，或者在docker run中使用-u some_other_user参数。
  
  在装入卷之前，必须在主机中设置正确的权限 sudo chown 1000 volume_dir