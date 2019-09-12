# Jenkins docker部署

<a href="https://hub.docker.com/_/jenkins/">https://hub.docker.com/_/jenkins/</a>


# 1.jenkins 安装是无法加载本地目录报错无权限问题。

  ```shell
  sudo docker run --restart=always --name jenkins -it -p 8088:8080 -p 50000:50000 -v /data/docker/volumes/jenkins:/var/jenkins_home jenkins
  ```

  报错: touch: cannot touch '/var/jenkins_home/copy_reference_file.log': Permission denied
  
  这会将jenkins数据存储在主机上的/ your / home中。确保容器中的jenkins用户（jenkins user - uid 1000）可以访问/ your / home，或者在docker run中使用-u some_other_user参数。
  
  在装入卷之前，必须在主机中设置正确的权限 sudo chown 1000 volume_dir


#  2. 在安装jenkins2.151以后，进入系统无法创建项目以及文件夹，查看错误信息

```shell
org.apache.commons.jelly.JellyTagException: jar:file:/usr/local/tomcat8/webapps/ROOT/WEB-INF/lib/jenkins-core-2.152.jar!/hudson/model/UpdateCenter/CoreUpdateMonitor/message.jelly:53:20: <j:otherwise> can't parse argument number: changelog.url
    at org.apache.commons.jelly.impl.TagScript.handleException(TagScript.java:726)
    at org.apache.commons.jelly.impl.TagScript.run(TagScript.java:281)
    at org.apache.commons.jelly.impl.ScriptBlock.run(ScriptBlock.java:95)
    at org.apache.commons.jelly.TagSupport.invokeBody(TagSupport.java:161)
    at org.apache.commons.jelly.tags.core.ChooseTag.doTag(ChooseTag.java:38
  ....
```

这个问题应该是jenkins BUG造成，好像好插件项目下面的多语言有关系，只需要进入jenkins/plugins， 删除localization-zh-cn*, 重启Jenkins可解决