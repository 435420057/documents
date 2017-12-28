# JDK安装

## 直接下载jdk压缩包方式安装(这里只介绍jdk7的,jdk8 的原理完全一致)
　分为下面5个步骤

   1.官网下载JDK

   2.解压缩,放到指定目录

   3.配置环境变量

   4.设置系统默认JDK

　5. 测试jdk

1.官网下载JDK　　　
     地址: http://www.oracle.com/technetwork/articles/javase/index-jsp-138363.html
　　选择相应的 .gz包下载 

2. 解压缩,放到指定目录(以jdk-7u60-linux-x64.gz为例)
　　创建目录:
```bash
sudo mkdir /usr/lib/jvm
```

　加压缩到该目录:

```bash
 sudo tar -zxvf jdk-7u60-linux-x64.gz -C /usr/lib/jvm
 ```

3.修改环境变量:　　
```bash
sudo vim ~/.bashrc
```
　文件的末尾追加下面内容:
```bash
#set oracle jdk environment
export JAVA_HOME=/usr/lib/jvm/jdk1.7.0_60  ## 这里要注意目录要换成自己解压的jdk 目录
export JRE_HOME=${JAVA_HOME}/jre  
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib  
export PATH=${JAVA_HOME}/bin:$PATH  
```
　使环境变量马上生效
```bash
 source ~/.bashrc
 ```

4.设置系统默认jdk 版本
```bash
sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/jdk1.7.0_60/bin/java 300  
sudo update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/jdk1.7.0_60/bin/javac 300  
sudo update-alternatives --install /usr/bin/jar jar /usr/lib/jvm/jdk1.7.0_60/bin/jar 300   
sudo update-alternatives --install /usr/bin/javah javah /usr/lib/jvm/jdk1.7.0_60/bin/javah 300   
sudo update-alternatives --install /usr/bin/javap javap /usr/lib/jvm/jdk1.7.0_60/bin/javap 300  
``` 
　然后执行:
```bash
sudo update-alternatives --config java
```
 若是初次安装jdk,会有下面的提示     

   There is only one alternative in link group java (providing /usr/bin/java): 
    /usr/lib/jvm/jdk1.7.0_60/bin/java

　否者,选择合适的jdk

5.测试jdk
```bash
java -version

java version "1.7.0_60"
  Java(TM) SE Runtime Environment (build 1.7.0_60-b19)
  Java HotSpot(TM) 64-Bit Server VM (build 24.60-b09, mixed mode)
```
  jdk 安装成功
