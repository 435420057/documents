# 手动安装oracle JDK

1.去oracle官网下载,链接点此。也可以使用wget http://download.oracle.com/otn-pub/java/jdk/8u111-b14/jdk-8u111-linux-x64.tar.gz 安装包在你执行这个命令时所在的文件夹位置。

　 2.解压 tar -zxvf jdk-8u111-linux-x64.tar.gz

　 3.移动到自己想放的位置：

　　　　mkdir /usr/lib/jdk    

　　　　mv jdk1.8.0_111  /usr/lib/jdk/jdk1.8

　 4.设置环境变量：

　　　　方案一：修改全局配置文件，作用于所有用户：vim /etc/profile 

　　　　export JAVA_HOME=/usr/lib/jdk/jdk1.8

　　　　export JRE_HOME=${JAVA_HOME}/jre

　　　　export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib

　　　　export PATH=.:${JAVA_HOME}/bin:$PATH

　　　　方案二：修改当前用户配置文件，只作用于当前用户：vim ~/.bashrc 设置与上一样

　 5.使修改的配置立刻生效：source /etc/profile 或者 source ~/.bashrc

　 6.检查是否安装成功：java -version