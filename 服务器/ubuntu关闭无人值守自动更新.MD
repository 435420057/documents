# ubuntu关闭无人值守自动更新

通过 /var/log/apt/history.log 文件查看是否有更新过

通过以下命令信息

```sh
sudo vim /etc/apt/apt.conf.d/10periodic
```
将 APT::Periodic::Update-Package-Lists "1" 改为 APT::Periodic::Update-Package-Lists "0"
