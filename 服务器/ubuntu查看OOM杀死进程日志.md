# ubuntu查看OOM杀死进程日志

```sh
sudo dmesg |grep -E 'kill|oom|out of memory'
```