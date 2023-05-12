# 将mp4转换为m3u8

```sh
ffmpeg -i home_loop.mp4 -codec: copy -start_number 0 -hls_time 5 -hls_list_size 0 -f hls home_loop/home_loop.m3u8
```