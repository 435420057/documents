# Nginx代理Go程序并支持跨域处理

```conf
server
{
    listen 80;
	listen 443 ssl http2;
    server_name apiv1.xxxx.com;
    index index.php index.html index.htm default.php default.htm default.html;
    root /www/wwwroot/apiv1.xxxx.com;
    
     #允许跨域请求的域，* 代表所有
      #add_header 'Access-Control-Allow-Origin' 'https://www.xxx.com';
      #允许带上cookie请求
      add_header 'Access-Control-Allow-Credentials' 'true';
      #允许请求的方法，比如 GET/POST/PUT/DELETE
      add_header 'Access-Control-Allow-Methods' 'POST,GET,OPTIONS,PUT,DELETE';
      #允许请求的header
      add_header 'Access-Control-Allow-Headers' 'Authorization,Content-Type,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,X-Requested-With';
      
      # 预检命令的缓存，如果不缓存每次会发送两次请求，单位为秒。
      # 第一次是浏览器使用OPTIONS方法发起一个预检请求，第二次才是真正的异步请求
      add_header 'Access-Control-Max-Age' 7776000;
      

    #SSL-START SSL相关配置，请勿删除或修改下一行带注释的404规则
    #error_page 404/404.html;
    ssl_certificate    /www/server/panel/vhost/cert/apiv1.xxxx.com/fullchain.pem;
    ssl_certificate_key    /www/server/panel/vhost/cert/apiv1.xxxx.com/privkey.pem;
    ssl_protocols TLSv1.1 TLSv1.2 TLSv1.3;
    ssl_ciphers EECDH+CHACHA20:EECDH+CHACHA20-draft:EECDH+AES128:RSA+AES128:EECDH+AES256:RSA+AES256:EECDH+3DES:RSA+3DES:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    add_header Strict-Transport-Security "max-age=31536000";
    error_page 497  https://$host$request_uri;

    #SSL-END

    #ERROR-PAGE-START  错误页配置，可以注释、删除或修改
    #error_page 404 /404.html;
    #error_page 502 /502.html;
    #ERROR-PAGE-END

    #PHP-INFO-START  PHP引用配置，可以注释或修改
    include enable-php-00.conf;
    #PHP-INFO-END

    #REWRITE-START URL重写规则引用,修改后将导致面板设置的伪静态规则失效
    include /www/server/panel/vhost/rewrite/apiv1.xxxx.com.conf;
    #REWRITE-END

    #禁止访问的文件或目录
    location ~ ^/(\.user.ini|\.htaccess|\.git|\.env|\.svn|\.project|LICENSE|README.md)
    {
        return 404;
    }

    #一键申请SSL证书验证目录相关设置
    location ~ \.well-known{
        allow all;
    }

    #禁止在证书验证目录放入敏感文件
    if ( $uri ~ "^/\.well-known/.*\.(php|jsp|py|js|css|lua|ts|go|zip|tar\.gz|rar|7z|sql|bak)$" ) {
        return 403;
    }

    location ~ .*\.(gif|jpg|jpeg|png|bmp|swf)$
    {
        expires      30d;
        error_log /dev/null;
        access_log /dev/null;
    }

    location ~ .*\.(js|css)?$
    {
        expires      12h;
        error_log /dev/null;
        access_log /dev/null;
    }
    
    location / {
        if ($request_method = 'OPTIONS') {
          add_header 'Access-Control-Allow-Origin' *;
          add_header 'Access-Control-Allow-Credentials' 'true';
          #允许请求的方法，比如 GET/POST/PUT/DELETE
          add_header 'Access-Control-Allow-Methods' 'POST,GET,OPTIONS,PUT,DELETE';
           #允许请求的header
          add_header 'Access-Control-Allow-Headers' 'Authorization,Content-Type,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,X-Requested-With';
          add_header 'Access-Control-Max-Age' 86400;
          return 204;
        }
      
        proxy_pass http://127.0.0.1:7350;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        #proxy_connect_timeout 600;
        #proxy_send_timeout 600;
        #proxy_read_timeout 600;
    }
    
    location /game {
        if ($request_method = 'OPTIONS') {
          add_header 'Access-Control-Allow-Origin' *;
          add_header 'Access-Control-Allow-Credentials' 'true';
          #允许请求的方法，比如 GET/POST/PUT/DELETE
          add_header 'Access-Control-Allow-Methods' 'POST,GET,OPTIONS,PUT,DELETE';
           #允许请求的header
          add_header 'Access-Control-Allow-Headers' 'Authorization,Content-Type,Accept,Origin,User-Agent,DNT,Cache-Control,X-Mx-ReqToken,X-Requested-With';
          add_header 'Access-Control-Max-Age' 86400;
          return 204;
        }
      
        proxy_pass http://127.0.0.1:3350;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        #proxy_connect_timeout 600;
        #proxy_send_timeout 600;
        #proxy_read_timeout 600;
    }
    access_log  /www/wwwlogs/apiv1.xxxx.com.log;
    error_log  /www/wwwlogs/apiv1.xxxx.com.error.log;
}
```