# 用nginx缩短Kubernetes dashboard访问url

Kubernetes dashboard以API Server方式访问的url很长，对纠结的人不大友好。所以想使用nginx来缩短它。
我们现在使用的是自签证书，nginx作反向代理意味着后端也是https方式，而且需要客户端证书和CA证书来验证。
否则nginx访问后面时也是报如下错误：

```json
{
  "kind": "Status",
  "apiVersion": "v1",
  "metadata": {
 
  },
  "status": "Failure",
  "message": "services \"https:kubernetes-dashboard:\" is forbidden: User \"system:anonymous\" cannot get services/proxy in the namespace \"kube-system\"",
  "reason": "Forbidden",
  "details": {
    "name": "https:kubernetes-dashboard:",
    "kind": "services"
  },
  "code": 403
}

```

## 解决

首先，我们需要需要访问的域名或IP的证书，我使用的是自签证书签发的。这里不另作说明，只是提醒，当前需要使用openssl v3扩展才能让浏览器完全信任出现绿色证书标识。

然后，需要将master节点的客户端证书提取出来，ca证书也准备好。


```sh
# 生成client-certificate-data
grep 'client-certificate-data' ~/.kube/config | head -n 1 | awk '{print $2}' | base64 -d >> kubecfg.crt
 
# 生成client-key-data
grep 'client-key-data' ~/.kube/config | head -n 1 | awk '{print $2}' | base64 -d >> kubecfg.key

```

最后，附上我的nginx配置：

```yaml
############负载均衡配置###########  
upstream k8s_dev {    
    server 192.168.105.92:8443 ;    
    server 192.168.105.92:8443 ;    
    server 192.168.105.92:8443 ;    
}    
############负载均衡配置########### 
 
server {
    listen       80 ;
    server_name  192.168.105.99  ;
 
    #access_log  logs/host.access.log  main;
 
    location / {
        root   html;
        index  index.html index.htm;
    }
 
 
    location /k8s/ {
        return 301 https://$host$request_uri;  # 强制使用https方式
    }
}
 
 
 
# HTTPS server
 
server {
    listen       443 ssl;
    server_name  192.168.105.99 ;
 
    ssl_certificate      certs/192.168.105.99.crt;  # 被访问域名证书
    ssl_certificate_key  certs/192.168.105.99.key; 
 
    # Recommendations from https://raymii.org/s/tutorials/Strong_SSL_Security_On_nginx.html
    ssl_protocols TLSv1.1 TLSv1.2;
    ssl_ciphers '!aNULL:kECDH+AESGCM:ECDH+AESGCM:RSA+AESGCM:kECDH+AES:ECDH+AES:RSA+AES:';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
 
    location / {
        root   html;
        index  index.html index.htm;
    }
 
    location /k8s/ {
        proxy_ssl_trusted_certificate /etc/kubernetes/pki/ca.crt;   # Kubernetes CA证书
        proxy_ssl_certificate certs/kubecfg.crt;                    # 客户端证书
        proxy_ssl_certificate_key certs/kubecfg.key; 
        proxy_ssl_session_reuse on;
        proxy_pass  https://k8s_dev/api/v1/namespaces/kube-system/services/https:kubernetes-dashboard:/proxy/ ;                                                                   # 跳转
    }
}
```

这样，即可通过http://192.168.105.99/k8s或者https://192.168.105.99/k8s登录dashboard。