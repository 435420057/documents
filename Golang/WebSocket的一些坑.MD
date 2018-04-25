# WebSocket 相关错误处理

  前些时间需要做一个视频直播HTML5版本，所以就研究了一下。当采用HLS进行流传送时这个延迟实在太高，不能满足项目要求。

所以，就打起WebScoket + JS本地解码的主意，经过一翻研究发现github已经有很多JS解码视频的库。只需要提供个Go WebSocket服务器就行了。

我们决定使用 github.com\gorilla\websocket 来实现服务端，JSMpeg 做测试前端，最终实现H264解码。所以就有下面两个问题：

## 1、 WebSocket connection to .. failed: Error in connection establishment: net::ERR_CONNECTION_REFUSED
   
    服务器需要监听可以解决这个问题，服务不能监听0.0.0.0:port JSMpeg 客户端做了设置，所以就会报这个错误，把监听改127.0.0.1:port

## 2、WebSocket: Error during WebSocket handshake: Sent non-empty 'Sec-WebSocket-Protocol' header but no response was received
    
    这个由于客户端在创建websocket的时候设置协议参数，如果服务器没有做对应处理那个就会报这个错误， JSMpeg 测试Demo是这个样写的 this.websocket = new WebSocket(url, this.options.protocols||null),因为他设置了第二参数，不管有没有设置服务器收到Sec-WebSocket-Protocol 这个header 那个服务器也做对应回应。比如：Sec-WebSocket-Protocol:null

    ```go
      func ws(w http.ResponseWriter, r *http.Request) {
	      h := make(http.Header)
          h.Set("Sec-WebSocket-Protocol", "null")
          conn, err := upgrader.Upgrade(w, r, h)
          ....
          ....
      }
    ```