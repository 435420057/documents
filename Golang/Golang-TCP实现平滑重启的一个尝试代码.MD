# 关于TCP实现平滑重启的一次尝试

经过这次尝试发现平滑处理比较合适在处理HTTP的时候，如果是长连TCP那么玩家还是会出现闪断的情况，但是玩家可以及时重新连接回来。

这种重启方式不支持在Windows平台上实现，主要实现方式主要通知在主程序中调用Restart接口,通过net.FileListener继承父进程的监听以子进程方式重启程序。

具体实现代码：

## 配置文件代码 socket_config.go
```go
package monet

import (
	"net"
)

// SocketConfig 配置
type SocketConfig struct {
	Addr            string         // 监听地址
	ReadBufferSize  int            // 读取缓存大小 32767
	WriteBufferSize int            // 写入缓存大小 32767
	IsGraceful      bool           // 优雅重启
	CallBack        func(net.Conn) //
}

func NewSocketConfig() *SocketConfig {
	return &SocketConfig{
		ReadBufferSize:  32767,
		WriteBufferSize: 32767,
		IsGraceful:      false,
	}
}

```

## Socket实现代码
```go
package monet

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

const (
	StateStoped  = iota // 服务器已经停止
	StateRunning        // 服务器运行中
)

// SocketServer TCP服务器
type SocketServer struct {
	Config   *SocketConfig    // 配置
	sigChan  chan os.Signal   //
	status   int              //
	done     chan error       //
	wg       sync.WaitGroup   //
	mux      sync.Mutex       //
	listener *net.TCPListener //
}

func (ss *SocketServer) Serve() error {
	defer func() {
		close(ss.done)
	}()

	ss.mux.Lock()
	if ss.status == StateRunning {
		return nil
	}

	ss.status = StateRunning
	ss.mux.Unlock()

	go ss.signals()
	go ss.serve()

	err := <-ss.done
	ss.listener.SetDeadline(time.Now())
	ss.wg.Wait()

	ss.mux.Lock()
	ss.status = StateStoped
	ss.mux.Unlock()
	ss.listener.Close()
	return err
}

func (ss *SocketServer) serve() {
	if err := ss.listen(ss.Config.Addr); err != nil {
		ss.done <- err
		return
	}

	if ss.Config.CallBack == nil {
		ss.done <- fmt.Errorf("CallBack is undefined")
		return
	}

	log.Println("TCP listening on:", ss.listener.Addr())
	for {
		conn, err := ss.listener.AcceptTCP()
		if err != nil {
			return
		}

		conn.SetReadBuffer(ss.Config.ReadBufferSize)
		conn.SetWriteBuffer(ss.Config.WriteBufferSize)
		go ss.handle(conn)
	}
}

func (ss *SocketServer) Shutdown() {
	ss.mux.Lock()
	if ss.status != StateRunning {
		return
	}

	ss.status = StateStoped
	ss.mux.Unlock()
	ss.done <- nil
}

func (ss *SocketServer) signals() {
	var sig os.Signal
	signal.Notify(ss.sigChan, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM)
	pid := syscall.Getpid()
	for {
		sig = <-ss.sigChan
		switch sig {
		case syscall.SIGHUP:
			log.Println(pid, "Received SIGHUP. forking.")
			ss.fork()

		case syscall.SIGINT:
			log.Println(pid, "Received SIGINT.")
			ss.Shutdown()

		case syscall.SIGTERM:
			log.Println(pid, "Received SIGTERM.")
			ss.Shutdown()

		default:
			log.Printf("Received %v: nothing i care about...\n", sig)
		}
	}
}

func (ss *SocketServer) listen(addr string) error {
	if ss.Config.IsGraceful {
		f := os.NewFile(uintptr(3), "")
		lis, err := net.FileListener(f)
		if err != nil {
			f.Close()
			return fmt.Errorf("net.FileListener: %s", err)
		}

		if err := f.Close(); err != nil {
			return fmt.Errorf("error closing socket fd %d: %s", 3, err)
		}

		ss.listener = lis.(*net.TCPListener)
		process, err := os.FindProcess(os.Getppid())
		if err != nil {
			return fmt.Errorf(" os.FindProcess %d: %s", 3, err)
		}

		err = process.Signal(syscall.SIGTERM)
		if err != nil {
			return fmt.Errorf("rocess.Signal %d: %s", 3, err)
		}

		return nil
	}

	resolveAddr, err := net.ResolveTCPAddr("tcp", addr)
	if err != nil {
		return err
	}

	lis, err := net.ListenTCP("tcp", resolveAddr)
	if err != nil {
		return fmt.Errorf("generateListener (net.Listen): %v", err)
	}

	ss.listener = lis
	return nil
}

func (ss *SocketServer) handle(conn net.Conn) {
	defer func() {
		ss.wg.Done()
		conn.Close()
	}()

	ss.wg.Add(1)
	ss.Config.CallBack(conn)
}

func (ss *SocketServer) fork() {
	ss.mux.Lock()
	defer ss.mux.Unlock()

	var args []string
	path := os.Args[0]
	if len(os.Args) > 1 {
		for _, arg := range os.Args[1:] {
			if arg == "--graceful" {
				break
			}
			args = append(args, arg)
		}
	}

	args = append(args, "--graceful=true")
	log.Println("FORK:", path, args)
	cmd := exec.Command(path, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	ff, _ := ss.listener.File()
	cmd.ExtraFiles = []*os.File{ff}
	log.Println("cmd.Start:", cmd.Start())
}

func (ss *SocketServer) Restart() {
	pid := syscall.Getpid()
	p, err := os.FindProcess(pid)
	if err != nil {
		log.Println(err)
		return
	}

	log.Println("SEND Process:", p)
	p.Signal(syscall.SIGHUP)
}

func NewSocketServer(c *SocketConfig) *SocketServer {
	return &SocketServer{
		sigChan: make(chan os.Signal),
		done:    make(chan error),
		Config:  c,
	}
}

```

## 演示代码 main.go
```go
package main

import (
	"bufio"
	"flag"
	"log"
	"net"
	"time"
	"monet"
)

var (
	graceful = flag.Bool("graceful", false, "graceful")
)

func main() {
	flag.Parse()
	config := monet.NewSocketConfig()
	config.IsGraceful = *graceful
	config.Addr = ":8220"
	config.CallBack = func(conn net.Conn) {
		host, port, err := net.SplitHostPort(conn.RemoteAddr().String())
		log.Println("new connection from :", host, port, err)
		defer func() {
			log.Println("shutdown:", host, port)
		}()

		for {
			conn.SetReadDeadline(time.Now().Add(time.Duration(30) * time.Second))
			conn.SetWriteDeadline(time.Now().Add(time.Duration(30) * time.Second))
			message, err := bufio.NewReader(conn).ReadString('\n')
			if err != nil {
				return
			}

			daytime := time.Now().String()
			conn.Write([]byte(daytime + "--ll----ll--" + message + "\n"))
			time.Sleep(time.Second * 1)
		}
	}

	log.Println("Config:", config)

	s := monet.NewSocketServer(config)
	if !config.IsGraceful {
		time.AfterFunc(time.Second*30, func() {
			s.Restart()
		})
	}

	log.Println(s.Serve())
}

```