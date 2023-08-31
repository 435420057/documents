# ubuntu下交叉编译arm64 golang插件
在ubuntu交叉编译arm64 golang 插件时如果设置不正确会收到报错：

```sh
loadinternal: cannot find runtime/cgo
/var/lib/jenkins/tools/org.jenkinsci.plugins.golang.GolangInstallation/Go1192/pkg/tool/linux_amd64/link: running gcc failed: exit status 1
/usr/bin/ld.gold: fatal error: /tmp/go-link-1440669699/go.o: unsupported ELF machine number 183
collect2: error: ld returned 1 exit statusb
```

收到这个报错，说明没有打开 CGO_ENABLED ， 设置环境变量 export CGO_ENABLED=1 继续编译，会收到：
```sh
# runtime/cgo
gcc_arm64.S: Assembler messages:
gcc_arm64.S:28: Error: no such instruction: `stp x29,x30,[sp,'
gcc_arm64.S:32: Error: too many memory references for `mov'
gcc_arm64.S:34: Error: no such instruction: `stp x19,x20,[sp,'
gcc_arm64.S:37: Error: no such instruction: `stp x21,x22,[sp,'
gcc_arm64.S:40: Error: no such instruction: `stp x23,x24,[sp,'
gcc_arm64.S:43: Error: no such instruction: `stp x25,x26,[sp,'
gcc_arm64.S:46: Error: no such instruction: `stp x27,x28,[sp,'
gcc_arm64.S:50: Error: too many memory references for `mov'
gcc_arm64.S:51: Error: too many memory references for `mov'
gcc_arm64.S:52: Error: too many memory references for `mov'
gcc_arm64.S:54: Error: no such instruction: `blr x20'
gcc_arm64.S:55: Error: no such instruction: `blr x19'
gcc_arm64.S:57: Error: no such instruction: `ldp x27,x28,[sp,'
gcc_arm64.S:60: Error: no such instruction: `ldp x25,x26,[sp,'
gcc_arm64.S:63: Error: no such instruction: `ldp x23,x24,[sp,'
gcc_arm64.S:66: Error: no such instruction: `ldp x21,x22,[sp,'
gcc_arm64.S:69: Error: no such instruction: `ldp x19,x20,[sp,'
gcc_arm64.S:72: Error: no such instruction: `ldp x29,x30,[sp],'
Build step 'Execute shell' marked build as failure
```

这是因为当前机器是amd64架构下的，所以安装的gcc也是x86的，这时需要安装：aarch64-linux-gnu-gcc

```sh
sudo apt-get install aarch64-linux-gnu-gcc
```

设置编译命令：
```sh
 export CGO_ENABLED=1
+ export GOOS=linux
+ export GOARCH=arm64
+ export GOPROXY=https://goproxy.cn
+ export CC=/usr/bin/aarch64-linux-gnu-gcc
+ go mod vendor
+ go build -buildmode=plugin -trimpath -o ./w-arm64.so
```
