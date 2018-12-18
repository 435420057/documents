# 关于Golang -ldflag 用法

ldflag 能让我们在编译时调协程序里面的变量，这样一来我们将可以将如版本信息写入到程序中。更重要的是我们可以Git编译版本写到程序中，这样可以方法我们知道程序运行的是什么版本。

ldflag用法：

【路径，非必需，除非你有目录层次】包名.变量

以下是提供一下例子：
```shell
VERSION = "0.0.1" # 定义版本号
COMMIT = $(shell git rev-parse HEAD) # --short 获取git commit信息
BUILDDATE = $(shell date +%Y:%m:%d:%T) # 获取当期日期

debug:
	go install -v

build:
	go build -race

test:
	go test -cover .

clean:
	go clean
	find . -type f -name \*.test | xargs rm >/dev/null

pull:
	git pull --verbose

linux:
	# strip information
	# GOOS=linux GOARCH=amd64 go install -ldflags "-w -s"
	GOOS=linux GOARCH=amd64 go install -ldflags "-X main.version=$(VERSION) -X main.commitid=$(COMMIT) -X main.builddate=$(BUILDDATE)"	

format:
	find . -type f -name \*.go | xargs gofmt -w

deploy: debug
	cp ../../bin/x.exe  //192.168.x.x/x/x

dptest: debug
	cp ../../bin/x.exe  //192.168.x.x/x/x/testsrv/

dpall: deploy dptest

.PHONY:\
	debug\
	build\
	test\
	run\
	clean\
	deploy\
	format
```

Golang 程序：

```golang
package main

import "fmt"

var (
    version    string
    commitid string
    builddate string
)

func main() {
    fmt.Printf("%s\n%s\n%s\n", version, commitid, builddate)
}
```