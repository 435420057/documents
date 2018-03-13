# Golang跨平台编译

Mac下编译Linux, Windows平台的64位可执行程序：
```bash
set CGO_ENABLED=0 
set GOOS=linux 
set GOARCH=amd64 
go build gofile.go


set CGO_ENABLED=0 
set GOOS=windows 
set GOARCH=amd64 
go build gofile.go
```

Linux下编译Mac, Windows平台的64位可执行程序：
```bash
set CGO_ENABLED=0 
set GOOS=darwin 
set GOARCH=amd64 
go build gofile.go

set CGO_ENABLED=0 
set GOOS=windows 
set GOARCH=amd64 
go build gofile.go
```
Windows下编译Mac, Linux平台的64位可执行程序：
```bash
set CGO_ENABLED=0 
set GOOS=darwin 
set GOARCH=amd64 
go build gofile.go

set CGO_ENABLED=0 
set GOOS=linux 
set GOARCH=amd64 
go build gofile.go
```