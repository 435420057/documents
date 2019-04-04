# 怎么在Javascript中使用ProtoBuffer

在Javascript使用ProtoBuffer目前常接触的主要有两种ProtoBuf.js和Google Buffer，大数项目中常看到是使用ProtoBuf.js,但是我个觉得像这种需要加载proto文件进行动态解析的做法，会造成效率的一些浪费。而Google Buffer 是直接造成JS文件这样的效率明显要高些，至少不要在解析proto文件。

## 一、先来说说怎么用Google Buffer生成JS的方式

这个方式有个问题。当通过生成来的js是不能直接在html中直接使用的，原因很简单因为生成后的js文件需要Google的库才能运行。我们可以在生成后JS中的看很类似与这样的东西
```js
var jspb = require('google-protobuf');
var goog = jspb;
var global = Function('return this')();

goog.exportSymbol('...', null, global);
```

这说明面要求google-protobuf这个库才能运行。

下面我们来说说怎么样让他在html中能直接运行起来。

### 需要安装一些软件和库

> 1、 下载安装nodejs http://nodejs.cn/download/ 下载系统对应的版本

  nodejs下载主要会用他的一些库，用来转换js。我主要会用到npm库管理工具。所以这里要安装 nodejs和npm。

> 2、 下载google buffer https://github.com/protocolbuffers/protobuf/releases 找对应的版本

 下载后解决到系统环境变量能找到目录或放到你的工作目录中等待使用。


> 3、需要安装nodejs的几库
```shell
npm install -g require  #加载其它库文件的方法库，我们常见的require('')
npm install -g browserify  #用于转换js的库
npm install google-protobuf #这个应该一看就明白
```

> 4、编写一个用于测试的proto文件

proto所使用的是google buffer 3 有语法，相关的详细语法 https://developers.google.com/protocol-buffers/?hl=zh-CN 这里看.如果你访问不了，请百度。

```js
syntax = "proto3";  
package pb;  
message MyMessage {  
    string Name =  1;  
}  
```

保存为test.proto

> 5、proto文件生成js

首先确认我们下载的google buffer 可执行文件可以在环境变量中找到

打在命令行:
```shell
> protoc --version
# libprotoc 3.7.1
```

现在我们在命令行进行存方proto的目录执行:
```shell
> protoc --js_out=import_style=commonjs,binary:. test.proto
```
命令执行后如果没有报错，那么在对应目录下就会出现test_pb.js

这个js在nodejs中运行比较简单。:)， 如果想在html直接运行还需要转换一次加对应的库

> 6、 编写一个导出文件

exports.js
```js

const pb = require('./test_pb');

module.exports = {
	DataProto:pb
}
```

文件写好后保存为exports.js,打开命令行执行以下命令:

```shell
> browserify exports.js > testPB.js
```

命令执行后在目录，就会看到一个testPB.js这个文件明显要比test_pb.js文件要大不少。因为通过转换已经打了对应需要用到的库

> 7、编写一个测试html文件
```html
<!DOCTYPE html>
<html>
<head>
	<title>Client</title>
	<meta charset="utf-8" />
	<style type="text/css">

	</style>
	
</head>
<body>
	<script src="./testPB.js"></script>
	<script type="text/javascript">
		var m = new proto.pb.MyMessage();
		m.setName("OK");

         var bytes = m.serializeBinary(); 
         console.log(bytes);
	</script>
</body>
</html>
```


## 二、ProtoBuf.js又怎么用呢?

ProtoBuf.js用起来就比较方法了.

> 1、下载ProtoBuf.js库
 
 ProtoBuf.js库在这里： https://github.com/dcodeIO/ProtoBuf.js 

 官网在这里http://dcode.io/protobuf.js/

 具体怎么用这个就不用多说了，上面的文档写的很清楚也很简单
