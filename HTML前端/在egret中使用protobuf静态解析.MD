# 在Egret中使用ProtoBuf.js 的静态解析方案

最近在弄一个聊天软件服务端已经写OK了，就没有客户端来进行测试。用机器人测试一轮又轮。感觉还是应该有个客户端来测试一下更好。

为快速弄个客户端来试试，我寻思用H5来弄个客户端试试。最终，我选择用Egret来快速弄一个，但是项目开始每一个问题就让我为难了，由于服务器这边的协议用的protobuf实现的，客户端也需要支持这东西,看了半天文档发现Egret没有protobuf库，在网上查了半天，大家都用第三方库实现的。

网上实现ProtoBuf都基于github上一个项目ProtoBuf.js来完成的，使用这个项目有两种方法可以实现proto文件的解析，一种是动态proto文件的加载解析,还有一个种就是通过将prtoto文件生成静态的ts或js文件来实现。但是网上的文档好多的说得模棱两可的，我折腾了半天终于成功。

现在详细说一下，我是如何做到的。

### 1 、通用npm安装protobujs

这个安装非常方便，如果电脑没有安装npm，那就先安装一个，至npm是什么东西。可以去看看nodejs。

安装好npm以后在命令行执行：
```shell
npm install protobufjs@6.8.4 -g
```

然后安装，对应egret可以的库

```shell
npm install @egret/protobuf -g
```

安装好后。等待项目中使用

### 2、在Egret项目使用ProtoBuf
首页我们创建好我们项目后，进入项目的要根目中,开启命令行。记录命令要进到项目的根目中。

执行以下命令：

```shell
pb-egret add
```

这个命令执行完成后，你会发现你Egret项目根目下会多一个protobuf目录，只要看到这个目录就证明成功了。

打开protobuf目录你会看到以下一些目录：

bundles 这个目录是用于存储proto文件生成后的ts和js文件

library 这个ProtoBuf.js项目的核心库
 
protofile 这个就是你需要将你的proto文件复制到这个目录中用于命令生成使用的

pbconfig.json 这个不用说

### 3、生成proto文件

将项目所需要使用到的proto文件复制到protobuf/protofile目录中。

执行以下命令：
```shell
pb-egret generate
```

注意这些命令都是在根目下执行的。

执行完成后，你可以在protobuf/bundles 目录到你的文件

### 4、如何使用

首页当执行完生成proto命令后，你编辑器输入你的proto中定义的命名空间名称，应该就会看到提示了。

但是,这里面有个地方要注意:

由于我们执行命令添加protobuf库后，会修改egret 项目根目中的tsconfig.json,egretProperties.json 。如下：

egretProperties.json

```json
{
		"engineVersion": "5.2.27",
		"compilerVersion": "5.2.27",
		"template": {},
		"target": {
				"current": "web"
		},
		"modules": [
				{
						"name": "egret"
				},
				{
						"name": "socket"
				},
				{
						"name": "eui"
				},
				{
						"name": "assetsmanager"
				},
				{
						"name": "dragonBones"
				},
				{
						"name": "game"
				},
				{
						"name": "tween"
				},
				{
						"name": "promise"
				},
				{
						"name": "WXCloudApi",
						"path": "./libs/WXCloudApi"
				},
				{
						"name": "egretmgobe",
						"path": "./libs/egretmgobe"
				},
                // 多了下面的东西
				{
						"name": "protobuf-library",
						"path": "./protobuf/library" // 这里前面一定要加./要不然运行会找不到定义
				},
				{
						"name": "protobuf-bundles",
						"path": "./protobuf/bundles"
				}
		]
}
```

tsconfig.json

```json
{
		"compilerOptions": {
				"target": "es5",
				"outDir": "bin-debug",
				"experimentalDecorators": true,
				"lib": [
						"es5",
						"dom",
						"es2015.promise"
				],
				"types": []
		},
		"include": [
				"src",
				"libs",
				"protobuf/**/*.d.ts" // 这里
		]
}
```

这里要注意在自动添加protobuf-library,protobuf-bundles的路径的时候，默认是没有加前面的./的。就这个问题坑了半天。这里要加上要不然运行的时候会找不到定义。



我的proto文件是这样写的
```proto
syntax = "proto3";

package pb;

message FoxChatAuth {
     message Form {
        message SearchUserRequest {
            string Name  = 1; // 名称
            string Phone = 2; // 手机
        }
     }
}
```

在使用的时候，我是这样用的:

```ts
let d:pb.FoxChatAuth.Form.SearchUserRequest = new pb.FoxChatAuth.Form.SearchUserRequest({"Name":"kxm", "Phone":"13588888888"});

let bytes = pb.FoxChatAuth.Form.SearchUserRequest.encode(d).finish();

let bytes = pb.FoxChatAuth.Form.SearchUserRequest.encode(d).finish()
console.log(bytes);

let req = pb.FoxChatAuth.Form.SearchUserRequest.decode(bytes);
console.log(req);
```

非常方便。