# MongoDB 聚合 -- 连表查询

在使用MongoDB存储数据的时候，我们查询的时候，有时候难免会需要进行连表查询。但是MongoDB本身是非关系性数据库，连表查询，很多时候，需要我们自己在代码里手工操作。但是从 MongoDB 3.2 版本过后，我们可以使用 $lookup 进行连表查询。下面就简单介绍一下 MongoDB 的 $lookup 的简单使用。

 比如现在我们有两张表, user 和 order 表。其中 user 表中的字段有 _id、uid、name、age；order 表中的字段有：_id、uid、product、money; 两张表存储的数据为：

```javascript

users = [{
  _id: ObjectId("5af2b2c6b138c267e414c072"),
  uid: "uid000",
  name: "小红",
  age: 26
}, {
  _id: ObjectId("5af2b2c6b138c267e414c073"),
  uid: "uid001",
  name: "小芳",
  age: 27
}]
```
```javascript
orders = [{
  _id: ObjectId("4af2b2c6b138c267e414c071"),
  uid: "uid000",
  product: "产品1",
  money: 100
}, {
  _id: ObjectId("4af2b2c6b138c267e414c072"),
  uid: "uid000",
  product: "产品2",
  money: 200
}, {
  _id: ObjectId("4af2b2c6b138c267e414c073"),
  uid: "uid001",
  product: "产品1",
  money: 100
}, {
  _id: ObjectId("4af2b2c6b138c267e414c074"),
  uid: "uid001",
  product: "产品2",
  money: 200
}]
```

假如现在有两个需求：

查询用户信息并且显示该用户的总消费金额(用户名、年龄、总消费金额)

查询用户的订单信息(订单id、产品、价格、用户名)

> 1. 首先来看第一个需求：

这个需求如果我们不考虑连表，只考虑关联的话，应该是


先查询出用户表所有的数据

在订单表中求出每一个用户的消费总金额

遍历用户和订单数据，然后一一通过 uid 进行匹配对应。

 如果按照我们的数据库连表来说：那应该是我们查询 user 表关联到 order 表，然后分组根据 uid 统计求和；下面来看一看具体的实现方式。

> 1.1 连表查询
```javascript
db.user.aggregate([{
  $lookup: { // 左连接
    from: "order", // 关联到order表
    localField: "uid", // user 表关联的字段
    foreignField: "uid", // order 表关联的字段
    as: "orders"
  }
}]);

```

这个时候出来的结果应该为：

```javascript
users = [{
  _id: ObjectId("5af2b2c6b138c267e414c072"),
  uid: "uid000",
  name: "小红",
  age: 26,
  orders: [{
    _id: ObjectId("4af2b2c6b138c267e414c071"),
    uid: "uid000",
    product: "产品1",
    money: 100
  }, {
    _id: ObjectId("4af2b2c6b138c267e414c072"),
    uid: "uid000",
    product: "产品2",
    money: 200
  }]
}, {
  _id: ObjectId("5af2b2c6b138c267e414c073"),
  uid: "uid001",
  name: "小芳",
  age: 27,
  orders: [{
    _id: ObjectId("4af2b2c6b138c267e414c073"),
    uid: "uid001",
    product: "产品1",
    money: 100
  }, {
    _id: ObjectId("4af2b2c6b138c267e414c073"),
    uid: "uid001",
    product: "产品1",
    money: 200
  }]
}]
```

> 1.2 拆分 orders 数组

```javascript
{
  $unwind: { // 拆分子数组
    path: "$orders",
    preserveNullAndEmptyArrays: true // 空的数组也拆分
  }
}
```

这个时候的数据结果应该是这样的


```javascript
[{
  _id: ObjectId("5af2b2c6b138c267e414c072"),
  uid: "uid000",
  name: "小红",
  age: 26,
  orders: {
    _id: ObjectId("4af2b2c6b138c267e414c071"),
    uid: "uid000",
    product: "产品1",
    money: 100
  }
}, {
  _id: ObjectId("5af2b2c6b138c267e414c072"),
  uid: "uid000",
  name: "小红",
  age: 26,
  orders: {
    _id: ObjectId("4af2b2c6b138c267e414c072"),
    uid: "uid000",
    product: "产品2",
    money: 200
  }
} …… ]

```

> 1.3 分组求和并返回字段数据


```javascript
{
  $group: { // 分组查询
    _id: "$_id",
    name: { $first: "$name" },
    age: { $first: "$age" },
    money: {$sum: "$orders.money"}
  }
}
```

这样就查询出了我们所需要的数据。将代码总结一下为：


```javascript
db.user.aggregate([{
  $lookup: { // 左连接
    from: "order", // 关联到order表
    localField: "uid", // user 表关联的字段
    foreignField: "uid", // order 表关联的字段
    as: "orders"
  }
}, {
  $unwind: { // 拆分子数组
    path: "$orders",
    preserveNullAndEmptyArrays: true // 空的数组也拆分
  }
}, { // 分组求和并返回
  $group: { // 分组查询
    _id: "$_id",
    name: { $first: "$name" },
    age: { $first: "$age" },
    money: {$sum: "$orders.money"}
  }
}]);
```

> 2. 查询用户的订单信息

> 2.1 连表查询

这个时候的连表是 order 表 跟 user 表关联(上一个是 user 表 和 order 表关联)


```javascript
{
  $lookup: {
    from: "users",
    localField: "openid",
    foreignField: "openid",
    as: "u"
  }
}
```

> 2.2 拆分子数组


```javascript
{ $unwind: "$u" }
```

> 2.3 只返回需要的字段

将 user 中需要返回的字段，提到子目录来


```javascript
{$addFields: {  name: "$u.name" }}
```

> 2.4 返回最终需要的字段结果


```javascript
{ 
  $project: {
    _id: 1,
    product: 1,
    money: 1,
    name: 1
  }
}
```

最终的代码为：


```javascript
db.order.aggregate([{
  $lookup: {
    from: "users",
    localField: "openid",
    foreignField: "openid",
    as: "u"
  }
}, {
  $unwind: "$u"
}, {
  $addFields: {  name: "$u.name" }
}, {
  $project: {
    _id: 1,
    product: 1,
    money: 1,
    name: 1
  }
}]);
```

虽然在 MongoDB 3.2 后我们能够进行连表查询了，方便了很多。但是其实 MongoDB 
本身是非关系性数据库。如果需要进行频繁的这种连表查询，我们可以考虑优化我们的数据库表。比如在订单表里面，每一条的订单记录都把我们的用户信息放进去。


```javascript
[{
  _id: ObjectId("4af2b2c6b138c267e414c071"),
  uid: "uid000",
  product: "产品1",
  money: 100,
  user: {
    _id: ObjectId("5af2b2c6b138c267e414c072"),
    uid: "uid000",
    name: "小红",
    age: 26
  }
}, {
  _id: ObjectId("4af2b2c6b138c267e414c071"),
  uid: "uid000",
  product: "产品1",
  money: 100,
  user: {
    _id: ObjectId("5af2b2c6b138c267e414c072"),
    uid: "uid000",
    name: "小红",
    age: 26
  }
}]

```
这个时候，在实现两个需求就很简单了：


```javascript

// 1. 查询用户信息并且显示该用户的总消费金额(用户名、年龄、总消费金额)
db.order.aggregate([{ // 根据 uid 求和
  $group: {
    _id: '$user.uid',
    money: {
      $sum: "$money"
    }，
    name: { $first: "$user.name" },
    age: { $first: "$user.age" }
  }
}]);

// 2. 查询用户的订单信息(订单id、产品、价格、用户名)
db.order.aggregate([{
  {$addFields: {  name: "$user.name" }}
}, { // 根据 uid 求和
  $project: {
    _id: 1,
    money: 1，
    product: 1,
    name: 1
  }
}]);
```


那么我们如何加条件语句呢？ 看看下面这个查询：
```javascript
db.player.aggregate([
{
  $lookup:{
    from:"club",
    localField:"_id",
    foreignField:"playerid",
    as:"cmsta"
  }
},{
  $unwind:{
    path:"$cmsta",
    preserveNullAndEmptyArrays:true,
  }
},
{$addFields:{pid:"$cmsta.playerid", mid:"$cmsta._id"}}
,
{
   $match:{pid:1003190, mid:6100}, // 条件匹配，分别匹配两张表的内容
},{
  $group:{
    _id:"$_id",
    nickname:{$first:"$nickname"},
    headimgurl:{$first:"$headimgurl"},
    name:{$first:"$cmsta.name"},
    mid:{$first:"$cmsta._id"},
  }
}
])
```

但是不得不说一句,如果真出现需要这样来查询数据，应该认真考虑一下自己的数据结构是否存在问题了,毕竟非关系数据这样操作成本是很高的。一但数据大了，会非常的卡。