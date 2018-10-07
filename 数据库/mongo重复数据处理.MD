# Mongodb 重复数据查询处理

## 查询表中相同字段

查询表中字段为'bqyId'相同的字段

```sql
db.resToMinPrice.aggregate([{'$group':{
                                    '_id':{'bqyId':'$bqyId'},
                                    'uniqueIds':{'$addToSet':'$_id'},
                                    'count':{'$sum':1}
                            }},
                            {'$match':{'count':{'$gt':1} 
                            }}],
                           {allowDiskUse:true}
                        )

```

## 删除表中相同数据

查询和删除语句,删除语句后面添加forEach循环

```sql
db.resToMinPrice.aggregate([{'$group':{
                                    '_id':{'bqyId':'$bqyId'},
                                    'uniqueIds':{'$addToSet':'$_id'},
                                    'count':{'$sum':1}
                            }},
                            {'$match':{'count':{'$gt':1} 
                            }}],
                           {allowDiskUse:true}
                        ).forEach(function(doc){
                            doc.uniqueIds.shift();
                            db.resToMinPrice.remove({_id: {$in: doc.uniqueIds}});
                        });

```

1.根据bqyId分组并统计数量，$group只会返回参与分组的字段，使用$addToSet在返回结果数组中增加_id字段

2.使用$match匹配数量大于1的数据

3.doc.uniqueIds.shift();表示从数组第一个值开始删除；作用是踢除重复数据其中一个_id，让后面的删除语句不会删除所有数据

4.使用forEach循环根据_id删除数据 $addToSet 操作符只有在值没有存在于数组中时才会向数组中添加一个值。如果值已经存在于数组中，$addToSet返回，不会修改数组。
