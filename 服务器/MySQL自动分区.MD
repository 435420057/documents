# MySQL自动分区

以timestamp类型字段创建每日数据分区

## 创建分区数据库
```sql
CREATE TABLE `t_log`  (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '日志id，自增',
  `uid` bigint UNSIGNED NOT NULL COMMENT '玩家id',
  `ticket_amount` bigint NOT NULL DEFAULT 0 COMMENT '使用奖券金额',
  `ticket_id` bigint UNSIGNED NOT NULL DEFAULT 0 COMMENT '使用奖券ID',
  PRIMARY KEY (`id`, `created_time`) USING BTREE,
  INDEX `IDX_CTIME`(`created_time` ASC) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 0 CHARACTER SET = utf8mb3 COLLATE = utf8mb3_general_ci COMMENT = '日志记录' ROW_FORMAT = DYNAMIC PARTITION BY RANGE (UNIX_TIMESTAMP(created_time))
(
   PARTITION p20250102 VALUES LESS THAN (UNIX_TIMESTAMP('2025-01-03 00:00:00'))
 );
```

## 自动创建分区过程
```sql
CREATE PROCEDURE `auto_create_partition`()
BEGIN
	  DECLARE next_partition_name VARCHAR(50);
    DECLARE next_partition_value INT;
		DECLARE next_partition_exists INT;
		DECLARE expire_partition_name VARCHAR(50);
    DECLARE expire_partition_value INT;
		DECLARE expire_partition_exists INT;
		
		SET @TimeAt = DATE_ADD(CURDATE(),INTERVAL 1 DAY);
		SET @NextTimeAt = DATE_ADD(CURDATE(),INTERVAL 2 DAY);
		SET @ExpireTimeAt = DATE_SUB(CURDATE(),INTERVAL 3 DAY);
		
		 -- 计算下一个分区的名称和值
    SET next_partition_value = YEAR(@TimeAt) * 10000 + MONTH(@TimeAt) * 100 + DAY(@TimeAt);
    SET next_partition_name = CONCAT('p', next_partition_value);
		SET expire_partition_value = YEAR(@ExpireTimeAt) * 10000 + MONTH(@ExpireTimeAt) * 100 + DAY(@ExpireTimeAt);
		SET expire_partition_name = CONCAT('p', expire_partition_value);
		
		-- 检查分区是否已存在
		SELECT COUNT(1) INTO next_partition_exists FROM information_schema.partitions WHERE table_schema = DATABASE() AND table_name = 't_log' AND partition_name = next_partition_name;
    IF next_partition_exists = 0 THEN
        -- 创建新的分区
     SET @alter_table_sql = CONCAT('ALTER TABLE t_log ADD PARTITION (PARTITION ', next_partition_name, ' VALUES LESS THAN (',ROUND(UNIX_TIMESTAMP(CONCAT(@NextTimeAt,' 00:00:00'))), '))');
		 
        PREPARE stmt FROM @alter_table_sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
		
		-- 删除过期的分区
		SELECT COUNT(1) INTO expire_partition_exists FROM information_schema.partitions WHERE table_schema = DATABASE() AND table_name = 't_log' AND partition_name = expire_partition_name;
		IF expire_partition_exists > 0 THEN
			ALTER TABLE t_log_game_bill DROP PARTITION expire_partition_name;
		END IF;
END
```

### 创建每日8点自动运行的事件
```sql
CREATE EVENT `create_partition_event` 
ON SCHEDULE EVERY 1 DAY STARTS '2025-01-02 08:00:00' 
ON COMPLETION NOT PRESERVE ENABLE COMMENT '自动分区' 
DO CALL auto_create_partition();
```

### 查询已存在的分区
```sql
SELECT * FROM information_schema.partitions WHERE table_schema = DATABASE() AND table_name = 't_log'
```