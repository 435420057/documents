# postgresql 函数例子

```sql

CREATE OR REPLACE FUNCTION UserGoodsTicketGroupToCoupon(VARIADIC groups INTEGER[])
RETURNS INTEGER AS $$
  DECLARE
	  i INTEGER;
		records RECORD;
		users RECORD;
	BEGIN
	
	-- 下架所有市场商品券
	UPDATE v3_goods_tickets_market SET order_status=2 WHERE order_status = 1 AND ticket_group IN (select * FROM unnest(groups)) AND deleted_at IS NULL;
	
	-- 修改用户在交易中的商品券为正常
	UPDATE v3_user_goods_tickets SET ticket_status=1 WHERE ticket_status = 2 AND ticket_group IN (select * FROM unnest(groups)) AND deleted_at IS NULL;
	
	
	i=0;
	-- 循环为用户转换星豆与积分
	for records in(SELECT A.ticket_user_id,SUM(B.ticket_price) AS ticket_price, SUM(B.ticket_score) AS ticket_score FROM v3_user_goods_tickets AS A LEFT JOIN v3_goods_tickets AS B ON B.id = A.ticket_id WHERE A.ticket_status = 1 AND A.ticket_group IN (select * FROM unnest(groups)) AND A.deleted_at IS NULL GROUP BY A.ticket_user_id) LOOP
	
	  -- 更新用户星豆与积分
	  UPDATE v3_user_accounts SET current_coupon=current_coupon + records.ticket_price, score = score + records.ticket_score WHERE user_id::uuid=records.ticket_user_id;
		
		-- 查的前用户的账户信息
		SELECT * FROM v3_user_accounts WHERE user_id::uuid = records.ticket_user_id LIMIT 1 INTO users;
		
		-- 为用户插入兑换日志
		INSERT INTO g_amount_logs(created_at,updated_at,user_id, change_type, change_value, before_value,after_value,related_no,trade_date, change_score,before_score,after_score) VALUES(NOW(),NOW(),records.ticket_user_id, 24, records.ticket_price, users.current_coupon, users.current_coupon + records.ticket_price, '', NOW(), records.ticket_score, users.score, users.score+ records.ticket_score);
		
	  i=i+1;
	END LOOP;
	
  -- 将所有转过的券软删除,设置为当前时间
  UPDATE v3_user_goods_tickets SET deleted_at = NOW() WHERE ticket_group IN (select * FROM unnest(groups)) AND deleted_at IS NULL;
	RETURN i;
	END; $$ LANGUAGE plpgsql;

-- 调用 
SELECT UserGoodsTicketGroupToCoupon(8,9);
```