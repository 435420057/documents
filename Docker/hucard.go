package memory

import (
	"encoding/json"
	"errors"
	"huapai/internal/pai"

	kredis "github.com/doublemo/koala/db/redis"
	redis "github.com/go-redis/redis"
)

// HuCardMember 胡牌玩家信息
type HuCardMember struct {
	PlayerID int64   `json:"id"`
	Fan      int32   `json:"f"`
	Score    float64 `json:"s"`
	BScore   float64 `json:"bs"` // 当前积分
}

// HuCard 胡牌结构
type HuCard struct {
	Card      int32            `json:"c"`  // 胡的牌
	HuType    int32            `json:"ht"` // 胡牌类型
	HuMember  []*HuCardMember  `json:"hm"` // 点炮玩家
	PlayerID  int64            `json:"id"` // 玩家ID
	Score     float64          `json:"s"`  // 获取分值
	BScore    float64          `json:"bs"` // 当前积分
	FanNumber int32            `json:"fn"` // 胡牌番数
	ZhuCard   int32            `json:"zc"` // 胡主精
	Cards     pai.OneHuaHuInfo `json:"oi"` // 胡牌排列
	HaiDi     bool             `json:"hd"` // 是否海底
}

// AddHuCard 增加胡牌
func AddHuCard(conn *kredis.Client, id int64, hu ...*HuCard) error {
	data := make([]interface{}, 0)
	for _, h := range hu {
		str, err := json.Marshal(h)
		if err != nil {
			return err
		}

		data = append(data, str)
	}

	var ret *redis.IntCmd
	if conn.IsCluster() {
		ret = conn.CClient().RPush(GenHuCardKey(id), data...)
	} else {
		ret = conn.Client().RPush(GenHuCardKey(id), data...)
	}

	if ret.Err() != nil || ret.Val() < 1 {
		return errors.New("save failed")
	}

	return nil
}

// GetHuCard 获取胡牌信息
func GetHuCard(conn *kredis.Client, id int64) (map[int64][]*HuCard, error) {
	var ret *redis.StringSliceCmd
	if conn.IsCluster() {
		ret = conn.CClient().LRange(GenHuCardKey(id), 0, -1)
	} else {
		ret = conn.CClient().LRange(GenHuCardKey(id), 0, -1)
	}

	data := make(map[int64][]*HuCard)
	err := ret.Err()
	if err == redis.Nil {
		return data, nil
	}

	if err != nil {
		return nil, err
	}

	for _, v := range ret.Val() {
		h := HuCard{}
		if err = json.Unmarshal([]byte(v), &h); err != nil {
			continue
		}

		if _, ok := data[h.PlayerID]; !ok {
			data[h.PlayerID] = make([]*HuCard, 0)
		}

		data[h.PlayerID] = append(data[h.PlayerID], &h)
	}

	return data, nil
}

// GenHuCardKey 房间记录玩家胡牌信息
func GenHuCardKey(id int64) string {
	return GenKey("room_hu", id)
}
