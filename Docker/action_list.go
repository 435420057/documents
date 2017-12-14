package memory

import (
	"container/list"
	"errors"
	"fmt"
	"game/chesscards/mahjong"
	"huapai/internal/pai"
	"strconv"
	"strings"

	kredis "github.com/doublemo/koala/db/redis"
	"github.com/go-redis/redis"
)

// ActionElement 动作对像
type ActionElement struct {
	PlayerID int64            `json:"pid"`    // 玩家ID
	Action   int32            `json:"action"` // 动作
	Card     int32            `json:"card"`   // 牌
	Fan      int              `json:"fan"`    // 胡牌番
	ZhuCard  int32            `json:"zcard"`  // 胡主精
	Cards    pai.OneHuaHuInfo `json:"oi"`     // 胡牌排列
	DianPao  []int64          `json:"dp"`     // 点炮玩家
}

// Marshal 序列化
func (element *ActionElement) Marshal() string {
	if element.Cards == nil {
		element.Cards = make(pai.OneHuaHuInfo, 0)
	}

	ohinfo := make([]string, 0)
	for _, m := range element.Cards {
		mdata := make([]string, 0)
		for _, n := range m {
			mdata = append(mdata, fmt.Sprint(n))
		}

		ohinfo = append(ohinfo, strings.Join(mdata, "_"))
	}

	if element.DianPao == nil {
		element.DianPao = make([]int64, 0)
	}

	dianPao := make([]string, len(element.DianPao))
	for i, m := range element.DianPao {
		dianPao[i] = fmt.Sprint(m)
	}

	return fmt.Sprintf("%d,%d,%d,%d,%d,%s,%s", element.PlayerID, element.Action, element.Card, element.Fan, element.ZhuCard, strings.Join(ohinfo, "|"), strings.Join(dianPao, "_"))
}

// Unmarshal 反序列化
func (element *ActionElement) Unmarshal(s string) error {
	data := strings.Split(s, ",")
	if len(data) != 7 {
		return errors.New("Invalid string")
	}

	m, err := strconv.ParseInt(data[0], 10, 64)
	if err != nil {
		return err
	}

	element.PlayerID = m

	m, err = strconv.ParseInt(data[1], 10, 32)
	if err != nil {
		return err
	}

	element.Action = int32(m)

	m, err = strconv.ParseInt(data[2], 10, 32)
	if err != nil {
		return err
	}

	element.Card = int32(m)

	m, err = strconv.ParseInt(data[3], 10, 32)
	if err != nil {
		return err
	}

	element.Fan = int(m)

	m, err = strconv.ParseInt(data[4], 10, 32)
	if err != nil {
		return err
	}

	element.ZhuCard = int32(m)

	if len(data[5]) > 0 {
		vdata := make(pai.OneHuaHuInfo, 0)
		mdata := strings.Split(data[5], "|")
		for _, m := range mdata {
			ndata := strings.Split(m, "_")
			vvdata := make(mahjong.CardsQueue, 0)
			for _, n := range ndata {
				x, err := strconv.ParseInt(n, 10, 32)
				if err != nil {
					return err
				}
				vvdata.Push(mahjong.Card(x))
			}

			vdata = append(vdata, vvdata)
		}

		element.Cards = vdata
	} else {
		element.Cards = make(pai.OneHuaHuInfo, 0)
	}

	if len(data[6]) > 0 {
		ndata := strings.Split(data[6], "_")
		vdata := make([]int64, len(ndata))
		for i, n := range ndata {
			x, err := strconv.ParseInt(n, 10, 64)
			if err != nil {
				return err
			}

			vdata[i] = x
		}

	} else {
		element.DianPao = make([]int64, 0)
	}

	return nil
}

// ActionValue 动作返回值
type ActionValue struct {
	Value *ActionElement // 当前值
	Next  *ActionElement // 下一个值
}

// ActionList 执行队列
type ActionList struct {
	front     *list.List     // 优先队列
	back      *list.List     // 尾部队列
	active    *list.List     // 已经激活的队列(正在运行队列)
	conn      *kredis.Client // redis 连接
	keyFrefix string         // redis 键前缀
}

// PushBack 往尾部列表进行追加
// redis 要保持一致
func (acl *ActionList) PushBack(elements ...*ActionElement) error {
	if len(elements) < 1 {
		return errors.New("elements empty")
	}

	data := make([]interface{}, 0)
	for _, element := range elements {
		data = append(data, element.Marshal())
	}

	f := func(pipe redis.Pipeliner) error {
		pipe.RPush(acl.genBackKey(), data...)
		return nil
	}

	if acl.conn.IsCluster() {
		if _, err := acl.conn.CClient().Pipelined(f); err != nil {
			return err
		}
	} else {
		if _, err := acl.conn.Client().Pipelined(f); err != nil {
			return err
		}
	}

	for _, element := range elements {
		acl.back.PushBack(element)
	}
	return nil
}

// PushFBack 往尾部列表进行头部追加
// redis 要保持一致
func (acl *ActionList) PushFBack(elements ...*ActionElement) error {
	if len(elements) < 1 {
		return errors.New("elements empty")
	}

	data := make([]interface{}, 0)
	for _, element := range elements {
		data = append(data, element.Marshal())
	}

	f := func(pipe redis.Pipeliner) error {
		pipe.LPush(acl.genBackKey(), data...)
		return nil
	}

	if acl.conn.IsCluster() {
		if _, err := acl.conn.CClient().Pipelined(f); err != nil {
			return err
		}
	} else {
		if _, err := acl.conn.Client().Pipelined(f); err != nil {
			return err
		}
	}

	for _, element := range elements {
		acl.back.PushFront(element)
	}
	return nil
}

// PushFront 往前部列表进行追加
// redis 要保持一致
func (acl *ActionList) PushFront(elements ...*ActionElement) error {
	if len(elements) < 1 {
		return errors.New("elements empty")
	}

	data := make([]interface{}, 0)
	for _, element := range elements {
		data = append(data, element.Marshal())
	}

	f := func(pipe redis.Pipeliner) error {
		pipe.RPush(acl.genFrontKey(), data...)
		return nil
	}

	if acl.conn.IsCluster() {
		if _, err := acl.conn.CClient().Pipelined(f); err != nil {
			return err
		}
	} else {
		if _, err := acl.conn.Client().Pipelined(f); err != nil {
			return err
		}
	}

	for _, element := range elements {
		acl.front.PushBack(element)
	}
	return nil
}

// PushFFront 往前部列表进行头部追加
// redis 要保持一致
func (acl *ActionList) PushFFront(elements ...*ActionElement) error {
	if len(elements) < 1 {
		return errors.New("elements empty")
	}

	data := make([]interface{}, 0)
	for _, element := range elements {
		data = append(data, element.Marshal())
	}

	f := func(pipe redis.Pipeliner) error {
		pipe.LPush(acl.genFrontKey(), data...)
		return nil
	}

	if acl.conn.IsCluster() {
		if _, err := acl.conn.CClient().Pipelined(f); err != nil {
			return err
		}
	} else {
		if _, err := acl.conn.Client().Pipelined(f); err != nil {
			return err
		}
	}

	for _, element := range elements {
		acl.front.PushFront(element)
	}
	return nil
}

// PushActive 激活队列
func (acl *ActionList) PushActive(elements ...*ActionElement) error {
	if len(elements) < 1 {
		return errors.New("elements empty")
	}

	data := make([]interface{}, 0)
	for _, element := range elements {
		data = append(data, element.Marshal())
	}

	f := func(pipe redis.Pipeliner) error {
		pipe.RPush(acl.genActiveKey(), data...)
		return nil
	}

	if acl.conn.IsCluster() {
		if _, err := acl.conn.CClient().Pipelined(f); err != nil {
			return err
		}
	} else {
		if _, err := acl.conn.Client().Pipelined(f); err != nil {
			return err
		}
	}

	for _, element := range elements {
		acl.active.PushBack(element)
	}
	return nil
}

// Pop 弹出左边第一个
func (acl *ActionList) Pop() *ActionValue {
	v := &ActionValue{}
	if acl.front.Len() > 0 {
		element := acl.front.Front()
		if m, ok := element.Value.(*ActionElement); ok {
			v.Value = m
			next := element.Next()
			if next == nil {
				next = acl.back.Front()
			}

			if next != nil {
				if m, ok := next.Value.(*ActionElement); ok {
					v.Next = m
				}
			}
		}

		acl.front.Remove(element)
		if acl.conn.IsCluster() {
			acl.conn.CClient().LPop(acl.genFrontKey())
		} else {
			acl.conn.Client().LPop(acl.genFrontKey())
		}
		return v
	}

	if acl.back.Len() > 0 {
		element := acl.back.Front()
		if m, ok := element.Value.(*ActionElement); ok {
			v.Value = m
			next := element.Next()
			if next != nil {
				if m, ok := next.Value.(*ActionElement); ok {
					v.Next = m
				}
			}
		}

		acl.back.Remove(element)
		if acl.conn.IsCluster() {
			acl.conn.CClient().LPop(acl.genBackKey())
		} else {
			acl.conn.Client().LPop(acl.genBackKey())
		}
		return v
	}

	return v
}

// Front 获取
func (acl *ActionList) Front() *ActionValue {
	v := &ActionValue{}
	if acl.front.Len() > 0 {
		element := acl.front.Front()
		if m, ok := element.Value.(*ActionElement); ok {
			v.Value = m
			next := element.Next()
			if next == nil {
				next = acl.back.Front()
			}

			if next != nil {
				if m, ok := next.Value.(*ActionElement); ok {
					v.Next = m
				}
			}
		}

		return v
	}

	if acl.back.Len() > 0 {
		element := acl.back.Front()
		if m, ok := element.Value.(*ActionElement); ok {
			v.Value = m
			next := element.Next()
			if next != nil {
				if m, ok := next.Value.(*ActionElement); ok {
					v.Next = m
				}
			}
		}
		return v
	}

	return v
}

// ActiveFront 获取激活
func (acl *ActionList) ActiveFront() *ActionValue {
	if acl.active.Len() < 1 {
		return nil
	}

	v := &ActionValue{}
	element := acl.active.Front()
	if m, ok := element.Value.(*ActionElement); ok {
		v.Value = m
		next := element.Next()
		if next != nil {
			if m, ok := next.Value.(*ActionElement); ok {
				v.Next = m
			}
		}
	}
	return v
}

// ActiveBack 获取激活
func (acl *ActionList) ActiveBack() *ActionValue {
	if acl.active.Len() < 1 {
		return nil
	}

	v := &ActionValue{}
	element := acl.active.Back()
	if m, ok := element.Value.(*ActionElement); ok {
		v.Value = m
		v.Next = nil
	}

	return v
}

// ActiveAll 获取激活
func (acl *ActionList) ActiveAll() []*ActionElement {
	data := make([]*ActionElement, 0)
	for e := acl.active.Front(); e != nil; e = e.Next() {
		if m, ok := e.Value.(*ActionElement); ok {
			data = append(data, m)
		}
	}

	return data
}

func (acl *ActionList) ActiveReverseAll() []*ActionElement {
	data := make([]*ActionElement, 0)
	for e := acl.active.Back(); e != nil; e = e.Prev() {
		if m, ok := e.Value.(*ActionElement); ok {
			data = append(data, m)
		}
	}

	return data
}

func (acl *ActionList) GetActiveListByPlayerID(playerID int64) []*ActionElement {
	data := make([]*ActionElement, 0)
	for e := acl.active.Front(); e != nil; e = e.Next() {
		if m, ok := e.Value.(*ActionElement); ok && m.PlayerID == playerID {
			data = append(data, m)
		}
	}

	return data
}

func (acl *ActionList) GetActiveListByPlayerIDAction(playerID int64, action int32) *ActionElement {
	for e := acl.active.Front(); e != nil; e = e.Next() {
		if m, ok := e.Value.(*ActionElement); ok && m.PlayerID == playerID && m.Action == action {
			return m
		}
	}

	return nil
}

// genFrontKey 尾部列表redis存储键值
func (acl *ActionList) genFrontKey() string {
	return acl.keyFrefix + "_" + "front"
}

// genBackKey 尾部列表redis存储键值
func (acl *ActionList) genBackKey() string {
	return acl.keyFrefix + "_" + "back"
}

// genBackKey 尾部列表redis存储键值
func (acl *ActionList) genActiveKey() string {
	return acl.keyFrefix + "_" + "Active"
}

// Reload 加载数据
func (acl *ActionList) Reload() error {
	if err := acl.ReloadActive(); err != nil {
		return err
	}

	if err := acl.ReloadFront(); err != nil {
		return err
	}

	if err := acl.ReloadBack(); err != nil {
		return err
	}

	return nil
}

// ReloadFront 加载前部数据
func (acl *ActionList) ReloadFront() error {
	var ret *redis.StringSliceCmd
	acl.front.Init()
	if acl.conn.IsCluster() {
		ret = acl.conn.CClient().LRange(acl.genFrontKey(), 0, -1)
	} else {
		ret = acl.conn.Client().LRange(acl.genFrontKey(), 0, -1)
	}

	err := ret.Err()
	if err == redis.Nil {
		return nil
	}

	if err != nil {
		return err
	}

	for _, item := range ret.Val() {
		element := &ActionElement{}
		if err := element.Unmarshal(item); err != nil {
			return err
		}

		acl.front.PushBack(element)
	}

	return nil
}

// ReloadBack 加载前部数据
func (acl *ActionList) ReloadBack() error {
	var ret *redis.StringSliceCmd
	acl.back.Init()
	if acl.conn.IsCluster() {
		ret = acl.conn.CClient().LRange(acl.genBackKey(), 0, -1)
	} else {
		ret = acl.conn.Client().LRange(acl.genBackKey(), 0, -1)
	}

	err := ret.Err()
	if err == redis.Nil {
		return nil
	}

	if err != nil {
		return err
	}

	for _, item := range ret.Val() {
		element := &ActionElement{}
		if err := element.Unmarshal(item); err != nil {
			return err
		}
		acl.back.PushBack(element)
	}
	return nil
}

// ReloadActive 加载前部数据
func (acl *ActionList) ReloadActive() error {
	var ret *redis.StringSliceCmd
	acl.active.Init()
	if acl.conn.IsCluster() {
		ret = acl.conn.CClient().LRange(acl.genActiveKey(), 0, -1)
	} else {
		ret = acl.conn.Client().LRange(acl.genActiveKey(), 0, -1)
	}
	err := ret.Err()
	if err == redis.Nil {
		return nil
	}

	if err != nil {
		return err
	}

	for _, item := range ret.Val() {
		element := &ActionElement{}
		if err := element.Unmarshal(item); err != nil {
			return err
		}
		acl.active.PushBack(element)
	}
	return nil
}

// Len 可用数据长度
func (acl *ActionList) Len() int {
	return acl.front.Len() + acl.back.Len()
}

// ActiceLen 激活数据长度
func (acl *ActionList) ActiceLen() int {
	return acl.active.Len()
}

// Init  初始化
func (acl *ActionList) Init() {
	acl.front.Init()
	acl.back.Init()

	if acl.conn.IsCluster() {
		acl.conn.CClient().Del(acl.genFrontKey())
		acl.conn.CClient().Del(acl.genBackKey())
	} else {
		acl.conn.Client().Del(acl.genFrontKey())
		acl.conn.Client().Del(acl.genBackKey())
	}
}

// ActiveInit .
func (acl *ActionList) ActiveInit() {
	acl.active.Init()
	if acl.conn.IsCluster() {
		acl.conn.CClient().Del(acl.genActiveKey())
	} else {
		acl.conn.Client().Del(acl.genActiveKey())
	}
}

// ActiveInit .
func (acl *ActionList) InitAll() {
	acl.front.Init()
	acl.back.Init()
	acl.active.Init()

	f := func(pipe redis.Pipeliner) error {
		pipe.Del(acl.genFrontKey())
		pipe.Del(acl.genBackKey())
		pipe.Del(acl.genActiveKey())
		return nil
	}

	if acl.conn.IsCluster() {
		acl.conn.CClient().Pipelined(f)
	} else {
		acl.conn.Client().Pipelined(f)
	}
}

// NewActionList 执行队列
func NewActionList(conn *kredis.Client, key string) *ActionList {
	return &ActionList{
		front:     list.New(),
		back:      list.New(),
		active:    list.New(),
		conn:      conn,
		keyFrefix: key,
	}
}
