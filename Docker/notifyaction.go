package pai

import (
	"container/list"
)

type NotifyActionElementHuMember struct {
	PlayerID int64
	Fan      int32
	Score    float64
	BScore   float64
}

// NotifyActionElement 信息
type NotifyActionElement struct {
	Action   int32
	Card     int32
	Fan      int
	HuInfo   OneHuaHuInfo
	ZhuCard  int32
	Cards    []int32
	HuType   int32
	PlayerID int64
	Score    float64
	BScore   float64
	HuMember []*NotifyActionElementHuMember
	HaiDi    bool
}

// NotifyActionValue 返回值
type NotifyActionValue struct {
	Prev  *NotifyActionElement
	Value *NotifyActionElement
	Next  *NotifyActionElement
}

// NotifyAction 所有玩家的操作集合
type NotifyAction struct {
	records map[int64]*list.List
}

// PushBack 加入后
func (nf *NotifyAction) PushBack(pid int64, element *NotifyActionElement) {
	if m, ok := nf.records[pid]; ok {
		m.PushBack(element)
	} else {
		m := list.New()
		m.PushBack(element)
		nf.records[pid] = m
	}
}

// PushFront 加入前
func (nf *NotifyAction) PushFront(pid int64, element *NotifyActionElement) {
	if m, ok := nf.records[pid]; ok {
		m.PushFront(element)
	} else {
		m := list.New()
		m.PushBack(element)
		nf.records[pid] = m
	}
}

// Get 取得某个玩家的操作选项
func (nf *NotifyAction) Get(pid int64) []*NotifyActionValue {
	data := make([]*NotifyActionValue, 0)
	if m, ok := nf.records[pid]; ok {
		for e := m.Front(); e != nil; e = e.Next() {
			value := NotifyActionValue{}
			if n, ok := e.Value.(*NotifyActionElement); ok {
				value.Value = n
			} else {
				continue
			}

			prev := e.Prev()
			next := e.Next()

			if prev != nil {
				if n, ok := prev.Value.(*NotifyActionElement); ok {
					value.Prev = n
				}
			}

			if next != nil {
				if n, ok := next.Value.(*NotifyActionElement); ok {
					value.Next = n
				}
			}

			data = append(data, &value)
		}
	}

	return data
}

// GetAll 取得所有玩家的操作选项
func (nf *NotifyAction) GetAll() map[int64][]*NotifyActionValue {
	data := make(map[int64][]*NotifyActionValue)
	for id, m := range nf.records {
		if _, ok := data[id]; !ok {
			data[id] = make([]*NotifyActionValue, 0)
		}

		for e := m.Front(); e != nil; e = e.Next() {
			value := NotifyActionValue{}
			if n, ok := e.Value.(*NotifyActionElement); ok {
				value.Value = n
			} else {
				continue
			}

			prev := e.Prev()
			next := e.Next()

			if prev != nil {
				if n, ok := prev.Value.(*NotifyActionElement); ok {
					value.Prev = n
				}
			}

			if next != nil {
				if n, ok := next.Value.(*NotifyActionElement); ok {
					value.Next = n
				}
			}

			data[id] = append(data[id], &value)
		}
	}

	return data
}

func (nf *NotifyAction) Len() int {
	return len(nf.records)
}

func (nf *NotifyAction) KeyLen(pid int64) int {
	if m, ok := nf.records[pid]; ok {
		return m.Len()
	}

	return 0
}

// NewNotifyAction 创建新的所有玩家的操作集合
func NewNotifyAction() *NotifyAction {
	return &NotifyAction{
		records: make(map[int64]*list.List),
	}
}
