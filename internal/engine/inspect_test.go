package engine

import (
	"fmt"
	"testing"
)

func TestInspectZeros(t *testing.T) {
	m, _, err := Load("../../bin/model.bin")
	if err != nil {
		t.Skip(err)
	}
	var zeros, neg int64
	for k := 1; k <= m.N; k++ {
		for _, tk := range m.Orders[k].Toks {
			if tk == 0 {
				zeros++
			}
			if tk < 0 {
				neg++
			}
		}
	}
	fmt.Printf("zero toks=%d neg=%d\n", zeros, neg)
	// scan for a row containing 0 and print its key + context
	for k := 1; k <= m.N; k++ {
		o := &m.Orders[k]
		for r := 0; r < len(o.Keys); r++ {
			row := o.Toks[o.Off[r]:o.Off[r+1]]
			for _, tk := range row {
				if tk <= 0 {
					fmt.Printf("order %d key %x row %v\n", k, o.Keys[r], row)
					goto next
				}
			}
		}
	next:
	}
}
