package model

// Vocab interns token strings to dense uint32 ids.
type Vocab struct {
	strToID map[string]uint32
	idToStr []string
}

func NewVocab() *Vocab {
	v := &Vocab{strToID: make(map[string]uint32)}
	v.idToStr = append(v.idToStr, "<pad>") // id 0 reserved
	v.strToID["<pad>"] = 0
	return v
}

func (v *Vocab) ID(s string) uint32 {
	if id, ok := v.strToID[s]; ok {
		return id
	}
	id := uint32(len(v.idToStr))
	v.strToID[s] = id
	v.idToStr = append(v.idToStr, s)
	return id
}

func (v *Vocab) Lookup(s string) (uint32, bool) {
	id, ok := v.strToID[s]
	return id, ok
}

func (v *Vocab) Str(id uint32) string {
	if int(id) >= len(v.idToStr) {
		return ""
	}
	return v.idToStr[id]
}

func (v *Vocab) Len() int { return len(v.idToStr) }

// Strings returns the id-ordered token table.
func (v *Vocab) Strings() []string { return v.idToStr }

func (v *Vocab) Intern(toks []string) []uint32 {
	out := make([]uint32, len(toks))
	for i, t := range toks {
		out[i] = v.ID(t)
	}
	return out
}

func (v *Vocab) InternLookup(toks []string) ([]uint32, bool) {
	out := make([]uint32, len(toks))
	for i, t := range toks {
		id, ok := v.strToID[t]
		if !ok {
			return nil, false
		}
		out[i] = id
	}
	return out, true
}
