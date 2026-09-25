//go:build !unix

package engine

import "os"

func mapFile(path string) (data []byte, mapped bool, err error) {
	b, err := os.ReadFile(path)
	return b, false, err
}

// madviseDontNeed is a no-op where madvise is unavailable.
func madviseDontNeed([]byte) {}
