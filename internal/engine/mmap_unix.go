//go:build unix && !linux

package engine

import (
	"io"
	"os"
	"syscall"
)

func mapFile(path string) (data []byte, mapped bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if fi.Size() < 4096 {
		b, err := io.ReadAll(f)
		return b, false, err
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		f.Seek(0, 0)
		b, err2 := io.ReadAll(f)
		return b, false, err2
	}
	return b, true, nil
}

func madviseDontNeed(b []byte) {
	syscall.Madvise(b, syscall.MADV_DONTNEED)
}
