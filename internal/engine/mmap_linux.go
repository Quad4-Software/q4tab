//go:build linux

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
	// The access pattern is binary-search probes scattered across the
	// file. Without MADV_RANDOM the kernel readahead faults ~128KB per
	// probe and hundreds of MB go resident for no benefit.
	syscall.Madvise(b, syscall.MADV_RANDOM)
	return b, true, nil
}
