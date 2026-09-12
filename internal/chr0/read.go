package chr0

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// File is an open .chr. The JSON header is in RAM; blobs are read with ReadAt.
type File struct {
	f   *os.File
	n   int64
	hdr Header
}

// Open parses the header and checks every blob range. Blobs are not loaded.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	ok := false
	defer func() {
		if !ok {
			f.Close()
		}
	}()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	size := st.Size()
	if size < 8 {
		return nil, fmt.Errorf("truncated")
	}
	var nbuf [8]byte
	if err := readAtFull(f, nbuf[:], 0); err != nil {
		return nil, err
	}
	ns := binary.LittleEndian.Uint64(nbuf[:])
	if ns > maxHeader {
		return nil, fmt.Errorf("header_too_large")
	}
	if ns == 0 || ns == 1 {
		return nil, fmt.Errorf("header_nbytes")
	}
	if int64(8+ns) > size {
		return nil, fmt.Errorf("header_nbytes")
	}
	js := make([]byte, ns)
	if err := readAtFull(f, js, 8); err != nil {
		return nil, err
	}
	h, err := parseHeaderJSON(js)
	if err != nil {
		return nil, err
	}
	if err := validateHeader(h); err != nil {
		return nil, err
	}
	if err := validateAgainstFile(h, int64(ns), size); err != nil {
		return nil, err
	}
	ok = true
	return &File{f: f, n: int64(ns), hdr: h}, nil
}

func (f *File) Close() error {
	if f == nil || f.f == nil {
		return nil
	}
	err := f.f.Close()
	f.f = nil
	return err
}

// Header returns the parsed JSON header.
func (f *File) Header() Header {
	return f.hdr
}

// ReadAt reads from the underlying file (offsets from file start).
func (f *File) ReadAt(p []byte, off int64) (int, error) {
	if f.f == nil {
		return 0, os.ErrClosed
	}
	return f.f.ReadAt(p, off)
}

// Get copies every blob of one tensor. It does not decode nf4/vq.
func (f *File) Get(name string) (Tensor, map[string][]byte, error) {
	t, ok := f.hdr.Tensors[name]
	if !ok {
		return Tensor{}, nil, fmt.Errorf("not_found: %s", name)
	}
	blobs := make(map[string][]byte)
	read := func(key string, span []int64) error {
		if len(span) != 2 {
			return nil
		}
		n := span[1] - span[0]
		buf := make([]byte, n)
		if err := readAtFull(f.f, buf, span[0]); err != nil {
			return err
		}
		blobs[key] = buf
		return nil
	}
	var err error
	switch t.Codec {
	case "bf16":
		err = read("data", t.Data)
	case "nf4":
		err = read("data", t.Data)
		if err == nil {
			err = read("scale", t.Scale)
		}
	case "int4":
		err = read("data", t.Data)
		if err == nil {
			err = read("scale", t.Scale)
		}
		if err == nil && len(t.Zero) == 2 {
			err = read("zero", t.Zero)
		}
	case "vq":
		err = read("codebook", t.Codebook)
		if err == nil {
			err = read("index", t.Index)
		}
	default:
		return t, nil, fmt.Errorf("unsupported codec %s (%s)", t.Codec, name)
	}
	if err != nil {
		return t, nil, err
	}
	return t, blobs, nil
}

func readAtFull(r io.ReaderAt, buf []byte, off int64) error {
	n, err := r.ReadAt(buf, off)
	if n == len(buf) {
		return nil
	}
	if err == nil || err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return fmt.Errorf("truncated: %w", err)
}
