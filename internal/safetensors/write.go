package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Tensor is one payload for Write.
type Tensor struct {
	Name  string
	DType DType
	Shape []int
	Data  []byte // raw little-endian
}

// Write writes a safetensors file. Blob order follows tensors; JSON keys are sorted.
func Write(path string, tensors []Tensor) error {
	if len(tensors) == 0 {
		return fmt.Errorf("no tensors")
	}
	metas := make(map[string]stJSON, len(tensors))
	var off int64
	for _, t := range tensors {
		if t.Name == "" || t.Name == "__metadata__" {
			return fmt.Errorf("invalid tensor name %q", t.Name)
		}
		if _, dup := metas[t.Name]; dup {
			return fmt.Errorf("duplicate tensor %q", t.Name)
		}
		es, err := t.DType.ElemSize()
		if err != nil {
			return err
		}
		n, err := numel(t.Shape)
		if err != nil {
			return err
		}
		want := n * int64(es)
		if int64(len(t.Data)) != want {
			return fmt.Errorf("payload size mismatch for %s: got %d want %d", t.Name, len(t.Data), want)
		}
		metas[t.Name] = stJSON{
			DType:       string(t.DType),
			Shape:       append([]int(nil), t.Shape...),
			DataOffsets: [2]int64{off, off + want},
		}
		off += want
	}
	js, err := marshalCompact(metas)
	if err != nil {
		return err
	}
	if len(js) < 2 || len(js) > maxHeader {
		if len(js) > maxHeader {
			return fmt.Errorf("header_too_large")
		}
		return fmt.Errorf("header_nbytes")
	}
	return writeAtomic(path, func(f *os.File) error {
		var nbuf [8]byte
		binary.LittleEndian.PutUint64(nbuf[:], uint64(len(js)))
		if _, err := f.Write(nbuf[:]); err != nil {
			return err
		}
		if _, err := f.Write(js); err != nil {
			return err
		}
		for _, t := range tensors {
			if _, err := f.Write(t.Data); err != nil {
				return err
			}
		}
		return nil
	})
}

// WriteF32 writes tensors as dtype F32 (chr decode and tests).
func WriteF32(path string, tensors []F32Tensor) error {
	st := make([]Tensor, len(tensors))
	for i, t := range tensors {
		n, err := numel(t.Shape)
		if err != nil {
			return err
		}
		if int64(len(t.Data)) != n {
			return fmt.Errorf("F32 count mismatch for %s: got %d want %d", t.Name, len(t.Data), n)
		}
		st[i] = Tensor{
			Name:  t.Name,
			DType: F32,
			Shape: append([]int(nil), t.Shape...),
			Data:  EncodeF32(t.Data),
		}
	}
	return Write(path, st)
}

func marshalCompact(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	return b, nil
}

func writeAtomic(path string, fn func(*os.File) error) error {
	dir := filepath.Dir(path)
	part := filepath.Join(dir, filepath.Base(path)+".part")
	f, err := os.OpenFile(part, os.O_CREATE|os.O_RDWR|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	remove := true
	defer func() {
		if f != nil {
			f.Close()
		}
		if remove {
			os.Remove(part)
		}
	}()
	if err := fn(f); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		f = nil
		return err
	}
	f = nil
	if err := os.Rename(part, path); err != nil {
		return err
	}
	remove = false
	return nil
}
