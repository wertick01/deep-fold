package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

const maxHeader = 100_000_000

// TensorMeta describes one tensor in a safetensors file.
// Offsets are [start,end) relative to the start of the data section (byte 8+N).
type TensorMeta struct {
	Name    string
	DType   DType
	Shape   []int
	Offsets [2]int64
}

// File is an open safetensors shard. Reads use ReadAt; there is no mmap.
type File struct {
	f      *os.File
	data0  int64
	order  []TensorMeta
	byName map[string]int
}

type stJSON struct {
	DType       string   `json:"dtype"`
	Shape       []int    `json:"shape"`
	DataOffsets [2]int64 `json:"data_offsets"`
}

// Open opens a safetensors file and parses the header. Payloads are not loaded.
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
	if ns < 2 {
		return nil, fmt.Errorf("header_nbytes")
	}
	if int64(8+ns) > size {
		return nil, fmt.Errorf("header_nbytes")
	}
	js := make([]byte, ns)
	if err := readAtFull(f, js, 8); err != nil {
		return nil, err
	}
	if js[0] != '{' {
		return nil, fmt.Errorf("json_invalid")
	}
	order, err := parseSTHeader(js)
	if err != nil {
		return nil, err
	}
	data0 := int64(8 + ns)
	byName := make(map[string]int, len(order))
	for i, m := range order {
		if m.Offsets[0] < 0 || m.Offsets[1] < m.Offsets[0] {
			return nil, fmt.Errorf("bad data_offsets for %s", m.Name)
		}
		if data0+m.Offsets[1] > size {
			return nil, fmt.Errorf("truncated")
		}
		es, err := m.DType.ElemSize()
		if err != nil {
			return nil, err
		}
		n, err := numel(m.Shape)
		if err != nil {
			return nil, err
		}
		if m.Offsets[1]-m.Offsets[0] != n*int64(es) {
			return nil, fmt.Errorf("data_offsets size mismatch for %s", m.Name)
		}
		if len(m.Shape) > 2 {
			return nil, fmt.Errorf("rank must be 1 or 2")
		}
		byName[m.Name] = i
	}
	ok = true
	return &File{f: f, data0: data0, order: order, byName: byName}, nil
}

func (f *File) Close() error {
	if f == nil || f.f == nil {
		return nil
	}
	err := f.f.Close()
	f.f = nil
	return err
}

// List returns tensor metadata in JSON key order (as in the file).
func (f *File) List() []TensorMeta {
	out := make([]TensorMeta, len(f.order))
	for i, m := range f.order {
		out[i] = m
		out[i].Shape = append([]int(nil), m.Shape...)
	}
	return out
}

// Read copies one tensor's payload.
func (f *File) Read(name string) (TensorMeta, []byte, error) {
	i, ok := f.byName[name]
	if !ok {
		return TensorMeta{}, nil, fmt.Errorf("not_found: %s", name)
	}
	m := f.order[i]
	n := m.Offsets[1] - m.Offsets[0]
	buf := make([]byte, n)
	if err := readAtFull(f.f, buf, f.data0+m.Offsets[0]); err != nil {
		return m, nil, err
	}
	m.Shape = append([]int(nil), m.Shape...)
	return m, buf, nil
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

func parseSTHeader(js []byte) ([]TensorMeta, error) {
	dec := json.NewDecoder(bytes.NewReader(js))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, fmt.Errorf("json_invalid")
	}
	var order []TensorMeta
	seen := make(map[string]struct{})
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		key, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("json_invalid")
		}
		if key == "__metadata__" {
			var skip json.RawMessage
			if err := dec.Decode(&skip); err != nil {
				return nil, fmt.Errorf("json_invalid: %w", err)
			}
			continue
		}
		if _, dup := seen[key]; dup {
			return nil, fmt.Errorf("duplicate tensor %q", key)
		}
		dec.DisallowUnknownFields()
		var e stJSON
		if err := dec.Decode(&e); err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		dt := DType(e.DType)
		if _, err := dt.ElemSize(); err != nil {
			return nil, err
		}
		if len(e.Shape) == 0 || len(e.Shape) > 2 {
			return nil, fmt.Errorf("rank must be 1 or 2")
		}
		if _, err := numel(e.Shape); err != nil {
			return nil, err
		}
		if e.DataOffsets[0] < 0 || e.DataOffsets[1] < e.DataOffsets[0] {
			return nil, fmt.Errorf("bad data_offsets for %s", key)
		}
		order = append(order, TensorMeta{
			Name:    key,
			DType:   dt,
			Shape:   append([]int(nil), e.Shape...),
			Offsets: e.DataOffsets,
		})
		seen[key] = struct{}{}
	}
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if err := trailingJSON(js, dec.InputOffset()); err != nil {
		return nil, err
	}
	return order, nil
}

func trailingJSON(js []byte, off int64) error {
	if off < 0 || off > int64(len(js)) {
		return fmt.Errorf("json_invalid")
	}
	rest := js[off:]
	for _, c := range rest {
		if c != ' ' && c != '\t' && c != '\n' && c != '\r' {
			return fmt.Errorf("json_invalid: trailing data")
		}
	}
	return nil
}
