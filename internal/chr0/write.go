package chr0

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

type blobRef struct {
	item  int
	field string
	data  []byte
}

// Write writes a .chr with two-pass metadata (stabilize JSON, then blobs).
// Blob layout follows items order; JSON tensor keys are sorted by encoding/json.
func Write(path string, h Header, items []WriteTensor) error {
	if len(items) == 0 {
		return fmt.Errorf("no tensors")
	}
	h.Magic = Magic
	h.Version = Version
	h.Tile = Tile{Row: TileRow, ColGroup: TileCol}
	if h.Arch == "" {
		h.Arch = "unknown"
	}

	local := make([]WriteTensor, len(items))
	copy(local, items)
	items = local

	names := make(map[string]struct{}, len(items))
	plan := make([]blobRef, 0, len(items))
	tensors := make(map[string]Tensor, len(items))
	for i, it := range items {
		if err := checkName(it.Name); err != nil {
			return err
		}
		if _, dup := names[it.Name]; dup {
			return fmt.Errorf("duplicate tensor name %q", it.Name)
		}
		names[it.Name] = struct{}{}
		t := it.Tensor
		if t.Codec == "int4" {
			return fmt.Errorf("writer does not create codec=int4")
		}
		switch t.Codec {
		case "nf4":
			if t.GroupSize == 0 {
				t.GroupSize = 64
			}
		case "vq":
			if t.GroupSize == 0 {
				t.GroupSize = 8
			}
			if t.NCodebooks == 0 {
				t.NCodebooks = 2
			}
			if t.CodebookBits == 0 {
				t.CodebookBits = 8
			}
		}
		t.Shape = append([]int(nil), t.Shape...)
		it.Tensor = t
		items[i] = it
		blobs, err := expectedBlobs(it)
		if err != nil {
			return err
		}
		t.Data, t.Scale, t.Zero, t.Codebook, t.Index = nil, nil, nil, nil, nil
		tensors[it.Name] = t
		for _, b := range blobs {
			plan = append(plan, blobRef{item: i, field: b.field, data: b.data})
		}
	}

	js, err := stabilize(h, items, tensors, plan)
	if err != nil {
		return err
	}
	h.Tensors = tensors
	if err := validateHeader(h); err != nil {
		return err
	}
	n := int64(len(js))
	var maxEnd int64
	for _, t := range h.Tensors {
		for _, sp := range tensorSpans("", t) {
			if sp.end > maxEnd {
				maxEnd = sp.end
			}
		}
	}
	fileSize := Align64(maxEnd)
	if err := validateAgainstFile(h, n, fileSize); err != nil {
		return err
	}

	return writeAtomic(path, func(f *os.File) error {
		var nbuf [8]byte
		binary.LittleEndian.PutUint64(nbuf[:], uint64(n))
		if _, err := f.Write(nbuf[:]); err != nil {
			return err
		}
		if _, err := f.Write(js); err != nil {
			return err
		}
		pad := Align64(8+n) - (8 + n)
		if pad > 0 {
			if _, err := f.Write(make([]byte, pad)); err != nil {
				return err
			}
		}
		for _, b := range plan {
			if _, err := f.Write(b.data); err != nil {
				return err
			}
			// start is 64-aligned, so pad after a blob of n bytes is Align64(n)-n.
			posPad := Align64(int64(len(b.data))) - int64(len(b.data))
			if posPad > 0 {
				if _, err := f.Write(make([]byte, posPad)); err != nil {
					return err
				}
			}
		}
		off, err := f.Seek(0, io.SeekCurrent)
		if err != nil {
			return err
		}
		if off != fileSize {
			return fmt.Errorf("internal size mismatch: got %d want %d", off, fileSize)
		}
		return nil
	})
}

type namedBlob struct {
	field string
	data  []byte
}

func expectedBlobs(it WriteTensor) ([]namedBlob, error) {
	t := it.Tensor
	switch t.Codec {
	case "bf16":
		want, err := BF16BlobBytes(t.Shape)
		if err != nil {
			return nil, err
		}
		if int64(len(it.Data)) != want {
			return nil, fmt.Errorf("%s: data size got %d want %d", it.Name, len(it.Data), want)
		}
		if len(it.Scale)+len(it.Zero)+len(it.Codebook)+len(it.Index) != 0 {
			return nil, fmt.Errorf("%s: extra blobs", it.Name)
		}
		return []namedBlob{{"data", it.Data}}, nil
	case "nf4":
		d, sc, err := NF4BlobBytes(t.Shape)
		if err != nil {
			return nil, err
		}
		if int64(len(it.Data)) != d || int64(len(it.Scale)) != sc {
			return nil, fmt.Errorf("%s: nf4 blob size", it.Name)
		}
		if len(it.Zero)+len(it.Codebook)+len(it.Index) != 0 {
			return nil, fmt.Errorf("%s: extra blobs", it.Name)
		}
		if t.GroupSize != 64 {
			return nil, fmt.Errorf("%s: group_size must be 64", it.Name)
		}
		return []namedBlob{{"data", it.Data}, {"scale", it.Scale}}, nil
	case "vq":
		cb, idx, err := VQBlobBytes(t.Shape)
		if err != nil {
			return nil, err
		}
		if int64(len(it.Codebook)) != cb || int64(len(it.Index)) != idx {
			return nil, fmt.Errorf("%s: vq blob size", it.Name)
		}
		if len(it.Data)+len(it.Scale)+len(it.Zero) != 0 {
			return nil, fmt.Errorf("%s: extra blobs", it.Name)
		}
		return []namedBlob{{"codebook", it.Codebook}, {"index", it.Index}}, nil
	default:
		return nil, fmt.Errorf("unsupported codec %s (%s)", t.Codec, it.Name)
	}
}

func stabilize(h Header, items []WriteTensor, tensors map[string]Tensor, plan []blobRef) ([]byte, error) {
	var n int64
	var js []byte
	for iter := 0; iter < 8; iter++ {
		// clear spans
		for name, t := range tensors {
			t.Data, t.Scale, t.Zero, t.Codebook, t.Index = nil, nil, nil, nil, nil
			tensors[name] = t
		}
		pos := Align64(8 + n)
		for _, b := range plan {
			start := pos
			end := start + int64(len(b.data))
			name := items[b.item].Name
			t := tensors[name]
			span := []int64{start, end}
			switch b.field {
			case "data":
				t.Data = span
			case "scale":
				t.Scale = span
			case "zero":
				t.Zero = span
			case "codebook":
				t.Codebook = span
			case "index":
				t.Index = span
			}
			tensors[name] = t
			pos = Align64(end)
		}
		h.Tensors = tensors
		var err error
		js, err = marshalCompact(h)
		if err != nil {
			return nil, err
		}
		if int64(len(js)) == n {
			return js, nil
		}
		n = int64(len(js))
		if n > maxHeader {
			return nil, fmt.Errorf("header_too_large")
		}
	}
	return nil, fmt.Errorf("header_not_stable")
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
