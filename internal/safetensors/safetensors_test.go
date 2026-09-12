package safetensors

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestWriteReadF32BF16(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "toy.safetensors")
	f32 := EncodeF32([]float32{1, 2, 3, 4})
	bf16 := []byte{0x80, 0x3f} // 1.0
	tensors := []Tensor{
		{Name: "a", DType: F32, Shape: []int{2, 2}, Data: f32},
		{Name: "b", DType: BF16, Shape: []int{1}, Data: bf16},
	}
	if err := Write(path, tensors); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	list := f.List()
	if len(list) != 2 {
		t.Fatalf("list %d", len(list))
	}
	if list[0].Name != "a" || list[1].Name != "b" {
		t.Fatalf("json key order: %+v", list)
	}
	ma, rawA, err := f.Read("a")
	if err != nil {
		t.Fatal(err)
	}
	if ma.DType != F32 || !bytes.Equal(rawA, f32) {
		t.Fatalf("a mismatch")
	}
	got, err := ToF32(F32, rawA)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 1 || got[3] != 4 {
		t.Fatalf("f32 %v", got)
	}
	mb, rawB, err := f.Read("b")
	if err != nil {
		t.Fatal(err)
	}
	if mb.DType != BF16 || !bytes.Equal(rawB, bf16) {
		t.Fatalf("b mismatch")
	}
}

func TestRoundtripBytes(t *testing.T) {
	dir := t.TempDir()
	p1 := filepath.Join(dir, "a.safetensors")
	p2 := filepath.Join(dir, "b.safetensors")
	tensors := []Tensor{
		{Name: "a", DType: F32, Shape: []int{1, 2}, Data: EncodeF32([]float32{0.5, -2})},
		{Name: "z", DType: BF16, Shape: []int{2}, Data: []byte{0x80, 0x3f, 0x00, 0x00}},
	}
	if err := Write(p1, tensors); err != nil {
		t.Fatal(err)
	}
	b1, err := os.ReadFile(p1)
	if err != nil {
		t.Fatal(err)
	}
	f, err := Open(p1)
	if err != nil {
		t.Fatal(err)
	}
	var again []Tensor
	for _, m := range f.List() {
		_, raw, err := f.Read(m.Name)
		if err != nil {
			t.Fatal(err)
		}
		again = append(again, Tensor{Name: m.Name, DType: m.DType, Shape: m.Shape, Data: raw})
	}
	f.Close()
	if err := Write(p2, again); err != nil {
		t.Fatal(err)
	}
	b2, err := os.ReadFile(p2)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("file bytes differ")
	}
}

func TestToF32BF16One(t *testing.T) {
	raw := []byte{0x80, 0x3f} // bits 0x3F80 LE
	got, err := ToF32(BF16, raw)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0] != 1 {
		t.Fatalf("bf16 1.0 got %v", got)
	}
}

func TestWriteF32(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "f32.safetensors")
	if err := WriteF32(path, []F32Tensor{{
		Name:  "w",
		Shape: []int{2, 1},
		Data:  []float32{3, 4},
	}}); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	m, raw, err := f.Read("w")
	if err != nil {
		t.Fatal(err)
	}
	if m.DType != F32 {
		t.Fatalf("dtype %s", m.DType)
	}
	got, err := ToF32(m.DType, raw)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 3 || got[1] != 4 {
		t.Fatalf("%v", got)
	}
}

func TestUnsupportedDtype(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.safetensors")
	js := `{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}`
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.WriteString(js)
	buf.Write(make([]byte, 8))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error")
	}
	if !bytes.Contains([]byte(err.Error()), []byte("unsupported dtype")) {
		t.Fatalf("err %v", err)
	}
}

func TestResolveInputSingleAndIndex(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a.safetensors")
	b := filepath.Join(dir, "b.safetensors")
	if err := Write(a, []Tensor{{Name: "t1", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{1})}}); err != nil {
		t.Fatal(err)
	}
	if err := Write(b, []Tensor{
		{Name: "t2", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{2})},
		{Name: "extra", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{9})},
	}); err != nil {
		t.Fatal(err)
	}

	r, err := ResolveInput(a)
	if err != nil {
		t.Fatal(err)
	}
	if r.Single == "" || len(r.Shards) != 1 {
		t.Fatalf("single %+v", r)
	}

	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	idx := `{"metadata":{"total_size":8},"weight_map":{"t2":"b.safetensors","t1":"a.safetensors"}}`
	if err := os.WriteFile(idxPath, []byte(idx), 0o644); err != nil {
		t.Fatal(err)
	}
	rd, err := ResolveInput(dir)
	if err != nil {
		t.Fatal(err)
	}
	if rd.WeightMap["t1"] != "a.safetensors" {
		t.Fatalf("map %+v", rd.WeightMap)
	}
	if len(rd.Shards) != 2 {
		t.Fatalf("shards %v", rd.Shards)
	}
	if filepath.Base(rd.Shards[0]) != "a.safetensors" || filepath.Base(rd.Shards[1]) != "b.safetensors" {
		t.Fatalf("shard order %v", rd.Shards)
	}

	var names []string
	if err := rd.ForEachTensor(func(hfName string, f *File, meta TensorMeta) error {
		names = append(names, hfName)
		_, raw, err := f.Read(hfName)
		if err != nil {
			return err
		}
		if len(raw) != 4 {
			t.Fatalf("raw %s", hfName)
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 || names[0] != "t1" || names[1] != "t2" {
		t.Fatalf("iter %v", names)
	}

	ri, err := ResolveInput(idxPath)
	if err != nil {
		t.Fatal(err)
	}
	if ri.WeightMap == nil {
		t.Fatal("index file")
	}
}

func TestResolveInputModelSafetensors(t *testing.T) {
	dir := t.TempDir()
	if err := Write(filepath.Join(dir, "model.safetensors"), []Tensor{{Name: "m", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{1})}}); err != nil {
		t.Fatal(err)
	}
	if err := Write(filepath.Join(dir, "other.safetensors"), []Tensor{{Name: "o", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{2})}}); err != nil {
		t.Fatal(err)
	}
	r, err := ResolveInput(dir)
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(r.Single) != "model.safetensors" {
		t.Fatalf("got %s", r.Single)
	}
}

func TestIndexEscape(t *testing.T) {
	dir := t.TempDir()
	js := `{"weight_map":{"t":"../escape.safetensors"}}`
	p := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(p, []byte(js), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err := ResolveInput(p)
	if err == nil {
		t.Fatal("expected escape error")
	}
}

func TestDuplicateWeightMap(t *testing.T) {
	js := []byte(`{"weight_map":{"a":"x.safetensors","a":"y.safetensors"}}`)
	_, err := ParseIndex(js)
	if err == nil {
		t.Fatal("expected duplicate")
	}
}

func TestToF32F16One(t *testing.T) {
	raw := []byte{0x00, 0x3c} // binary16 1.0
	got, err := ToF32(F16, raw)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 1 {
		t.Fatalf("f16 1.0 got %v", got)
	}
}

func TestListSkipsMetadata(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "m.safetensors")
	js := `{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.WriteString(js)
	buf.Write(EncodeF32([]float32{1}))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	list := f.List()
	if len(list) != 1 || list[0].Name != "w" {
		t.Fatalf("%+v", list)
	}
}

func TestJSONKeyOrderPreserved(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "order.safetensors")
	js := `{"z":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.WriteString(js)
	buf.Write(EncodeF32([]float32{1, 2}))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	list := f.List()
	if len(list) != 2 || list[0].Name != "z" || list[1].Name != "a" {
		t.Fatalf("order %+v", list)
	}
	_, raw, err := f.Read("a")
	if err != nil {
		t.Fatal(err)
	}
	got, err := ToF32(F32, raw)
	if err != nil {
		t.Fatal(err)
	}
	if got[0] != 2 {
		t.Fatalf("offsets not relative to data section: %v", got)
	}
}

func TestDuplicateSTKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "dup.safetensors")
	js := `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"a":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}`
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.WriteString(js)
	buf.Write(EncodeF32([]float32{1, 2}))
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil {
		t.Fatal("expected duplicate")
	}
}

func TestMissingMappedTensor(t *testing.T) {
	dir := t.TempDir()
	a := filepath.Join(dir, "a.safetensors")
	if err := Write(a, []Tensor{{Name: "t1", DType: F32, Shape: []int{1}, Data: EncodeF32([]float32{1})}}); err != nil {
		t.Fatal(err)
	}
	idxPath := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(idxPath, []byte(`{"weight_map":{"t1":"a.safetensors","ghost":"a.safetensors"}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	r, err := ResolveInput(dir)
	if err != nil {
		t.Fatal(err)
	}
	err = r.ForEachTensor(func(string, *File, TensorMeta) error { return nil })
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("missing")) {
		t.Fatalf("%v", err)
	}
}
