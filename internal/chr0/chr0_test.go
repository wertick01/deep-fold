package chr0

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func ip(n int) *int { return &n }

func TestVQSlotsRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "vq.chr")
	cb := bytes.Repeat([]byte{0xAB}, 8192)
	idx := []byte{1, 2, 3, 4}
	norm := []byte{0x80, 0x3f, 0x00, 0x40} // 1.0, 2.0 bf16
	items := []WriteTensor{
		{
			Name:     "model.layers.0.self_attn.q_proj",
			Tensor:   Tensor{Layer: ip(0), Kind: "q", Codec: "vq", Shape: []int{2, 8}},
			Codebook: cb,
			Index:    idx,
		},
		{
			Name:   "model.norm",
			Tensor: Tensor{Kind: "norm", Codec: "bf16", Shape: []int{2}},
			Data:   norm,
		},
	}
	if err := Write(path, Header{Arch: "toy", HiddenSize: 1, NumLayers: 1}, items); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, blobs, err := f.Get("model.layers.0.self_attn.q_proj")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(blobs["codebook"], cb) || !bytes.Equal(blobs["index"], idx) {
		t.Fatal("vq blobs")
	}
	info := f.Header().Tensors["model.layers.0.self_attn.q_proj"]
	if info.GroupSize != 8 || info.NCodebooks != 2 || info.CodebookBits != 8 {
		t.Fatalf("%+v", info)
	}
}

func TestDuplicateTensorJSON(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "dup.chr")
	js := `{"magic":"CHR0","version":1,"arch":"toy","hidden_size":1,"intermediate_size":0,"num_layers":0,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":{"a":{"kind":"other","codec":"bf16","shape":[1],"data":[64,66]},"a":{"kind":"other","codec":"bf16","shape":[1],"data":[64,66]}}}`
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.WriteString(js)
	if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	err := mustOpenErr(t, p)
	if !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("%v", err)
	}
}

func TestAlign64(t *testing.T) {
	if Align64(524) != 576 || Align64(576) != 576 || Align64(582) != 640 {
		t.Fatalf("align")
	}
	if Align64(0) != 0 || Align64(8) != 64 {
		t.Fatalf("align small")
	}
}

func TestContainerRoundtripBlobs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "toy.chr")
	q := bytes.Repeat([]byte{0xA5}, 64*64*2)
	down := bytes.Repeat([]byte{0xA5}, 32*128*2)
	norm := bytes.Repeat([]byte{0xA5}, 64*2)
	items := []WriteTensor{
		{
			Name:   "model.layers.0.self_attn.q_proj",
			Tensor: Tensor{Layer: ip(0), Kind: "q", Codec: "bf16", Shape: []int{64, 64}},
			Data:   q,
		},
		{
			Name:   "model.layers.0.mlp.down_proj",
			Tensor: Tensor{Layer: ip(0), Kind: "down", Codec: "bf16", Shape: []int{32, 128}},
			Data:   down,
		},
		{
			Name:   "model.norm",
			Tensor: Tensor{Kind: "norm", Codec: "bf16", Shape: []int{64}},
			Data:   norm,
		},
	}
	h := Header{
		Arch:             "toy",
		HiddenSize:       64,
		IntermediateSize: 128,
		NumLayers:        1,
		VocabSize:        0,
	}
	if err := Write(path, h, items); err != nil {
		t.Fatal(err)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	hdr := f.Header()
	if hdr.Magic != Magic || hdr.Version != Version {
		t.Fatalf("header %+v", hdr)
	}
	want := map[string][]byte{
		"model.layers.0.self_attn.q_proj": q,
		"model.layers.0.mlp.down_proj":    down,
		"model.norm":                      norm,
	}
	for name, payload := range want {
		info := hdr.Tensors[name]
		if len(info.Data) != 2 {
			t.Fatalf("%s offsets", name)
		}
		start, end := info.Data[0], info.Data[1]
		buf := make([]byte, end-start)
		n, err := f.ReadAt(buf, start)
		if err != nil || n != len(buf) {
			t.Fatalf("ReadAt %s: %v", name, err)
		}
		if !bytes.Equal(buf, payload) {
			t.Fatalf("blob %s", name)
		}
		buf2 := make([]byte, end-start)
		if _, err := f.ReadAt(buf2, start); err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(buf, buf2) {
			t.Fatalf("not idempotent %s", name)
		}
		_, blobs, err := f.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(blobs["data"], payload) {
			t.Fatalf("Get %s", name)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	N := binary.LittleEndian.Uint64(raw[:8])
	js := raw[8 : 8+N]
	if !bytes.Contains(js, []byte(`"layer":0`)) {
		t.Fatalf("layer:0 missing: %s", js)
	}
	if js[0] != '{' {
		t.Fatal("json start")
	}
}

func TestWriteMatchesSpecExampleNF4(t *testing.T) {
	const want = `{"magic":"CHR0","version":1,"arch":"toy","hidden_size":64,"intermediate_size":128,"num_layers":1,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":{"model.layers.0.mlp.down_proj":{"layer":0,"kind":"down","codec":"nf4","shape":[32,128],"group_size":64,"data":[2752,4800],"scale":[4800,4928]},"model.layers.0.self_attn.q_proj":{"layer":0,"kind":"q","codec":"nf4","shape":[64,64],"group_size":64,"data":[576,2624],"scale":[2624,2752]},"model.norm":{"kind":"norm","codec":"bf16","shape":[64],"data":[4928,5056]}}}`
	if len(want) != 516 {
		t.Fatalf("spec N=%d", len(want))
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "toy.chr")
	items := []WriteTensor{
		{
			Name:   "model.layers.0.self_attn.q_proj",
			Tensor: Tensor{Layer: ip(0), Kind: "q", Codec: "nf4", Shape: []int{64, 64}, GroupSize: 64},
			Data:   bytes.Repeat([]byte{0x11}, 2048),
			Scale:  bytes.Repeat([]byte{0x22}, 128),
		},
		{
			Name:   "model.layers.0.mlp.down_proj",
			Tensor: Tensor{Layer: ip(0), Kind: "down", Codec: "nf4", Shape: []int{32, 128}, GroupSize: 64},
			Data:   bytes.Repeat([]byte{0x33}, 2048),
			Scale:  bytes.Repeat([]byte{0x44}, 128),
		},
		{
			Name:   "model.norm",
			Tensor: Tensor{Kind: "norm", Codec: "bf16", Shape: []int{64}},
			Data:   bytes.Repeat([]byte{0xA5}, 128),
		},
	}
	h := Header{Arch: "toy", HiddenSize: 64, IntermediateSize: 128, NumLayers: 1}
	if err := Write(path, h, items); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	N := binary.LittleEndian.Uint64(raw[:8])
	got := string(raw[8 : 8+N])
	if got != want {
		t.Fatalf("json mismatch\n got %s\nwant %s", got, want)
	}
	if N != 516 {
		t.Fatalf("N=%d", N)
	}
	if binary.LittleEndian.Uint64(raw[:8]) != 0x204 {
		t.Fatalf("header nbytes bytes")
	}
	f, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, blobs, err := f.Get("model.norm")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(blobs["data"], bytes.Repeat([]byte{0xA5}, 128)) {
		t.Fatal("norm payload")
	}
}

func TestReaderRejects(t *testing.T) {
	dir := t.TempDir()
	t.Run("truncated", func(t *testing.T) {
		for _, n := range []int{0, 3, 7} {
			p := filepath.Join(dir, "t.bin")
			if err := os.WriteFile(p, make([]byte, n), 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := Open(p); err == nil || !strings.Contains(err.Error(), "truncated") {
				t.Fatalf("n=%d err=%v", n, err)
			}
		}
	})
	t.Run("header_nbytes", func(t *testing.T) {
		p := filepath.Join(dir, "n.bin")
		var buf bytes.Buffer
		_ = binary.Write(&buf, binary.LittleEndian, uint64(100))
		buf.WriteString("{")
		if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(p); err == nil || !strings.Contains(err.Error(), "header_nbytes") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("bom", func(t *testing.T) {
		p := filepath.Join(dir, "bom.chr")
		js := "\xef\xbb\xbf{}"
		var buf bytes.Buffer
		_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
		buf.WriteString(js)
		if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := Open(p); err == nil || !strings.Contains(err.Error(), "json_invalid") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("magic_version_tile", func(t *testing.T) {
		path := filepath.Join(dir, "ok.chr")
		mustWriteTiny(t, path)
		raw, _ := os.ReadFile(path)
		n := binary.LittleEndian.Uint64(raw[:8])
		js := append([]byte(nil), raw[8:8+n]...)

		badMagic := bytes.Replace(js, []byte(`"CHR0"`), []byte(`"chr0"`), 1)
		p := filepath.Join(dir, "magic.chr")
		writePatched(t, p, raw, badMagic)
		if _, err := Open(p); err == nil {
			t.Fatal("magic")
		}

		badVer := bytes.Replace(js, []byte(`"version":1`), []byte(`"version":2`), 1)
		p = filepath.Join(dir, "ver.chr")
		writePatched(t, p, raw, badVer)
		err := mustOpenErr(t, p)
		if !strings.Contains(err.Error(), "unsupported version") {
			t.Fatalf("%v", err)
		}

		badTile := bytes.Replace(js, []byte(`"row":64`), []byte(`"row":32`), 1)
		p = filepath.Join(dir, "tile.chr")
		writePatched(t, p, raw, badTile)
		if err := mustOpenErr(t, p); !strings.Contains(err.Error(), "tile") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("empty_tensors", func(t *testing.T) {
		js := `{"magic":"CHR0","version":1,"arch":"toy","hidden_size":64,"intermediate_size":0,"num_layers":0,"vocab_size":0,"tile":{"row":64,"col_group":8},"tensors":{}}`
		p := filepath.Join(dir, "empty.chr")
		var buf bytes.Buffer
		_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
		buf.WriteString(js)
		if err := os.WriteFile(p, buf.Bytes(), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := mustOpenErr(t, p); !strings.Contains(err.Error(), "no tensors") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("unaligned_start", func(t *testing.T) {
		path := filepath.Join(dir, "align.chr")
		mustWriteTiny(t, path)
		raw, _ := os.ReadFile(path)
		n := binary.LittleEndian.Uint64(raw[:8])
		js := raw[8 : 8+n]
		// data:[start,end] → nudge start by +1 if digits allow
		var hdr Header
		if err := json.Unmarshal(js, &hdr); err != nil {
			t.Fatal(err)
		}
		t0 := hdr.Tensors["n"]
		t0.Data[0]++
		t0.Data[1]++
		hdr.Tensors["n"] = t0
		js2, err := json.Marshal(hdr)
		if err != nil || len(js2) != int(n) {
			t.Fatalf("len %d vs %d", len(js2), n)
		}
		p := filepath.Join(dir, "unal.chr")
		writePatched(t, p, raw, js2)
		err = mustOpenErr(t, p)
		if err == nil || !strings.Contains(err.Error(), "64") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("norm_nf4", func(t *testing.T) {
		p := filepath.Join(dir, "normnf4.chr")
		data, scale := make([]byte, 64), make([]byte, 4)
		h := Header{
			Arch: "toy", HiddenSize: 1,
			Tensors: map[string]Tensor{
				"model.norm": {Kind: "norm", Codec: "nf4", Shape: []int{2, 64}, GroupSize: 64},
			},
		}
		writeRaw(t, p, h, []rawBlob{{"model.norm", "data", data}, {"model.norm", "scale", scale}})
		err := mustOpenErr(t, p)
		if !strings.Contains(err.Error(), "not allowed") && !strings.Contains(err.Error(), "codec") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("layer0_present", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "l.chr")
		items := []WriteTensor{{
			Name:   "model.layers.0.mlp.down_proj",
			Tensor: Tensor{Layer: ip(0), Kind: "down", Codec: "bf16", Shape: []int{2, 2}},
			Data:   bytes.Repeat([]byte{0xA5}, 8),
		}}
		if err := Write(path, Header{Arch: "toy", HiddenSize: 1, NumLayers: 1}, items); err != nil {
			t.Fatal(err)
		}
		raw, _ := os.ReadFile(path)
		N := binary.LittleEndian.Uint64(raw[:8])
		if !bytes.Contains(raw[8:8+N], []byte(`"layer":0`)) {
			t.Fatal("layer 0 omitted")
		}
	})
	t.Run("missing_layer", func(t *testing.T) {
		p := filepath.Join(dir, "nolayer.chr")
		h := Header{
			Arch: "toy", HiddenSize: 1, NumLayers: 1,
			Tensors: map[string]Tensor{
				"model.layers.0.mlp.down_proj": {Kind: "down", Codec: "bf16", Shape: []int{2, 2}},
			},
		}
		writeRaw(t, p, h, []rawBlob{{"model.layers.0.mlp.down_proj", "data", bytes.Repeat([]byte{0xA5}, 8)}})
		err := mustOpenErr(t, p)
		if !strings.Contains(err.Error(), "layer") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("mixed_codec", func(t *testing.T) {
		p := filepath.Join(dir, "mix.chr")
		q := bytes.Repeat([]byte{0xA5}, 8)
		d, sc := make([]byte, 64), make([]byte, 4)
		h := Header{
			Arch: "toy", HiddenSize: 1, NumLayers: 1,
			Tensors: map[string]Tensor{
				"model.layers.0.self_attn.q_proj": {Layer: ip(0), Kind: "q", Codec: "bf16", Shape: []int{2, 2}},
				"model.layers.0.mlp.down_proj":    {Layer: ip(0), Kind: "down", Codec: "nf4", Shape: []int{2, 64}, GroupSize: 64},
			},
		}
		writeRaw(t, p, h, []rawBlob{
			{"model.layers.0.self_attn.q_proj", "data", q},
			{"model.layers.0.mlp.down_proj", "data", d},
			{"model.layers.0.mlp.down_proj", "scale", sc},
		})
		err := mustOpenErr(t, p)
		if !strings.Contains(err.Error(), "mixed") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("overlap", func(t *testing.T) {
		p := filepath.Join(dir, "ov.chr")
		h := Header{
			Arch: "toy", HiddenSize: 1,
			Tensors: map[string]Tensor{
				"a": {Kind: "other", Codec: "bf16", Shape: []int{1}},
				"b": {Kind: "other", Codec: "bf16", Shape: []int{1}},
			},
		}
		writeRaw(t, p, h, []rawBlob{
			{"a", "data", []byte{0x80, 0x3f}},
			{"b", "data", []byte{0x00, 0x40}},
		})
		raw, err := os.ReadFile(p)
		if err != nil {
			t.Fatal(err)
		}
		n := binary.LittleEndian.Uint64(raw[:8])
		var hdr Header
		if err := json.Unmarshal(raw[8:8+n], &hdr); err != nil {
			t.Fatal(err)
		}
		a := hdr.Tensors["a"]
		b := hdr.Tensors["b"]
		b.Data = append([]int64(nil), a.Data...)
		hdr.Tensors["b"] = b
		js2, err := json.Marshal(hdr)
		if err != nil || len(js2) != int(n) {
			t.Fatalf("len %d vs %d", len(js2), n)
		}
		p2 := filepath.Join(dir, "ov2.chr")
		writePatched(t, p2, raw, js2)
		err = mustOpenErr(t, p2)
		if !strings.Contains(err.Error(), "overlap") {
			t.Fatalf("%v", err)
		}
	})
	t.Run("tail_ignored", func(t *testing.T) {
		path := filepath.Join(dir, "tail.chr")
		mustWriteTiny(t, path)
		raw, _ := os.ReadFile(path)
		raw = append(raw, 0xFF, 0xEE)
		p := filepath.Join(dir, "tail2.chr")
		if err := os.WriteFile(p, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		f, err := Open(p)
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
	})
}

func mustWriteTiny(t *testing.T, path string) {
	t.Helper()
	items := []WriteTensor{{
		Name:   "n",
		Tensor: Tensor{Kind: "other", Codec: "bf16", Shape: []int{1}},
		Data:   []byte{0x80, 0x3f},
	}}
	if err := Write(path, Header{Arch: "toy", HiddenSize: 1}, items); err != nil {
		t.Fatal(err)
	}
}

func writePatched(t *testing.T, path string, orig, js []byte) {
	t.Helper()
	n := binary.LittleEndian.Uint64(orig[:8])
	if uint64(len(js)) != n {
		t.Fatalf("patch len %d != %d", len(js), n)
	}
	out := append([]byte(nil), orig...)
	copy(out[8:], js)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

func mustOpenErr(t *testing.T, path string) error {
	t.Helper()
	f, err := Open(path)
	if err == nil {
		f.Close()
		t.Fatal("expected error")
	}
	return err
}

type rawBlob struct {
	name, field string
	data        []byte
}

func writeRaw(t *testing.T, path string, h Header, blobs []rawBlob) {
	t.Helper()
	h.Magic = Magic
	h.Version = Version
	h.Tile = Tile{Row: TileRow, ColGroup: TileCol}
	if h.Tensors == nil {
		h.Tensors = make(map[string]Tensor)
	}
	var n int
	var js []byte
	for iter := 0; iter < 8; iter++ {
		for name, tt := range h.Tensors {
			tt.Data, tt.Scale, tt.Zero, tt.Codebook, tt.Index = nil, nil, nil, nil, nil
			h.Tensors[name] = tt
		}
		pos := Align64(8 + int64(n))
		for _, b := range blobs {
			start := pos
			end := start + int64(len(b.data))
			tt := h.Tensors[b.name]
			span := []int64{start, end}
			switch b.field {
			case "data":
				tt.Data = span
			case "scale":
				tt.Scale = span
			case "zero":
				tt.Zero = span
			case "codebook":
				tt.Codebook = span
			case "index":
				tt.Index = span
			}
			h.Tensors[b.name] = tt
			pos = Align64(end)
		}
		var err error
		js, err = json.Marshal(h)
		if err != nil {
			t.Fatal(err)
		}
		if len(js) == n {
			break
		}
		n = len(js)
	}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, uint64(len(js)))
	buf.Write(js)
	pad := Align64(8+int64(len(js))) - (8 + int64(len(js)))
	buf.Write(make([]byte, pad))
	for _, b := range blobs {
		buf.Write(b.data)
		p := Align64(int64(len(b.data))) - int64(len(b.data))
		buf.Write(make([]byte, p))
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}
