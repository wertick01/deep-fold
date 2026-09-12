package verify

import (
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"chr/internal/chr0"
	"chr/internal/f16"
	"chr/internal/safetensors"
)

func fillMat(nOut, nIn int) []float32 {
	w := make([]float32, nOut*nIn)
	for i := 0; i < nOut; i++ {
		for j := 0; j < nIn; j++ {
			w[i*nIn+j] = float32((17*i+13*j)%100)/50 - 1
		}
	}
	return w
}

func bf16Payload(v []float32) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[i*2:], f16.ToBF16Bits(x))
	}
	return b
}

func writeBF16ST(t *testing.T, path string, tensors []safetensors.Tensor) {
	t.Helper()
	if err := safetensors.Write(path, tensors); err != nil {
		t.Fatal(err)
	}
}

func fakeThree(t *testing.T, dir string) string {
	t.Helper()
	q := fillMat(64, 64)
	down := fillMat(32, 128)
	norm := make([]float32, 64)
	for i := range norm {
		norm[i] = 1
	}
	p := filepath.Join(dir, "model.safetensors")
	writeBF16ST(t, p, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{64, 64}, Data: bf16Payload(q)},
		{Name: "model.layers.0.mlp.down_proj.weight", DType: safetensors.BF16, Shape: []int{32, 128}, Data: bf16Payload(down)},
		{Name: "model.norm.weight", DType: safetensors.BF16, Shape: []int{64}, Data: bf16Payload(norm)},
	})
	return p
}

func TestFakeModelThreeTensors(t *testing.T) {
	dir := t.TempDir()
	orig := fakeThree(t, dir)
	for _, codec := range []string{"nf4", "vq"} {
		chrPath := filepath.Join(dir, codec+".chr")
		if err := Compress(CompressOptions{
			In: orig, Out: chrPath, Codec: codec, Iters: 20, Chunk: 256, Quiet: true,
		}); err != nil {
			t.Fatalf("%s compress: %v", codec, err)
		}
		cf, err := chr0.Open(chrPath)
		if err != nil {
			t.Fatal(err)
		}
		h := cf.Header()
		if len(h.Tensors) != 3 {
			t.Fatalf("%s tensors %d", codec, len(h.Tensors))
		}
		if h.Tensors["model.layers.0.self_attn.q_proj"].Kind != "q" {
			t.Fatal("kind q")
		}
		if h.Tensors["model.layers.0.mlp.down_proj"].Kind != "down" {
			t.Fatal("kind down")
		}
		if h.Tensors["model.norm"].Codec != "bf16" {
			t.Fatal("norm codec")
		}
		if h.Tensors["model.layers.0.self_attn.q_proj"].Codec != codec {
			t.Fatal("linear codec")
		}
		cf.Close()

		dec := filepath.Join(dir, codec+".safetensors")
		if err := Decode(chrPath, dec); err != nil {
			t.Fatal(err)
		}
		sf, err := safetensors.Open(dec)
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range sf.List() {
			if m.DType != safetensors.F32 {
				t.Fatalf("decode dtype %s", m.DType)
			}
		}
		sf.Close()

		r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, JSON: true, Quiet: true, Stdout: ioDiscard{}})
		if r.ExitCode != 0 {
			t.Fatalf("%s verify %d err=%v fail=%v", codec, r.ExitCode, r.Err, r.Report.Failed)
		}
		if r.Report.Summary.Tensors != 3 || r.Report.Summary.BF16 != 1 || r.Report.Summary.Lossy != 2 {
			t.Fatalf("summary %+v", r.Report.Summary)
		}
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }

func TestVerifyFailCode(t *testing.T) {
	dir := t.TempDir()
	w := fillMat(8, 64)
	orig := filepath.Join(dir, "o.safetensors")
	writeBF16ST(t, orig, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{8, 64}, Data: bf16Payload(w)},
	})
	chrPath := filepath.Join(dir, "m.chr")
	if err := Compress(CompressOptions{In: orig, Out: chrPath, Codec: "nf4", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	tiny := 1e-12
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, FailRMSE: &tiny, FailMax: &tiny, Stdout: ioDiscard{}})
	if r.ExitCode != 2 || r.Report.Summary.Fail < 1 {
		t.Fatalf("want 2 got %d err=%v", r.ExitCode, r.Err)
	}
	r2 := Verify(VerifyOptions{Orig: orig, CHR: chrPath, Stdout: ioDiscard{}})
	if r2.ExitCode != 0 {
		t.Fatalf("default want 0 got %d %v", r2.ExitCode, r2.Report.Failed)
	}
}

func TestPad(t *testing.T) {
	dir := t.TempDir()
	nf := fillMat(4, 100)
	orig := filepath.Join(dir, "p.safetensors")
	writeBF16ST(t, orig, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{4, 100}, Data: bf16Payload(nf)},
	})
	chrPath := filepath.Join(dir, "p.chr")
	if err := Compress(CompressOptions{In: orig, Out: chrPath, Codec: "nf4", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	cf, err := chr0.Open(chrPath)
	if err != nil {
		t.Fatal(err)
	}
	sh := cf.Header().Tensors["model.layers.0.self_attn.q_proj"].Shape
	if sh[0] != 4 || sh[1] != 100 {
		t.Fatalf("shape %v", sh)
	}
	cf.Close()
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, Stdout: ioDiscard{}})
	if r.ExitCode != 0 {
		t.Fatalf("verify %d %v", r.ExitCode, r.Err)
	}
	if r.Report.Tensors[0].N != 400 {
		t.Fatalf("n=%d", r.Report.Tensors[0].N)
	}

	vqW := fillMat(2, 12)
	orig2 := filepath.Join(dir, "v.safetensors")
	writeBF16ST(t, orig2, []safetensors.Tensor{
		{Name: "model.layers.0.mlp.down_proj.weight", DType: safetensors.BF16, Shape: []int{2, 12}, Data: bf16Payload(vqW)},
	})
	chr2 := filepath.Join(dir, "v.chr")
	if err := Compress(CompressOptions{In: orig2, Out: chr2, Codec: "vq", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	r = Verify(VerifyOptions{Orig: orig2, CHR: chrPath, Stdout: ioDiscard{}})
	// wrong chr on purpose skipped — verify matching pair
	r = Verify(VerifyOptions{Orig: orig2, CHR: chr2, Stdout: ioDiscard{}})
	if r.ExitCode != 0 {
		t.Fatalf("vq pad verify %d %v", r.ExitCode, r.Err)
	}
	if r.Report.Tensors[0].N != 24 {
		t.Fatalf("vq n=%d", r.Report.Tensors[0].N)
	}
}

func TestVerifyMissingCompressable(t *testing.T) {
	dir := t.TempDir()
	orig := fakeThree(t, dir)
	norm := make([]float32, 64)
	for i := range norm {
		norm[i] = 1
	}
	chrPath := filepath.Join(dir, "partial.chr")
	items := []chr0.WriteTensor{
		{
			Name:   "model.layers.0.self_attn.q_proj",
			Tensor: chr0.Tensor{Kind: "q", Codec: "bf16", Shape: []int{64, 64}, Layer: intPtr(0)},
			Data:   bf16Payload(fillMat(64, 64)),
		},
		{
			Name:   "model.norm",
			Tensor: chr0.Tensor{Kind: "norm", Codec: "bf16", Shape: []int{64}},
			Data:   bf16Payload(norm),
		},
	}
	h := chr0.Header{Arch: "toy", HiddenSize: 64, IntermediateSize: 128, NumLayers: 1}
	if err := chr0.Write(chrPath, h, items); err != nil {
		t.Fatal(err)
	}
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, Stdout: ioDiscard{}})
	if r.ExitCode != 1 || r.Err == nil || !strings.Contains(r.Err.Error(), "missing tensor in chr: model.layers.0.mlp.down_proj") {
		t.Fatalf("got %d %v", r.ExitCode, r.Err)
	}
}

func TestVerifyExtraTensor(t *testing.T) {
	dir := t.TempDir()
	norm := make([]float32, 64)
	for i := range norm {
		norm[i] = 1
	}
	orig := filepath.Join(dir, "n.safetensors")
	writeBF16ST(t, orig, []safetensors.Tensor{
		{Name: "model.norm.weight", DType: safetensors.BF16, Shape: []int{64}, Data: bf16Payload(norm)},
	})
	chrPath := filepath.Join(dir, "e.chr")
	items := []chr0.WriteTensor{
		{
			Name:   "model.layers.0.self_attn.q_proj",
			Tensor: chr0.Tensor{Kind: "q", Codec: "bf16", Shape: []int{8, 64}, Layer: intPtr(0)},
			Data:   bf16Payload(fillMat(8, 64)),
		},
		{
			Name:   "model.norm",
			Tensor: chr0.Tensor{Kind: "norm", Codec: "bf16", Shape: []int{64}},
			Data:   bf16Payload(norm),
		},
	}
	h := chr0.Header{Arch: "toy", HiddenSize: 64, NumLayers: 1}
	if err := chr0.Write(chrPath, h, items); err != nil {
		t.Fatal(err)
	}
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, Stdout: ioDiscard{}})
	if r.ExitCode != 1 || r.Err == nil || !strings.Contains(r.Err.Error(), "extra tensor in chr:") {
		t.Fatalf("got %d %v", r.ExitCode, r.Err)
	}
}

func TestVerifySkipInvFreq(t *testing.T) {
	dir := t.TempDir()
	q := fillMat(8, 64)
	inv := []float32{1, 2, 3, 4}
	orig := filepath.Join(dir, "i.safetensors")
	writeBF16ST(t, orig, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{8, 64}, Data: bf16Payload(q)},
		{Name: "model.layers.0.self_attn.rotary_emb.inv_freq", DType: safetensors.BF16, Shape: []int{4}, Data: bf16Payload(inv)},
	})
	chrPath := filepath.Join(dir, "i.chr")
	if err := Compress(CompressOptions{In: orig, Out: chrPath, Codec: "nf4", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	cf, err := chr0.Open(chrPath)
	if err != nil {
		t.Fatal(err)
	}
	for name := range cf.Header().Tensors {
		if strings.Contains(name, "inv_freq") {
			t.Fatalf("inv_freq stored: %s", name)
		}
	}
	cf.Close()
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, Stdout: ioDiscard{}})
	if r.ExitCode != 0 {
		t.Fatalf("verify %d %v", r.ExitCode, r.Err)
	}
	if r.Report.Summary.Skipped != 1 {
		t.Fatalf("skipped=%d", r.Report.Summary.Skipped)
	}
}

func TestDecodeWritesF32(t *testing.T) {
	dir := t.TempDir()
	orig := fakeThree(t, dir)
	chrPath := filepath.Join(dir, "d.chr")
	if err := Compress(CompressOptions{In: orig, Out: chrPath, Codec: "nf4", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "out.safetensors")
	if err := Decode(chrPath, out); err != nil {
		t.Fatal(err)
	}
	sf, err := safetensors.Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer sf.Close()
	if len(sf.List()) != 3 {
		t.Fatal("count")
	}
	for _, m := range sf.List() {
		if m.DType != safetensors.F32 {
			t.Fatalf("%s %s", m.Name, m.DType)
		}
	}
}

func TestVerifyJSON(t *testing.T) {
	dir := t.TempDir()
	orig := fakeThree(t, dir)
	chrPath := filepath.Join(dir, "j.chr")
	if err := Compress(CompressOptions{In: orig, Out: chrPath, Codec: "nf4", Iters: 20, Chunk: 256, Quiet: true}); err != nil {
		t.Fatal(err)
	}
	var buf strings.Builder
	r := Verify(VerifyOptions{Orig: orig, CHR: chrPath, JSON: true, Stdout: &buf})
	if r.ExitCode != 0 {
		t.Fatal(r.Err)
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(buf.String()), &obj); err != nil {
		t.Fatal(err, buf.String())
	}
	if obj["ok"] != true {
		t.Fatal(obj)
	}
}

func intPtr(n int) *int { return &n }

func TestClassifyInvFreq(t *testing.T) {
	_, c := Classify("model.layers.0.self_attn.rotary_emb.inv_freq")
	if c != ClassSkip {
		t.Fatal(c)
	}
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
