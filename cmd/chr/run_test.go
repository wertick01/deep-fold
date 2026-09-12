package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"chr/internal/f16"
	"chr/internal/safetensors"
	"encoding/binary"
)

func TestRunCompressRequiresCodec(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "m.safetensors")
	writeTiny(t, in)
	out := filepath.Join(dir, "x.chr")
	code := run([]string{"compress", "--in", in, "--out", out}, ioDiscard{}, ioDiscard{})
	if code != 1 {
		t.Fatalf("code %d", code)
	}
}

func TestRunHelpExitZero(t *testing.T) {
	if run([]string{"-h"}, ioDiscard{}, ioDiscard{}) != 0 {
		t.Fatal("root help")
	}
	if run([]string{"verify", "-h"}, ioDiscard{}, ioDiscard{}) != 0 {
		t.Fatal("verify help")
	}
}

func TestRunVerifyExitCodes(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "m.safetensors")
	writeTiny(t, in)
	chrPath := filepath.Join(dir, "m.chr")
	var errBuf bytes.Buffer
	if c := run([]string{"compress", "--in", in, "--out", chrPath, "--codec", "nf4", "--quiet"}, ioDiscard{}, &errBuf); c != 0 {
		t.Fatalf("compress %d %s", c, errBuf.String())
	}
	var out bytes.Buffer
	c := run([]string{"verify", "--orig", in, "--chr", chrPath, "--fail-rmse", "1e-12", "--fail-maxabs", "1e-12"}, &out, ioDiscard{})
	if c != 2 {
		t.Fatalf("want 2 got %d %s", c, out.String())
	}
	if !strings.Contains(out.String(), "FAIL") {
		t.Fatalf("stdout %q", out.String())
	}
	c = run([]string{"verify", "--orig", filepath.Join(dir, "nope"), "--chr", chrPath}, ioDiscard{}, &errBuf)
	if c != 1 {
		t.Fatalf("missing orig want 1 got %d", c)
	}
}

func TestRunFakeRoundtrip(t *testing.T) {
	dir := t.TempDir()
	in := filepath.Join(dir, "m.safetensors")
	writeThree(t, in)
	chrPath := filepath.Join(dir, "m.chr")
	if c := run([]string{"compress", "--in", in, "--out", chrPath, "--codec", "nf4", "--quiet"}, ioDiscard{}, ioDiscard{}); c != 0 {
		t.Fatal(c)
	}
	if c := run([]string{"verify", "--orig", in, "--chr", chrPath, "--quiet"}, ioDiscard{}, ioDiscard{}); c != 0 {
		t.Fatal(c)
	}
	dec := filepath.Join(dir, "out.safetensors")
	if c := run([]string{"decode", "--in", chrPath, "--out", dec}, ioDiscard{}, ioDiscard{}); c != 0 {
		t.Fatal(c)
	}
}

type ioDiscard struct{}

func (ioDiscard) Write(p []byte) (int, error) { return len(p), nil }

func writeTiny(t *testing.T, path string) {
	t.Helper()
	w := make([]float32, 8*64)
	for i := 0; i < 8; i++ {
		for j := 0; j < 64; j++ {
			w[i*64+j] = float32((17*i+13*j)%100)/50 - 1
		}
	}
	if err := safetensors.Write(path, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{8, 64}, Data: bf16(w)},
	}); err != nil {
		t.Fatal(err)
	}
}

func writeThree(t *testing.T, path string) {
	t.Helper()
	q := make([]float32, 64*64)
	down := make([]float32, 32*128)
	norm := make([]float32, 64)
	for i := range q {
		q[i] = float32(i%50)/25 - 1
	}
	for i := range down {
		down[i] = float32(i%40)/20 - 1
	}
	for i := range norm {
		norm[i] = 1
	}
	if err := safetensors.Write(path, []safetensors.Tensor{
		{Name: "model.layers.0.self_attn.q_proj.weight", DType: safetensors.BF16, Shape: []int{64, 64}, Data: bf16(q)},
		{Name: "model.layers.0.mlp.down_proj.weight", DType: safetensors.BF16, Shape: []int{32, 128}, Data: bf16(down)},
		{Name: "model.norm.weight", DType: safetensors.BF16, Shape: []int{64}, Data: bf16(norm)},
	}); err != nil {
		t.Fatal(err)
	}
}

func bf16(v []float32) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[i*2:], f16.ToBF16Bits(x))
	}
	return b
}

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}
