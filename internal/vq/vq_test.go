package vq

import (
	"math"
	"math/rand/v2"
	"testing"

	"chr/internal/f16"
)

func TestVQToy(t *testing.T) {
	nOut, nIn := 4, 16
	templates := [4][8]float32{
		{0, 0, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
	}
	w := make([]float32, nOut*nIn)
	for n := 0; n < nOut*(nIn/B); n++ {
		r := n / 2
		g := n % 2
		tmpl := templates[n%4]
		copy(w[r*nIn+g*B:], tmpl[:])
	}

	cb, idx, err := Encode(w, nOut, nIn, Options{Seed: 0, Iters: 20, Chunk: 256})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(cb) != codebookLen {
		t.Fatalf("codebook len %d", len(cb))
	}
	G := (nIn + B - 1) / B
	if len(idx) != nOut*G*M {
		t.Fatalf("index len %d want %d", len(idx), nOut*G*M)
	}

	recon, err := Decode(cb, idx, nOut, nIn)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	ref := reconstructFromBook(cb, idx, nOut, nIn)
	bitsEqual(t, recon, ref)

	gotMSE := mse(w, recon)
	if gotMSE > 1e-4 {
		t.Fatalf("MSE=%g want <= 1e-4", gotMSE)
	}

	for i, v := range templates {
		if d := minInfNormBook0(cb, v); d >= 1e-4 {
			t.Fatalf("template %d not in book0 inf-norm %g", i, d)
		}
	}

	// Identical vectors share book-0 index (tie → smaller j).
	i00 := idx[0]
	i20 := idx[(2*G+0)*2]
	if i00 != i20 {
		t.Fatalf("repeated v0 got book0 indices %d vs %d", i00, i20)
	}
}

func TestVQPad(t *testing.T) {
	nOut, nIn := 2, 12
	w := make([]float32, nOut*nIn)
	for i := range w {
		w[i] = float32((17*i + 3) % 17)
	}
	cb, idx, err := Encode(w, nOut, nIn, Options{Seed: 0, Iters: 20, Chunk: 64})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	G := (nIn + B - 1) / B
	if G != 2 {
		t.Fatalf("G=%d want 2", G)
	}
	if len(idx) != nOut*G*M {
		t.Fatalf("index len %d want %d", len(idx), nOut*G*M)
	}
	recon, err := Decode(cb, idx, nOut, nIn)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	if len(recon) != nOut*nIn {
		t.Fatalf("decode len %d want logical %d", len(recon), nOut*nIn)
	}
	ref := reconstructFromBook(cb, idx, nOut, nIn)
	bitsEqual(t, recon, ref)
}

func TestVQRejectNaN(t *testing.T) {
	w := make([]float32, 8)
	w[3] = float32(math.NaN())
	_, _, err := Encode(w, 1, 8, Options{})
	if err == nil {
		t.Fatal("expected error on NaN")
	}
}

func TestVQRejectInf(t *testing.T) {
	w := make([]float32, 8)
	w[0] = float32(math.Inf(1))
	_, _, err := Encode(w, 1, 8, Options{})
	if err == nil {
		t.Fatal("expected error on Inf")
	}
}

func TestVQZeroish(t *testing.T) {
	nOut, nIn := 32, 128
	w := make([]float32, nOut*nIn)
	for i := 0; i < nOut; i++ {
		for j := 0; j < nIn; j++ {
			w[i*nIn+j] = float32((17*i+13*j)%100)/50 - 1
		}
	}
	cb, idx, err := Encode(w, nOut, nIn, Options{Seed: 0, Iters: 20, Chunk: 512})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	recon, err := Decode(cb, idx, nOut, nIn)
	if err != nil {
		t.Fatalf("Decode: %v", err)
	}
	bitsEqual(t, recon, reconstructFromBook(cb, idx, nOut, nIn))

	got := mse(w, recon)
	v := popVar(w)
	if got >= 0.5*v {
		t.Fatalf("MSE=%g var=%g; want MSE < 0.5*var", got, v)
	}
}

func TestVQDecodeMatchesBook(t *testing.T) {
	t.Run("encode", func(t *testing.T) {
		nOut, nIn := 3, 16
		w := make([]float32, nOut*nIn)
		for i := range w {
			w[i] = float32((i*7)%11) / 4
		}
		cb, idx, err := Encode(w, nOut, nIn, Options{Seed: 1, Iters: 8, Chunk: 32})
		if err != nil {
			t.Fatalf("Encode: %v", err)
		}
		got, err := Decode(cb, idx, nOut, nIn)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		bitsEqual(t, got, reconstructFromBook(cb, idx, nOut, nIn))
	})

	t.Run("layout1x16", func(t *testing.T) {
		cb := make([]uint16, codebookLen)
		cb[0] = 0x3C00         // C[0,0,0] = 1
		cb[1*B+1] = 0x3C00     // C[0,1,1] = 1
		cb[K*B+7*B+2] = 0x3800 // C[1,7,2] = 0.5
		index := []byte{0x00, 0x07, 0x01, 0x00}
		got, err := Decode(cb, index, 1, 16)
		if err != nil {
			t.Fatalf("Decode: %v", err)
		}
		want := []float32{1, 0, 0.5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0}
		bitsEqual(t, got, want)
		if math.Float32bits(got[0]) != 0x3f800000 || math.Float32bits(got[2]) != 0x3f000000 {
			t.Fatalf("golden bits: %08x %08x", math.Float32bits(got[0]), math.Float32bits(got[2]))
		}
	})
}

func TestVQEmpty(t *testing.T) {
	_, _, err := Encode([]float32{}, 0, 8, Options{})
	if err == nil {
		t.Fatal("expected error on n_out=0")
	}
	_, _, err = Encode([]float32{}, 4, 0, Options{})
	if err == nil {
		t.Fatal("expected error on n_in=0")
	}
}

func TestVQSeedDeterminism(t *testing.T) {
	nOut, nIn := 8, 16
	w := make([]float32, nOut*nIn)
	for i := range w {
		w[i] = float32((i*19)%23)/10 - 1
	}
	opt := Options{Seed: 1, Iters: 4, Chunk: 64}
	cb1, idx1, err := Encode(w, nOut, nIn, opt)
	if err != nil {
		t.Fatal(err)
	}
	cb2, idx2, err := Encode(w, nOut, nIn, opt)
	if err != nil {
		t.Fatal(err)
	}
	if len(cb1) != len(cb2) || len(idx1) != len(idx2) {
		t.Fatal("length mismatch")
	}
	for i := range cb1 {
		if cb1[i] != cb2[i] {
			t.Fatalf("codebook[%d] %04x vs %04x", i, cb1[i], cb2[i])
		}
	}
	for i := range idx1 {
		if idx1[i] != idx2[i] {
			t.Fatalf("index[%d] %d vs %d", i, idx1[i], idx2[i])
		}
	}
}

func TestVQArgmin(t *testing.T) {
	C := make([]float32, K*B)
	for j := 0; j < K; j++ {
		C[j*B] = 10
	}
	x := []float32{1, 0, 0, 0, 0, 0, 0, 0}
	copy(C[0:B], x)
	copy(C[1*B:2*B], []float32{0, 1, 0, 0, 0, 0, 0, 0})
	copy(C[2*B:3*B], []float32{0.5, 0, 0, 0, 0, 0, 0, 0})

	var cnorm2 [K]float32
	centroidNorm2(C, &cnorm2)
	if j := argminFused(x, C, &cnorm2); j != 0 {
		t.Fatalf("dist example argmin=%d want 0", j)
	}

	copy(C[1*B:2*B], x)
	centroidNorm2(C, &cnorm2)
	if j := argminFused(x, C, &cnorm2); j != 0 {
		t.Fatalf("tie argmin=%d want 0", j)
	}
}

func TestVQItersZero(t *testing.T) {
	nOut, nIn := 4, 16
	w := make([]float32, nOut*nIn)
	templates := [4][8]float32{
		{0, 0, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 0, 0, 0, 0},
		{0, 1, 0, 0, 0, 0, 0, 0},
		{0, 0, 1, 0, 0, 0, 0, 0},
	}
	for n := 0; n < nOut*2; n++ {
		r, g := n/2, n%2
		tmpl := templates[n%4]
		copy(w[r*nIn+g*B:], tmpl[:])
	}
	G := (nIn + B - 1) / B
	R := packVectors(w, nOut, nIn, G)
	rng := rand.New(rand.NewPCG(0, 0))
	idxSub := reservoir(len(R)/B, rng)
	S := make([]float32, len(idxSub)*B)
	C, pi := kmeans(R, S, idxSub, 0, 256, rng)
	cb := make([]uint16, codebookLen)
	if err := storeFP16(cb, 0, C); err != nil {
		t.Fatal(err)
	}
	if len(pi) != nOut*G {
		t.Fatalf("pi len %d", len(pi))
	}
}

func TestVQZeroMatrix(t *testing.T) {
	nOut, nIn := 8, 16
	w := make([]float32, nOut*nIn)
	cb, idx, err := Encode(w, nOut, nIn, Options{Seed: 0, Iters: 4, Chunk: 32})
	if err != nil {
		t.Fatal(err)
	}
	recon, err := Decode(cb, idx, nOut, nIn)
	if err != nil {
		t.Fatal(err)
	}
	if mse(w, recon) != 0 {
		t.Fatalf("zero matrix MSE=%g", mse(w, recon))
	}
}

func reconstructFromBook(codebook []uint16, index []byte, nOut, nIn int) []float32 {
	G := (nIn + B - 1) / B
	out := make([]float32, nOut*nIn)
	for r := 0; r < nOut; r++ {
		for j := 0; j < G; j++ {
			off := (r*G + j) * M
			i1 := int(index[off])
			i2 := int(index[off+1])
			for d := 0; d < B; d++ {
				col := j*B + d
				if col >= nIn {
					continue
				}
				out[r*nIn+col] = f16.ToFloat32(codebook[i1*B+d]) + f16.ToFloat32(codebook[K*B+i2*B+d])
			}
		}
	}
	return out
}

func minInfNormBook0(cb []uint16, v [8]float32) float32 {
	best := float32(math.Inf(1))
	for j := 0; j < K; j++ {
		var inf float32
		for d := 0; d < B; d++ {
			diff := f16.ToFloat32(cb[j*B+d]) - v[d]
			if diff < 0 {
				diff = -diff
			}
			if diff > inf {
				inf = diff
			}
		}
		if inf < best {
			best = inf
		}
	}
	return best
}

func mse(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s / float64(len(a))
}

func popVar(w []float32) float64 {
	n := float64(len(w))
	var mean float64
	for _, v := range w {
		mean += float64(v)
	}
	mean /= n
	var s float64
	for _, v := range w {
		d := float64(v) - mean
		s += d * d
	}
	return s / n
}

func bitsEqual(t *testing.T, a, b []float32) {
	t.Helper()
	if len(a) != len(b) {
		t.Fatalf("len %d != %d", len(a), len(b))
	}
	for i := range a {
		ba, bb := math.Float32bits(a[i]), math.Float32bits(b[i])
		if ba != bb {
			t.Fatalf("i=%d bits %08x vs %08x (%v vs %v)", i, ba, bb, a[i], b[i])
		}
	}
}
