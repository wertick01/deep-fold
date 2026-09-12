package nf4

import (
	"encoding/binary"
	"encoding/hex"
	"math"
	"strings"
	"testing"
)

// LUT bits from docs/spec/nf4.md §1.
var nf4Bits = [16]uint32{
	0xBF800000,
	0xBF3239B1,
	0xBF066B30,
	0xBECA32A0,
	0xBE91A24D,
	0xBE3D353F,
	0xBDBA7871,
	0x00000000,
	0x3DA2FAFF,
	0x3E24CAE3,
	0x3E7C04DD,
	0x3EAD033A,
	0x3EE1A4B8,
	0x3F1007AB,
	0x3F3913B3,
	0x3F800000,
}

// Packed hex from docs/spec/nf4.md §8 (row W[i]=i/63).
const golden64Hex = "778788889999A9AAAABABBBBCBCCCCCCDDDDDDDDEDEEEEEEEEEEEEFEFFFFFFFF"

// Packed hex from docs/spec/nf4.md §9.4 row 1.
const golden2x64Row1Hex = "00001011111121222233434454556676878899AABABBCCCCDDDDEEEEEEFEFFFF"

// Packed hex from docs/spec/nf4.md §6.1 for W=[0,1].
const pad01Hex = "F777777777777777777777777777777777777777777777777777777777777777"

// Indices from docs/spec/nf4.md §8 (W[i]=i/63).
var golden64Idx = [64]byte{
	7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 10, 10, 10,
	10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12,
	13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14,
	14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15,
}

func TestNF4TableBits(t *testing.T) {
	for i, want := range nf4Bits {
		got := math.Float32bits(NF4[i])
		if got != want {
			t.Errorf("NF4[%d] bits=%08X want %08X (val=%v)", i, got, want, NF4[i])
		}
	}
}

func TestNF4Golden64(t *testing.T) {
	w := rowIOver63()
	data, scale, err := Encode(w, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	assertHex(t, data, golden64Hex)
	if len(scale) != 1 || scale[0] != 0x3C00 {
		t.Fatalf("scale=%v want [0x3C00]", scale)
	}
}

func TestNF4Golden2x64(t *testing.T) {
	w := fixture2x64()
	data, scale, err := Encode(w, 2, 64)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != 64 {
		t.Fatalf("len(data)=%d want 64", len(data))
	}
	assertHex(t, data[:32], golden64Hex)
	assertHex(t, data[32:], golden2x64Row1Hex)
	if len(scale) != 2 || scale[0] != 0x3C00 || scale[1] != 0x3C00 {
		t.Fatalf("scale=%v want [0x3C00, 0x3C00]", scale)
	}

	hat, err := Decode(data, scale, 2, 64)
	if err != nil {
		t.Fatal(err)
	}
	rmse, maxabs, mae := reconMetrics(w, hat)
	if rmse > 0.05185 {
		t.Errorf("rmse=%g want ≤ 0.05185", rmse)
	}
	if maxabs > 0.14508 {
		t.Errorf("maxabs=%g want ≤ 0.14508", maxabs)
	}
	if mae > 0.03980 {
		t.Errorf("mae=%g want ≤ 0.03980", mae)
	}
}

func TestNF4PackNibble(t *testing.T) {
	t.Run("0_1", func(t *testing.T) {
		data, _, err := Encode([]float32{0, 1}, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		if data[0] != 0xF7 {
			t.Fatalf("byte=0x%02X want 0xF7 (bnb packing would be 0x7F)", data[0])
		}
	})
	t.Run("1_0", func(t *testing.T) {
		data, _, err := Encode([]float32{1, 0}, 1, 2)
		if err != nil {
			t.Fatal(err)
		}
		if data[0] != 0x7F {
			t.Fatalf("byte=0x%02X want 0x7F", data[0])
		}
	})
	t.Run("L3_L12", func(t *testing.T) {
		// Keep scale=1 so u equals the LUT values (max |g| = 1).
		w := make([]float32, 64)
		w[0] = NF4[3]
		w[1] = NF4[12]
		w[2] = 1
		data, scale, err := Encode(w, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if scale[0] != 0x3C00 {
			t.Fatalf("scale=%04X want 0x3C00", scale[0])
		}
		if data[0] != 0xC3 {
			t.Fatalf("byte=0x%02X want 0xC3", data[0])
		}
	})
}

func TestNF4ZeroMatrix(t *testing.T) {
	t.Run("3x64", func(t *testing.T) {
		w := make([]float32, 3*64)
		data, scale, err := Encode(w, 3, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i, b := range data {
			if b != 0x77 {
				t.Fatalf("data[%d]=0x%02X want 0x77", i, b)
			}
		}
		if len(scale) != 3 {
			t.Fatalf("len(scale)=%d", len(scale))
		}
		for i, s := range scale {
			if s != 0x3C00 {
				t.Fatalf("scale[%d]=%04X want 0x3C00", i, s)
			}
		}
		hat, err := Decode(data, scale, 3, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range hat {
			if v != 0 || math.Float32bits(v) != 0 {
				t.Fatalf("hat[%d]=%v bits=%08X", i, v, math.Float32bits(v))
			}
		}
	})
	t.Run("1x65", func(t *testing.T) {
		w := make([]float32, 65)
		data, scale, err := Encode(w, 1, 65)
		if err != nil {
			t.Fatal(err)
		}
		if len(data) != 64 {
			t.Fatalf("len(data)=%d want 64 (two padded groups)", len(data))
		}
		for i, b := range data {
			if b != 0x77 {
				t.Fatalf("data[%d]=0x%02X want 0x77", i, b)
			}
		}
		if len(scale) != 2 || scale[0] != 0x3C00 || scale[1] != 0x3C00 {
			t.Fatalf("scale=%v want [0x3C00, 0x3C00]", scale)
		}
		hat, err := Decode(data, scale, 1, 65)
		if err != nil {
			t.Fatal(err)
		}
		if len(hat) != 65 {
			t.Fatalf("len(hat)=%d want 65", len(hat))
		}
		for i, v := range hat {
			if v != 0 {
				t.Fatalf("hat[%d]=%v", i, v)
			}
		}
	})
}

func TestNF4Constant(t *testing.T) {
	t.Run("neg_2.25", func(t *testing.T) {
		w := fill(64, -2.25)
		data, scale, err := Encode(w, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if scale[0] != 0x4080 {
			t.Fatalf("scale=%04X want 0x4080", scale[0])
		}
		for i, b := range data {
			if b != 0x00 {
				t.Fatalf("data[%d]=0x%02X want 0x00 (all idx 0)", i, b)
			}
		}
		hat, err := Decode(data, scale, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range hat {
			if v != -2.25 {
				t.Fatalf("hat[%d]=%v want -2.25", i, v)
			}
		}
	})
	t.Run("pos_0.5", func(t *testing.T) {
		w := fill(64, 0.5)
		data, scale, err := Encode(w, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		if scale[0] != 0x3800 {
			t.Fatalf("scale=%04X want 0x3800", scale[0])
		}
		for i, b := range data {
			if b != 0xFF {
				t.Fatalf("data[%d]=0x%02X want 0xFF (all idx 15)", i, b)
			}
		}
		hat, err := Decode(data, scale, 1, 64)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range hat {
			if v != 0.5 {
				t.Fatalf("hat[%d]=%v want 0.5", i, v)
			}
		}
	})
}

func TestNF4Idempotent(t *testing.T) {
	t.Run("golden64", func(t *testing.T) {
		assertIdempotent(t, rowIOver63(), 1, 64)
	})
	t.Run("golden2x64", func(t *testing.T) {
		assertIdempotent(t, fixture2x64(), 2, 64)
	})
}

func TestNF4Pad65(t *testing.T) {
	w := []float32{0, 1}
	data, scale, err := Encode(w, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	assertHex(t, data, pad01Hex)
	if len(scale) != 1 || scale[0] != 0x3C00 {
		t.Fatalf("scale=%v want [0x3C00]", scale)
	}
	hat, err := Decode(data, scale, 1, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(hat) != 2 {
		t.Fatalf("decode shape len=%d want 2", len(hat))
	}
	if hat[0] != 0 || hat[1] != 1 {
		t.Fatalf("hat=%v want [0, 1]", hat)
	}
}

func TestNF4RejectNaN(t *testing.T) {
	w := rowIOver63()
	w[3] = float32(math.NaN())
	if _, _, err := Encode(w, 1, 64); err == nil {
		t.Fatal("expected error for NaN")
	}
}

func TestNF4RejectInf(t *testing.T) {
	w := rowIOver63()
	w[3] = float32(math.Inf(1))
	if _, _, err := Encode(w, 1, 64); err == nil {
		t.Fatal("expected error for +Inf")
	}
	w[3] = float32(math.Inf(-1))
	if _, _, err := Encode(w, 1, 64); err == nil {
		t.Fatal("expected error for -Inf")
	}
}

func TestNF4RejectEmpty(t *testing.T) {
	if _, _, err := Encode(nil, 0, 64); err == nil {
		t.Fatal("expected error for n_out=0")
	}
	if _, _, err := Encode(nil, 1, 0); err == nil {
		t.Fatal("expected error for n_in=0")
	}
	if _, err := Decode(nil, nil, 0, 64); err == nil {
		t.Fatal("expected decode error for n_out=0")
	}
	if _, err := Decode(nil, nil, 1, 0); err == nil {
		t.Fatal("expected decode error for n_in=0")
	}
}

func TestNF4GroupSize(t *testing.T) {
	if GroupSize != 64 {
		t.Fatalf("GroupSize=%d want 64", GroupSize)
	}
	if err := CheckGroupSize(64); err != nil {
		t.Fatal(err)
	}
	if err := CheckGroupSize(32); err == nil {
		t.Fatal("expected error for group_size=32")
	}
	if err := CheckGroupSize(8); err == nil {
		t.Fatal("expected error for group_size=8")
	}
	if err := CheckGroupSize(128); err == nil {
		t.Fatal("expected error for group_size=128")
	}
}

func TestNF4DecodeExact(t *testing.T) {
	w := rowIOver63()
	data, scale, err := Encode(w, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	hat, err := Decode(data, scale, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	s := float32(1)
	for i := 0; i < 64; i++ {
		want := NF4[golden64Idx[i]] * s
		if math.Float32bits(hat[i]) != math.Float32bits(want) {
			t.Fatalf("i=%d hat bits=%08X want L[%d]=%08X", i,
				math.Float32bits(hat[i]), golden64Idx[i], math.Float32bits(want))
		}
	}
	// Second decode of the same blobs is bit-stable.
	hat2, err := Decode(data, scale, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	for i := range hat {
		if math.Float32bits(hat[i]) != math.Float32bits(hat2[i]) {
			t.Fatalf("decode not stable at %d", i)
		}
	}
}

func TestNF4ScaleLE(t *testing.T) {
	w := rowIOver63()
	_, scale, err := Encode(w, 1, 64)
	if err != nil {
		t.Fatal(err)
	}
	if scale[0] != 0x3C00 {
		t.Fatalf("scale bits=%04X want 0x3C00", scale[0])
	}
	buf := make([]byte, 2)
	binary.LittleEndian.PutUint16(buf, scale[0])
	if buf[0] != 0x00 || buf[1] != 0x3C {
		t.Fatalf("LE blob %02X %02X want 00 3C", buf[0], buf[1])
	}
}

func rowIOver63() []float32 {
	w := make([]float32, 64)
	for i := 0; i < 64; i++ {
		w[i] = float32(i) / float32(63)
	}
	return w
}

func fixture2x64() []float32 {
	w := make([]float32, 2*64)
	for i := 0; i < 64; i++ {
		w[i] = float32(i) / float32(63)
		w[64+i] = float32(2*i-63) / float32(63)
	}
	return w
}

func fill(n int, c float32) []float32 {
	w := make([]float32, n)
	for i := range w {
		w[i] = c
	}
	return w
}

func assertHex(t *testing.T, data []byte, want string) {
	t.Helper()
	got := strings.ToUpper(hex.EncodeToString(data))
	if got != want {
		t.Fatalf("packed\n got %s\nwant %s", got, want)
	}
}

func assertIdempotent(t *testing.T, w []float32, nOut, nIn int) {
	t.Helper()
	data, scale, err := Encode(w, nOut, nIn)
	if err != nil {
		t.Fatal(err)
	}
	hat, err := Decode(data, scale, nOut, nIn)
	if err != nil {
		t.Fatal(err)
	}
	data2, scale2, err := Encode(hat, nOut, nIn)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != len(data2) {
		t.Fatalf("len data %d vs %d", len(data), len(data2))
	}
	for i := range data {
		if data[i] != data2[i] {
			t.Fatalf("nibble byte %d: %02X vs %02X", i, data[i], data2[i])
		}
	}
	if len(scale) != len(scale2) {
		t.Fatalf("len scale %d vs %d", len(scale), len(scale2))
	}
	for i := range scale {
		if scale[i] != scale2[i] {
			t.Fatalf("scale[%d]=%04X vs %04X", i, scale[i], scale2[i])
		}
	}
	hat2, err := Decode(data2, scale2, nOut, nIn)
	if err != nil {
		t.Fatal(err)
	}
	for i := range hat {
		if math.Float32bits(hat[i]) != math.Float32bits(hat2[i]) {
			t.Fatalf("second decode bits differ at %d", i)
		}
	}
}

func reconMetrics(orig, hat []float32) (rmse, maxabs, mae float64) {
	n := len(orig)
	var sumSq, sumAbs float64
	for i := range orig {
		e := hat[i] - orig[i]
		ef := float64(e)
		sumSq += ef * ef
		ae := math.Abs(ef)
		sumAbs += ae
		if ae > maxabs {
			maxabs = ae
		}
	}
	rmse = math.Sqrt(sumSq / float64(n))
	mae = sumAbs / float64(n)
	return
}
