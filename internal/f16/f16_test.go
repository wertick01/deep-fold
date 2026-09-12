package f16

import (
	"math"
	"testing"
)

func TestRoundtripExactValues(t *testing.T) {
	vals := []float32{0, 1, -1, 0.5, 2, 2.25, 2.5, 65504}
	for _, v := range vals {
		h := FromFloat32(v)
		got := ToFloat32(h)
		if got != v {
			t.Fatalf("%v: bits=%04x got %v", v, h, got)
		}
	}
	if FromFloat32(1) != 0x3C00 {
		t.Fatalf("1.0 bits %04x", FromFloat32(1))
	}
	if FromFloat32(2.25) != 0x4080 {
		t.Fatalf("2.25 bits %04x", FromFloat32(2.25))
	}
}

func TestBF16Expand(t *testing.T) {
	if FromBF16Bits(0x3F80) != 1 {
		t.Fatalf("bf16 1.0 got %v", FromBF16Bits(0x3F80))
	}
}

func TestSubnormal(t *testing.T) {
	h := FromFloat32(1e-5)
	if h&0x7c00 == 0x7c00 {
		t.Fatalf("1e-5 → inf/nan %04x", h)
	}
	got := ToFloat32(h)
	if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || got <= 0 || got > 2e-5 {
		t.Fatalf("1e-5 roundtrip %v fp16=%04x", got, h)
	}
}

func TestNaNInf(t *testing.T) {
	inf := FromFloat32(float32(math.Inf(1)))
	if inf != 0x7c00 {
		t.Fatalf("inf %04x", inf)
	}
	n := FromFloat32(float32(math.NaN()))
	if n&0x7c00 != 0x7c00 || n&0x3ff == 0 {
		t.Fatalf("nan %04x", n)
	}
}
