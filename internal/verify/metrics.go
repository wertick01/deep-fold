package verify

import (
	"encoding/binary"
	"math"

	"chr/internal/f16"
)

// Stats are orig vs reconstruct metrics over logical elements.
type Stats struct {
	N      int
	RMSE   float64
	MAE    float64
	MaxAbs float64
	RMS    float64
	Rel    float64
}

// ComputeStats compares orig and hat in float32. Padding must already be stripped.
func ComputeStats(orig, hat []float32) Stats {
	n := len(orig)
	if len(hat) < n {
		n = len(hat)
	}
	var sumSqErr, sumAbs, sumSqOrig float64
	var maxAbs float64
	for i := 0; i < n; i++ {
		e := float64(hat[i]) - float64(orig[i])
		if e < 0 {
			e = -e
		}
		sumAbs += e
		d := float64(hat[i]) - float64(orig[i])
		sumSqErr += d * d
		sumSqOrig += float64(orig[i]) * float64(orig[i])
		if e > maxAbs {
			maxAbs = e
		}
	}
	s := Stats{N: n, MaxAbs: maxAbs}
	if n > 0 {
		s.MAE = sumAbs / float64(n)
		s.RMSE = math.Sqrt(sumSqErr / float64(n))
		s.RMS = math.Sqrt(sumSqOrig / float64(n))
		switch {
		case s.RMS > 0:
			s.Rel = s.RMSE / s.RMS
		case s.RMSE == 0:
			s.Rel = 0
		default:
			s.Rel = math.Inf(1)
		}
	}
	return s
}

// BF16Exact reports whether hat equals the BF16 projection of orig (bitwise float32).
func BF16Exact(orig, hat []float32) bool {
	if len(orig) != len(hat) {
		return false
	}
	for i := range orig {
		ref := f16.FromBF16Bits(f16.ToBF16Bits(orig[i]))
		if math.Float32bits(hat[i]) != math.Float32bits(ref) {
			return false
		}
	}
	return true
}

func uint16ToLE(u []uint16) []byte {
	b := make([]byte, len(u)*2)
	for i, x := range u {
		binary.LittleEndian.PutUint16(b[i*2:], x)
	}
	return b
}

func leToUint16(b []byte) []uint16 {
	n := len(b) / 2
	u := make([]uint16, n)
	for i := 0; i < n; i++ {
		u[i] = binary.LittleEndian.Uint16(b[i*2:])
	}
	return u
}

func encodeBF16(v []float32) []byte {
	b := make([]byte, len(v)*2)
	for i, x := range v {
		binary.LittleEndian.PutUint16(b[i*2:], f16.ToBF16Bits(x))
	}
	return b
}
