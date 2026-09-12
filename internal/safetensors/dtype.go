package safetensors

import (
	"encoding/binary"
	"fmt"
	"math"

	"chr/internal/f16"
)

// DType is a safetensors dtype string. Register is exact: BF16, not bf16.
type DType string

const (
	F32  DType = "F32"
	F16  DType = "F16"
	BF16 DType = "BF16"
)

// ElemSize is the on-disk size of one element in bytes.
func (d DType) ElemSize() (int, error) {
	switch d {
	case F32:
		return 4, nil
	case F16, BF16:
		return 2, nil
	default:
		return 0, fmt.Errorf("unsupported dtype %s", d)
	}
}

// ToF32 converts a little-endian payload to float32.
func ToF32(dtype DType, raw []byte) ([]float32, error) {
	es, err := dtype.ElemSize()
	if err != nil {
		return nil, err
	}
	if len(raw)%es != 0 {
		return nil, fmt.Errorf("payload length %d is not a multiple of %d for %s", len(raw), es, dtype)
	}
	n := len(raw) / es
	out := make([]float32, n)
	switch dtype {
	case F32:
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case F16:
		for i := 0; i < n; i++ {
			out[i] = f16.ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case BF16:
		for i := 0; i < n; i++ {
			out[i] = f16.FromBF16Bits(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	}
	return out, nil
}

// EncodeF32 encodes float32 values as little-endian F32 bytes.
func EncodeF32(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(x))
	}
	return b
}

// F32Tensor is a convenience payload for WriteF32 (tests and chr decode).
type F32Tensor struct {
	Name  string
	Shape []int
	Data  []float32
}

func numel(shape []int) (int64, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("rank 0")
	}
	var p int64 = 1
	for _, d := range shape {
		if d < 1 {
			return 0, fmt.Errorf("shape axis must be >= 1")
		}
		if d > (1<<24)-1 {
			return 0, fmt.Errorf("shape axis too large")
		}
		if p > (1<<63-1)/int64(d) {
			return 0, fmt.Errorf("tensor too large")
		}
		p *= int64(d)
	}
	if p > (1<<32)/4 {
		return 0, fmt.Errorf("tensor too large")
	}
	return p, nil
}
