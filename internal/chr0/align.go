package chr0

import (
	"fmt"
	"math"
)

// Align64 rounds x up to a multiple of 64. Offsets are from file start.
func Align64(x int64) int64 {
	if x < 0 {
		return x
	}
	return (x + 63) &^ 63
}

func nInPad(nIn, g int) int {
	return ((nIn + g - 1) / g) * g
}

func checkShape(shape []int) error {
	if len(shape) != 1 && len(shape) != 2 {
		return fmt.Errorf("rank must be 1 or 2")
	}
	var p int64 = 1
	for _, d := range shape {
		if d < 1 {
			return fmt.Errorf("shape axis must be >= 1")
		}
		if d > (1<<24)-1 {
			return fmt.Errorf("shape axis too large")
		}
		if p > math.MaxInt64/int64(d) {
			return fmt.Errorf("tensor too large")
		}
		p *= int64(d)
	}
	if p > (1<<32)/4 {
		return fmt.Errorf("tensor too large")
	}
	return nil
}

func shapeProduct(shape []int) (int64, error) {
	if err := checkShape(shape); err != nil {
		return 0, err
	}
	var p int64 = 1
	for _, d := range shape {
		p *= int64(d)
	}
	return p, nil
}

// BF16BlobBytes is 2 * Π shape[i].
func BF16BlobBytes(shape []int) (int64, error) {
	p, err := shapeProduct(shape)
	if err != nil {
		return 0, err
	}
	return p * 2, nil
}

// NF4BlobBytes returns data and scale lengths for rank-2 nf4/int4.
func NF4BlobBytes(shape []int) (data, scale int64, err error) {
	if len(shape) != 2 {
		return 0, 0, fmt.Errorf("quantized tensor must be rank 2")
	}
	if err := checkShape(shape); err != nil {
		return 0, 0, err
	}
	nOut, nIn := shape[0], shape[1]
	pad := nInPad(nIn, 64)
	data = int64(nOut) * int64(pad) / 2
	scale = int64(nOut) * int64(pad/64) * 2
	return data, scale, nil
}

// VQBlobBytes returns codebook (always 8192) and index lengths for rank-2 vq.
func VQBlobBytes(shape []int) (codebook, index int64, err error) {
	if len(shape) != 2 {
		return 0, 0, fmt.Errorf("quantized tensor must be rank 2")
	}
	if err := checkShape(shape); err != nil {
		return 0, 0, err
	}
	nOut, nIn := shape[0], shape[1]
	pad := nInPad(nIn, 8)
	return 8192, int64(nOut) * int64(pad/8) * 2, nil
}
