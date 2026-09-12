// Package nf4 implements group-wise NF4 (group size 64) for CHR0 codec "nf4".
package nf4

import (
	"fmt"
	"math"

	"chr/internal/f16"
)

// GroupSize is the only valid NF4 group length along n_in (CHR0 group_size).
const GroupSize = 64

// NF4 is the QLoRA / bitsandbytes 16-level codebook (index 0 = −1 … 15 = +1).
// Literals are the canonical get_4bit_type("nf4") strings; IEEE binary32 bits
// must match docs/spec/nf4.md §1.
var NF4 = [16]float32{
	float32(-1.0),
	float32(-0.6961928009986877),
	float32(-0.5250730514526367),
	float32(-0.39491748809814453),
	float32(-0.28444138169288635),
	float32(-0.18477343022823334),
	float32(-0.09105003625154495),
	float32(0.0),
	float32(0.07958029955625534),
	float32(0.16093020141124725),
	float32(0.24611230194568634),
	float32(0.33791524171829224),
	float32(0.44070982933044434),
	float32(0.5626170039176941),
	float32(0.7229568362236023),
	float32(1.0),
}

// CheckGroupSize reports an error unless n equals GroupSize.
// CHR0 readers and the CLI encoder call this when group_size ≠ 64.
func CheckGroupSize(n int) error {
	if n != GroupSize {
		return fmt.Errorf("nf4: group_size must be %d, got %d", GroupSize, n)
	}
	return nil
}

// Encode quantizes row-major W[nOut, nIn] to packed NF4 nibbles and FP16 scales.
func Encode(w []float32, nOut, nIn int) (data []byte, scale []uint16, err error) {
	if nOut < 1 || nIn < 1 {
		return nil, nil, fmt.Errorf("nf4: empty shape n_out=%d n_in=%d", nOut, nIn)
	}
	if len(w) != nOut*nIn {
		return nil, nil, fmt.Errorf("nf4: len(w)=%d want %d", len(w), nOut*nIn)
	}

	nInPadded, nGroups := padded(nIn)
	data = make([]byte, nOut*(nInPadded/2))
	scale = make([]uint16, nOut*nGroups)

	var grp [GroupSize]float32
	for r := 0; r < nOut; r++ {
		row := w[r*nIn : (r+1)*nIn]
		for g := 0; g < nGroups; g++ {
			for k := 0; k < GroupSize; k++ {
				c := g*GroupSize + k
				if c < nIn {
					grp[k] = row[c]
				} else {
					grp[k] = 0
				}
			}
			idx, s16, err := encodeGroup64(&grp)
			if err != nil {
				return nil, nil, err
			}
			scale[r*nGroups+g] = s16
			dout := data[r*(nInPadded/2)+g*(GroupSize/2):]
			for k := 0; k < GroupSize/2; k++ {
				dout[k] = (idx[2*k+1] << 4) | idx[2*k]
			}
		}
	}
	return data, scale, nil
}

// Decode reconstructs logical W[nOut, nIn] as float32 (padding columns dropped).
func Decode(data []byte, scale []uint16, nOut, nIn int) ([]float32, error) {
	if nOut < 1 || nIn < 1 {
		return nil, fmt.Errorf("nf4: empty shape n_out=%d n_in=%d", nOut, nIn)
	}
	nInPadded, nGroups := padded(nIn)
	wantData := nOut * (nInPadded / 2)
	wantScale := nOut * nGroups
	if len(data) != wantData {
		return nil, fmt.Errorf("nf4: len(data)=%d want %d", len(data), wantData)
	}
	if len(scale) != wantScale {
		return nil, fmt.Errorf("nf4: len(scale)=%d want %d", len(scale), wantScale)
	}

	out := make([]float32, nOut*nIn)
	for r := 0; r < nOut; r++ {
		for g := 0; g < nGroups; g++ {
			s := f16.ToFloat32(scale[r*nGroups+g])
			base := r*(nInPadded/2) + g*(GroupSize/2)
			for k := 0; k < GroupSize; k++ {
				c := g*GroupSize + k
				if c >= nIn {
					continue
				}
				b := data[base+k/2]
				var nib byte
				if k%2 == 0 {
					nib = b & 0x0F
				} else {
					nib = (b >> 4) & 0x0F
				}
				out[r*nIn+c] = NF4[nib] * s
			}
		}
	}
	return out, nil
}

func padded(nIn int) (nInPadded, nGroups int) {
	nInPadded = ((nIn + GroupSize - 1) / GroupSize) * GroupSize
	nGroups = nInPadded / GroupSize
	return
}

func encodeGroup64(g *[GroupSize]float32) (idx [GroupSize]byte, s16 uint16, err error) {
	var s32 float32
	for i := 0; i < GroupSize; i++ {
		if !finite(g[i]) {
			return idx, 0, fmt.Errorf("nf4: non-finite weight")
		}
		a := abs32(g[i])
		if a > s32 {
			s32 = a
		}
	}
	if s32 == 0 {
		s32 = 1
	}
	s16 = f16.FromFloat32(s32)
	s := f16.ToFloat32(s16)
	if !finite(s) || s <= 0 {
		return idx, 0, fmt.Errorf("nf4: scale is not finite positive")
	}
	for i := 0; i < GroupSize; i++ {
		u := g[i] / s
		if u > 1 {
			u = 1
		} else if u < -1 {
			u = -1
		}
		idx[i] = nearest(u)
	}
	return idx, s16, nil
}

func nearest(u float32) byte {
	bestJ := byte(0)
	d := u - NF4[0]
	best := d * d
	for j := 1; j < 16; j++ {
		d := u - NF4[j]
		d2 := d * d
		if d2 < best {
			best = d2
			bestJ = byte(j)
		}
	}
	return bestJ
}

func finite(x float32) bool {
	return !math.IsNaN(float64(x)) && !math.IsInf(float64(x), 0)
}

func abs32(x float32) float32 {
	return float32(math.Abs(float64(x)))
}
