// Package vq implements additive residual vector quantization (M=2, k=256, B=8).
package vq

import (
	"fmt"
	"math"
	"math/rand/v2"

	"chr/internal/f16"
)

const (
	// M is the number of codebooks.
	M = 2
	// K is the number of centroids per codebook.
	K = 256
	// B is the group size along n_in.
	B = 8

	// DefaultIters is the Lloyd iteration count when Options.Iters is 0.
	DefaultIters = 20
	// DefaultChunk is the assignment chunk size when Options.Chunk is 0.
	DefaultChunk = 262144

	subsampleCap = 65536
	codebookLen  = M * K * B
)

// Options controls Encode. Zero Iters/Chunk mean DefaultIters / DefaultChunk.
// Seed 0 is the default PCG seed. A new PCG(seed, 0) is created per matrix.
type Options struct {
	Seed  uint64
	Iters int
	Chunk int
}

// Encode fits two residual k-means codebooks on W[nOut, nIn] (row-major float32).
// codebook is FP16 with shape [M, 256, 8]; index is uint8 [nOut, G, 2] row-major.
func Encode(w []float32, nOut, nIn int, opt Options) (codebook []uint16, index []byte, err error) {
	if nOut < 1 || nIn < 1 {
		return nil, nil, fmt.Errorf("vq: empty matrix n_out=%d n_in=%d", nOut, nIn)
	}
	if len(w) != nOut*nIn {
		return nil, nil, fmt.Errorf("vq: weight length %d != n_out*n_in %d", len(w), nOut*nIn)
	}
	if opt.Iters < 0 {
		return nil, nil, fmt.Errorf("vq: iters must be >= 0")
	}
	if opt.Chunk < 0 {
		return nil, nil, fmt.Errorf("vq: chunk must be >= 1")
	}
	if opt.Iters == 0 {
		opt.Iters = DefaultIters
	}
	if opt.Chunk == 0 {
		opt.Chunk = DefaultChunk
	}

	for i, v := range w {
		if math.IsNaN(float64(v)) {
			return nil, nil, fmt.Errorf("vq: NaN at index %d", i)
		}
		if math.IsInf(float64(v), 0) {
			return nil, nil, fmt.Errorf("vq: Inf at index %d", i)
		}
	}

	G := (nIn + B - 1) / B
	N := nOut * G
	R := packVectors(w, nOut, nIn, G)

	rng := rand.New(rand.NewPCG(opt.Seed, 0))
	idxSub := reservoir(N, rng)
	S := make([]float32, len(idxSub)*B)
	codebook = make([]uint16, codebookLen)
	piStore := [M][]uint8{}

	for m := 0; m < M; m++ {
		C, pi := kmeans(R, S, idxSub, opt.Iters, opt.Chunk, rng)
		if err := storeFP16(codebook, m, C); err != nil {
			return nil, nil, err
		}
		subtractBook(R, codebook, m, pi)
		piStore[m] = pi
	}

	index = packIndex(piStore[0], piStore[1], nOut, G)
	return codebook, index, nil
}

// Decode reconstructs W_hat[nOut, nIn] as C1[i1]+C2[i2] in float32 (logical shape, no pad columns).
func Decode(codebook []uint16, index []byte, nOut, nIn int) ([]float32, error) {
	if nOut < 1 || nIn < 1 {
		return nil, fmt.Errorf("vq: empty matrix n_out=%d n_in=%d", nOut, nIn)
	}
	if len(codebook) != codebookLen {
		return nil, fmt.Errorf("vq: codebook length %d != %d", len(codebook), codebookLen)
	}
	G := (nIn + B - 1) / B
	want := nOut * G * M
	if len(index) != want {
		return nil, fmt.Errorf("vq: index length %d != n_out*G*2 %d", len(index), want)
	}

	out := make([]float32, nOut*nIn)
	for r := 0; r < nOut; r++ {
		for j := 0; j < G; j++ {
			off := (r*G + j) * M
			i1 := int(index[off])
			i2 := int(index[off+1])
			b0 := i1 * B
			b1 := K*B + i2*B
			for d := 0; d < B; d++ {
				col := j*B + d
				if col >= nIn {
					break
				}
				out[r*nIn+col] = f16.ToFloat32(codebook[b0+d]) + f16.ToFloat32(codebook[b1+d])
			}
		}
	}
	return out, nil
}

func packVectors(w []float32, nOut, nIn, G int) []float32 {
	N := nOut * G
	R := make([]float32, N*B)
	for r := 0; r < nOut; r++ {
		row := w[r*nIn : (r+1)*nIn]
		for j := 0; j < G; j++ {
			dst := R[(r*G+j)*B : (r*G+j)*B+B]
			srcOff := j * B
			for d := 0; d < B; d++ {
				col := srcOff + d
				if col < nIn {
					dst[d] = row[col]
				}
			}
		}
	}
	return R
}

func packIndex(pi0, pi1 []uint8, nOut, G int) []byte {
	index := make([]byte, nOut*G*M)
	for n := 0; n < nOut*G; n++ {
		index[n*2] = pi0[n]
		index[n*2+1] = pi1[n]
	}
	return index
}

func storeFP16(codebook []uint16, m int, C []float32) error {
	off := m * K * B
	for i, v := range C {
		h := f16.FromFloat32(v)
		if h&0x7fff == 0x7c00 {
			return fmt.Errorf("vq: centroid overflows binary16")
		}
		codebook[off+i] = h
	}
	return nil
}

func subtractBook(R []float32, codebook []uint16, m int, pi []uint8) {
	book := codebook[m*K*B : (m+1)*K*B]
	N := len(pi)
	for n := 0; n < N; n++ {
		cbase := int(pi[n]) * B
		rbase := n * B
		for d := 0; d < B; d++ {
			R[rbase+d] -= f16.ToFloat32(book[cbase+d])
		}
	}
}
