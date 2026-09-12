package vq

import "math/rand/v2"

func reservoir(N int, rng *rand.Rand) []uint32 {
	nSub := N
	if nSub > subsampleCap {
		nSub = subsampleCap
	}
	idx := make([]uint32, nSub)
	for i := 0; i < nSub; i++ {
		idx[i] = uint32(i)
	}
	if N <= subsampleCap {
		return idx
	}
	for t := nSub; t < N; t++ {
		j := rng.IntN(t + 1)
		if j < nSub {
			idx[j] = uint32(t)
		}
	}
	return idx
}

func gather(S, R []float32, idx []uint32) {
	for s, n := range idx {
		copy(S[s*B:(s+1)*B], R[int(n)*B:int(n)*B+B])
	}
}

func kmeans(R, S []float32, idxSub []uint32, iters, chunk int, rng *rand.Rand) ([]float32, []uint8) {
	nSub := len(idxSub)
	gather(S, R, idxSub)
	C := make([]float32, K*B)
	kmeansPlusPlus(S[:nSub*B], nSub, C, rng)

	N := len(R) / B
	pi := make([]uint8, N)
	for iter := 0; iter < iters; iter++ {
		var sum [K][B]float64
		var count [K]int64
		assignAccum(R, C, pi, chunk, &sum, &count)
		updateMeans(C, &sum, &count)
		resplitEmpty(C, &count, rng)
	}
	assignOnly(R, C, pi, chunk)
	return C, pi
}

func kmeansPlusPlus(S []float32, nSub int, C []float32, rng *rand.Rand) {
	j0 := rng.IntN(nSub)
	copy(C[0:B], S[j0*B:(j0+1)*B])

	d2 := make([]float32, nSub)
	c0 := C[0:B]
	for s := 0; s < nSub; s++ {
		d2[s] = l2sq(S[s*B:(s+1)*B], c0)
	}

	for t := 1; t < K; t++ {
		var sigma float64
		for s := 0; s < nSub; s++ {
			sigma += float64(d2[s])
		}
		var pick int
		if sigma > 0 {
			r := rng.Float64() * sigma
			acc := 0.0
			pick = nSub - 1
			for s := 0; s < nSub; s++ {
				acc += float64(d2[s])
				if acc >= r {
					pick = s
					break
				}
			}
		} else {
			pick = rng.IntN(nSub)
		}
		copy(C[t*B:(t+1)*B], S[pick*B:(pick+1)*B])
		if t+1 == K {
			break
		}
		ct := C[t*B : (t+1)*B]
		for s := 0; s < nSub; s++ {
			dist := l2sq(S[s*B:(s+1)*B], ct)
			if dist < d2[s] {
				d2[s] = dist
			}
		}
	}
}

func l2sq(a, b []float32) float32 {
	var s float32
	for d := 0; d < B; d++ {
		diff := a[d] - b[d]
		s += diff * diff
	}
	return s
}

func centroidNorm2(C []float32, cnorm2 *[K]float32) {
	for j := 0; j < K; j++ {
		var s float32
		base := j * B
		for d := 0; d < B; d++ {
			v := C[base+d]
			s += v * v
		}
		cnorm2[j] = s
	}
}

// argminFused is fused argmin over 256 centroids. Ties keep the smaller index (strict <).
func argminFused(x []float32, C []float32, cnorm2 *[K]float32) uint8 {
	var dot0 float32
	for d := 0; d < B; d++ {
		dot0 += x[d] * C[d]
	}
	bestD := cnorm2[0] - (dot0 + dot0)
	bestJ := 0
	for j := 1; j < K; j++ {
		var dot float32
		base := j * B
		for d := 0; d < B; d++ {
			dot += x[d] * C[base+d]
		}
		d := cnorm2[j] - (dot + dot)
		if d < bestD {
			bestD = d
			bestJ = j
		}
	}
	return uint8(bestJ)
}

func assignOnly(R, C []float32, pi []uint8, chunk int) {
	N := len(pi)
	var cnorm2 [K]float32
	centroidNorm2(C, &cnorm2)
	for offset := 0; offset < N; {
		T := chunk
		if T > N-offset {
			T = N - offset
		}
		for t := 0; t < T; t++ {
			n := offset + t
			x := R[n*B : n*B+B]
			pi[n] = argminFused(x, C, &cnorm2)
		}
		offset += T
	}
}

func assignAccum(R, C []float32, pi []uint8, chunk int, sum *[K][B]float64, count *[K]int64) {
	N := len(pi)
	var cnorm2 [K]float32
	centroidNorm2(C, &cnorm2)
	for offset := 0; offset < N; {
		T := chunk
		if T > N-offset {
			T = N - offset
		}
		for t := 0; t < T; t++ {
			n := offset + t
			x := R[n*B : n*B+B]
			j := argminFused(x, C, &cnorm2)
			pi[n] = j
			count[j]++
			for d := 0; d < B; d++ {
				sum[j][d] += float64(x[d])
			}
		}
		offset += T
	}
}

func updateMeans(C []float32, sum *[K][B]float64, count *[K]int64) {
	for j := 0; j < K; j++ {
		cj := count[j]
		if cj == 0 {
			continue
		}
		den := float64(cj)
		base := j * B
		for d := 0; d < B; d++ {
			C[base+d] = float32(sum[j][d] / den)
		}
	}
}

func resplitEmpty(C []float32, count *[K]int64, rng *rand.Rand) {
	t := 0
	for j := 1; j < K; j++ {
		if count[j] > count[t] {
			t = j
		}
	}
	tbase := t * B
	const eps = 1e-5
	for j := 0; j < K; j++ {
		if count[j] != 0 {
			continue
		}
		base := j * B
		for d := 0; d < B; d++ {
			u := rng.Float64()
			C[base+d] = C[tbase+d] + float32((2*u-1)*eps)
		}
	}
}
