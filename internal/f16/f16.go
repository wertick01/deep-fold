// Package f16 converts IEEE 754 binary16 (FP16) and bfloat16 to/from float32.
package f16

import "math"

// FromFloat32 converts f to IEEE 754 binary16 (round-to-nearest-even).
// Overflow becomes Inf; underflow becomes zero or a subnormal.
func FromFloat32(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b >> 23) & 0xff)
	mant := b & 0x7fffff

	if exp == 0xff { // Inf / NaN
		if mant != 0 {
			n := uint16(mant >> 13)
			if n == 0 {
				n = 1
			}
			return sign | 0x7c00 | n
		}
		return sign | 0x7c00
	}

	// Unbias float32, rebias for binary16.
	e := exp - 127 + 15
	switch {
	case e >= 0x1f:
		return sign | 0x7c00 // overflow → Inf
	case e <= 0:
		// Subnormal or zero in binary16.
		if e < -10 {
			return sign
		}
		mant |= 1 << 23
		// Hidden 1 sits at 2^-14 when e=0, so the extra shift is 13+(1-e)=14-e
		// (not 1-e). ε=1e-5 used by VQ resplit is a binary16 subnormal.
		shift := uint32(14 - e)
		round := (mant >> (shift - 1)) & 1
		tail := mant & ((1 << (shift - 1)) - 1)
		m := mant >> shift
		if round != 0 && (tail != 0 || m&1 != 0) {
			m++
		}
		return sign | uint16(m)
	default:
		half := sign | uint16(e<<10) | uint16(mant>>13)
		// Round: remaining bits of mantissa.
		remainder := mant & 0x1fff
		halfway := uint32(0x1000)
		if remainder > halfway || (remainder == halfway && half&1 != 0) {
			half++
		}
		return half
	}
}

// ToFloat32 expands a binary16 value to float32 (exact).
func ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32((h >> 10) & 0x1f)
	mant := uint32(h & 0x3ff)

	var f uint32
	switch exp {
	case 0:
		if mant == 0 {
			f = sign
		} else {
			// Subnormal: normalize.
			e := uint32(127 - 15 + 1)
			for mant&0x400 == 0 {
				mant <<= 1
				e--
			}
			mant &= 0x3ff
			f = sign | (e << 23) | (mant << 13)
		}
	case 0x1f:
		f = sign | 0x7f800000 | (mant << 13)
	default:
		f = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(f)
}

// FromBF16Bits expands bfloat16 bits to float32 (low 16 bits of mantissa zero).
func FromBF16Bits(h uint16) float32 {
	return math.Float32frombits(uint32(h) << 16)
}

// ToBF16Bits converts float32 to bfloat16 with round-to-nearest-even.
func ToBF16Bits(f float32) uint16 {
	b := math.Float32bits(f)
	if b&0x7fffffff > 0x7f800000 { // NaN
		n := uint16(b >> 16)
		if n&0x7f == 0 {
			n |= 1
		}
		return n
	}
	// round: add 0x7FFF + sticky LSB of result
	l := (b >> 16) & 1
	b += 0x7fff + l
	return uint16(b >> 16)
}
