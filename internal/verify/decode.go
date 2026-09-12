package verify

import (
	"chr/internal/chr0"
	"chr/internal/safetensors"
)

// Decode writes every tensor in the .chr as one F32 safetensors file.
func Decode(inCHR, outST string) error {
	cf, err := chr0.Open(inCHR)
	if err != nil {
		return err
	}
	defer cf.Close()
	h := cf.Header()
	names := make([]string, 0, len(h.Tensors))
	for name := range h.Tensors {
		names = append(names, name)
	}
	sortStrings(names)
	out := make([]safetensors.F32Tensor, 0, len(names))
	for _, name := range names {
		t, hat, err := decodeTensor(cf, name)
		if err != nil {
			return err
		}
		out = append(out, safetensors.F32Tensor{Name: name, Shape: append([]int(nil), t.Shape...), Data: hat})
	}
	return safetensors.WriteF32(outST, out)
}
