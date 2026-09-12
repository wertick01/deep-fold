package verify

import (
	"fmt"

	"chr/internal/chr0"
	"chr/internal/nf4"
	"chr/internal/safetensors"
	"chr/internal/vq"
)

func decodeTensor(cf *chr0.File, name string) (chr0.Tensor, []float32, error) {
	t, blobs, err := cf.Get(name)
	if err != nil {
		return t, nil, err
	}
	switch t.Codec {
	case "bf16":
		hat, err := safetensors.ToF32(safetensors.BF16, blobs["data"])
		return t, hat, err
	case "nf4":
		if err := nf4.CheckGroupSize(t.GroupSize); err != nil {
			return t, nil, err
		}
		if len(t.Shape) != 2 {
			return t, nil, fmt.Errorf("nf4 rank")
		}
		hat, err := nf4.Decode(blobs["data"], leToUint16(blobs["scale"]), t.Shape[0], t.Shape[1])
		return t, hat, err
	case "vq":
		if len(t.Shape) != 2 {
			return t, nil, fmt.Errorf("vq rank")
		}
		hat, err := vq.Decode(leToUint16(blobs["codebook"]), blobs["index"], t.Shape[0], t.Shape[1])
		return t, hat, err
	case "int4":
		return t, nil, fmt.Errorf("unsupported codec: %s (%s)", t.Codec, name)
	default:
		return t, nil, fmt.Errorf("unsupported codec: %s (%s)", t.Codec, name)
	}
}

func encodeLossy(codec string, w []float32, nOut, nIn int, seed uint64, iters, chunk int) (chr0.WriteTensor, error) {
	wt := chr0.WriteTensor{
		Tensor: chr0.Tensor{
			Codec: codec,
			Shape: []int{nOut, nIn},
		},
	}
	switch codec {
	case "nf4":
		if err := nf4.CheckGroupSize(64); err != nil {
			return wt, err
		}
		data, scale, err := nf4.Encode(w, nOut, nIn)
		if err != nil {
			return wt, err
		}
		wt.Tensor.GroupSize = 64
		wt.Data = data
		wt.Scale = uint16ToLE(scale)
	case "vq":
		opt := vq.Options{Seed: seed, Iters: iters, Chunk: chunk}
		cb, idx, err := vq.Encode(w, nOut, nIn, opt)
		if err != nil {
			return wt, err
		}
		wt.Tensor.GroupSize = 8
		wt.Tensor.NCodebooks = 2
		wt.Tensor.CodebookBits = 8
		wt.Codebook = uint16ToLE(cb)
		wt.Index = idx
	default:
		return wt, fmt.Errorf("unsupported codec: %s", codec)
	}
	return wt, nil
}
