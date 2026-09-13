package verify

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	"chr/internal/chr0"
	"chr/internal/safetensors"
)

// CompressOptions is the compress command.
type CompressOptions struct {
	In, Out, Codec string
	GroupSize      int
	Seed           uint64
	Iters, Chunk   int
	StripeBytes    int64
	StripeRows     int
	Arch           string
	HiddenSize     int
	Intermediate   int
	NumLayers      int
	VocabSize      int
	Quiet          bool
	Log            io.Writer
}

func (o CompressOptions) logf(format string, args ...any) {
	if o.Quiet || o.Log == nil {
		return
	}
	fmt.Fprintf(o.Log, format, args...)
}

// Compress reads safetensors (possibly sharded) and writes a .chr.
func Compress(opt CompressOptions) error {
	if opt.Codec != "nf4" && opt.Codec != "vq" {
		return fmt.Errorf("unsupported codec %s", opt.Codec)
	}
	if opt.Codec == "nf4" && opt.GroupSize != 0 && opt.GroupSize != 64 {
		return fmt.Errorf("nf4 group-size must be 64")
	}
	if opt.Codec == "vq" && opt.GroupSize != 0 && opt.GroupSize != 8 {
		return fmt.Errorf("vq group-size must be 8")
	}
	if opt.Iters < 1 {
		return fmt.Errorf("iters must be >= 1")
	}
	if opt.Chunk < 256 {
		return fmt.Errorf("chunk must be >= 256")
	}
	res, err := safetensors.ResolveInput(opt.In)
	if err != nil {
		return err
	}
	var items []chr0.WriteTensor
	var skipped int
	err = res.ForEachTensor(func(hfName string, f *safetensors.File, meta safetensors.TensorMeta) error {
		canon := CanonicalName(hfName)
		kind, class := Classify(canon)
		kind, class = ApplyRank(kind, class, len(meta.Shape))
		if class == ClassSkip {
			skipped++
			return nil
		}
		raw, err := readMeta(f, meta)
		if err != nil {
			return err
		}
		w, err := safetensors.ToF32(meta.DType, raw)
		if err != nil {
			return fmt.Errorf("unsupported dtype %s (%s)", meta.DType, canon)
		}
		start := time.Now()
		var it chr0.WriteTensor
		switch class {
		case ClassStoreBF16:
			var blob []byte
			if meta.DType == safetensors.BF16 {
				blob = append([]byte(nil), raw...)
			} else {
				blob = encodeBF16(w)
			}
			it = chr0.WriteTensor{
				Name:   canon,
				Tensor: chr0.Tensor{Kind: kind, Codec: "bf16", Shape: append([]int(nil), meta.Shape...)},
				Data:   blob,
			}
		case ClassCompress:
			if len(meta.Shape) != 2 {
				return fmt.Errorf("compressable tensor must be rank 2: %s", canon)
			}
			it, err = encodeLossy(opt.Codec, w, meta.Shape[0], meta.Shape[1], opt.Seed, opt.Iters, opt.Chunk)
			if err != nil {
				if isNonFinite(err) {
					return fmt.Errorf("non-finite value in tensor %s", canon)
				}
				return fmt.Errorf("%s: %w", canon, err)
			}
			it.Name = canon
			it.Tensor.Kind = kind
		default:
			return fmt.Errorf("internal class")
		}
		if n, ok := chr0.LayerIndex(canon); ok {
			n := n
			it.Tensor.Layer = &n
		}
		items = append(items, it)
		opt.logf("compress  %s  %v  %s  %s\n", canon, meta.Shape, it.Tensor.Codec, time.Since(start).Truncate(time.Millisecond))
		return nil
	})
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return fmt.Errorf("no tensors to write")
	}
	h := chr0.Header{
		Arch:             opt.Arch,
		HiddenSize:       opt.HiddenSize,
		IntermediateSize: opt.Intermediate,
		NumLayers:        opt.NumLayers,
		VocabSize:        opt.VocabSize,
	}
	if h.Arch == "" {
		h.Arch = "unknown"
	}
	applyConfigJSON(res.Base, &h)
	deriveHeader(&h, items)
	if err := chr0.Write(opt.Out, h, items); err != nil {
		return err
	}
	opt.logf("wrote %s tensors=%d\n", opt.Out, len(items))
	_ = skipped
	return nil
}

func readMeta(f *safetensors.File, meta safetensors.TensorMeta) ([]byte, error) {
	_, raw, err := f.Read(meta.Name)
	return raw, err
}

func isNonFinite(err error) bool {
	if err == nil {
		return false
	}
	s := err.Error()
	return strings.Contains(s, "NaN") || strings.Contains(s, "Inf") || strings.Contains(s, "non-finite")
}

type hfConfig struct {
	ModelType        string `json:"model_type"`
	HiddenSize       int    `json:"hidden_size"`
	NEmbd            int    `json:"n_embd"`
	DModel           int    `json:"d_model"`
	IntermediateSize int    `json:"intermediate_size"`
	FFNDim           int    `json:"ffn_dim"`
	NInner           int    `json:"n_inner"`
	NumHiddenLayers  int    `json:"num_hidden_layers"`
	NLayer           int    `json:"n_layer"`
	NumLayers        int    `json:"num_layers"`
	VocabSize        int    `json:"vocab_size"`
}

func applyConfigJSON(dir string, h *chr0.Header) {
	b, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return
	}
	var c hfConfig
	if json.Unmarshal(b, &c) != nil {
		return
	}
	if h.Arch == "" || h.Arch == "unknown" {
		if c.ModelType != "" {
			h.Arch = c.ModelType
		}
	}
	if h.HiddenSize == 0 {
		h.HiddenSize = firstNonZero(c.HiddenSize, c.NEmbd, c.DModel)
	}
	if h.IntermediateSize == 0 {
		h.IntermediateSize = firstNonZero(c.IntermediateSize, c.FFNDim, c.NInner)
	}
	if h.NumLayers == 0 {
		h.NumLayers = firstNonZero(c.NumHiddenLayers, c.NLayer, c.NumLayers)
	}
	if h.VocabSize == 0 {
		h.VocabSize = c.VocabSize
	}
}

func firstNonZero(xs ...int) int {
	for _, x := range xs {
		if x != 0 {
			return x
		}
	}
	return 0
}

func deriveHeader(h *chr0.Header, items []chr0.WriteTensor) {
	maxL := -1
	for _, it := range items {
		if it.Tensor.Layer != nil && *it.Tensor.Layer > maxL {
			maxL = *it.Tensor.Layer
		}
		switch it.Tensor.Kind {
		case "norm":
			if h.HiddenSize == 0 && len(it.Tensor.Shape) == 1 {
				h.HiddenSize = it.Tensor.Shape[0]
			}
		case "embed":
			if h.HiddenSize == 0 && len(it.Tensor.Shape) >= 2 {
				h.HiddenSize = it.Tensor.Shape[1]
			}
			if h.VocabSize == 0 && len(it.Tensor.Shape) >= 1 {
				h.VocabSize = it.Tensor.Shape[0]
			}
		case "lm_head":
			if h.VocabSize == 0 && len(it.Tensor.Shape) >= 1 {
				h.VocabSize = it.Tensor.Shape[0]
			}
		case "q":
			if h.HiddenSize == 0 && len(it.Tensor.Shape) >= 2 {
				h.HiddenSize = it.Tensor.Shape[1]
			}
		case "down":
			if h.IntermediateSize == 0 && len(it.Tensor.Shape) >= 2 {
				h.IntermediateSize = it.Tensor.Shape[1]
			}
		case "up", "gate":
			if h.IntermediateSize == 0 && len(it.Tensor.Shape) >= 1 {
				h.IntermediateSize = it.Tensor.Shape[0]
			}
		}
	}
	if h.NumLayers == 0 && maxL >= 0 {
		h.NumLayers = maxL + 1
	}
	if h.HiddenSize < 1 {
		h.HiddenSize = 1
		for _, it := range items {
			if len(it.Tensor.Shape) >= 1 && it.Tensor.Shape[len(it.Tensor.Shape)-1] > 1 {
				h.HiddenSize = it.Tensor.Shape[len(it.Tensor.Shape)-1]
				break
			}
			if len(it.Tensor.Shape) == 1 {
				h.HiddenSize = it.Tensor.Shape[0]
				break
			}
		}
	}
	if h.Arch == "" {
		h.Arch = "unknown"
	}
}
