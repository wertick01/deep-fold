package chr0

import (
	"strconv"
	"strings"
)

const (
	Magic   = "CHR0"
	Version = 1
	TileRow = 64
	TileCol = 8
)

const maxHeader = 100_000_000
const maxTensors = 1_000_000
const maxName = 1024

// Header is the CHR0 v1 JSON root.
type Header struct {
	Magic            string            `json:"magic"`
	Version          int               `json:"version"`
	Arch             string            `json:"arch"`
	HiddenSize       int               `json:"hidden_size"`
	IntermediateSize int               `json:"intermediate_size"`
	NumLayers        int               `json:"num_layers"`
	VocabSize        int               `json:"vocab_size"`
	Tile             Tile              `json:"tile"`
	Tensors          map[string]Tensor `json:"tensors"`
}

// Tile is the Ampere 64×8 constant of this slice.
type Tile struct {
	Row      int `json:"row"`
	ColGroup int `json:"col_group"`
}

// Tensor is one tensors[name] object. Offsets are [start,end) from file start.
type Tensor struct {
	Layer        *int    `json:"layer,omitempty"`
	Kind         string  `json:"kind"`
	Codec        string  `json:"codec"`
	Shape        []int   `json:"shape"`
	GroupSize    int     `json:"group_size,omitempty"`
	NCodebooks   int     `json:"n_codebooks,omitempty"`
	CodebookBits int     `json:"codebook_bits,omitempty"`
	Data         []int64 `json:"data,omitempty"`
	Scale        []int64 `json:"scale,omitempty"`
	Zero         []int64 `json:"zero,omitempty"`
	Codebook     []int64 `json:"codebook,omitempty"`
	Index        []int64 `json:"index,omitempty"`
}

// WriteTensor is one tensor for Write: metadata without offsets, plus raw blobs.
type WriteTensor struct {
	Name     string
	Tensor   Tensor
	Data     []byte
	Scale    []byte
	Zero     []byte
	Codebook []byte
	Index    []byte
}

func isQKind(kind string) bool {
	switch kind {
	case "q", "k", "v", "o", "qkv", "gate", "up", "down", "embed", "lm_head":
		return true
	}
	return false
}

func inQ(name string, t Tensor) bool {
	return isQKind(t.Kind) && !strings.HasSuffix(name, ".bias")
}

// LayerIndex returns n if name has a dotted pair layers.<n>, first match left to right.
func LayerIndex(name string) (int, bool) {
	parts := strings.Split(name, ".")
	for i := 0; i+1 < len(parts); i++ {
		if parts[i] != "layers" {
			continue
		}
		n, err := strconv.Atoi(parts[i+1])
		if err != nil || n < 0 {
			continue
		}
		if parts[i+1] != strconv.Itoa(n) {
			continue
		}
		return n, true
	}
	return 0, false
}

func validKind(kind string) bool {
	switch kind {
	case "q", "k", "v", "o", "qkv", "gate", "up", "down", "embed", "lm_head", "norm", "other":
		return true
	}
	return false
}
