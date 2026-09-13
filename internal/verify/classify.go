package verify

import "strings"

// Class is how compress/verify treat a tensor.
type Class int

const (
	ClassSkip Class = iota
	ClassStoreBF16
	ClassCompress
)

// CanonicalName strips a trailing ".weight" component (once).
func CanonicalName(hf string) string {
	i := strings.LastIndex(hf, ".")
	if i >= 0 && hf[i+1:] == "weight" {
		return hf[:i]
	}
	return hf
}

func lastComponent(name string) string {
	i := strings.LastIndex(name, ".")
	if i < 0 {
		return name
	}
	return name[i+1:]
}

// Classify maps a CHR0-name (no .weight) to kind and class.
// Rank is applied afterwards: unknown rank-1 → store BF16; rank-2 other → compress.
func Classify(canon string) (kind string, class Class) {
	if strings.Contains(canon, "inv_freq") || strings.Contains(canon, "rotary_emb") ||
		strings.HasSuffix(canon, ".sin") || strings.HasSuffix(canon, ".cos") {
		return "other", ClassSkip
	}
	if strings.HasSuffix(canon, ".bias") {
		return "other", ClassStoreBF16
	}
	last := lastComponent(canon)
	switch last {
	case "embed_tokens", "tok_embeddings", "wte":
		return "embed", ClassCompress
	case "lm_head":
		return "lm_head", ClassCompress
	case "output":
		if !strings.Contains(canon, "norm") {
			return "lm_head", ClassCompress
		}
	case "q_proj":
		return "q", ClassCompress
	case "k_proj":
		return "k", ClassCompress
	case "v_proj":
		return "v", ClassCompress
	case "o_proj", "wo":
		return "o", ClassCompress
	case "wqkv":
		return "qkv", ClassCompress
	case "gate_proj", "w1":
		return "gate", ClassCompress
	case "up_proj", "w3":
		return "up", ClassCompress
	case "down_proj", "w2":
		return "down", ClassCompress
	}
	if strings.Contains(canon, "norm") || strings.Contains(canon, "layernorm") ||
		strings.Contains(canon, "layer_norm") || strings.Contains(canon, "ln_f") {
		return "norm", ClassStoreBF16
	}
	return "other", ClassCompress
}

// ApplyRank turns a name-only class into a store/compress decision using tensor rank.
func ApplyRank(kind string, class Class, rank int) (string, Class) {
	if class == ClassSkip {
		return kind, class
	}
	if class == ClassStoreBF16 {
		return kind, class
	}
	if rank == 2 {
		return kind, ClassCompress
	}
	return kind, ClassStoreBF16
}
