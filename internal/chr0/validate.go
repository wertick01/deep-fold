package chr0

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

func isJSONSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

func marshalCompact(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	b := buf.Bytes()
	if n := len(b); n > 0 && b[n-1] == '\n' {
		b = b[:n-1]
	}
	return b, nil
}

func parseHeaderJSON(js []byte) (Header, error) {
	if len(js) < 2 || js[0] != '{' {
		return Header{}, fmt.Errorf("json_invalid")
	}
	for _, c := range js {
		if c == 0 {
			return Header{}, fmt.Errorf("json_invalid")
		}
	}
	dec := json.NewDecoder(bytes.NewReader(js))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return Header{}, fmt.Errorf("json_invalid: %w", err)
	}
	if tok != json.Delim('{') {
		return Header{}, fmt.Errorf("json_invalid")
	}
	var h Header
	seen := make(map[string]struct{})
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return Header{}, fmt.Errorf("json_invalid: %w", err)
		}
		key, ok := kt.(string)
		if !ok {
			return Header{}, fmt.Errorf("json_invalid")
		}
		if _, ok := seen[key]; ok {
			return Header{}, fmt.Errorf("json_invalid: duplicate key %q", key)
		}
		seen[key] = struct{}{}
		switch key {
		case "magic":
			if err := dec.Decode(&h.Magic); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "version":
			if err := dec.Decode(&h.Version); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "arch":
			if err := dec.Decode(&h.Arch); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "hidden_size":
			if err := dec.Decode(&h.HiddenSize); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "intermediate_size":
			if err := dec.Decode(&h.IntermediateSize); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "num_layers":
			if err := dec.Decode(&h.NumLayers); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "vocab_size":
			if err := dec.Decode(&h.VocabSize); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "tile":
			dec.DisallowUnknownFields()
			if err := dec.Decode(&h.Tile); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
		case "tensors":
			var raw json.RawMessage
			if err := dec.Decode(&raw); err != nil {
				return Header{}, fmt.Errorf("json_invalid: %w", err)
			}
			ts, err := parseTensors(raw)
			if err != nil {
				return Header{}, err
			}
			h.Tensors = ts
		default:
			return Header{}, fmt.Errorf("json_invalid: unknown field %q", key)
		}
	}
	if _, err := dec.Token(); err != nil {
		return Header{}, fmt.Errorf("json_invalid: %w", err)
	}
	rest := js[dec.InputOffset():]
	for _, c := range rest {
		if !isJSONSpace(c) {
			return Header{}, fmt.Errorf("json_invalid: trailing data")
		}
	}
	required := []string{"magic", "version", "arch", "hidden_size", "intermediate_size", "num_layers", "vocab_size", "tile", "tensors"}
	for _, k := range required {
		if _, ok := seen[k]; !ok {
			return Header{}, fmt.Errorf("json_invalid: missing %s", k)
		}
	}
	return h, nil
}

func parseTensors(raw []byte) (map[string]Tensor, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	dec.DisallowUnknownFields()
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, fmt.Errorf("json_invalid: tensors must be object")
	}
	out := make(map[string]Tensor)
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		name, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("json_invalid")
		}
		if err := checkName(name); err != nil {
			return nil, err
		}
		if _, dup := out[name]; dup {
			return nil, fmt.Errorf("duplicate tensor name %q", name)
		}
		var t Tensor
		if err := dec.Decode(&t); err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		t.Shape = append([]int(nil), t.Shape...)
		out[name] = t
	}
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no tensors")
	}
	if len(out) > maxTensors {
		return nil, fmt.Errorf("too many tensors")
	}
	return out, nil
}

func checkName(name string) error {
	if !utf8.ValidString(name) {
		return fmt.Errorf("invalid tensor name")
	}
	n := len(name)
	if n < 1 || n > maxName {
		return fmt.Errorf("tensor name length")
	}
	for i := 0; i < n; i++ {
		if name[i] < 0x20 {
			return fmt.Errorf("tensor name control char")
		}
	}
	return nil
}

type interval struct {
	start, end int64
	who        string
}

func validateHeader(h Header) error {
	if h.Magic != Magic {
		return fmt.Errorf("bad magic %q", h.Magic)
	}
	if h.Version != Version {
		return fmt.Errorf("unsupported version")
	}
	if h.Arch == "" || !utf8.ValidString(h.Arch) {
		return fmt.Errorf("arch")
	}
	if h.HiddenSize < 1 {
		return fmt.Errorf("hidden_size")
	}
	if h.IntermediateSize < 0 || h.NumLayers < 0 || h.VocabSize < 0 {
		return fmt.Errorf("negative size field")
	}
	if h.Tile.Row != TileRow || h.Tile.ColGroup != TileCol {
		return fmt.Errorf("tile")
	}
	if len(h.Tensors) == 0 {
		return fmt.Errorf("no tensors")
	}
	if len(h.Tensors) > maxTensors {
		return fmt.Errorf("too many tensors")
	}

	var qCodec string
	var spans []interval
	for name, t := range h.Tensors {
		if err := checkName(name); err != nil {
			return err
		}
		if err := validateTensor(name, t, h.NumLayers); err != nil {
			return err
		}
		if inQ(name, t) {
			if qCodec == "" {
				qCodec = t.Codec
			} else if t.Codec != qCodec {
				return fmt.Errorf("mixed codec in Q: %s and %s", qCodec, t.Codec)
			}
		}
		for _, sp := range tensorSpans(name, t) {
			spans = append(spans, sp)
		}
	}
	if qCodec != "" {
		switch qCodec {
		case "nf4", "vq", "int4", "bf16":
		default:
			return fmt.Errorf("unsupported codec %s", qCodec)
		}
	}
	sort.Slice(spans, func(i, j int) bool {
		if spans[i].start != spans[j].start {
			return spans[i].start < spans[j].start
		}
		return spans[i].end < spans[j].end
	})
	for i := 1; i < len(spans); i++ {
		if spans[i].start < spans[i-1].end {
			return fmt.Errorf("overlapping blobs %s and %s", spans[i-1].who, spans[i].who)
		}
	}
	return nil
}

func tensorSpans(name string, t Tensor) []interval {
	var out []interval
	add := func(field string, sp []int64) {
		if len(sp) == 2 {
			out = append(out, interval{start: sp[0], end: sp[1], who: name + "." + field})
		}
	}
	add("data", t.Data)
	add("scale", t.Scale)
	add("zero", t.Zero)
	add("codebook", t.Codebook)
	add("index", t.Index)
	return out
}

func validateTensor(name string, t Tensor, numLayers int) error {
	if !validKind(t.Kind) {
		return fmt.Errorf("kind %q (%s)", t.Kind, name)
	}
	if err := checkShape(t.Shape); err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	wantLayer, hasLayerSeg := LayerIndex(name)
	if hasLayerSeg {
		if t.Layer == nil {
			return fmt.Errorf("layer missing for %s", name)
		}
		if *t.Layer != wantLayer {
			return fmt.Errorf("layer mismatch for %s", name)
		}
		if *t.Layer < 0 || *t.Layer >= numLayers {
			return fmt.Errorf("layer out of range for %s", name)
		}
	} else if t.Layer != nil {
		return fmt.Errorf("unexpected layer for %s", name)
	}

	bias := strings.HasSuffix(name, ".bias")
	if t.Kind == "norm" || t.Kind == "other" || bias {
		if t.Codec != "bf16" {
			return fmt.Errorf("codec %s not allowed for %s", t.Codec, name)
		}
	}

	switch t.Codec {
	case "bf16":
		if t.GroupSize != 0 || t.NCodebooks != 0 || t.CodebookBits != 0 {
			return fmt.Errorf("%s: bf16 extra fields", name)
		}
		if len(t.Scale) != 0 || len(t.Zero) != 0 || len(t.Codebook) != 0 || len(t.Index) != 0 {
			return fmt.Errorf("%s: bf16 extra blobs", name)
		}
		want, err := BF16BlobBytes(t.Shape)
		if err != nil {
			return err
		}
		return checkSpan(t.Data, want, name+".data")
	case "nf4":
		return validateNF4(name, t, false)
	case "int4":
		return validateNF4(name, t, true)
	case "vq":
		return validateVQ(name, t)
	default:
		return fmt.Errorf("unsupported codec %s (%s)", t.Codec, name)
	}
}

func validateNF4(name string, t Tensor, int4 bool) error {
	if len(t.Shape) != 2 {
		return fmt.Errorf("%s: quantized tensor must be rank 2", name)
	}
	if t.GroupSize != 64 {
		return fmt.Errorf("%s: group_size must be 64", name)
	}
	if t.NCodebooks != 0 || t.CodebookBits != 0 {
		return fmt.Errorf("%s: extra codebook fields", name)
	}
	if len(t.Codebook) != 0 || len(t.Index) != 0 {
		return fmt.Errorf("%s: extra vq blobs", name)
	}
	if !int4 && len(t.Zero) != 0 {
		return fmt.Errorf("%s: zero not allowed", name)
	}
	data, scale, err := NF4BlobBytes(t.Shape)
	if err != nil {
		return err
	}
	if err := checkSpan(t.Data, data, name+".data"); err != nil {
		return err
	}
	if err := checkSpan(t.Scale, scale, name+".scale"); err != nil {
		return err
	}
	if int4 && len(t.Zero) != 0 {
		if err := checkSpan(t.Zero, scale, name+".zero"); err != nil {
			return err
		}
	}
	return nil
}

func validateVQ(name string, t Tensor) error {
	if len(t.Shape) != 2 {
		return fmt.Errorf("%s: quantized tensor must be rank 2", name)
	}
	if t.GroupSize != 8 {
		return fmt.Errorf("%s: group_size must be 8", name)
	}
	if t.NCodebooks != 2 {
		return fmt.Errorf("%s: n_codebooks must be 2", name)
	}
	if t.CodebookBits != 8 {
		return fmt.Errorf("%s: codebook_bits must be 8", name)
	}
	if len(t.Data) != 0 || len(t.Scale) != 0 || len(t.Zero) != 0 {
		return fmt.Errorf("%s: extra blobs", name)
	}
	cb, idx, err := VQBlobBytes(t.Shape)
	if err != nil {
		return err
	}
	if err := checkSpan(t.Codebook, cb, name+".codebook"); err != nil {
		return err
	}
	return checkSpan(t.Index, idx, name+".index")
}

func checkSpan(span []int64, want int64, who string) error {
	if len(span) != 2 {
		return fmt.Errorf("%s: offset pair", who)
	}
	start, end := span[0], span[1]
	if start < 0 || end <= start {
		return fmt.Errorf("%s: bad range", who)
	}
	if start%64 != 0 {
		return fmt.Errorf("%s: start not aligned 64", who)
	}
	if end-start != want {
		return fmt.Errorf("%s: blob size mismatch: got %d want %d", who, end-start, want)
	}
	return nil
}

func validateAgainstFile(h Header, nJSON, fileSize int64) error {
	headerEnd := Align64(8 + nJSON)
	for name, t := range h.Tensors {
		for _, sp := range tensorSpans(name, t) {
			if sp.start < headerEnd {
				return fmt.Errorf("%s overlaps header", sp.who)
			}
			if sp.end > fileSize {
				return fmt.Errorf("truncated")
			}
		}
	}
	return nil
}
