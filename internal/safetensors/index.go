package safetensors

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// Index is model.safetensors.index.json. The metadata key is ignored.
type Index struct {
	WeightMap map[string]string // HF name → cleaned relative shard path
}

// ResolvedInput is a single shard or an index of shards (LRU=1).
type ResolvedInput struct {
	Base      string            // directory containing shards
	Single    string            // absolute path when the input is one file
	WeightMap map[string]string // HF name → cleaned relative path; nil for a single file
	Shards    []string          // absolute shard paths, unique, sorted by relative UTF-8
	shardRel  []string          // cleaned relative paths, parallel to Shards
}

// ResolveInput implements integrity-cli §1.7 and chr0.md §5.3–5.4.
func ResolveInput(p string) (*ResolvedInput, error) {
	fi, err := os.Stat(p)
	if err != nil {
		return nil, err
	}
	if fi.IsDir() {
		return resolveDir(p)
	}
	name := filepath.Base(p)
	if strings.HasSuffix(name, ".safetensors.index.json") {
		return resolveIndexFile(p)
	}
	if strings.HasSuffix(name, ".safetensors") {
		return resolveSingle(p)
	}
	// File whose header is safetensors (integrity-cli §1.7).
	sf, err := Open(p)
	if err != nil {
		return nil, fmt.Errorf("ambiguous or unsupported input: %s", p)
	}
	sf.Close()
	return resolveSingle(p)
}

func resolveSingle(p string) (*ResolvedInput, error) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return nil, err
	}
	return &ResolvedInput{
		Base:   filepath.Dir(abs),
		Single: abs,
		Shards: []string{abs},
	}, nil
}

func resolveDir(dir string) (*ResolvedInput, error) {
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if st, err := os.Stat(indexPath); err == nil && !st.IsDir() {
		return resolveIndexFile(indexPath)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	var sts []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(e.Name(), ".safetensors") {
			sts = append(sts, e.Name())
		}
	}
	if len(sts) == 1 {
		return resolveSingle(filepath.Join(dir, sts[0]))
	}
	model := filepath.Join(dir, "model.safetensors")
	if st, err := os.Stat(model); err == nil && !st.IsDir() {
		return resolveSingle(model)
	}
	return nil, fmt.Errorf("ambiguous safetensors directory")
}

func resolveIndexFile(p string) (*ResolvedInput, error) {
	abs, err := filepath.Abs(p)
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return nil, err
	}
	idx, err := ParseIndex(data)
	if err != nil {
		return nil, err
	}
	dir := filepath.Dir(abs)
	uniq := make(map[string]struct{})
	var rels []string
	for _, rel := range idx.WeightMap {
		if _, ok := uniq[rel]; ok {
			continue
		}
		uniq[rel] = struct{}{}
		rels = append(rels, rel)
	}
	sort.Strings(rels)
	shards := make([]string, len(rels))
	for i, rel := range rels {
		absShard := filepath.Join(dir, filepath.FromSlash(rel))
		st, err := os.Stat(absShard)
		if err != nil {
			return nil, fmt.Errorf("shard %s: %w", absShard, err)
		}
		if st.IsDir() {
			return nil, fmt.Errorf("shard %s: is a directory", absShard)
		}
		shards[i] = absShard
	}
	return &ResolvedInput{
		Base:      dir,
		WeightMap: idx.WeightMap,
		Shards:    shards,
		shardRel:  rels,
	}, nil
}

// ParseIndex parses index.json bytes. Duplicate weight_map keys are an error.
func ParseIndex(data []byte) (*Index, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, fmt.Errorf("index.json: not an object")
	}
	var wm json.RawMessage
	found := false
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		key, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("json_invalid")
		}
		if key == "weight_map" {
			if found {
				return nil, fmt.Errorf("duplicate key weight_map")
			}
			found = true
			if err := dec.Decode(&wm); err != nil {
				return nil, fmt.Errorf("json_invalid: %w", err)
			}
			continue
		}
		var skip json.RawMessage
		if err := dec.Decode(&skip); err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
	}
	if !found {
		return nil, fmt.Errorf("index.json: missing weight_map")
	}
	m, err := parseWeightMap(wm)
	if err != nil {
		return nil, err
	}
	if len(m) == 0 {
		return nil, fmt.Errorf("index.json: empty weight_map")
	}
	return &Index{WeightMap: m}, nil
}

func parseWeightMap(raw []byte) (map[string]string, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	if tok != json.Delim('{') {
		return nil, fmt.Errorf("weight_map must be object")
	}
	out := make(map[string]string)
	for dec.More() {
		kt, err := dec.Token()
		if err != nil {
			return nil, fmt.Errorf("json_invalid: %w", err)
		}
		name, ok := kt.(string)
		if !ok {
			return nil, fmt.Errorf("json_invalid")
		}
		if _, dup := out[name]; dup {
			return nil, fmt.Errorf("duplicate weight_map key %q", name)
		}
		var val string
		if err := dec.Decode(&val); err != nil {
			return nil, fmt.Errorf("weight_map[%s]: %w", name, err)
		}
		cleaned, err := cleanShardRel(val)
		if err != nil {
			return nil, err
		}
		out[name] = cleaned
	}
	if _, err := dec.Token(); err != nil {
		return nil, fmt.Errorf("json_invalid: %w", err)
	}
	return out, nil
}

func cleanShardRel(p string) (string, error) {
	if p == "" {
		return "", fmt.Errorf("empty shard path")
	}
	slashed := filepath.ToSlash(p)
	if strings.HasPrefix(slashed, "/") {
		return "", fmt.Errorf("shard path must be relative: %s", p)
	}
	cleaned := path.Clean(slashed)
	if cleaned == ".." || strings.HasPrefix(cleaned, "../") {
		return "", fmt.Errorf("shard path escapes: %s", p)
	}
	return cleaned, nil
}

// ForEachShard opens each shard in order, one at a time (LRU=1).
func (r *ResolvedInput) ForEachShard(fn func(absPath string, f *File) error) error {
	for _, abs := range r.Shards {
		f, err := Open(abs)
		if err != nil {
			return err
		}
		err = fn(abs, f)
		cErr := f.Close()
		if err != nil {
			return err
		}
		if cErr != nil {
			return cErr
		}
	}
	return nil
}

// ForEachTensor walks tensors with one open shard (LRU=1).
// With an index: ST header order inside each sorted shard, only names mapped to that shard.
// A tensor in weight_map that never appears is an error. Extra tensors in a shard are skipped.
func (r *ResolvedInput) ForEachTensor(fn func(hfName string, f *File, meta TensorMeta) error) error {
	seen := make(map[string]struct{})
	for i, abs := range r.Shards {
		f, err := Open(abs)
		if err != nil {
			return err
		}
		var rel string
		if r.WeightMap != nil {
			rel = r.shardRel[i]
		}
		err = func() error {
			for _, meta := range f.List() {
				if r.WeightMap != nil {
					mapped, ok := r.WeightMap[meta.Name]
					if !ok || mapped != rel {
						continue
					}
				}
				seen[meta.Name] = struct{}{}
				if err := fn(meta.Name, f, meta); err != nil {
					return err
				}
			}
			return nil
		}()
		cErr := f.Close()
		if err != nil {
			return err
		}
		if cErr != nil {
			return cErr
		}
	}
	if r.WeightMap != nil {
		var missing []string
		for name := range r.WeightMap {
			if _, ok := seen[name]; !ok {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			sort.Strings(missing)
			return fmt.Errorf("missing tensors in shards: %s", strings.Join(missing, ", "))
		}
	}
	return nil
}
