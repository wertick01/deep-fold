package verify

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"sort"

	"chr/internal/chr0"
	"chr/internal/safetensors"
)

// Thresholds are lossy gates. BF16 is always bit-exact and ignores these.
type Thresholds struct {
	RMSE   float64
	MaxAbs float64
}

func DefaultThresholds(codec string) Thresholds {
	switch codec {
	case "vq":
		return Thresholds{RMSE: 0.40, MaxAbs: 2.50}
	default:
		return Thresholds{RMSE: 0.08, MaxAbs: 0.50}
	}
}

// TensorReport is one verify row.
type TensorReport struct {
	Name   string  `json:"name"`
	Shape  []int   `json:"shape"`
	Codec  string  `json:"codec"`
	N      int     `json:"n"`
	RMSE   float64 `json:"rmse"`
	MAE    float64 `json:"mae"`
	MaxAbs float64 `json:"maxabs"`
	RMS    float64 `json:"rms"`
	Rel    float64 `json:"rel"`
	OK     bool    `json:"ok"`
}

// Report is the JSON object for --json.
type Report struct {
	Orig        string         `json:"orig"`
	CHR         string         `json:"chr"`
	LinearCodec string         `json:"linear_codec"`
	Thresholds  jsonThresholds `json:"thresholds"`
	Tensors     []TensorReport `json:"tensors"`
	Worst       *worstJSON     `json:"worst"`
	Summary     summaryJSON    `json:"summary"`
	Failed      []string       `json:"failed"`
	OK          bool           `json:"ok"`
	Exit        int            `json:"exit"`
}

type jsonThresholds struct {
	RMSE      float64 `json:"rmse"`
	MaxAbs    float64 `json:"maxabs"`
	BF16Exact bool    `json:"bf16_exact"`
}

type worstJSON struct {
	Name   string  `json:"name"`
	RMSE   float64 `json:"rmse"`
	MaxAbs float64 `json:"maxabs"`
}

type summaryJSON struct {
	Tensors       int `json:"tensors"`
	Lossy         int `json:"lossy"`
	BF16          int `json:"bf16"`
	Skipped       int `json:"skipped"`
	Fail          int `json:"fail"`
	MeanRMSELossy any `json:"mean_rmse_lossy"`
}

// Result is returned by Verify.
type Result struct {
	ExitCode int
	Report   Report
	Err      error
}

// VerifyOptions is the verify command.
type VerifyOptions struct {
	Orig, CHR string
	FailRMSE  *float64
	FailMax   *float64
	JSON      bool
	Quiet     bool
	Stdout    io.Writer
	Stderr    io.Writer
}

// Verify compares orig safetensors to a .chr. Exit 1 = contract, 2 = threshold.
func Verify(opt VerifyOptions) Result {
	cf, err := chr0.Open(opt.CHR)
	if err != nil {
		return Result{ExitCode: 1, Err: err}
	}
	defer cf.Close()
	h := cf.Header()
	linear := linearCodec(h)
	th := DefaultThresholds(linear)
	if opt.FailRMSE != nil {
		th.RMSE = *opt.FailRMSE
	}
	if opt.FailMax != nil {
		th.MaxAbs = *opt.FailMax
	}
	if th.RMSE < 0 || th.MaxAbs < 0 || math.IsNaN(th.RMSE) || math.IsNaN(th.MaxAbs) {
		return Result{ExitCode: 1, Err: fmt.Errorf("negative or NaN threshold")}
	}

	res, err := safetensors.ResolveInput(opt.Orig)
	if err != nil {
		return Result{ExitCode: 1, Err: err}
	}

	type origRef struct {
		hf    string
		canon string
		class Class
		kind  string
	}
	var origList []origRef
	skipped := 0
	err = res.ForEachTensor(func(hfName string, f *safetensors.File, meta safetensors.TensorMeta) error {
		canon := CanonicalName(hfName)
		kind, class := Classify(canon)
		kind, class = ApplyRank(kind, class, len(meta.Shape))
		if class == ClassSkip {
			skipped++
			if _, ok := h.Tensors[canon]; ok {
				return fmt.Errorf("extra tensor in chr: %s", canon)
			}
			return nil
		}
		origList = append(origList, origRef{hf: hfName, canon: canon, class: class, kind: kind})
		return nil
	})
	if err != nil {
		return Result{ExitCode: 1, Err: err}
	}

	visited := map[string]struct{}{}
	var rows []TensorReport
	fail := 0
	var sumRMSE float64
	var nLossy, nBF16 int

	// Re-walk orig to load one tensor at a time (same order as names we'll sort later).
	byCanon := map[string]origRef{}
	for _, o := range origList {
		byCanon[o.canon] = o
		if _, ok := h.Tensors[o.canon]; !ok {
			return Result{ExitCode: 1, Err: fmt.Errorf("missing tensor in chr: %s", o.canon)}
		}
	}
	for name := range h.Tensors {
		if _, ok := byCanon[name]; !ok {
			return Result{ExitCode: 1, Err: fmt.Errorf("extra tensor in chr: %s", name)}
		}
	}

	err = res.ForEachTensor(func(hfName string, f *safetensors.File, meta safetensors.TensorMeta) error {
		canon := CanonicalName(hfName)
		ref, ok := byCanon[canon]
		if !ok {
			return nil // skip
		}
		raw, err := readMeta(f, meta)
		if err != nil {
			return err
		}
		orig, err := safetensors.ToF32(meta.DType, raw)
		if err != nil {
			return fmt.Errorf("unsupported dtype %s (%s)", meta.DType, canon)
		}
		t, hat, err := decodeTensor(cf, canon)
		if err != nil {
			return err
		}
		if !shapeEq(t.Shape, meta.Shape) {
			return fmt.Errorf("shape mismatch: %s orig=%v chr=%v", canon, meta.Shape, t.Shape)
		}
		st := ComputeStats(orig, hat)
		row := TensorReport{
			Name:   canon,
			Shape:  append([]int(nil), t.Shape...),
			Codec:  t.Codec,
			N:      st.N,
			RMSE:   st.RMSE,
			MAE:    st.MAE,
			MaxAbs: st.MaxAbs,
			RMS:    st.RMS,
			Rel:    st.Rel,
			OK:     true,
		}
		if t.Codec == "bf16" {
			nBF16++
			if !BF16Exact(orig, hat) {
				row.OK = false
			}
		} else {
			nLossy++
			sumRMSE += st.RMSE
			if st.RMSE > th.RMSE || st.MaxAbs > th.MaxAbs {
				row.OK = false
			}
		}
		if !row.OK {
			fail++
		}
		rows = append(rows, row)
		visited[canon] = struct{}{}
		_ = ref
		return nil
	})
	if err != nil {
		return Result{ExitCode: 1, Err: err}
	}

	sort.Slice(rows, func(i, j int) bool { return rows[i].Name < rows[j].Name })
	var failed []string
	for _, r := range rows {
		if !r.OK {
			failed = append(failed, r.Name)
		}
	}
	var mean any = "n/a"
	if nLossy > 0 {
		mean = sumRMSE / float64(nLossy)
	}
	rep := Report{
		Orig:        opt.Orig,
		CHR:         opt.CHR,
		LinearCodec: linear,
		Thresholds:  jsonThresholds{RMSE: th.RMSE, MaxAbs: th.MaxAbs, BF16Exact: true},
		Tensors:     rows,
		Summary: summaryJSON{
			Tensors:       len(rows),
			Lossy:         nLossy,
			BF16:          nBF16,
			Skipped:       skipped,
			Fail:          fail,
			MeanRMSELossy: mean,
		},
		Failed: failed,
	}
	rep.Worst = pickWorst(rows)
	code := 0
	if fail > 0 {
		code = 2
	}
	rep.Exit = code
	rep.OK = code == 0
	if opt.JSON && opt.Stdout != nil {
		enc := json.NewEncoder(opt.Stdout)
		enc.SetEscapeHTML(false)
		if err := enc.Encode(rep); err != nil {
			return Result{ExitCode: 1, Err: err, Report: rep}
		}
	} else if opt.Stdout != nil {
		writeHuman(opt.Stdout, rep, opt.Quiet)
	}
	return Result{ExitCode: code, Report: rep}
}

func linearCodec(h chr0.Header) string {
	for name, t := range h.Tensors {
		if t.Codec == "nf4" || t.Codec == "vq" {
			return t.Codec
		}
		_ = name
	}
	return "nf4"
}

func shapeEq(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func pickWorst(rows []TensorReport) *worstJSON {
	if len(rows) == 0 {
		return nil
	}
	best := 0
	for i := 1; i < len(rows); i++ {
		r, b := rows[i], rows[best]
		if r.RMSE > b.RMSE || (r.RMSE == b.RMSE && r.MaxAbs > b.MaxAbs) || (r.RMSE == b.RMSE && r.MaxAbs == b.MaxAbs && r.Name < b.Name) {
			best = i
		}
	}
	return &worstJSON{Name: rows[best].Name, RMSE: rows[best].RMSE, MaxAbs: rows[best].MaxAbs}
}

func writeHuman(w io.Writer, rep Report, quiet bool) {
	if !quiet {
		fmt.Fprintf(w, "chr verify\n")
		fmt.Fprintf(w, "orig:        %s\n", rep.Orig)
		fmt.Fprintf(w, "chr:         %s\n", rep.CHR)
		fmt.Fprintf(w, "linear_codec: %s\n", rep.LinearCodec)
		fmt.Fprintf(w, "thresholds:  rmse<=%.2f  maxabs<=%.2f  bf16=exact\n\n", rep.Thresholds.RMSE, rep.Thresholds.MaxAbs)
		fmt.Fprintf(w, "name  shape  codec  n  rmse  mae  maxabs  rms  rel  status\n")
		for _, r := range rep.Tensors {
			st := "ok"
			if !r.OK {
				st = "FAIL"
			}
			fmt.Fprintf(w, "%s  %v  %s  %d  %.6e  %.6e  %.6e  %.6e  %.2e  %s\n",
				r.Name, r.Shape, r.Codec, r.N, r.RMSE, r.MAE, r.MaxAbs, r.RMS, r.Rel, st)
		}
		fmt.Fprintln(w)
	}
	if rep.Worst != nil {
		fmt.Fprintf(w, "worst: %s  rmse=%.6e  maxabs=%.6e\n", rep.Worst.Name, rep.Worst.RMSE, rep.Worst.MaxAbs)
	} else {
		fmt.Fprintf(w, "worst: (none)\n")
	}
	mean := "n/a"
	if m, ok := rep.Summary.MeanRMSELossy.(float64); ok {
		mean = fmt.Sprintf("%.6e", m)
	}
	fmt.Fprintf(w, "summary: tensors=%d  lossy=%d  bf16=%d  skipped=%d  fail=%d  mean_rmse_lossy=%s\n",
		rep.Summary.Tensors, rep.Summary.Lossy, rep.Summary.BF16, rep.Summary.Skipped, rep.Summary.Fail, mean)
	if rep.OK {
		fmt.Fprintf(w, "PASS\n")
	} else {
		fmt.Fprintf(w, "FAIL\n")
	}
}

func sortStrings(n []string) {
	sort.Strings(n)
}
