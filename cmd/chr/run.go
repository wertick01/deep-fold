package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"

	"chr/internal/verify"
)

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintf(stderr, "usage: chr <compress|decode|verify> [flags]\n")
		return 1
	}
	if args[0] == "-h" || args[0] == "--help" {
		fmt.Fprintf(stdout, "chr compress|decode|verify\n")
		return 0
	}
	switch args[0] {
	case "compress":
		return runCompress(args[1:], stdout, stderr)
	case "decode":
		return runDecode(args[1:], stdout, stderr)
	case "verify":
		return runVerify(args[1:], stdout, stderr)
	default:
		fmt.Fprintf(stderr, "chr: unknown command %s\n", args[0])
		return 1
	}
}

func newFS(name string, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.SetOutput(stderr)
	return fs
}

func parseFS(fs *flag.FlagSet, args []string) int {
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return 0
		}
		return 1
	}
	return -1
}

func runCompress(args []string, stdout, stderr io.Writer) int {
	fs := newFS("compress", stderr)
	in := fs.String("in", "", "safetensors file, dir, or index.json")
	out := fs.String("out", "", "output .chr")
	codec := fs.String("codec", "", "nf4 or vq")
	group := fs.Int("group-size", 0, "must match codec canonical size")
	seed := fs.Int64("seed", 0, "VQ k-means seed")
	iters := fs.Int("iters", 20, "VQ Lloyd iterations")
	chunk := fs.Int("chunk", 262144, "VQ assignment chunk")
	stripeB := fs.Int64("stripe-bytes", 268435456, "F32 stripe threshold")
	stripeR := fs.Int("stripe-rows", 4096, "preferred stripe height")
	arch := fs.String("arch", "unknown", "CHR0 arch")
	hidden := fs.Int("hidden-size", 0, "header hidden_size")
	inter := fs.Int("intermediate-size", 0, "header intermediate_size")
	layers := fs.Int("num-layers", 0, "header num_layers")
	vocab := fs.Int("vocab-size", 0, "header vocab_size")
	quiet := fs.Bool("quiet", false, "no progress")
	if c := parseFS(fs, args); c >= 0 {
		return c
	}
	if *in == "" || *out == "" || *codec == "" {
		if *codec == "" && *in != "" && *out != "" {
			fmt.Fprintf(stderr, "chr: --codec is required\n")
			return 1
		}
		fmt.Fprintf(stderr, "chr: compress requires --in --out --codec\n")
		return 1
	}
	gs := *group
	if gs == 0 {
		if *codec == "nf4" {
			gs = 64
		} else if *codec == "vq" {
			gs = 8
		}
	}
	if *codec == "nf4" && gs != 64 {
		fmt.Fprintf(stderr, "chr: nf4 group-size must be 64\n")
		return 1
	}
	if *codec == "vq" && gs != 8 {
		fmt.Fprintf(stderr, "chr: vq group-size must be 8\n")
		return 1
	}
	if *iters < 1 {
		fmt.Fprintf(stderr, "chr: iters must be >= 1\n")
		return 1
	}
	if *chunk < 256 {
		fmt.Fprintf(stderr, "chr: chunk must be >= 256\n")
		return 1
	}
	err := verify.Compress(verify.CompressOptions{
		In: *in, Out: *out, Codec: *codec,
		GroupSize: gs, Seed: uint64(*seed), Iters: *iters, Chunk: *chunk,
		StripeBytes: *stripeB, StripeRows: *stripeR,
		Arch: *arch, HiddenSize: *hidden, Intermediate: *inter, NumLayers: *layers, VocabSize: *vocab,
		Quiet: *quiet, Log: stderr,
	})
	if err != nil {
		fmt.Fprintf(stderr, "chr: %v\n", err)
		_ = os.Remove(*out)
		return 1
	}
	_ = stdout
	return 0
}

func runDecode(args []string, stdout, stderr io.Writer) int {
	fs := newFS("decode", stderr)
	in := fs.String("in", "", "input .chr")
	out := fs.String("out", "", "output .safetensors")
	if c := parseFS(fs, args); c >= 0 {
		return c
	}
	if *in == "" || *out == "" {
		fmt.Fprintf(stderr, "chr: decode requires --in --out\n")
		return 1
	}
	if err := verify.Decode(*in, *out); err != nil {
		fmt.Fprintf(stderr, "chr: %v\n", err)
		return 1
	}
	_ = stdout
	return 0
}

func runVerify(args []string, stdout, stderr io.Writer) int {
	fs := newFS("verify", stderr)
	orig := fs.String("orig", "", "original safetensors")
	chrPath := fs.String("chr", "", "compressed .chr")
	failRMSE := fs.String("fail-rmse", "", "RMSE threshold")
	failMax := fs.String("fail-maxabs", "", "maxabs threshold")
	js := fs.Bool("json", false, "JSON report on stdout")
	quiet := fs.Bool("quiet", false, "summary only")
	if c := parseFS(fs, args); c >= 0 {
		return c
	}
	if *orig == "" || *chrPath == "" {
		fmt.Fprintf(stderr, "chr: verify requires --orig --chr\n")
		return 1
	}
	opt := verify.VerifyOptions{
		Orig: *orig, CHR: *chrPath, JSON: *js, Quiet: *quiet,
		Stdout: stdout, Stderr: stderr,
	}
	if *failRMSE != "" {
		v, err := strconv.ParseFloat(*failRMSE, 64)
		if err != nil || v != v || v < 0 {
			fmt.Fprintf(stderr, "chr: bad --fail-rmse\n")
			return 1
		}
		opt.FailRMSE = &v
	}
	if *failMax != "" {
		v, err := strconv.ParseFloat(*failMax, 64)
		if err != nil || v != v || v < 0 {
			fmt.Fprintf(stderr, "chr: bad --fail-maxabs\n")
			return 1
		}
		opt.FailMax = &v
	}
	r := verify.Verify(opt)
	if r.Err != nil && r.ExitCode == 1 {
		fmt.Fprintf(stderr, "chr: %v\n", r.Err)
	}
	return r.ExitCode
}
