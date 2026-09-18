Decode V2 CUDA-core GEMV / SwiGLU ncu
ncu=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.BAT
python=C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe
l2_bytes=25165824 iters=8
L2-rotated isolated shapes. Not TokenLoop. Not chr_nf4_gemm.
launch__grid_size omitted: ncu --import --csv dies (bad conversion) on (2048,1,1).
Do not paste these percentages into the README tok/s line.

export of existing .ncu-rep

Median over profiled launches after dropping ID 0 (warmup). Locale on this box uses a decimal comma in ncu CSV; summary.csv uses dots.
case                         kernel                         DRAM%  tensor%  occ%  regs  block  grid
qwen25-3b-o_proj-gemv        gemv_splitk                   23.45     0.00  82.37    40    128  2048
qwen25-3b-down_proj-gemv     gemv_splitk                   50.84     0.00  85.93    40    128  2048
qwen25-3b-swiglu-gemv        gemv_swiglu                   49.03     0.00  72.20    48    128  2752

These are ncu counters on isolated GEMV, not live tok/s.
