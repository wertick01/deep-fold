WAVE 10 K2 ncu — measured 2026-09-13 on RTX 3080 (sm_86, 70 SMs)

ncu=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.BAT
python=C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe
l2_bytes=25165824 iters=8
Weight buffers L2-rotated. Isolated one shape x N per ncu process.
Not TokenLoop. Not bench.py microseconds. Do not paste these into the README speed line.

After reboot, non-admin ncu works (RmProfilingAdminOnly=0). This plate is
that run: python -m gpu.nf4.ncu --profile, no UAC.

.ncu-rep is the Nsight report. *.metrics.csv is ncu --import --page details.
summary.csv is the median over launches after dropping ID 0 (warmup).
ncu CSV on this box uses a decimal comma; summary.csv uses dots.

metrics:
  dram__throughput.avg.pct_of_peak_sustained_elapsed
  dram__bytes_read.sum
  dram__bytes_write.sum
  sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__warps_active.avg.pct_of_peak_sustained_active
  launch__shared_mem_per_block_dynamic
  launch__registers_per_thread
  launch__block_size
  launch__grid_size

OK qwen25-3b-q_proj-n1
OK qwen25-3b-q_proj-n16
OK qwen25-3b-k_proj-n1
OK qwen25-3b-k_proj-n16

plan.py (same shapes):
  q_proj N=1  path=1 decode_small  grid=(32,4) ctas=128 smem=13056
  q_proj N=16 path=2 prefill_n16    grid=(32,4) ctas=128 smem=24576
  k_proj N=1  path=1 decode_small  grid=(4,16) ctas=64  smem=13056
  k_proj N=16 path=2 prefill_n16    grid=(4,16) ctas=64  smem=24576

case                         kernel                         DRAM%  tensor%  occ%  regs  block  grid
qwen25-3b-q_proj-n1          chr_nf4_gemm_decode_small      5.12     1.38  15.73    55    128   128
qwen25-3b-q_proj-n16         chr_nf4_gemm_prefill_n16       6.60     2.72  28.80    96    256   128
qwen25-3b-k_proj-n1          chr_nf4_gemm_decode_small      3.35     0.85  8.33    55    128    64
qwen25-3b-k_proj-n16         chr_nf4_gemm_prefill_n16       4.28     1.65  16.45    96    256    64

Re-export without profiling: python -m gpu.nf4.ncu --export
Elevated helper if counters lock again: gpu/nf4/ncu_admin.ps1
