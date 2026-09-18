@echo off
set NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.2.0\ncu.BAT
set PY=C:\Users\Professional\anaconda3\envs\torch-gpu\python.exe
set METRICS=dram__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__shared_mem_per_block_dynamic,launch__registers_per_thread,launch__block_size
call "%NCU%" --target-processes all --kernel-name-base demangled --kernel-name regex:gemv_splitk --metrics "%METRICS%" --log-file C:\dev\deep-fold\docs\runs\ncu-gemv\qwen25-3b-down_proj-gemv.log -o C:\dev\deep-fold\docs\runs\ncu-gemv\qwen25-3b-down_proj-gemv.ncu-rep --force-overwrite %PY% -m gpu.nf4.bench_gemv_ncu --kind down_proj --iters 4 --reps 1 --l2-bytes 25165824
