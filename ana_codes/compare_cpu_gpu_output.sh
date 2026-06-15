cd /home/tzhou/workdir/stack-run5gpu-260515
#conda activate rootenv
python3 compare_root_outputs_cpu_gpu.py \
  --cpu-input histograms_1000evts_gausintegral_float_cpu.root \
  --gpu-input histograms_1000evts_gausintegral_single_thread_pv_gpu.root \
  --max-events 1000 \
  --output-dir compare_cpu_gpu_results/floatcpu_gpu_compare_1000evt