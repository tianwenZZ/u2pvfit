#!/usr/bin/env bash

#set -euo pipefail

output_dir="computing_performance"
profile_base="${output_dir}/allen_profile"
stats_base="${output_dir}/allen_stats"
gpu_stats_base="${output_dir}/allen_gpu_stats"
nsys_bin="/home/tzhou/opt/cuda-12.4.1/bin/nsys"
ncu_bin="/home/tzhou/opt/cuda-12.4.1/bin/ncu"
kernels=(
  pv_beamline_prepare_tracks
  pv_beamline_histo
  pv_beamline_peak
  pv_beamline_assign_tracks
  pv_beamline_fitter
  pv_beamline_merge_vertices
)

mkdir -p "${output_dir}"

Allen/run "${nsys_bin}" profile \
   --output "${profile_base}" \
   --force-overwrite true \
   --trace=cuda,nvtx,osrt \
   --sample=cpu \
   python Allen/Dumpers/BinaryDumpers/options/allen.py \
    --binary-geometry \
    -g Allen/input/allen_geometries/geometry_sim-20231017-vc-mu100 \
    --sequence upgrade2_tv_cheated_tracking \
    --register-monitoring-counters 1 \
    --python-hlt1-node node \
    --enable-monitoring-printing 1 \
    -s 1 \
    --device 0 \
    --params "$PARAMFILESROOT" \
    --mdf merged_t50ps.mdf \
    --tags "dddb-20171122,sim-20180530-vc-md100"

# Export the default Nsight Systems summary reports.
"${nsys_bin}" stats \
   --force-export true \
   --force-overwrite true \
   --output "${stats_base}" \
   "${profile_base}.nsys-rep"

# Export kernel summary, GPU timeline trace, and CUDA API summary.
"${nsys_bin}" stats \
   --force-export true \
   --force-overwrite true \
   --report cuda_gpu_kern_sum,cuda_gpu_trace,cuda_api_sum \
   --format column \
   --output "${gpu_stats_base}" \
   "${profile_base}.nsys-rep"


for tag in "${kernels[@]}"; do
  Allen/run "${ncu_bin}" \
    --target-processes all \
    --kernel-name regex:".*${tag}.*" \
    --set full \
    --import-source yes \
    --source-folders /home/tzhou/workdir/stack-run5gpu-260515/Allen \
    -f -o "${output_dir}/ncu_${tag}_full" \
     python Allen/Dumpers/BinaryDumpers/options/allen.py \
      --binary-geometry \
      -g Allen/input/allen_geometries/geometry_sim-20231017-vc-mu100 \
      --sequence upgrade2_tv_cheated_tracking \
      --register-monitoring-counters 1 \
      --python-hlt1-node node \
      --enable-monitoring-printing 1 \
      -s 1 \
      --device 0 \
      --params "$PARAMFILESROOT" \
      --mdf merged_t50ps.mdf \
      --tags "dddb-20171122,sim-20180530-vc-md100"
done
