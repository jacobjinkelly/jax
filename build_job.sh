#!/bin/bash

python build/build.py --enable_cuda --cuda_path /pkgs/cuda-10.0 --cudnn_path /pkgs/cudnn-10.0-v7.4.2 &> build_out
pip install -e build --no-cache-dir -b /scratch/hdd001/home/jkelly/jax_nodes -t /scratch/hdd001/home/jkelly/jax_nodes &> install_build # installs jaxlib (includes XLA)
pip install -e . --no-cache-dir -b /scratch/hdd001/home/jkelly/jax_nodes -t /scratch/hdd001/home/jkelly/jax_nodes &> install_jax     # installs jax (pure Python)
