2019-11-09 09:32:11.289575: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /pkgs/cuda-10.0/lib64:/pkgs/cudnn-10.0-v7.4.2/lib64:
2019-11-09 09:32:11.289637: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_driver.cc:334] failed call to cuInit: UNKNOWN ERROR (303)
2019-11-09 10:35:10.795627: E external/org_tensorflow/tensorflow/compiler/xla/service/slow_operation_alarm.cc:55] 
********************************
Very slow compile?  If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2019-11-09 11:55:01.778948: E external/org_tensorflow/tensorflow/compiler/xla/service/slow_operation_alarm.cc:55] 
********************************
Very slow compile?  If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
/scratch/hdd001/home/jkelly/jax_nodes/jax/lib/xla_bridge.py:115: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
Reg: none	Lambda 0.0000e+00
Iter 0010 | Total (Regularized) Loss 18.260595 | Loss 18.260595 | r0 3.578821 | r1 4.644607
Iter 0020 | Total (Regularized) Loss 13.789402 | Loss 13.789402 | r0 3.331477 | r1 4.277373
Iter 0030 | Total (Regularized) Loss 11.053827 | Loss 11.053827 | r0 3.500993 | r1 4.732568
Iter 0040 | Total (Regularized) Loss 10.104459 | Loss 10.104459 | r0 3.857987 | r1 5.554670
Iter 0050 | Total (Regularized) Loss 8.441788 | Loss 8.441788 | r0 3.793057 | r1 5.441615
Iter 0060 | Total (Regularized) Loss 7.243951 | Loss 7.243951 | r0 3.740577 | r1 5.365346
Iter 0070 | Total (Regularized) Loss 6.439753 | Loss 6.439753 | r0 3.715109 | r1 5.370077
Iter 0080 | Total (Regularized) Loss 5.651610 | Loss 5.651610 | r0 3.634275 | r1 5.250111
Iter 0090 | Total (Regularized) Loss 5.566908 | Loss 5.566908 | r0 3.704619 | r1 5.429256
Iter 0100 | Total (Regularized) Loss 4.939416 | Loss 4.939416 | r0 3.627313 | r1 5.310201
