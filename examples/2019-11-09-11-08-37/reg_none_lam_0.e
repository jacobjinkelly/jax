2019-11-09 11:08:41.920399: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /pkgs/cuda-10.0/lib64:/pkgs/cudnn-10.0-v7.4.2/lib64:
2019-11-09 11:08:41.920449: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_driver.cc:334] failed call to cuInit: UNKNOWN ERROR (303)
2019-11-09 12:12:03.796123: E external/org_tensorflow/tensorflow/compiler/xla/service/slow_operation_alarm.cc:55] 
********************************
Very slow compile?  If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
2019-11-09 13:30:13.491049: E external/org_tensorflow/tensorflow/compiler/xla/service/slow_operation_alarm.cc:55] 
********************************
Very slow compile?  If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
/scratch/hdd001/home/jkelly/jax_nodes/jax/lib/xla_bridge.py:115: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
Reg: none	Lambda 0.0000e+00
Iter 0010 | Total (Regularized) Loss 0.135869 | Loss 0.135869 | r0 0.863248 | r1 0.232109
Iter 0020 | Total (Regularized) Loss 0.127429 | Loss 0.127429 | r0 0.864143 | r1 0.234512
Iter 0030 | Total (Regularized) Loss 0.113632 | Loss 0.113632 | r0 0.863563 | r1 0.234983
Iter 0040 | Total (Regularized) Loss 0.089545 | Loss 0.089545 | r0 0.858447 | r1 0.229418
Iter 0050 | Total (Regularized) Loss 0.057270 | Loss 0.057270 | r0 0.845148 | r1 0.249311
Iter 0060 | Total (Regularized) Loss 0.121982 | Loss 0.121982 | r0 0.696832 | r1 0.170210
Iter 0070 | Total (Regularized) Loss 0.056581 | Loss 0.056581 | r0 0.755608 | r1 0.216939
Iter 0080 | Total (Regularized) Loss 0.050957 | Loss 0.050957 | r0 0.743830 | r1 0.241909
Iter 0090 | Total (Regularized) Loss 0.042026 | Loss 0.042026 | r0 0.742172 | r1 0.268714
Iter 0100 | Total (Regularized) Loss 0.036699 | Loss 0.036699 | r0 0.738385 | r1 0.289380
