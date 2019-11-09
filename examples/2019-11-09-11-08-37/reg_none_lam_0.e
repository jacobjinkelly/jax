2019-11-09 11:08:41.920399: W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /pkgs/cuda-10.0/lib64:/pkgs/cudnn-10.0-v7.4.2/lib64:
2019-11-09 11:08:41.920449: E external/org_tensorflow/tensorflow/stream_executor/cuda/cuda_driver.cc:334] failed call to cuInit: UNKNOWN ERROR (303)
2019-11-09 12:12:03.796123: E external/org_tensorflow/tensorflow/compiler/xla/service/slow_operation_alarm.cc:55] 
********************************
Very slow compile?  If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.
********************************
