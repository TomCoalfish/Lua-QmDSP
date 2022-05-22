swig -lua -c++ qm-dsp.i
gcc -I.  -DMKL_ILP64  -m64  -I/opt/intel/oneapi/mkl/2022.0.2/include -O2 -march=native -mavx2 -fPIC -shared -oqm_dsp.so qm-dsp_wrap.cxx libqm-dsp.a -lstdc++ -lm -lluajit  -L/opt/intel/oneapi/mkl/2022.0.2/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -llapacke -lkissfft-float
