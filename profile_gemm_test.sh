sudo LD_LIBRARY_PATH=/usr/local/cuda/lib:.build_release/lib \
/usr/local/cuda/bin/nvprof -m global_hit_rate,local_hit_rate,shared_efficiency,shared_utilization,flop_count_sp,flop_count_dp,inst_fp_32,inst_fp_64,inst_executed,shared_load_transactions,local_load_transactions,gld_transactions,l2_read_transactions,l2_write_transactions \
./gemm_test.bin 161 4096 25088
