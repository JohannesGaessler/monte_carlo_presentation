#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE   = np.array([    1,     2,     4,     8,    16,   32,    64,  128,  256,  512, 1024, 2048, 4096, 8192])
TS_FP16      = np.array([56.23, 102.2, 205.7, 406.1, 797.7, 1508,  2747, 4553, 7287, 8591, 8695, 8751, 8781, 8797])
TS_Q4_0_GEMM = np.array([28.21, 54.63, 109.6, 217.3, 430.2, 839.7, 1562, 2907, 4799, 6400, 6547, 6624, 6663, 6678])
TS_Q8_0_GEMM = np.array([26.84, 51.93, 104.1, 206.7, 408.2, 798.8, 1485, 2753, 4593, 6148, 6296, 6368, 6408, 6425])
TS_Q4_0_INT8 = np.array([148.9, 275.1, 543.1, 944.8,  1616, 2764,  4860, 6677, 9061, 9829, 9894, 9916, 9931, 9937])
TS_Q8_0_INT8 = np.array([95.10, 179.5, 357.4, 664.4,  1196, 2149,  3752, 5672, 8465, 9571, 9645, 9673, 9705, 9702])

plt.yscale("log")
plt.xscale("log")
plt.title("LLaMA 3 8b, single user, 8192 context, RTX 4090", fontsize=12)
plt.xlabel("Batch size", fontsize=12)
plt.ylabel("Throughput [tokens / second]", fontsize=12)
plt.xlim(1,  8192)
plt.ylim(10, 11000)

plt.plot(BATCH_SIZE, TS_FP16,      label="FP16 cuBLAS GEMM")
plt.plot(BATCH_SIZE, TS_Q4_0_GEMM, label="Q4_0 cuBLAS GEMM")
plt.plot(BATCH_SIZE, TS_Q8_0_GEMM, label="Q8_0 cuBLAS GEMM")
plt.plot(BATCH_SIZE, TS_Q4_0_INT8, label="Q4_0 llama.cpp custom int8")
plt.plot(BATCH_SIZE, TS_Q8_0_INT8, label="Q8_0 llama.cpp custom int8")
plt.legend(loc="lower right", fontsize=12)
plt.savefig("020_mul_mat_scaling.png", dpi=240)
