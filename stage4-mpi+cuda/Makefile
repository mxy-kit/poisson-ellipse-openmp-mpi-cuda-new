# ==============================
# Makefile for MPI + CUDA (SpectrumMPI)
# ==============================

# 题目要求的两个变量 —— 一定要有
ARCH      ?= sm_60        # GPU 架构，可按老师要求改
HOST_COMP ?= mpicxx       # host 端 C++ 编译器
NVCC      = nvcc
LINKER    = mpicxx        # 链接阶段也用 mpicxx

# nvcc 编译 .cu -> .o 时用 HOST_COMP 作为 host compiler
NVCCFLAGS = -O3 -std=c++11 -arch=$(ARCH) -ccbin $(HOST_COMP)
LDFLAGS  = -O3 -std=c++11 -L/usr/local/cuda-10.2/lib64 -lcudart -lm

TARGET = poisson_mpi_cuda_f
OBJS   = poisson_mpi_cuda_f.o

all: $(TARGET)

# 1) .cu -> .o 由 nvcc 完成
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# 2) 最终链接 —— 用 mpicxx 来链接 MPI 和 CUDA 库
$(TARGET): $(OBJS)
	$(LINKER) $(LDFLAGS) $(OBJS) -o $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS)

