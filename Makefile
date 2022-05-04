CC = /usr/local/cuda/bin/nvcc
EXE_NAME = conv_fwd.out
MAIN = main.cu
OBJS = src/debug.cu src/conv_fwd_direct.cu
INCS = conv_fwd_algo.cuh
COMPILER_ARGS = -lcudnn

.PHONY : run clean

all: ${EXE_NAME}

${EXE_NAME}: ${MAIN} ${OBJS} ${INCS} Makefile
	${CC} ${COMPILER_ARGS} -o ${EXE_NAME} ${MAIN} ${OBJS} 


run : ${EXE_NAME}
	./${EXE_NAME}

clean :
	rm ${EXE_NAME}