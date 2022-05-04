#include "../conv_fwd_algo.cuh"

std::string MODE_NAME[] = {
    "IMPLICIT_GEMM",
    "IMPLICIT_PRECOMP_GEMM",
    "GEMM",
    "DIRECT",
    "FFT",
    "FFT_TILING",
    "WINOGRAD",
    "WINOGRAD_NONFUSED"
};


/***************************************************************
 * Debug code
 ***************************************************************/
 void cudnnAssert(cudnnStatus_t code, const char *file, int line) {
    if (code != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr,"cuDNN assert: %s %s %d\n", cudnnGetErrorString(code), file, line);
        exit(1);
    }
}

void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}

void print_performance(cudaEvent_t& start, cudaEvent_t& stop, const cudnnConvolutionFwdAlgo_t& mode, const size_t memory_size) {
    float msec_total=0.0f;
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );

    printf("[%s]\n", MODE_NAME[(int)(mode)].c_str());
    printf(" : %05.3f s , %f GB\n", 1.0f*msec_total/1024.0f, 1.0f*memory_size/1024.0f/1024.0f/1024.0f);

}