#include "../conv_fwd_algo.cuh"


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

    std::cout << "MODE : " << mode << "\n";
    std::cout << " -- elapsed time : " << msec_total*1e-3 << " s\n";
    std::cout << " -- workspace size : " << memory_size*1e-9 << " GB\n";

}