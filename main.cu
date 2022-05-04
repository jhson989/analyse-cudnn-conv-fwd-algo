#include <vector>
#include <algorithm>

#include "conv_fwd_algo.cuh"

int main(void) {

    cudaErrChk( cudaSetDevice(0) );
    cudnnHandle_t cudnn;
    cudnnErrChk( cudnnCreate(&cudnn) );
    


    /*******************************************************************************
     * Set input, output, filter
     ********************************************************************************/ 

    // Input configuration
    const int BATCH_NUM=128, INPUT_C=3, INPUT_H=256, INPUT_W=256;
    const int OUTPUT_C=3, FILTER_H=3, FILTER_W=3;
    const int PAD_H=0, PAD_W=0;
    const int STRIDE_H=1, STRIDE_W=1;
    const int DILATION_H=1, DILATION_W=1;
    int OUTPUT_H=(INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
    int OUTPUT_W=(INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;

    // Input 
    float* d_input;
    std::vector<float> input(BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    std::generate( input.begin(), input.end(), [](){ return ( (std::rand()%101-50)/5.0f); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyHostToDevice) );
    
    // Output
    float* d_output;
    std::vector<float> output(BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W);
    cudaErrChk( cudaMalloc(&d_output, sizeof(float)*BATCH_NUM*OUTPUT_C*OUTPUT_H*OUTPUT_W) );

    // Filter
    float* d_filter;
    std::vector<float> filter(INPUT_C*OUTPUT_C*FILTER_H*FILTER_W);
    std::generate(filter.begin(), filter.end(), [](){return (std::rand()%11-5)/1.f;});
    cudaErrChk( cudaMalloc(&d_filter, sizeof(float)*INPUT_C*OUTPUT_C*FILTER_H*FILTER_W) );
    cudaErrChk( cudaMemcpy(d_filter, filter.data(), sizeof(float)*INPUT_C*OUTPUT_C*FILTER_H*FILTER_W, cudaMemcpyHostToDevice) );


    
    /*******************************************************************************
     * Describe input, output, filter
     ********************************************************************************/ 

    // Input tensor descriptor
    cudnnTensorDescriptor_t input_desc;
    cudnnErrChk (cudnnCreateTensorDescriptor (&input_desc));
    cudnnErrChk (cudnnSetTensor4dDescriptor (input_desc, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/BATCH_NUM, /*C*/INPUT_C, /*H*/INPUT_H, /*W*/INPUT_W));

    // Output tensor descriptor
    cudnnTensorDescriptor_t output_desc;
    cudnnErrChk (cudnnCreateTensorDescriptor (&output_desc));
    cudnnErrChk (cudnnSetTensor4dDescriptor (output_desc, 
        /*LAYOUT*/CUDNN_TENSOR_NCHW, /*DATATYPE*/CUDNN_DATA_FLOAT, /*N*/BATCH_NUM, /*C*/OUTPUT_C, /*H*/OUTPUT_H, /*W*/OUTPUT_W));

    // Filter
    cudnnFilterDescriptor_t filter_desc;
    cudnnErrChk (cudnnCreateFilterDescriptor (&filter_desc));
    cudnnErrChk (cudnnSetFilter4dDescriptor (filter_desc, 
        /*DATATYPE*/CUDNN_DATA_FLOAT, /*LAYOUT*/CUDNN_TENSOR_NCHW, /*O_C*/OUTPUT_C, /*I_C*/INPUT_C, /*K_H*/FILTER_H, /*K_W*/FILTER_W));


    // 0. CUDNN_CONVOLUTION_FWD_ALGO_DIRECT
    
    // 1. CUDNN_CONVOLUTION_FWD_ALGO_GEMM
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 2. CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 3. CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 4. CUDNN_CONVOLUTION_FWD_ALGO_FFT
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 5. CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 6. CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );

    // 7. CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    launch_conv_fwd(
        /*CUDNN HANDLER*/cudnn,
        /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
        /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
        /*INPUT*/input_desc, d_input,
        /*OUTPUT*/output_desc, d_output,
        /*FILTER*/filter_desc, d_filter
    );



    /*******************************************************************************
     * Set input and filter
     ********************************************************************************/ 

    cudaErrChk( cudaFree(d_input) ); 
    cudaErrChk( cudaFree(d_output) ); 
    cudaErrChk( cudaFree(d_filter) ); 


    cudnnErrChk( cudnnDestroyTensorDescriptor (input_desc) );
    cudnnErrChk( cudnnDestroyTensorDescriptor (output_desc) );
    cudnnErrChk( cudnnDestroyFilterDescriptor (filter_desc) );

    cudnnErrChk( cudnnDestroy(cudnn) );
    return 0;       
}


