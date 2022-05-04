#include "../conv_fwd_algo.cuh"

    
void launch_conv_fwd(
    /*CUDNN HANDLER*/cudnnHandle_t& cudnn,
    /*MODE*/cudnnConvolutionFwdAlgo_t mode,
    /*LAYER CONFIG*/ const int PAD_H, const int PAD_W, const int STRIDE_H, const int STRIDE_W, const int DILATION_H, const int DILATION_W,
    /*INPUT*/ const cudnnTensorDescriptor_t& input_desc, const float* d_input,
    /*OUTPUT*/ const cudnnTensorDescriptor_t& output_desc, float* d_output,
    /*FILTER*/ const cudnnFilterDescriptor_t& filter_desc, const float* d_filter
) {

    /*******************************************************************************
     * Describe convolution forward layer
     ********************************************************************************/ 

    // Layer 
    cudnnConvolutionDescriptor_t conv2d_desc;
    cudnnErrChk( cudnnCreateConvolutionDescriptor(&conv2d_desc) );
    cudnnErrChk( cudnnSetConvolution2dDescriptor(
        conv2d_desc,
        /*PAD_H*/PAD_H, /*PAD_W*/PAD_W, /*STRIDE_VERTICAL*/STRIDE_H, /*STRIDE_HORIZONTAL*/STRIDE_W, /*DILATION_H*/DILATION_H, /*DILATION_W*/DILATION_W, /*MODE*/CUDNN_CROSS_CORRELATION, /*DATATYPE*/CUDNN_DATA_FLOAT
    ) );

    // Specify forward algorithm
    void* d_workspace_forward;
    size_t bytes_workspace_forward;
    cudnnErrChk( cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv2d_desc, output_desc, mode, &bytes_workspace_forward) );
    cudaErrChk( cudaMalloc(&d_workspace_forward, bytes_workspace_forward) );




    /******************************************************************
     * 6. Launch forward kernel
     *******************************************************************/    
    
    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );

    const float alpha=1.0f, beta=0.0f;
    cudaErrChk( cudaEventRecord(start, NULL) );
    for (int i=0; i<TEST_ITERATION; i++) {
        cudnnErrChk( cudnnConvolutionForward(cudnn
                                            , /*ALPHA*/&alpha
                                            , /*INPUT*/input_desc, d_input
                                            , /*KERNEL*/filter_desc, d_filter
                                            , /*LAYER*/conv2d_desc, mode, d_workspace_forward, bytes_workspace_forward
                                            , /*BETA*/&beta
                                            , /*OUTPUT*/output_desc, d_output
        ) );    
    }
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk(cudaEventRecord(stop, NULL));
    cudaErrChk( cudaEventSynchronize(stop) );

    print_performance(start, stop, mode, bytes_workspace_forward);

    cudaErrChk( cudaFree(d_workspace_forward) );
}
