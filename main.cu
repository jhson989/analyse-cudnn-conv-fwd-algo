#include <vector>
#include <algorithm>

#include "conv_fwd_algo.cuh"

void set_layer_config(
    /*ARGUMENT*/int argc, char** argv,
    /*INPUT*/int& BATCH_NUM, int& INPUT_C, int& INPUT_H, int& INPUT_W,
    /*FILTER*/int& OUTPUT_C, int& FILTER_H, int& FILTER_W,
    /*PAD*/int& PAD_H, int& PAD_W,
    /*STRIDE*/int& STRIDE_H, int& STRIDE_W,
    /*DILATION*/int& DILATION_H, int& DILATION_W
) {

    if (argc == 12) {
        BATCH_NUM = std::atoi(argv[1]);
        INPUT_C = std::atoi(argv[2]);
        INPUT_H = std::atoi(argv[3]);
        INPUT_W = std::atoi(argv[4]);
        OUTPUT_C = std::atoi(argv[5]);
        FILTER_H = std::atoi(argv[6]);
        FILTER_W = std::atoi(argv[7]);
        PAD_H = std::atoi(argv[8]);
        PAD_W = std::atoi(argv[9]);
        STRIDE_H = std::atoi(argv[10]);
        STRIDE_W = std::atoi(argv[11]);
        DILATION_H=1, DILATION_W=1;
    } 
    // Default setting
    else {
        BATCH_NUM=128; INPUT_C=3; INPUT_H=256; INPUT_W=256;
        OUTPUT_C=3; FILTER_H=3; FILTER_W=3;
        PAD_H=0; PAD_W=0;
        STRIDE_H=1; STRIDE_W=1;
        DILATION_H=1; DILATION_W=1;
    }

    printf("INPUT : [%d,%d,%d,%d]\n", BATCH_NUM, INPUT_C, INPUT_H, INPUT_W);
    printf("FILTER : [%d,%d,%d,%d]\n", OUTPUT_C, INPUT_C, FILTER_H, FILTER_W);
    printf("LAYER CONFIG : PAD[%d,%d], STRIDE[%d,%d], DILATION[%d,%d]\n", PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);
}


int main(int argc, char** argv) {

    std::cout<<std::endl;
    cudnnHandle_t cudnn;
    cudnnErrChk( cudnnCreate(&cudnn) );
    


    /*******************************************************************************
     * Set input, output, filter
     ********************************************************************************/ 

    // Input configuration
    int BATCH_NUM, INPUT_C, INPUT_H, INPUT_W;
    int OUTPUT_C, FILTER_H, FILTER_W;
    int PAD_H, PAD_W;
    int STRIDE_H, STRIDE_W;
    int DILATION_H, DILATION_W;
    set_layer_config(argc, argv, BATCH_NUM, INPUT_C, INPUT_H, INPUT_W, OUTPUT_C, FILTER_H, FILTER_W, PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W);    

    // Input 
    float* d_input;
    std::vector<float> input(BATCH_NUM*INPUT_C*INPUT_H*INPUT_W);
    std::generate( input.begin(), input.end(), [](){ return ( (std::rand()%101-50)/5.0f); } );
    cudaErrChk( cudaMalloc(&d_input, sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W) );
    cudaErrChk( cudaMemcpy(d_input, input.data(), sizeof(float)*BATCH_NUM*INPUT_C*INPUT_H*INPUT_W, cudaMemcpyHostToDevice) );
    
    // Output
    int OUTPUT_H=(INPUT_H-FILTER_H+2*PAD_H)/STRIDE_H + 1;
    int OUTPUT_W=(INPUT_W-FILTER_W+2*PAD_W)/STRIDE_W + 1;
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


    // 0. CUDNN_CONVOLUTION_FWD_ALGO_DIRECT -> Not presently supported by cuDNN
    //launch_conv_fwd(
    //    /*CUDNN HANDLER*/cudnn,
    //    /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
    //    /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
    //    /*INPUT*/input_desc, d_input,
    //    /*OUTPUT*/output_desc, d_output,
    //    /*FILTER*/filter_desc, d_filter
    //);   

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
    if (FILTER_H == 3 && FILTER_W ==3) {
        launch_conv_fwd(
            /*CUDNN HANDLER*/cudnn,
            /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
            /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
            /*INPUT*/input_desc, d_input,
            /*OUTPUT*/output_desc, d_output,
            /*FILTER*/filter_desc, d_filter
        );
    }
    

    // 7. CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED
    if ( (FILTER_H == 3 && FILTER_W ==3) || (FILTER_H == 5 && FILTER_W ==5) ) {
        launch_conv_fwd(
            /*CUDNN HANDLER*/cudnn,
            /*MODE*/CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
            /*LAYER CONFIG*/PAD_H, PAD_W, STRIDE_H, STRIDE_W, DILATION_H, DILATION_W,
            /*INPUT*/input_desc, d_input,
            /*OUTPUT*/output_desc, d_output,
            /*FILTER*/filter_desc, d_filter
        );
    }


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
    std::cout<<std::endl;
    return 0;       
}


