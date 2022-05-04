#pragma once
#include <iostream>
#include <cudnn.h>

// Direct algorithm 
void launch_conv_fwd_direct(
    /*LAYER CONFIG*/ const int PAD_H, const int PAD_W, const int STRIDE_H, const int STRIDE_W, const int DILATION_H, const int DILATION_W,
    /*INPUT*/ const cudnnTensorDescriptor_t& input_desc, const float* d_input,
    /*OUTPUT*/ const cudnnTensorDescriptor_t& output_desc, float* d_output,
    /*FILTER*/ const cudnnFilterDescriptor_t& filter_desc, const float* d_filter
);


// Debugging
#define cudnnErrChk(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
void cudnnAssert(cudnnStatus_t code, const char *file, int line);
void cudaAssert(cudaError_t code, const char *file, int line);
