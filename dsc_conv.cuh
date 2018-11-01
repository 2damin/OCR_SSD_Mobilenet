#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <sstream>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <math.h>

void depth_f(float *image, float *tmp, float *depthwiseResult, float *filter, const int input_h, const int input_w, const int depth, const int depthFilterW, const int depthwiseOutputW,
	const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks);

void point_f(float *depthwiseResult, float *pointwiseResult, float *pointwiseFilter, const int col, const int depth, const int depthFilterW, const int depthwiseOutputW, 
	const int outputDepth, const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks);

void pointbias_f(float *depthwiseResult, float *pointwiseResult, float *pointwiseFilter, float *bias, const int col, const int depth, const int depthFilterW, const int depthwiseOutputW,
	const int outputDepth, const int padding, const int stride, const int conv_blocks_num, dim3 threads_per_block, dim3 num_of_blocks);

void padding_f(float *image, float *tmp, const int depthwiseOutputW, const int col, const int depth, const int padding, dim3 threads_per_block, dim3 num_of_blocks);