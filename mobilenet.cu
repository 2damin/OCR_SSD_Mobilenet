#include <sstream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <vector>

#include <cuda.h>
#include <cudnn.h>

#include <FreeImage.h>
#include "fp16_dev.h"
#include "fp16_emu.h"
#include "FreeImage\gemv.h"
#include "error_util.h"
#include "dsc_conv.cuh"
#include <time.h>
#include "image_resizer.h"
#include "postprocess.cuh"
#include "anchorBox_generator.cuh"
#include "preprocess.cuh"
#include "convolution.cuh"


#define IMAGE_C 3
#define IMAGE_H 300
#define IMAGE_W 300
#define num_classes 16
#define box_code 4
#define box_total 1917

//#define minScale 0.2
//#define maxScale 0.95
//#define num_of_featuremaps 6
//#define num_of_boxChannel_per_layer 6
const float minScale = 0.1;
const float maxScale = 0.7;
const int num_of_boxChannel_per_layer = 6;
const int num_of_featuremaps = 6;


const char *second_image = "test_image/18.png";


const char *conv0_bin = "binData/conv0_d.bin";
const char *bnScale0 = "binData/conv0_dgamma.bin";
const char *bnBias0 = "binData/conv0_dbeta.bin";
const char *conv0_mMean = "binData/conv0_dMean.bin";
const char *conv0_mVar = "binData/conv0_dVar.bin";

const char *conv1_depth = "binData/conv1_d.bin";
const char *conv1_point = "binData/conv1_p.bin";
const char *conv1_dBeta = "binData/conv1_dbeta.bin";
const char *conv1_pBeta = "binData/conv1_pbeta.bin";
const char *conv1_dGamma = "binData/conv1_dgamma.bin";
const char *conv1_pGamma = "binData/conv1_pgamma.bin";
const char *conv1_dmMean = "binData/conv1_dMean.bin";
const char *conv1_dmVar = "binData/conv1_dVar.bin";
const char *conv1_pmMean = "binData/conv1_pMean.bin";
const char *conv1_pmVar = "binData/conv1_pVar.bin";

const char *conv2_depth = "binData/conv2_d.bin";
const char *conv2_point = "binData/conv2_p.bin";
const char *conv2_dBeta = "binData/conv2_dbeta.bin";
const char *conv2_pBeta = "binData/conv2_pbeta.bin";
const char *conv2_dGamma = "binData/conv2_dgamma.bin";
const char *conv2_pGamma = "binData/conv2_pgamma.bin";
const char *conv2_dmMean = "binData/conv2_dMean.bin";
const char *conv2_dmVar = "binData/conv2_dVar.bin";
const char *conv2_pmMean = "binData/conv2_pMean.bin";
const char *conv2_pmVar = "binData/conv2_pVar.bin";

const char *conv3_depth = "binData/conv3_d.bin";
const char *conv3_point = "binData/conv3_p.bin";
const char *conv3_dBeta = "binData/conv3_dbeta.bin";
const char *conv3_pBeta = "binData/conv3_pbeta.bin";
const char *conv3_dGamma = "binData/conv3_dgamma.bin";
const char *conv3_pGamma = "binData/conv3_pgamma.bin";
const char *conv3_dmMean = "binData/conv3_dMean.bin";
const char *conv3_dmVar = "binData/conv3_dVar.bin";
const char *conv3_pmMean = "binData/conv3_pMean.bin";
const char *conv3_pmVar = "binData/conv3_pVar.bin";

const char *conv4_depth = "binData/conv4_d.bin";
const char *conv4_point = "binData/conv4_p.bin";
const char *conv4_dBeta = "binData/conv4_dbeta.bin";
const char *conv4_pBeta = "binData/conv4_pbeta.bin";
const char *conv4_dGamma = "binData/conv4_dgamma.bin";
const char *conv4_pGamma = "binData/conv4_pgamma.bin";
const char *conv4_dmMean = "binData/conv4_dMean.bin";
const char *conv4_dmVar = "binData/conv4_dVar.bin";
const char *conv4_pmMean = "binData/conv4_pMean.bin";
const char *conv4_pmVar = "binData/conv4_pVar.bin";

const char *conv5_depth = "binData/conv5_d.bin";
const char *conv5_point = "binData/conv5_p.bin";
const char *conv5_dBeta = "binData/conv5_dbeta.bin";
const char *conv5_pBeta = "binData/conv5_pbeta.bin";
const char *conv5_dGamma = "binData/conv5_dgamma.bin";
const char *conv5_pGamma = "binData/conv5_pgamma.bin";
const char *conv5_dmMean = "binData/conv5_dMean.bin";
const char *conv5_dmVar = "binData/conv5_dVar.bin";
const char *conv5_pmMean = "binData/conv5_pMean.bin";
const char *conv5_pmVar = "binData/conv5_pVar.bin";

const char *conv6_depth = "binData/conv6_d.bin";
const char *conv6_point = "binData/conv6_p.bin";
const char *conv6_dBeta = "binData/conv6_dbeta.bin";
const char *conv6_pBeta = "binData/conv6_pbeta.bin";
const char *conv6_dGamma = "binData/conv6_dgamma.bin";
const char *conv6_pGamma = "binData/conv6_pgamma.bin";
const char *conv6_dmMean = "binData/conv6_dMean.bin";
const char *conv6_dmVar = "binData/conv6_dVar.bin";
const char *conv6_pmMean = "binData/conv6_pMean.bin";
const char *conv6_pmVar = "binData/conv6_pVar.bin";

const char *conv7_depth = "binData/conv7_d.bin";
const char *conv7_point = "binData/conv7_p.bin";
const char *conv7_dBeta = "binData/conv7_dbeta.bin";
const char *conv7_pBeta = "binData/conv7_pbeta.bin";
const char *conv7_dGamma = "binData/conv7_dgamma.bin";
const char *conv7_pGamma = "binData/conv7_pgamma.bin";
const char *conv7_dmMean = "binData/conv7_dMean.bin";
const char *conv7_dmVar = "binData/conv7_dVar.bin";
const char *conv7_pmMean = "binData/conv7_pMean.bin";
const char *conv7_pmVar = "binData/conv7_pVar.bin";

const char *conv8_depth = "binData/conv8_d.bin";
const char *conv8_point = "binData/conv8_p.bin";
const char *conv8_dBeta = "binData/conv8_dbeta.bin";
const char *conv8_pBeta = "binData/conv8_pbeta.bin";
const char *conv8_dGamma = "binData/conv8_dgamma.bin";
const char *conv8_pGamma = "binData/conv8_pgamma.bin";
const char *conv8_dmMean = "binData/conv8_dMean.bin";
const char *conv8_dmVar = "binData/conv8_dVar.bin";
const char *conv8_pmMean = "binData/conv8_pMean.bin";
const char *conv8_pmVar = "binData/conv8_pVar.bin";

const char *conv9_depth = "binData/conv9_d.bin";
const char *conv9_point = "binData/conv9_p.bin";
const char *conv9_dBeta = "binData/conv9_dbeta.bin";
const char *conv9_pBeta = "binData/conv9_pbeta.bin";
const char *conv9_dGamma = "binData/conv9_dgamma.bin";
const char *conv9_pGamma = "binData/conv9_pgamma.bin";
const char *conv9_dmMean = "binData/conv9_dMean.bin";
const char *conv9_dmVar = "binData/conv9_dVar.bin";
const char *conv9_pmMean = "binData/conv9_pMean.bin";
const char *conv9_pmVar = "binData/conv9_pVar.bin";

const char *conv10_depth = "binData/conv10_d.bin";
const char *conv10_point = "binData/conv10_p.bin";
const char *conv10_dBeta = "binData/conv10_dbeta.bin";
const char *conv10_pBeta = "binData/conv10_pbeta.bin";
const char *conv10_dGamma = "binData/conv10_dgamma.bin";
const char *conv10_pGamma = "binData/conv10_pgamma.bin";
const char *conv10_dmMean = "binData/conv10_dMean.bin";
const char *conv10_dmVar = "binData/conv10_dVar.bin";
const char *conv10_pmMean = "binData/conv10_pMean.bin";
const char *conv10_pmVar = "binData/conv10_pVar.bin";

const char *conv11_depth = "binData/conv11_d.bin";
const char *conv11_point = "binData/conv11_p.bin";
const char *conv11_dBeta = "binData/conv11_dbeta.bin";
const char *conv11_pBeta = "binData/conv11_pbeta.bin";
const char *conv11_dGamma = "binData/conv11_dgamma.bin";
const char *conv11_pGamma = "binData/conv11_pgamma.bin";
const char *conv11_dmMean = "binData/conv11_dMean.bin";
const char *conv11_dmVar = "binData/conv11_dVar.bin";
const char *conv11_pmMean = "binData/conv11_pMean.bin";
const char *conv11_pmVar = "binData/conv11_pVar.bin";

const char *conv12_depth = "binData/conv12_d.bin";
const char *conv12_point = "binData/conv12_p.bin";
const char *conv12_dBeta = "binData/conv12_dbeta.bin";
const char *conv12_pBeta = "binData/conv12_pbeta.bin";
const char *conv12_dGamma = "binData/conv12_dgamma.bin";
const char *conv12_pGamma = "binData/conv12_pgamma.bin";
const char *conv12_dmMean = "binData/conv12_dMean.bin";
const char *conv12_dmVar = "binData/conv12_dVar.bin";
const char *conv12_pmMean = "binData/conv12_pMean.bin";
const char *conv12_pmVar = "binData/conv12_pVar.bin";

const char *conv13_depth = "binData/conv13_d.bin";
const char *conv13_point = "binData/conv13_p.bin";
const char *conv13_dBeta = "binData/conv13_dbeta.bin";
const char *conv13_pBeta = "binData/conv13_pbeta.bin";
const char *conv13_dGamma = "binData/conv13_dgamma.bin";
const char *conv13_pGamma = "binData/conv13_pgamma.bin";
const char *conv13_dmMean = "binData/conv13_dMean.bin";
const char *conv13_dmVar = "binData/conv13_dVar.bin";
const char *conv13_pmMean = "binData/conv13_pMean.bin";
const char *conv13_pmVar = "binData/conv13_pVar.bin";

const char *conv14_point = "binData/conv14_p.bin";
const char *conv14_pBeta = "binData/conv14_pbeta.bin";
const char *conv14_pGamma = "binData/conv14_pgamma.bin";
const char *conv14_pmMean = "binData/conv14_pMean.bin";
const char *conv14_pmVar = "binData/conv14_pVar.bin";
const char *conv14_w = "binData/conv14_2_d.bin";
const char *conv14_wBeta = "binData/conv14_2_dbeta.bin";
const char *conv14_wGamma = "binData/conv14_2_dgamma.bin";
const char *conv14_wmMean = "binData/conv14_2_dMean.bin";
const char *conv14_wmVar = "binData/conv14_2_dVar.bin";

const char *conv15_point = "binData/conv15_p.bin";
const char *conv15_pBeta = "binData/conv15_pbeta.bin";
const char *conv15_pGamma = "binData/conv15_pgamma.bin";
const char *conv15_pmMean = "binData/conv15_pMean.bin";
const char *conv15_pmVar = "binData/conv15_pVar.bin";
const char *conv15_w = "binData/conv15_2_d.bin";
const char *conv15_wBeta = "binData/conv15_2_dbeta.bin";
const char *conv15_wGamma = "binData/conv15_2_dgamma.bin";
const char *conv15_wmMean = "binData/conv15_2_dMean.bin";
const char *conv15_wmVar = "binData/conv15_2_dVar.bin";

const char *conv16_point = "binData/conv16_p.bin";
const char *conv16_pBeta = "binData/conv16_pbeta.bin";
const char *conv16_pGamma = "binData/conv16_pgamma.bin";
const char *conv16_pmMean = "binData/conv16_pMean.bin";
const char *conv16_pmVar = "binData/conv16_pVar.bin";
const char *conv16_w = "binData/conv16_2_d.bin";
const char *conv16_wBeta = "binData/conv16_2_dbeta.bin";
const char *conv16_wGamma = "binData/conv16_2_dgamma.bin";
const char *conv16_wmMean = "binData/conv16_2_dMean.bin";
const char *conv16_wmVar = "binData/conv16_2_dVar.bin";

const char *conv17_point = "binData/conv17_p.bin";
const char *conv17_pBeta = "binData/conv17_pbeta.bin";
const char *conv17_pGamma = "binData/conv17_pgamma.bin";
const char *conv17_pmMean = "binData/conv17_pMean.bin";
const char *conv17_pmVar = "binData/conv17_pVar.bin";
const char *conv17_w = "binData/conv17_2_d.bin";
const char *conv17_wBeta = "binData/conv17_2_dbeta.bin";
const char *conv17_wGamma = "binData/conv17_2_dgamma.bin";
const char *conv17_wmMean = "binData/conv17_2_dMean.bin";
const char *conv17_wmVar = "binData/conv17_2_dVar.bin";

const char *box0_loc_b = "binData/box0_boxp_b.bin";
const char *box0_loc_w = "binData/box0_boxp_w.bin";
const char *box0_conf_b = "binData/box0_classp_b.bin";
const char *box0_conf_w = "binData/box0_classp_w.bin";

const char *box1_loc_b = "binData/box1_boxp_b.bin";
const char *box1_loc_w = "binData/box1_boxp_w.bin";
const char *box1_conf_b = "binData/box1_classp_b.bin";
const char *box1_conf_w = "binData/box1_classp_w.bin";

const char *box2_loc_b = "binData/box2_boxp_b.bin";
const char *box2_loc_w = "binData/box2_boxp_w.bin";
const char *box2_conf_b = "binData/box2_classp_b.bin";
const char *box2_conf_w = "binData/box2_classp_w.bin";

const char *box3_loc_b = "binData/box3_boxp_b.bin";
const char *box3_loc_w = "binData/box3_boxp_w.bin";
const char *box3_conf_b = "binData/box3_classp_b.bin";
const char *box3_conf_w = "binData/box3_classp_w.bin";

const char *box4_loc_b = "binData/box4_boxp_b.bin";
const char *box4_loc_w = "binData/box4_boxp_w.bin";
const char *box4_conf_b = "binData/box4_classp_b.bin";
const char *box4_conf_w = "binData/box4_classp_w.bin";

const char *box5_loc_b = "binData/box5_boxp_b.bin";
const char *box5_loc_w = "binData/box5_boxp_w.bin";
const char *box5_conf_b = "binData/box5_classp_b.bin";
const char *box5_conf_w = "binData/box5_classp_w.bin";
/********************************************************
* Prints the error message, and exits
* ******************************************************/

#define EXIT_WAIVED 0

using namespace std;

void get_path(std::string& sFilename, const char *fname, const char *pname)
{
	sFilename = (std::string("test_data/") + std::string(fname));
}

// Need the map, since scaling factor is of float type in half precision
// Also when one needs to use float instead of half, e.g. for printing
template <typename T>
struct ScaleFactorTypeMap { typedef T Type; };
template <> struct ScaleFactorTypeMap<half1> { typedef float Type; };

// float/double <-> half conversion class
template <class value_type>
class Convert
{
public:
	template <class T>
	value_type operator()(T x) { return value_type(x); }
	value_type operator()(half1 x) { return value_type(cpu_half2float(x)); }
};

template <>
class Convert<half1>
{
public:
	template <class T>
	half1 operator()(T x) { return cpu_float2half_rn(T(x)); }
	half1 operator()(half1 x) { return x; }
};

// IO utils
template <class value_type>
void readBinaryFile(const char* fname, int size, value_type* data_h)
{
	std::ifstream dataFile(fname, std::ios::in | std::ios::binary);
	std::stringstream error_s;
	if (!dataFile)
	{
		error_s << "Error opening file " << fname;
		FatalError(error_s.str());
	}
	// we assume the data stored is always in float precision
	float* data_tmp = new float[size];
	int size_b = size * sizeof(float);
	if (!dataFile.read((char*)data_tmp, size_b))
	{
		error_s << "Error reading file " << fname;
		FatalError(error_s.str());
	}
	// conversion
	Convert<value_type> fromReal;
	for (int i = 0; i < size; i++)
	{
		data_h[i] = fromReal(data_tmp[i]);
		//printf("%f \n", data_tmp[i]);
	}// system("pause");

	delete[] data_tmp;
}

template <class value_type>
void readAllocMemcpy(const char* fname, int size, value_type** data_h, value_type** data_d)
{
	*data_h = new value_type[size];

	readBinaryFile<value_type>(fname, size, *data_h);


	int size_b = size * sizeof(value_type);
	checkCudaErrors(cudaMalloc(data_d, size_b));
	checkCudaErrors(cudaMemcpy(*data_d, *data_h,
		size_b,
		cudaMemcpyHostToDevice));
	//delete[] *data_h;
}

void FreeImageErrorHandler(FREE_IMAGE_FORMAT oFif, const char *zMessage)
{
	//FatalError(zMessage);
}
template <class value_type>
void readImage(const char* fname, value_type* imgData_h)
{
	// declare a host image object for an 8-bit grayscale image
	std::string sFilename(fname);
	//std::cout << "Loading image " << sFilename << std::endl;
	// Take care of half precision
	Convert<value_type> fromReal;

	// load gray-scale image from disk    
	// set your own FreeImage error handler
	//FreeImage_SetOutputMessage(FreeImageErrorHandler);

	FREE_IMAGE_FORMAT eFormat = FreeImage_GetFileType(sFilename.c_str());

	// no signature? try to guess the file format from the file extension
	if (eFormat == FIF_UNKNOWN)
	{
		eFormat = FreeImage_GetFIFFromFilename(sFilename.c_str());
	}

	if (eFormat == FIF_UNKNOWN)
	{
		FatalError("Unknown image format");
	}
	// check that the plugin has reading capabilities ...

	FIBITMAP *pBitmap;
	if (FreeImage_FIFSupportsReading(eFormat))
	{
		pBitmap = FreeImage_Load(eFormat, sFilename.c_str());
	}

	if (pBitmap == 0)
	{
		FatalError("Error reading image");
	}

	// make sure this is an 8-bit single channel image
	if (FreeImage_GetColorType(pBitmap) != FIC_MINISBLACK)
	{
		FatalError("This is not 8-bit single channel imagee");
	}
	if (FreeImage_GetBPP(pBitmap) != 8)
	{
		FatalError("This is not 8-bit single channel imagee");
	}

	// create an ImageCPU to receive the loaded image data
	//ImageCPU_8u_C1 oImage(FreeImage_GetWidth(pBitmap), FreeImage_GetHeight(pBitmap));

	int width = FreeImage_GetWidth(pBitmap);
	int height = FreeImage_GetHeight(pBitmap);
	Convert<half1> toReal;

	if (width != IMAGE_W || height != IMAGE_H)
	{
		FatalError("Image dimensions missmatch");
	}

	// Normalize image to be in range [0,1]
	for (int i = 0; i < height; ++i)
	{
		unsigned char *pSrcLine = FreeImage_GetScanLine(pBitmap, height - i - 1);
		for (int j = 0; j < width; j++)
		{
			int idx = IMAGE_W*i + j;
			imgData_h[idx] = fromReal(double(2)*(*(pSrcLine + j) / double(255))-double(1));
			//printf("%0.2f  ",imgData_h[idx]);
		}
		//printf("\n");
	}
	FreeImage_Unload(pBitmap);

}

template <class value_type>
void printDeviceVector(int size, value_type* vec_d)
{
	typedef typename ScaleFactorTypeMap<value_type>::Type real_type;
	value_type *vec;
	vec = new value_type[size];
	cudaDeviceSynchronize();
	cudaMemcpy(vec, vec_d, size * sizeof(value_type), cudaMemcpyDeviceToHost);
	Convert<real_type> toReal;
	std::cout.precision(7);
	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	for (int i = 0; i < size; i++)
	{
		std::cout << toReal(vec[i]) << " ";
		//printf("%f", vec[i]);
	}
	std::cout << std::endl;
	delete[] vec;
}

typedef enum {
	FP16_HOST = 0,
	FP16_CUDA = 1,
	FP16_CUDNN = 2
} fp16Import_t;
template <class value_type>
struct Layer_t
{
	fp16Import_t fp16Import;
	int inputs, pointoutputs, outputs;
	// linear dimension (i.e. size is kernel_dim * kernel_dim)
	int depth_kernel_dim, point_kernel_dim, conv_kernel_dim;
	value_type *pointData_h, *pointData_d, *depthData_h, *depthData_d, *pointBias_h, *pointBias_d, *depthBias_h, *depthBias_d, *convData_h, *convData_d;
	value_type *depthBnScale_h, *depthBnBias_h, *pointBnScale_h, *pointBnBias_h, *depthBnScale_d, *depthBnBias_d, *pointBnScale_d, *pointBnBias_d, *convBnBias_h, *convBnBias_d, *convBnScale_h, *convBnScale_d;
	value_type *depthBnMean_h, *depthBnMean_d, *depthBnVar_h, *depthBnVar_d, *pointBnMean_h, *pointBnMean_d, *pointBnVar_h, *pointBnVar_d, *convBnMean_h, *convBnMean_d, *convBnVar_h, *convBnVar_d;
	//value_type *
	Layer_t() : pointData_h(NULL), pointData_d(NULL), pointBias_h(NULL), pointBias_d(NULL), depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), depthBnScale_h(NULL), 
		depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL), depthBnMean_h(NULL), 
		depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL), convData_h(NULL), 
		convData_d(NULL), convBnBias_d(NULL), convBnBias_h(NULL), convBnVar_d(NULL), convBnVar_h(NULL), convBnMean_d(NULL), convBnMean_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL),
		inputs(0), pointoutputs(0), outputs(0), depth_kernel_dim(0), point_kernel_dim(0), conv_kernel_dim(0), fp16Import(FP16_HOST) {};
	//box predictor Layer//
	Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bias, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(0), point_kernel_dim(_kernel_dim), conv_kernel_dim(0),
		depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL),
		pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL), depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), 
		depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL), convData_h(NULL), convData_d(NULL), convBnBias_d(NULL), convBnBias_h(NULL),
		convBnVar_d(NULL), convBnVar_h(NULL), convBnMean_d(NULL), convBnMean_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bias_path;
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			get_path(bias_path, fname_bias, pname);
		}
		else
		{
			weights_path = fname_weights; bias_path = fname_bias;
		}
		//printf("%s \n", weights_path.c_str()); 
		//printf("%s \n", fname_weights);
		readAllocInit(weights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
		//for (int i = 0; i < outputs * point_kernel_dim * point_kernel_dim; ++i)
		//{
		//	printf("%f \n", pointBias_h[i]);
		//}
		//system("pause");
	}
	//original convolution Layer//
	Layer_t(int _inputs, int _outputs, int _kernel_dim, const char* fname_weights, const char* fname_bnScale, const char* fname_bnBias,
		const char* fname_bnMean, const char* fname_bnVar, const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(0), point_kernel_dim(0), conv_kernel_dim(_kernel_dim),
	    depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), pointData_h(NULL), pointData_d(NULL), pointBias_h(NULL), pointBias_d(NULL),
		depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), pointBnScale_h(NULL), pointBnBias_h(NULL), pointBnScale_d(NULL), pointBnBias_d(NULL),
		depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL), pointBnMean_h(NULL), pointBnMean_d(NULL), pointBnVar_h(NULL), pointBnVar_d(NULL)
	{
		fp16Import = _fp16Import;
		std::string weights_path, bnScale_path, bnBias_path, bnMean_path, bnVar_path; //bias_path
		if (pname != NULL)
		{
			get_path(weights_path, fname_weights, pname);
			//get_path(bias_path, fname_bias, pname);
			get_path(bnScale_path, fname_bnScale, pname);
			get_path(bnBias_path, fname_bnBias, pname);
			get_path(bnMean_path, fname_bnMean, pname);
			get_path(bnVar_path, fname_bnVar, pname);
		}
		else
		{
			weights_path = fname_weights; //bias_path = fname_bias;
			bnScale_path = fname_bnScale; bnBias_path = fname_bnBias;
			bnMean_path = fname_bnMean; bnVar_path = fname_bnVar;
		}
		//printf("%s \n", weights_path.c_str()); 
		//printf("%s \n", fname_weights);
		readAllocInit(weights_path.c_str(), inputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		//printf("%s: \n", fname_bias);
		//readAllocInit(bias_path.c_str(), outputs, &pointBias_h, &pointBias_d);
		readAllocInit(bnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(bnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(bnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(bnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
	}
	//DSC Layer//
	Layer_t(int _inputs, int _outputs, int _depth_kernel_dim, int _point_kernel_dim, const char* fname_depthWeights, const char* fname_pointWeights, 
		const char* fname_depthBnScale, const char* fname_depthBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias , 
		const char* fname_depthBnMean, const char* fname_depthBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(0), outputs(_outputs), depth_kernel_dim(_depth_kernel_dim), point_kernel_dim(_point_kernel_dim), conv_kernel_dim(0), depthBias_h(NULL), depthBias_d(NULL),
		pointBias_h(NULL), pointBias_d(NULL), convData_h(NULL), convData_d(NULL), convBnMean_h(NULL), convBnMean_d(NULL), convBnVar_d(NULL), convBnVar_h(NULL), convBnBias_d(NULL),
		convBnBias_h(NULL), convBnScale_d(NULL), convBnScale_h(NULL)
	{
		fp16Import = _fp16Import;
		std::string depthWeights_path, pointWeights_path, depthBnScale_path, depthBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string depthBnMean_path, depthBnVar_path, pointBnMean_path, pointBnVar_path;
		if (pname != NULL)
		{
			get_path(depthWeights_path, fname_depthWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(depthBnScale_path, fname_depthBnScale, pname);
			get_path(depthBnBias_path, fname_depthBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(depthBnMean_path, fname_depthBnMean, pname);
			get_path(depthBnVar_path, fname_depthBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			depthWeights_path = fname_depthWeights; pointWeights_path = fname_pointWeights;
			depthBnScale_path = fname_depthBnScale; depthBnBias_path = fname_depthBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			depthBnMean_path = fname_depthBnMean; depthBnVar_path = fname_depthBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}
		//printf("%s \n", fname_weights);
		readAllocInit(depthWeights_path.c_str(), inputs * depth_kernel_dim * depth_kernel_dim,
			&depthData_h, &depthData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * outputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(depthBnScale_path.c_str(), inputs, &depthBnScale_h, &depthBnScale_d);
		readAllocInit(depthBnBias_path.c_str(), inputs, &depthBnBias_h, &depthBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), outputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), outputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(depthBnMean_path.c_str(), inputs, &depthBnMean_h, &depthBnMean_d);
		readAllocInit(depthBnVar_path.c_str(), inputs, &depthBnVar_h, &depthBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), outputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), outputs, &pointBnVar_h, &pointBnVar_d);
		//cout << "depthData :" << endl;
		//for (int i = 0; i < inputs * depth_kernel_dim * depth_kernel_dim; i++)
		//{
		//	printf("%0.2f \n", depthData_h[i]);
		//} system("pause");
		//cout << "pointData :" << endl;
		//for (int i = 0; i <inputs * outputs * point_kernel_dim * point_kernel_dim; i++)
		//{
		//	printf("%0.2f \n", pointData_h[i]);
		//}
	}
	//extra Layer//
	Layer_t(int _inputs, int _pointouts, int _outputs, int _conv_kernel_dim, int _point_kernel_dim, const char* fname_convWeights, const char* fname_pointWeights,
		const char* fname_convBnScale, const char* fname_convBnBias, const char* fname_pointBnScale, const char* fname_pointBnBias,
		const char* fname_convBnMean, const char* fname_convBnVar, const char* fname_pointBnMean, const char* fname_pointBnVar,
		const char* pname = NULL, fp16Import_t _fp16Import = FP16_HOST)
		: inputs(_inputs), pointoutputs(_pointouts), outputs(_outputs), conv_kernel_dim(_conv_kernel_dim), point_kernel_dim(_point_kernel_dim), depth_kernel_dim(0),
		depthData_h(NULL), depthData_d(NULL), depthBias_h(NULL), depthBias_d(NULL), pointBias_d(NULL), pointBias_h(NULL),
		depthBnScale_h(NULL), depthBnBias_h(NULL), depthBnScale_d(NULL), depthBnBias_d(NULL), depthBnMean_h(NULL), depthBnMean_d(NULL), depthBnVar_h(NULL), depthBnVar_d(NULL)
	{
		fp16Import = _fp16Import;
		std::string convWeights_path, pointWeights_path, convBnScale_path, convBnBias_path, pointBnScale_path, pointBnBias_path;
		std::string convBnMean_path, convBnVar_path, pointBnMean_path, pointBnVar_path;
		if (pname != NULL)
		{
			get_path(convWeights_path, fname_convWeights, pname);
			get_path(pointWeights_path, fname_pointWeights, pname);
			get_path(convBnScale_path, fname_convBnScale, pname);
			get_path(convBnBias_path, fname_convBnBias, pname);
			get_path(pointBnScale_path, fname_pointBnScale, pname);
			get_path(pointBnBias_path, fname_pointBnBias, pname);
			get_path(convBnMean_path, fname_convBnMean, pname);
			get_path(convBnVar_path, fname_convBnVar, pname);
			get_path(pointBnMean_path, fname_pointBnMean, pname);
			get_path(pointBnVar_path, fname_pointBnVar, pname);
		}
		else
		{
			convWeights_path = fname_convWeights; pointWeights_path = fname_pointWeights;
			convBnScale_path = fname_convBnScale; convBnBias_path = fname_convBnBias;
			pointBnScale_path = fname_pointBnScale; pointBnBias_path = fname_pointBnBias;
			convBnMean_path = fname_convBnMean; convBnVar_path = fname_convBnVar;
			pointBnMean_path = fname_pointBnMean; pointBnVar_path = fname_pointBnVar;
		}
		//printf("%s \n", fname_weights);
		readAllocInit(convWeights_path.c_str(), pointoutputs * outputs * conv_kernel_dim * conv_kernel_dim,
			&convData_h, &convData_d);
		readAllocInit(pointWeights_path.c_str(), inputs * pointoutputs * point_kernel_dim * point_kernel_dim,
			&pointData_h, &pointData_d);
		readAllocInit(convBnScale_path.c_str(), outputs, &convBnScale_h, &convBnScale_d);
		readAllocInit(convBnBias_path.c_str(), outputs, &convBnBias_h, &convBnBias_d);
		readAllocInit(pointBnScale_path.c_str(), pointoutputs, &pointBnScale_h, &pointBnScale_d);
		readAllocInit(pointBnBias_path.c_str(), pointoutputs, &pointBnBias_h, &pointBnBias_d);
		readAllocInit(convBnMean_path.c_str(), outputs, &convBnMean_h, &convBnMean_d);
		readAllocInit(convBnVar_path.c_str(), outputs, &convBnVar_h, &convBnVar_d);
		readAllocInit(pointBnMean_path.c_str(), pointoutputs, &pointBnMean_h, &pointBnMean_d);
		readAllocInit(pointBnVar_path.c_str(), pointoutputs, &pointBnVar_h, &pointBnVar_d);
		//cout << "convData :" << endl;
		//for (int i = 0; i < inputs * conv_kernel_dim * conv_kernel_dim; i++)
		//{
		//	printf("%0.2f \n", convData_h[i]);
		//}
		//cout << "pointData :" << endl;
		//for (int i = 0; i <inputs * outputs * point_kernel_dim * point_kernel_dim; i++)
		//{
		//	printf("%0.2f \n", pointData_h[i]);
		//}
	}
	~Layer_t()
	{

		if (pointData_h != NULL) delete[] pointData_h;
		if (depthData_h != NULL) delete[] depthData_h;
		if (convData_h != NULL) delete[] convData_h;
		if (pointData_d != NULL) checkCudaErrors(cudaFree(pointData_d));
		if (depthData_d != NULL) checkCudaErrors(cudaFree(depthData_d));
		if (convData_d != NULL) checkCudaErrors(cudaFree(convData_d));
		if (depthBias_h != NULL) delete[] depthBias_h;
		if (pointBias_h != NULL) delete[] pointBias_h;
		if (depthBias_d != NULL) checkCudaErrors(cudaFree(depthBias_d));
		if (pointBias_d != NULL) checkCudaErrors(cudaFree(pointBias_d));
		if (depthBnScale_h != NULL) delete[] depthBnScale_h;
		if (depthBnBias_h != NULL) delete[] depthBnBias_h;
		if (pointBnScale_h != NULL) delete[] pointBnScale_h;
		if (pointBnBias_h != NULL) delete[] pointBnBias_h;
		if (convBnScale_h != NULL) delete[] convBnScale_h;
		if (convBnBias_h != NULL) delete[] convBnBias_h;
		if (depthBnScale_d != NULL) checkCudaErrors(cudaFree(depthBnScale_d));
		if (depthBnBias_d != NULL) checkCudaErrors(cudaFree(depthBnBias_d));
		if (pointBnScale_d != NULL) checkCudaErrors(cudaFree(pointBnScale_d));
		if (pointBnBias_d != NULL) checkCudaErrors(cudaFree(pointBnBias_d));
		if (convBnScale_d != NULL) checkCudaErrors(cudaFree(convBnScale_d));
		if (convBnBias_d != NULL) checkCudaErrors(cudaFree(convBnBias_d));
		if (depthBnMean_h != NULL) delete[] depthBnMean_h;
		if (depthBnVar_h != NULL) delete[] depthBnVar_h;
		if (pointBnMean_h != NULL) delete[] pointBnMean_h;
		if (pointBnVar_h != NULL) delete[] pointBnVar_h;
		if (convBnMean_h != NULL) delete[] convBnMean_h;
		if (convBnVar_h != NULL) delete[] convBnVar_h;
		if (depthBnMean_d != NULL) checkCudaErrors(cudaFree(depthBnMean_d));
		if (depthBnVar_d != NULL) checkCudaErrors(cudaFree(depthBnVar_d));
		if (pointBnMean_d != NULL) checkCudaErrors(cudaFree(pointBnMean_d));
		if (pointBnVar_d != NULL) checkCudaErrors(cudaFree(pointBnVar_d));
		if (convBnMean_d != NULL) checkCudaErrors(cudaFree(convBnMean_d));
		if (convBnVar_d != NULL) checkCudaErrors(cudaFree(convBnVar_d));
	}

private:
	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
	{
		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
	}
};
//template <class value_type>
//struct extraLayer_t
//{
//	fp16Import_t fp16Import;
//	int inputs, pointouts, outputs;
//	// linear dimension (i.e. size is kernel_dim * kernel_dim)
//	int conv_kernel_dim, point_kernel_dim;
//	value_type *pointData_h, *pointData_d, *convData_h, *convData_d, *pointBias_h, *pointBias_d, *convBias_h, *convBias_d;
//	value_type *convBnScale_h, *convBnBias_h, *pointBnScale_h, *pointBnBias_h, *convBnScale_d, *convBnBias_d, *pointBnScale_d, *pointBnBias_d;
//	value_type *convBnMean_h, *convBnMean_d, *convBnVar_h, *convBnVar_d, *pointBnMean_h, *pointBnMean_d, *pointBnVar_h, *pointBnVar_d;
//	//value_type *
//	extraLayer_t() : pointData_h(NULL), pointData_d(NULL), convData_h(NULL), convData_d(NULL), pointBias_h(NULL), pointBias_d(NULL), convBias_h(NULL), convBias_d(NULL),
//		inputs(0), pointouts(0), outputs(0), conv_kernel_dim(0), point_kernel_dim(0), fp16Import(FP16_HOST) {};
//	
//	~extraLayer_t()
//	{
//		if (pointData_h != NULL) delete[] pointData_h;
//		if (convData_h != NULL) delete[] convData_h;
//		if (pointData_d != NULL) checkCudaErrors(cudaFree(pointData_d));
//		if (convData_d != NULL) checkCudaErrors(cudaFree(convData_d));
//		if (convBias_h != NULL) delete[] convBias_h;
//		if (pointBias_h != NULL) delete[] pointBias_h;
//		if (convBias_d != NULL) checkCudaErrors(cudaFree(convBias_d));
//		if (pointBias_d != NULL) checkCudaErrors(cudaFree(pointBias_d));
//		if (convBnScale_h != NULL) delete[] convBnScale_h;
//		if (convBnBias_h != NULL) delete[] convBnBias_h;
//		if (pointBnScale_d != NULL) checkCudaErrors(cudaFree(pointBnScale_d));
//		if (pointBnBias_d != NULL) checkCudaErrors(cudaFree(pointBnBias_d));
//		if (pointBnScale_h != NULL) delete[] pointBnScale_h;
//		if (pointBnBias_h != NULL) delete[] pointBnBias_h;
//		if (convBnScale_d != NULL) checkCudaErrors(cudaFree(convBnScale_d));
//		if (convBnBias_d != NULL) checkCudaErrors(cudaFree(convBnBias_d));
//		if (convBnMean_h != NULL) delete[] convBnMean_h;
//		if (convBnVar_h != NULL) delete[] convBnVar_h;
//		if (convBnMean_d != NULL) checkCudaErrors(cudaFree(convBnMean_d));
//		if (convBnVar_d != NULL) checkCudaErrors(cudaFree(convBnVar_d));
//		if (pointBnMean_h != NULL) delete[] pointBnMean_h;
//		if (pointBnVar_h != NULL) delete[] pointBnVar_h;
//		if (pointBnMean_d != NULL) checkCudaErrors(cudaFree(pointBnMean_d));
//		if (pointBnVar_d != NULL) checkCudaErrors(cudaFree(pointBnVar_d));
//	}
//private:
//	void readAllocInit(const char* fname, int size, value_type** data_h, value_type** data_d)
//	{
//		readAllocMemcpy<value_type>(fname, size, data_h, data_d);
//	}
//};


//template <>
//void Layer_t<half1>::readAllocInit(const char* fname, int size, half1** data_h, half1** data_d)
//{
//	*data_h = new half1[size];
//	int size_b = size * sizeof(half1);
//	checkCudaErrors(cudaMalloc(data_d, size_b));
//	float *data_tmp_h, *data_tmp_d;
//
//	switch (fp16Import)
//	{
//	case FP16_HOST:
//	{
//		readBinaryFile<half1>(fname, size, *data_h);
//		checkCudaErrors(cudaMemcpy(*data_d, *data_h, size_b,
//			cudaMemcpyHostToDevice));
//		break;
//	}
//	case FP16_CUDA:
//	{
//		readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
//
//		gpu_float2half_rn<float>(size, data_tmp_d, *data_d);
//
//		delete[] data_tmp_h;
//		checkCudaErrors(cudaFree(data_tmp_d));
//		break;
//	}
//	case FP16_CUDNN:
//	{
//		readAllocMemcpy<float>(fname, size, &data_tmp_h, &data_tmp_d);
//		delete[] data_tmp_h;
//		cudnnHandle_t cudnnHandle;
//		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
//		checkCUDNN(cudnnCreate(&cudnnHandle));
//		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
//		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
//		checkCUDNN(cudnnSetTensor4dDescriptorEx(srcTensorDesc,
//			CUDNN_DATA_FLOAT,
//			1, size,
//			1, 1,
//			size, 1, 1, 1));
//		checkCUDNN(cudnnSetTensor4dDescriptorEx(dstTensorDesc,
//			CUDNN_DATA_HALF,
//			1, size,
//			1, 1,
//			size, 1, 1, 1));
//		float alpha = 1.0f;
//		float beta = 0.0f;
//
//		checkCUDNN(cudnnTransformTensor(cudnnHandle, &alpha,
//			srcTensorDesc,
//			data_tmp_d, &beta,
//			dstTensorDesc,
//			*data_d));
//		checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
//		checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
//		checkCUDNN(cudnnDestroy(cudnnHandle));
//		checkCudaErrors(cudaFree(data_tmp_d));
//		break;
//	}
//	}
//}

// demonstrate different ways of setting tensor descriptor
//#define SIMPLE_TENSOR_DESCRIPTOR
#define ND_TENSOR_DESCRIPTOR
void setTensorDesc(cudnnTensorDescriptor_t& tensorDesc,
	cudnnTensorFormat_t& tensorFormat,
	cudnnDataType_t& dataType,
	int n,
	int c,
	int h,
	int w)
{
#if SIMPLE_TENSOR_DESCRIPTOR
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc,
		tensorFormat,
		dataType,
		n, c,
		h,
		w));
#elif defined(ND_TENSOR_DESCRIPTOR)
	const int nDims = 4;
	int dimA[nDims] = { n,c,h,w };
	int strideA[nDims] = { c*h*w, h*w, w, 1 };
	checkCUDNN(cudnnSetTensorNdDescriptor(tensorDesc,
		dataType,
		4,
		dimA,
		strideA));
#else
	checkCUDNN(cudnnSetTensor4dDescriptorEx(tensorDesc,
		dataType,
		n, c,
		h, w,
		c*h*w, h*w, w, 1));
#endif
}

template <class value_type>
class network_t
{
	typedef typename ScaleFactorTypeMap<value_type>::Type scaling_type;
	int convAlgorithm;
	cudnnDataType_t dataType;
	cudnnTensorFormat_t tensorFormat;
	cudnnHandle_t cudnnHandle;
	cudnnTensorDescriptor_t srcConvTensorDesc, dstConvTensorDesc, dstTensorDesc, srcTensorDesc, biasTensorDesc, bnTensorDesc, activTensorDesc;
	//cudnnTensorDescriptor_t srcPoolTensorDesc, dstPoolTensorDesc, srcSoftmaxTensorDesc, dstSoftmaxTensorDesc;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnPoolingDescriptor_t     poolingDesc;
	cudnnActivationDescriptor_t  activDesc, activBoxDesc;
	//cudnnLRNDescriptor_t   normDesc;
	cublasHandle_t cublasHandle;
	cudnnConvolutionFwdAlgo_t algo;
	size_t sizeInBytes = 0;
	void* workSpace = NULL;
	double epsilon = 0.001;


	void createHandles()
	{
		checkCUDNN(cudnnCreate(&cudnnHandle));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&srcConvTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&activTensorDesc));
		//checkCUDNN(cudnnCreateTensorDescriptor(&srcPoolTensorDesc));
		//checkCUDNN(cudnnCreateTensorDescriptor(&srcSoftmaxTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstConvTensorDesc));
		/*checkCUDNN(cudnnCreateTensorDescriptor(&dstActivTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstPoolTensorDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&dstSoftmaxTensorDesc));*/
		checkCUDNN(cudnnCreateTensorDescriptor(&biasTensorDesc));
		checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
		//checkCUDNN(cudnnCreatePoolingDescriptor(&poolingDesc));
		checkCUDNN(cudnnCreateActivationDescriptor(&activDesc));
		checkCUDNN(cudnnCreateActivationDescriptor(&activBoxDesc));
		//checkCUDNN(cudnnCreateLRNDescriptor(&normDesc));
		checkCUDNN(cudnnCreateTensorDescriptor(&bnTensorDesc));
		checkCublasErrors(cublasCreate(&cublasHandle));
	}
	void destroyHandles()
	{
		//checkCUDNN(cudnnDestroyLRNDescriptor(normDesc));
		//checkCUDNN(cudnnDestroyPoolingDescriptor(poolingDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(activDesc));
		checkCUDNN(cudnnDestroyActivationDescriptor(activBoxDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(activTensorDesc));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
		checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(srcConvTensorDesc));
		//checkCUDNN(cudnnDestroyTensorDescriptor(srcPoolTensorDesc));
		//checkCUDNN(cudnnDestroyTensorDescriptor(srcSoftmaxTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstConvTensorDesc));
		/*checkCUDNN(cudnnDestroyTensorDescriptor(dstActivTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstPoolTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(dstSoftmaxTensorDesc));*/
		checkCUDNN(cudnnDestroyTensorDescriptor(biasTensorDesc));
		checkCUDNN(cudnnDestroyTensorDescriptor(bnTensorDesc));
		checkCUDNN(cudnnDestroy(cudnnHandle));
		checkCublasErrors(cublasDestroy(cublasHandle));
	}
public:
	network_t()
	{
		convAlgorithm = -1;
		switch (sizeof(value_type))
		{
		case 2: dataType = CUDNN_DATA_HALF; break;
		case 4: dataType = CUDNN_DATA_FLOAT; break;
		case 8: dataType = CUDNN_DATA_DOUBLE; break;
		default: FatalError("Unsupported data type");
		}
		tensorFormat = CUDNN_TENSOR_NCHW;
		createHandles();
	};
	~network_t()
	{
		destroyHandles();
	}

	void resize(int size, value_type **data)
	{
		if (*data != NULL)
		{
			checkCudaErrors(cudaFree(*data));
		}
		checkCudaErrors(cudaMalloc(data, size * sizeof(value_type)));
		checkCudaErrors(cudaMemset(*data, 0, size * sizeof(value_type)));
	}
	void setConvolutionAlgorithm(const cudnnConvolutionFwdAlgo_t& algo)
	{
		convAlgorithm = (int)algo;
	}

	//void addBias(const cudnnTensorDescriptor_t& dstTensorDesc, const Layer_t<value_type>& layer, int c, value_type* data)
	//{
	//	setTensorDesc(biasTensorDesc, tensorFormat, dataType, 1, c, 1, 1);
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(1);
	//	checkCUDNN(cudnnAddTensor(cudnnHandle,
	//		&alpha, biasTensorDesc,
	//		layer.pointBias_d,
	//		&beta,
	//		dstTensorDesc,
	//		data));
	//}

	//void fullyConnectedForward(const Layer_t<value_type>& ip,
	//	int n, int c, int h, int w,
	//	value_type* srcData, value_type** dstData)
	//{
	//	if (n != 1)
	//	{
	//		FatalError("Not Implemented");
	//	}
	//	int dim_x = c*h*w;
	//	int dim_y = ip.outputs;
	//	resize(dim_y, dstData);
	//	scaling_type alpha = scaling_type(1), beta = scaling_type(1);
	//	//std::ifstream dataFile("simplenetworkData/new/W2_new.bin", std::ios::in | std::ios::binary);
	//	//std::stringstream error_s;
	//	//if (!dataFile)
	//	//{
	//	//	error_s << "Error opening file " << endl;
	//	//	FatalError(error_s.str());
	//	//}
	//	//// we assume the data stored is always in float precision
	//	//float* data_tmp2 = new float[7*7*10];
	//	//memset(data_tmp2, 0, sizeof(float) * 7 * 7 * 10);
	//	//int size_b = (7*7*10) * sizeof(float);
	//	//if (!dataFile.read((char*)data_tmp2, size_b))
	//	//{
	//	//	error_s << "Error reading file " << endl;
	//	//	FatalError(error_s.str());
	//	//}
	//	//float *data_tmpd;
	//	//checkCudaErrors(cudaMalloc((void**)&data_tmpd, (7 * 7 * 10) * sizeof(float)));
	//	//checkCudaErrors(cudaMemcpy(data_tmpd, data_tmp2, (7 * 7 * 10) * sizeof(float), cudaMemcpyHostToDevice));
	//	// place bias into dstData
	//	checkCudaErrors(cudaMemcpy(*dstData, ip.bias_d, dim_y * sizeof(value_type), cudaMemcpyDeviceToDevice));
	//	gemv(cublasHandle, dim_y, h*w, alpha, ip.data_d, srcData, beta, *dstData);
	//	//float* tmpDsc = new float[10 * 1];
	//	//float* tmpSrc = new float[7 * 7];
	//	//float* tmpSum = new float[10];
	//	//memset(tmpSum, 0, sizeof(float) * 10);
	//	//memset(tmpSrc, 0, 7 * 7 * sizeof(float));
	//	//memset(tmpDsc, 0, sizeof(float) * 10 * 1);
	//	//checkCudaErrors(cudaMemcpy(tmpSrc, srcData, sizeof(float) * 7 * 7, cudaMemcpyDeviceToHost));
	//	//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float) * 10, cudaMemcpyDeviceToHost));
	//	//int k = 0;
	//	//for (int j = 0; j < 10; j++)
	//	//{
	//	//	for (int i = 0; i <49; i++)
	//	//	{
	//	//		tmpSum[j] += tmpSrc[i] * data_tmp2[i * 10 + j];
	//	//		//printf("%f ", tmpDsc[w*j +i]);
	//	//		//cout << tmpDsc[j + i] << "\t";
	//	//	}
	//	//	printf("\n");
	//	//	cout << "sum result: " << tmpSum[j] << endl;
	//	//	cout << "prediction result: " << tmpDsc[j] << endl;
	//	//}
	//	h = 1; w = 1; c = dim_y;
	//}
	////int& n, int& c, int& h, int& w

	void depthSetting(int n, int c, int h, int w, int col, int padding, value_type** dstData, value_type** tmp)
	{
		int tmpsize = (col + padding) * (col + padding) * c;
		resize(tmpsize, tmp);
		//checkCudaErrors(cudaMalloc(tmp, tmpsize * sizeof(float)));
		//checkCudaErrors(cudaMemset(*tmp, 0.0, tmpsize * sizeof(float)));
		resize((n * c * h * w), dstData);
	}
	void pointSetting(int n, int c, int h, int w, value_type** dstData)
	{
		resize(n * c * h * w, dstData);
	}

	void anchorBox_generator(value_type* aspect_ratio_layer0 ,value_type* aspect_ratio, value_type** anchorShape0, value_type** anchorShape1to5)
	{
		//aspect_ratio[0] = 1.0;
		//aspect_ratio[1] = 2.0;
		//aspect_ratio[2] = 0.5;
		//aspect_ratio[3] = 3.0;
		//aspect_ratio[4] = 0.3333333;
		//aspect_ratio[5] = 1.0;
		aspect_ratio[0] = 1.0;
		aspect_ratio[1] = 1.5;
		aspect_ratio[2] = 0.3333;
		aspect_ratio[3] = 0.75;
		aspect_ratio[4] = 0.5;
		aspect_ratio[5] = 1.0;


		value_type* scales_layer0 = new value_type[3];
		//scales_layer0[0] = 0.1;
		//scales_layer0[1] = 0.2;
		//scales_layer0[2] = 0.2;

		scales_layer0[0] = 0.1;
		scales_layer0[1] = 0.1;
		scales_layer0[2] = 0.1;


		aspect_ratio_layer0[0] = 1.0;
		aspect_ratio_layer0[1] = 2.0;
		aspect_ratio_layer0[2] = 0.5;
		//aspect_ratio_layer0[0] = 1.0;
		//aspect_ratio_layer0[1] = 1.5;
		//aspect_ratio_layer0[2] = 0.3333;

		dim3 threads_per_block(32, 32, 1);
		dim3 num_of_blocks(1, 1, 1);

		checkCudaErrors(cudaMalloc(anchorShape0, sizeof(value_type) * (num_of_boxChannel_per_layer)));
		checkCudaErrors(cudaMemset(*anchorShape0, 0, sizeof(value_type) *  (num_of_boxChannel_per_layer)));

		checkCudaErrors(cudaMalloc(anchorShape1to5, sizeof(value_type) * (num_of_boxChannel_per_layer * 2 * (num_of_featuremaps-1))));
		checkCudaErrors(cudaMemset(*anchorShape1to5, 0, sizeof(value_type) *  (num_of_boxChannel_per_layer * 2 * (num_of_featuremaps - 1))));

		value_type* aspect_ratio_dev = NULL;
		checkCudaErrors(cudaMalloc(&aspect_ratio_dev, sizeof(value_type) * num_of_boxChannel_per_layer));
		checkCudaErrors(cudaMemcpy(aspect_ratio_dev, aspect_ratio, sizeof(value_type) * num_of_boxChannel_per_layer, cudaMemcpyHostToDevice));

		value_type* aspect_ratio_layer0_dev = NULL;
		checkCudaErrors(cudaMalloc(&aspect_ratio_layer0_dev, sizeof(value_type) * num_of_boxChannel_per_layer/2));
		checkCudaErrors(cudaMemcpy(aspect_ratio_layer0_dev, aspect_ratio_layer0, sizeof(value_type) * num_of_boxChannel_per_layer/2, cudaMemcpyHostToDevice));
		
		value_type* scales_layer0_dev = NULL;
		checkCudaErrors(cudaMalloc(&scales_layer0_dev, sizeof(value_type) * num_of_boxChannel_per_layer / 2));
		checkCudaErrors(cudaMemcpy(scales_layer0_dev, scales_layer0, sizeof(value_type)* num_of_boxChannel_per_layer / 2, cudaMemcpyHostToDevice));
		
		anchorbox_generate(aspect_ratio_layer0_dev ,aspect_ratio_dev, scales_layer0_dev, minScale, maxScale, num_of_boxChannel_per_layer, *anchorShape0, *anchorShape1to5, threads_per_block, num_of_blocks);
	}

	void convolutionSetting(int *n, int *c, int *h, int *w, const Layer_t<value_type>& conv, int padding, int stride, int *tensorOuputDimA, const int tensorDims, value_type** dstData)
	{
		////-----convolution pre-processing------////
		setTensorDesc(srcConvTensorDesc, tensorFormat, dataType, *n, *c, *h, *w);

		tensorOuputDimA[0] = *n;
		tensorOuputDimA[1] = *c;
		tensorOuputDimA[2] = *h;
		tensorOuputDimA[3] = *w;
		const int filterDimA[4] = { conv.outputs, *c, conv.conv_kernel_dim, conv.conv_kernel_dim };
		
		checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
			dataType,
			CUDNN_TENSOR_NCHW,
			tensorDims,
			filterDimA));

		const int convDims = 2;
		int padA[convDims] = { padding, padding };
		int filterStrideA[convDims] = { stride, stride };
		int upscaleA[convDims] = { 1,1 };
		cudnnDataType_t  convDataType = dataType;
		if (dataType == CUDNN_DATA_HALF) {
			convDataType = CUDNN_DATA_FLOAT; //Math are done in FP32 when tensor are in FP16
		}
		checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc,
			convDims,
			padA,
			filterStrideA,
			upscaleA,
			CUDNN_CROSS_CORRELATION,
			convDataType));
		// find dimension of convolution output
		checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc,
			srcConvTensorDesc,
			filterDesc,
			tensorDims,
			tensorOuputDimA));
		*n = tensorOuputDimA[0]; *c = tensorOuputDimA[1];
		*h = tensorOuputDimA[2]; *w = tensorOuputDimA[3];

		setTensorDesc(dstConvTensorDesc, tensorFormat, dataType, *n, *c, *h, *w);

		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
			srcConvTensorDesc,
			filterDesc,
			convDesc,
			dstConvTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,
			&algo
		));

		resize((*n)*(*c)*(*h)*(*w), dstData);
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
			srcConvTensorDesc,
			filterDesc,
			convDesc,
			dstConvTensorDesc,
			algo,
			&sizeInBytes));
		if (sizeInBytes != 0)
		{
			checkCudaErrors(cudaMalloc(&workSpace, sizeInBytes));
		}
	}

	void convoluteForward(const Layer_t<value_type>& conv, int n, int c, int h, int w, cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t sizeInBytes, value_type* srcData, value_type** dstData)
	{
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);

		checkCUDNN(cudnnConvolutionForward(cudnnHandle,
			&alpha,
			srcConvTensorDesc,
			srcData,
			filterDesc,
			conv.convData_d,
			convDesc,
			algo,
			workSpace,
			sizeInBytes,
			&beta,
			dstConvTensorDesc,
			*dstData));

		//addBias(dstConvTensorDesc, conv, c, *dstData);
		if (sizeInBytes != 0)
		{
			checkCudaErrors(cudaFree(workSpace));
		}
	}

	void batchNorm_depth(int n, int c, int h, int w ,const Layer_t<value_type>& conv , cudnnTensorDescriptor_t tensorDesc ,value_type* srcData, value_type** dstData )
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);
		
		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.depthBnScale_d,
			conv.depthBnBias_d,
			conv.depthBnMean_d,
			conv.depthBnVar_d,
			epsilon));
	}

	void batchNorm_point(int n, int c, int h, int w, const Layer_t<value_type>& conv, cudnnTensorDescriptor_t tensorDesc, value_type* srcData, value_type** dstData)
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);

		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.pointBnScale_d,
			conv.pointBnBias_d,
			conv.pointBnMean_d,
			conv.pointBnVar_d,
			epsilon));
	}

	void batchNorm_conv(int n, int c, int h, int w, const Layer_t<value_type>& conv, cudnnTensorDescriptor_t tensorDesc, value_type* srcData, value_type** dstData)
	{
		setTensorDesc(tensorDesc, tensorFormat, dataType, n, c, h, w);
		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		resize(n*c*h*w, dstData);

		checkCUDNN(cudnnDeriveBNTensorDescriptor(bnTensorDesc, tensorDesc, CUDNN_BATCHNORM_SPATIAL));

		checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
			CUDNN_BATCHNORM_SPATIAL,
			&alpha,
			&beta,
			tensorDesc,
			srcData,
			tensorDesc,
			*dstData,
			bnTensorDesc,
			conv.convBnScale_d,
			conv.convBnBias_d,
			conv.convBnMean_d,
			conv.convBnVar_d,
			epsilon));
	}

	//Pooling//

	//void poolForward(int n, int c, int h, int w,
	//	value_type* srcData, value_type** dstData)
	//{
	//	/*const int poolDims = 2;
	//	int windowDimA[poolDims] = { 2,2 };
	//	int paddingA[poolDims] = { 0,0 };
	//	int strideA[poolDims] = { 2,2 };
	//	checkCUDNN(cudnnSetPoolingNdDescriptor(poolingDesc,
	//	CUDNN_POOLING_MAX,
	//	CUDNN_PROPAGATE_NAN,
	//	poolDims,
	//	windowDimA,
	//	paddingA,
	//	strideA));
	//	setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
	//	const int tensorDims = 4;
	//	int tensorOuputDimA[tensorDims] = { n,c,h,w };
	//	checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc,
	//	srcTensorDesc,
	//	tensorDims,
	//	tensorOuputDimA));
	//	n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
	//	h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
	//	setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);*/
	//	resize(n*c*h*w, dstData);
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(0);
	//	checkCUDNN(cudnnPoolingForward(cudnnHandle,
	//		poolingDesc,
	//		&alpha,
	//		srcPoolTensorDesc,
	//		srcData,
	//		&beta,
	//		dstPoolTensorDesc,
	//		*dstData));
	//	//float* tmpDsc = new float[7 * 7];
	//	//memset(tmpDsc, 0, sizeof(float) * 7 * 7);
	//	//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float) * 7 * 7, cudaMemcpyDeviceToHost));
	//	//for (int j = 0; j < h; j++)
	//	//{
	//	//	for (int i = 0; i <w; i++)
	//	//	{
	//	//		//printf("%f ", tmpDsc[w*j +i]);
	//	//		cout << tmpDsc[w*j + i] << "\t";
	//	//	}
	//	//	printf("\n");
	//	//}
	//}

	//softmax//

	//void softmaxForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	//{
	//	resize(n*c*h*w, dstData);
	//	/*setTensorDesc(srcTensorDesc, tensorFormat, dataType, n, c, h, w);
	//	setTensorDesc(dstTensorDesc, tensorFormat, dataType, n, c, h, w);*/
	//	scaling_type alpha = scaling_type(1);
	//	scaling_type beta = scaling_type(0);
	//	checkCUDNN(cudnnSoftmaxForward(cudnnHandle,
	//		CUDNN_SOFTMAX_ACCURATE,
	//		CUDNN_SOFTMAX_MODE_CHANNEL,
	//		&alpha,
	//		srcSoftmaxTensorDesc,
	//		srcData,
	//		&beta,
	//		dstSoftmaxTensorDesc,
	//		*dstData));
	//	//float* tmpDsc = new float[10 * 1];
	//	//memset(tmpDsc, 0, sizeof(float) * 10 * 1);
	//	//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float) * 10 * 1, cudaMemcpyDeviceToHost));
	//	//for (int j = 0; j < 1; j++)
	//	//{
	//	//	for (int i = 0; i <10; i++)
	//	//	{
	//	//		//printf("%f ", tmpDsc[w*j +i]);
	//	//		cout << tmpDsc[w*j + i] << "\t";
	//	//	}
	//	//	printf("\n");
	//	//}
	//}

	void activationForward(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	{
		/*checkCUDNN(cudnnSetActivationDescriptor(activDesc,
		CUDNN_ACTIVATION_RELU,
		CUDNN_PROPAGATE_NAN,
		0.0));*/

		resize(n*c*h*w, dstData);
		//checkCudaErrors(cudaMemset(*dstData, 0, n*c*h*w * sizeof(value_type)));
		setTensorDesc(activTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		checkCUDNN(cudnnActivationForward(cudnnHandle,
			activDesc,
			&alpha,
			activTensorDesc,
			srcData,
			&beta,
			activTensorDesc,
			*dstData));
	}

	void activationForward_box(int n, int c, int h, int w, value_type* srcData, value_type** dstData)
	{
		/*checkCUDNN(cudnnSetActivationDescriptor(activBoxDesc,
		CUDNN_ACTIVATION_SIGMOID,
		CUDNN_PROPAGATE_NAN,
		0.0));*/

		resize(n*c*h*w, dstData);
		//checkCudaErrors(cudaMemset(*dstData, 0, n*c*h*w * sizeof(value_type)));
		setTensorDesc(activTensorDesc, tensorFormat, dataType, n, c, h, w);

		scaling_type alpha = scaling_type(1);
		scaling_type beta = scaling_type(0);
		checkCUDNN(cudnnActivationForward(cudnnHandle,
			activBoxDesc,
			&alpha,
			activTensorDesc,
			srcData,
			&beta,
			activTensorDesc,
			*dstData));
	}

	void depthwise_conv(int *n, int *c, int *h, int *w, int padding, int stride, const int depthwiseOutputW, const Layer_t<value_type>& conv, value_type* srcData, value_type** dstData, value_type** tmp, const int conv_blocks_num,
		dim3 threads_per_block, dim3 num_of_blocks)
	{
		int depthFilterW = 3;
		const int padding_along_hw = (int)fmaxf((float)((depthwiseOutputW - 1)*stride + depthFilterW - *h), 0);
		const int padding_topLeft = (int)(padding_along_hw / 2);
		const int padding_bottomRight = padding_along_hw - padding_topLeft;
		const int padding_blocks = *h*(*c) / 32 + 1;
		int padded_h = *h + padding_along_hw;
		int padded_w = *w + padding_along_hw;
		dim3 padding_threads_num(32, 32, 1);
		dim3 padding_blocks_num(padding_blocks, padding_blocks, 1);
		depthSetting(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, *h, padding_along_hw, dstData, tmp);

		//padding_f(srcData, *tmp, depthwiseOutputW, *h, *c, padding, padding_threads_num, padding_blocks_num);
		padding_conv(srcData, *tmp, depthwiseOutputW, *h, *c, padding_topLeft, padding_bottomRight, padding_threads_num, padding_blocks_num);

		depth_f(srcData, *tmp, *dstData, conv.depthData_d, padded_h, padded_w, *c, depthFilterW, depthwiseOutputW, padding_along_hw, stride, conv_blocks_num, threads_per_block, num_of_blocks); //dstData
		//checkCudaErrors(cudaDeviceSynchronize());

		//cout << "====depth conv2====" << endl;
		//float* tmpDsc_111 = new float[conv.inputs * (*h) * (*w)];
		//memset(tmpDsc_111, 0, sizeof(float) * conv.inputs * (*h) * (*w));
		//checkCudaErrors(cudaMemcpy(tmpDsc_111, *dstData, sizeof(float)*conv.inputs * (*h) * (*w), cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	//float tmpss = 0;
		//	for (int k = 0; k < 1; ++k) {
		//		for (int j = 0; j < *h; ++j) {
		//			for (int i = 0; i < *w; ++i) {
		//				cout << tmpDsc_111[(k) * *h * *w + *w * j + i] << "\t";
		//				//tmpss += tmpDsc_1[(k) * *h * *w + *w * (j + 1) + (i + 1)] * conv.depthData_h[k * 3 * 3 + j * 3 + i];
		//			}
		//		}
		//		printf("----------------------\n");
		//	}//cout << tmpss << endl;
		//	printf("=================\n");
		//}
		//system("pause");

	}

	void pointwise_conv(int *n, int *c, int *h, int *w, int padding, int stride, const int depthwiseOutputW, const Layer_t<value_type>& conv, value_type*srcData, value_type** dstData, const int conv_blocks_num,
		dim3 threads_per_block, dim3 num_of_blocks) 
	{
		int depthFilterW = 3;
		pointSetting(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, dstData);//&srcData
		point_f(srcData, *dstData, conv.pointData_d, *h, *c, depthFilterW, depthwiseOutputW, conv.outputs, padding, stride, conv_blocks_num, threads_per_block, num_of_blocks); //srcData
		checkCudaErrors(cudaDeviceSynchronize());
	}

	void dsc(int *n, int *c, int *h, int *w, int padding, int stride, const Layer_t<value_type>& conv, value_type** srcData, value_type** dstData, value_type** tmp)
	{
		int depthFilterW = 3;
		const int depthwiseOutputW = (int)((*h - depthFilterW + 2*padding) / stride + 1);
		/*const int conv_blocks_num = (depthwiseOutputW * conv.outputs) / 32 + 1;
		dim3 threads_per_block(32, 32, 1);
		dim3 num_of_blocks(conv_blocks_num, conv_blocks_num, 1);*/
		const int conv_blocks_num = (depthwiseOutputW) / 30 + 1;
		dim3 threads_per_block(30, 30, 1);
		dim3 num_of_blocks_depthwise(conv_blocks_num* *c, conv_blocks_num, 1);
		dim3 num_of_blocks_pointwise(conv_blocks_num* conv.outputs, conv_blocks_num, 1);
		//cout << "c : " << *c << " h : " << *h << " w : " << *w << "outFilterSize : " << depthwiseOutputW <<endl;
		depthwise_conv(n, c, h, w, padding, stride, depthwiseOutputW, conv, *srcData, dstData, tmp, conv_blocks_num, threads_per_block, num_of_blocks_depthwise);
		//if (depthwiseOutputW == 19) {
		//	cout << "====depth conv전====" << endl;
		//	float* tmpDsc_10 = new float[conv.inputs * (depthwiseOutputW) * (depthwiseOutputW)];
		//	memset(tmpDsc_10, 0, sizeof(float) * conv.inputs * (depthwiseOutputW) * (depthwiseOutputW));
		//	checkCudaErrors(cudaMemcpy(tmpDsc_10, *dstData, sizeof(float)*conv.inputs * (depthwiseOutputW) * (depthwiseOutputW), cudaMemcpyDeviceToHost));
		//	for (int l = 0; l < 1; ++l) {
		//		//float tmpss = 0;
		//		for (int k = 0; k < 1; ++k) {
		//			for (int j = 0; j < 2; ++j) {
		//				for (int i = 0; i < depthwiseOutputW; ++i) {
		//					cout << tmpDsc_10[(k) *depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i] << " ";
		//					//tmpss += tmpDsc_1[(k) * *h * *w + *w * (j + 1) + (i + 1)] * conv.depthData_h[k * 3 * 3 + j * 3 + i];
		//				} cout << endl; system("pause");
		//			}
		//			printf("----------------------\n");
		//		}//cout << tmpss << endl;
		//		printf("=================\n");
		//	}
		//}

		//float* tmDsc23 = new float[*c * *h * *w];
		//memset(tmDsc23, 0, sizeof(float) * *c * *h * *w);
		//readBinaryFile("test_data/conv1_depthout.bin", depthwiseOutputW*depthwiseOutputW*conv.inputs, tmDsc23);
		//cout << "Done" << endl;
		//system("pause");
		//cout << "====depth conv후====" << endl;
		//float* tmpDsc = new float[conv.inputs * (depthwiseOutputW) * (depthwiseOutputW)];
		//memset(tmpDsc, 0, sizeof(float) * conv.inputs * (depthwiseOutputW) * (depthwiseOutputW));
		//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float)*conv.inputs * (depthwiseOutputW) * (depthwiseOutputW), cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k <conv.inputs; ++k) {
		//		for (int j = 0; j < depthwiseOutputW; ++j) {
		//			for (int i = 0; i <depthwiseOutputW; ++i) {
		//				if (fabsf(tmDsc23[j*depthwiseOutputW*conv.inputs + i * conv.inputs + k] - tmpDsc[(k)* depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i]) > 0.001) {
		//					cout << tmDsc23[j*depthwiseOutputW*conv.inputs + i * conv.inputs + k] << "\t" <<tmpDsc[(k)* depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i] << endl;
		//				}
		//				
		//			}
		//		} 
		//	}
		//	printf("=================\n"); 
		//}
		//system("pause");

		//cout << "conv" << endl;
		//const int padding_along_hw = (int)fmaxf((float)((depthwiseOutputW - 1)*stride + depthFilterW - *h), 0);
		//const int padding_topLeft = (int)(padding_along_hw / 2);
		//const int padding_bottomRight = padding_along_hw - padding_topLeft;
		//cout << depthwiseOutputW << endl << padding << endl << padding_along_hw << endl << padding_topLeft << endl << padding_bottomRight << endl; system("pause");
		//float* tmpOut0 = new float[conv.inputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpOut0, 0, sizeof(float) * conv.inputs * depthwiseOutputW * depthwiseOutputW);
		//float* tmpPad_h = new float[conv.inputs * (*h + padding_along_hw) * (*w + padding_along_hw)];
		//memset(tmpPad_h, 0, sizeof(float) * conv.inputs *  (*h + padding_along_hw) * (*w + padding_along_hw));
		//float* tmpPad = NULL;
		//cudaMalloc(&tmpPad, sizeof(float) * conv.inputs *  (*h + padding_along_hw) * (*w + padding_along_hw));
		//cudaMemset(tmpPad, 0, sizeof(float) *conv.inputs * (*h + padding_along_hw) * (*w + padding_along_hw));
		//dim3 theadsPerBlock_padding(32, 32, 1);
		//dim3 numOfBlocks_padding(*w * *c / 32 + 1, *h * *c / 32 + 1, 1);
		//padding_conv(*srcData, tmpPad, depthwiseOutputW, *h, *c, padding_topLeft, padding_bottomRight, theadsPerBlock_padding, numOfBlocks_padding);
		//checkCudaErrors(cudaMemcpy(tmpPad_h, tmpPad, sizeof(float) *conv.inputs *  (*h + padding_along_hw) * (*w + padding_along_hw), cudaMemcpyDeviceToHost));
		//cout << "conv 전" << endl;
		//float* tmpDsc_1 = new float[*c *depthwiseOutputW * depthwiseOutputW];
		//memset(tmpDsc_1, 0, sizeof(float) * *c * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_1, *dstData, sizeof(float) * *c * depthwiseOutputW* depthwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//	for (int k = 0; k < *c; ++k) {
		//		for (int py = 0; py < depthwiseOutputW; ++py) {
		//			for (int px = 0; px < depthwiseOutputW; ++px) {
		//				for (int j = 0; j < 3; ++j) {
		//					for (int i = 0; i < 3; ++i) {
		//						tmpOut0[k * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] += 
		//							tmpPad_h[k * (*h + padding_along_hw) * (*w + padding_along_hw) + (*w + padding_along_hw) * (py * stride + j) + (px * stride + i)] * conv.depthData_h[k * 3 * 3 + j * 3 + i];
		//					}
		//				}//cout << tmpOut0[k * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] << "\t" << tmpDsc_1[k * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] << endl;
		//				if (fabsf(tmpOut0[k * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] - tmpDsc_1[k * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px]) > 0.001)
		//				{ cout << "ERROR " << endl; 
		//				system("pause"); }
		//			}cout << endl;
		//		}
		//	printf("n : %d =================\n", k);
		//}

		////cout << "====ConvOut====" << endl;
		////float* tmpDsc_2 = new float[conv.inputs * depthwiseOutputW * depthwiseOutputW];
		////memset(tmpDsc_2, 0, sizeof(float) * conv.inputs * depthwiseOutputW * depthwiseOutputW);
		////checkCudaErrors(cudaMemcpy(tmpDsc_2, *dstData, sizeof(float) * conv.inputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		////float totalE = 0;
		////for (int l = 0; l < conv.inputs; ++l) {
		////	float error = 0;
		////	for (int py = 0; py < depthwiseOutputW; ++py) {
		////		for (int px = 0; px < depthwiseOutputW; ++px) {
		////			error += fabsf(tmpOut0[l * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] - tmpDsc_2[l * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px]);
		////		}
		////	}
		////	printf("error : %f =================\n", error); totalE += error;
		////}
		////cout << totalE / (float)(conv.inputs * depthwiseOutputW * depthwiseOutputW) << endl;
		////system("pause");

		//float* srcData2 = NULL;
		//resize(conv.inputs *depthwiseOutputW *depthwiseOutputW, &srcData2);
		resize(conv.inputs * depthwiseOutputW *depthwiseOutputW, srcData);
		batchNormalization(*dstData, *srcData, conv.depthBnBias_d, conv.depthBnScale_d, conv.depthBnMean_d, conv.depthBnVar_d, (float)epsilon, *c, depthwiseOutputW, depthwiseOutputW);
		//batchNorm_depth(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, conv, srcTensorDesc, *dstData, srcData);

		//cout << "srcData : " << *srcData << endl << "dstData : " << *dstData << endl;
		//cout << "====batchNorm====" << endl;
		//float* tmpDsc22 = new float[conv.inputs * (depthwiseOutputW) * (depthwiseOutputW)];
		//memset(tmpDsc22, 0, sizeof(float) *conv.inputs * depthwiseOutputW * depthwiseOutputW);
		////float* tmpDsc333 = new float[conv.inputs * (depthwiseOutputW) * (depthwiseOutputW)];
		////memset(tmpDsc333, 0, sizeof(float) *conv.inputs * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc22, *srcData, sizeof(float)* conv.inputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));
		////checkCudaErrors(cudaMemcpy(tmpDsc333, srcData2, sizeof(float)* conv.inputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int i = 0; i <depthwiseOutputW; ++i) {
		//			for (int k = 0; k < conv.inputs; ++k) {
		//			
		//				for (int j = 0; j < depthwiseOutputW; ++j) {
		//				cout << tmpDsc22[(k)* depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i] << endl;
		//				//if (fabsf(tmpDsc22[k * depthwiseOutputW * depthwiseOutputW + j * depthwiseOutputW + i] - tmpDsc333[k * depthwiseOutputW * depthwiseOutputW + j * depthwiseOutputW + i]) > 0.001)
		//				//{
		//				//	cout << "ERROR " << endl; system("pause");
		//				//}
		//			}printf("----------------------\n"); system("pause");
		//			//printf("\n");
		//		}//system("pause");
		//		
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		activationForward(1, conv.inputs, depthwiseOutputW, depthwiseOutputW, *srcData, dstData);

		//cout << "activation 후" << endl;
		//cout << "srcData : " << *srcData << endl << "dstData : " << *dstData << endl;
		//cout << "====activation====" << endl;
		//float* tmpDsc = new float[conv.inputs * (depthwiseOutputW) * (depthwiseOutputW)];
		//memset(tmpDsc, 0, sizeof(float) * conv.inputs * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float)*conv.inputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 31; k < conv.inputs; ++k) {
		//		for (int j = 0; j <depthwiseOutputW; ++j) {
		//			for (int i = 0; i <depthwiseOutputW; ++i) {
		//				cout << tmpDsc[(k)* depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i] << "\t";
		//				
		//			}
		//			printf("\n");
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		//cout << "====POINT conv전====" << endl;
		//float* tmpDsc_1 = new float[conv.inputs *depthwiseOutputW *depthwiseOutputW];
		//memset(tmpDsc_1, 0, sizeof(float) * conv.inputs * depthwiseOutputW *depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_1, *dstData, sizeof(float)*conv.inputs * depthwiseOutputW* depthwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	float tmpss = 0;
		//	for (int k = 0; k < conv.inputs; ++k) {
		//		for (int j = 0; j < 1; ++j) {
		//			for (int i = 0; i <1; ++i) {
		//				tmpss += tmpDsc_1[(k) * depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * (j) + (i)] * conv.pointData_h[k+ conv.inputs];
		//			}
		//		}
		//	}cout << tmpss << endl;
		//	printf("=================\n");
		//}
		//system("pause");

		pointwise_conv(n, c, h, w, padding, stride, depthwiseOutputW, conv, *dstData, srcData, conv_blocks_num, threads_per_block, num_of_blocks_pointwise);
		
		//if (depthwiseOutputW == 19) {
		//	cout << "====pointConv====" << endl;
		//	float* tmpDsc2 = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//	memset(tmpDsc2, 0, sizeof(float) * conv.outputs * depthwiseOutputW * depthwiseOutputW);
		//	checkCudaErrors(cudaMemcpy(tmpDsc2, *srcData, sizeof(float)*conv.outputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//	for (int l = 0; l < 1; ++l) {
		//		for (int k = 0; k < 1; ++k) {
		//			for (int j = 0; j < 2; ++j) {
		//				for (int i = 0; i < depthwiseOutputW; ++i) {
		//					cout << tmpDsc2[k * depthwiseOutputW * depthwiseOutputW + depthwiseOutputW * j + i] << "  ";
		//				}
		//				printf("\n"); system("pause");
		//			}
		//			printf("----------------------\n");
		//		}
		//		printf("=================\n");
		//	}
		//	system("pause");
		//}
		//cout << "pointwise_conv" << endl;
		//float* tmpOut00 = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpOut00, 0, sizeof(float) * conv.outputs * depthwiseOutputW * depthwiseOutputW);
		//float* tmpDsc_11 = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpDsc_11, 0, sizeof(float) *conv.outputs * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_11, *srcData, sizeof(float) * conv.outputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < conv.outputs; ++l) {
		//		for (int py = 0; py < depthwiseOutputW; ++py) {
		//			for (int px = 0; px < depthwiseOutputW; ++px) {
		//				for (int k = 0; k < *c; ++k) {
		//				tmpOut00[l * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] +=
		//					tmpDsc3333[k * depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * py +px] * conv.pointData_h[l * *c + k];
		//				} if (fabsf(tmpOut00[l * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px] - tmpDsc_11[l * depthwiseOutputW * depthwiseOutputW + py * depthwiseOutputW + px]) > 0.001)
		//				{
		//					cout << "ERROR on Pointwise" << endl; system("pause");
		//			}
		//		}
		//	}printf("=================\n");
		//}
		
		resize(conv.outputs * depthwiseOutputW *depthwiseOutputW, dstData);
		batchNormalization(*srcData, *dstData, conv.pointBnBias_d, conv.pointBnScale_d, conv.pointBnMean_d, conv.pointBnVar_d, (float)epsilon, conv.outputs, depthwiseOutputW, depthwiseOutputW);
		//batchNorm_point(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, conv, dstTensorDesc, *srcData, dstData);

		//cout << "bathch 후" << endl;
		//cout << "srcData : " << *srcData << endl << "dstData : " << *dstData << endl;
		//cout << "====batchNorm-point====" << endl;
		//float* tmpDsc_b = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpDsc_b, 0, sizeof(float) * conv.outputs * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_b, *dstData, sizeof(float)*conv.outputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < conv.outputs; ++k) {
		//		for (int j = 0; j < depthwiseOutputW; ++j) {
		//			for (int i = 0; i <depthwiseOutputW; ++i) {
		//				cout << tmpDsc_1[k *depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * j + i] <<"\t" << tmpDsc_b[ k *depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * j + i] << "\n";
		//			}
		//			printf("\n");
		//		} system("pause");
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		activationForward(1, conv.outputs, depthwiseOutputW, depthwiseOutputW, *dstData, srcData);

		//cout << "====activation-point====" << endl;
		//float* tmpDsc = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpDsc, 0, sizeof(float) * conv.outputs * depthwiseOutputW * depthwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc, *srcData, sizeof(float)*conv.outputs * depthwiseOutputW * depthwiseOutputW, cudaMemcpyDeviceToHost)); //srcData
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < conv.outputs; ++k) {
		//		for (int j = 0; j < depthwiseOutputW; ++j) {
		//			for (int i = 0; i <depthwiseOutputW; ++i) {
		//				cout << tmpDsc[k *depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * j + i] << "\t" << tmpDsc_b[k *depthwiseOutputW *depthwiseOutputW + depthwiseOutputW * j + i] << endl;
		//			}
		//			printf("\n");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//delete tmpDsc;
		//system("pause");

		*n = 1; *c = conv.outputs;
		*h = *w = depthwiseOutputW;
		//cout << "dsc - n, c, h, w : " << *n <<"\t" << *c << "\t" << *h << "\t" << *w << endl;
	}

	void extraLayer(int *n, int *c, int *h, int *w, int padding, int stride, int *tensorOuputDimA, const int tensorDims, const Layer_t<value_type>& conv, value_type** srcData, value_type** dstData, value_type** tmp)
	{
		int convFilterW = 3;
		const int pointwiseOutputW = *h;
		const int convOutputW = (int)((*h - convFilterW + 2 * padding) / stride + 1);
		//const int conv_blocks_num = (pointwiseOutputW * conv.outputs) / 32 + 1;
		const int conv_blocks_num = (convOutputW) / 30 + 1;
		//dim3 threads_per_block(32, 32, 1);
		//dim3 num_of_blocks(conv_blocks_num, conv_blocks_num, 1);
		
		dim3 threads_per_block(30, 30, 1);
		dim3 num_of_blocks_pointwise(conv_blocks_num* conv.pointoutputs, conv_blocks_num, 1);

		pointSetting(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, dstData);
		point_f(*srcData, *dstData, conv.pointData_d, *h, *c, convFilterW, pointwiseOutputW, conv.pointoutputs, padding, stride, conv_blocks_num, threads_per_block, num_of_blocks_pointwise);
		

		//float* tmpOut0 = new float[conv.pointoutputs * pointwiseOutputW * pointwiseOutputW];
		//memset(tmpOut0, 0, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW);
		//cout << "conv 전" << endl;
		//float* tmpDsc_1 = new float[conv.inputs * pointwiseOutputW * pointwiseOutputW];
		//memset(tmpDsc_1, 0, sizeof(float) *conv.inputs * pointwiseOutputW * pointwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_1, *srcData, sizeof(float) * conv.inputs * pointwiseOutputW * pointwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < conv.pointoutputs; ++l) {
		//		for (int py = 0; py < pointwiseOutputW; ++py) {
		//			for (int px = 0; px < pointwiseOutputW; ++px) {
		//				for (int k = 0; k < conv.inputs; ++k) {
		//				tmpOut0[l * pointwiseOutputW * pointwiseOutputW + py * pointwiseOutputW + px] +=
		//					tmpDsc_1[k * pointwiseOutputW *pointwiseOutputW + pointwiseOutputW * py +px] * conv.pointData_h[l * conv.inputs + k];
		//			}
		//		}
		//	}
		//}
		//cout << "====ConvOut====" << endl;
		//float* tmpDsc_2 = new float[conv.pointoutputs * pointwiseOutputW * pointwiseOutputW];
		//memset(tmpDsc_2, 0, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc_2, *dstData, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//float totalE = 0;
		//for (int l = 0; l < conv.pointoutputs; ++l) {
		//	float error = 0;
		//	for (int py = 0; py < pointwiseOutputW; ++py) {
		//		for (int px = 0; px < pointwiseOutputW; ++px) {
		//			error += fabsf(tmpOut0[l * pointwiseOutputW * pointwiseOutputW + py * pointwiseOutputW + px] - tmpDsc_2[l * pointwiseOutputW * pointwiseOutputW + py * pointwiseOutputW + px]);
		//		}
		//	}
		//	printf("error : %f =================\n", error); totalE += error;
		//}
		//cout << totalE / (float)(conv.pointoutputs * pointwiseOutputW * pointwiseOutputW) << endl;
		//system("pause");


		//cout << "====point====" << endl;
		//float* tmpDsc = new float[conv.pointoutputs * pointwiseOutputW * pointwiseOutputW];
		//memset(tmpDsc, 0, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float)*conv.pointoutputs * pointwiseOutputW * pointwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 1; k <2 ; ++k) { //conv.pointoutputs
		//		for (int j = 0; j < 1; ++j) {
		//			for (int i = 0; i <1; ++i) { //pointwiseOutputW
		//				cout << tmpDsc[k *pointwiseOutputW *pointwiseOutputW + pointwiseOutputW * j + i] << "\t";
		//			}
		//			printf("\n");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//batchNorm_point(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, conv, srcTensorDesc, *dstData, srcData);

		batchNormalization(*dstData, *srcData, conv.pointBnBias_d, conv.pointBnScale_d, conv.pointBnMean_d, conv.pointBnVar_d, (float)epsilon, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW);

		//cout << "bathch 후" << endl;
		//cout << "srcData : " << *srcData << endl << "dstData : " << *dstData << endl;
		//cout << "====batchNorm-point====" << endl;
		//float* tmpDsc = new float[conv.outputs * depthwiseOutputW * depthwiseOutputW];
		//memset(tmpDsc, 0, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc, *srcData, sizeof(float)*conv.pointoutputs * pointwiseOutputW * pointwiseOutputW, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < conv.pointoutputs; ++k) {
		//		for (int j = 0; j < pointwiseOutputW; ++j) {
		//			for (int i = 0; i <pointwiseOutputW; ++i) {
		//				cout << tmpDsc[ k *pointwiseOutputW *pointwiseOutputW + pointwiseOutputW * j + i] << "\t";
		//			}
		//			printf("\n");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		activationForward(1, conv.pointoutputs, pointwiseOutputW, pointwiseOutputW, *srcData, dstData);
		
		//cout << "point 후" << endl;
		//cout << "srcData : " << *srcData << endl << "dstData : " << *dstData << endl;
		//cout << "====pointConv====" << endl;
		//float* tmpDsc = new float[conv.pointoutputs * pointwiseOutputW * pointwiseOutputW];
		//memset(tmpDsc, 0, sizeof(float) * conv.pointoutputs * pointwiseOutputW * pointwiseOutputW);
		//checkCudaErrors(cudaMemcpy(tmpDsc, *dstData, sizeof(float)*conv.pointoutputs * pointwiseOutputW * pointwiseOutputW, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < 1; ++l) {
		//	float tmpss = 0;
		//	for (int k = 0; k < conv.pointoutputs; ++k) {
		//		for (int j = 0; j < 3; ++j) {
		//			for (int i = 0; i <3; ++i) {
		//				tmpss += tmpDsc[k * pointwiseOutputW * pointwiseOutputW + pointwiseOutputW * (j+3) + (i+3)] * conv.convData_h[l* 3*3*conv.pointoutputs + k * 3 * 3 + j * 3 + i];
		//			}
		//		}
		//	} cout << tmpss << endl;
		//	printf("=================\n");
		//}
		//system("pause");
		*n = 1; *c = conv.pointoutputs;
		*h = *w = pointwiseOutputW;

		convolutionSetting(n, c, h, w, conv, padding, stride, tensorOuputDimA, tensorDims, srcData);
		//convoluteForward(conv, *n, *c, *h, *w, algo, workSpace, sizeInBytes, *dstData, srcData);
		resize(*c * *h * *w, srcData);
		float * filter = conv.convData_d;
		//resize(conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding), tmp);
		convolution(dstData, tmp, srcData, &filter, pointwiseOutputW, pointwiseOutputW, conv.pointoutputs, conv.outputs, 3, padding, stride);
		//cout << fixed;
		//cout.precision(9);
		//float* tmpOut0 = new float[conv.outputs * *h * *w];
		//memset(tmpOut0, 0, sizeof(float) * conv.outputs * *h * *w);
		//float* tmpPad_h = new float[conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding)];
		//memset(tmpPad_h, 0, sizeof(float) *conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding));
		//float* tmpPad = NULL;
		//cudaMalloc(&tmpPad, sizeof(float) * conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding));
		//cudaMemset(tmpPad, 0, sizeof(float) *conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding));
		//dim3 theadsPerBlock_padding(32, 32, 1);
		//dim3 numOfBlocks_padding(pointwiseOutputW*conv.pointoutputs / 32 +1, pointwiseOutputW*conv.pointoutputs / 32 + 1, 1);
		//padding_f(*dstData, tmpPad, *h, pointwiseOutputW, conv.pointoutputs, padding, theadsPerBlock_padding, numOfBlocks_padding);
		//checkCudaErrors(cudaMemcpy(tmpPad_h, tmpPad, sizeof(float) * conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding), cudaMemcpyDeviceToHost));
		//cout << "conv 전" << endl;
		//float* tmpDsc_1 = new float[conv.pointoutputs * (pointwiseOutputW) * (pointwiseOutputW)];
		//memset(tmpDsc_1, 0, sizeof(float) *conv.pointoutputs * (pointwiseOutputW) * (pointwiseOutputW));
		//checkCudaErrors(cudaMemcpy(tmpDsc_1, *dstData, sizeof(float)* conv.pointoutputs * (pointwiseOutputW) * (pointwiseOutputW), cudaMemcpyDeviceToHost));//srcData
		//for (int l =0; l < conv.outputs; ++l) {
		//		for (int py = 0; py < *h; ++py) {
		//			for (int px = 0; px < *w; ++px) {
		//				for (int k = 0; k < conv.pointoutputs; ++k) {
		//				for (int j = 0; j < 3; ++j) {
		//					for (int i = 0; i < 3; ++i) {
		//						tmpOut0[l * *h * *w + py* *w +px] +=
		//							tmpPad_h[k * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding) + (pointwiseOutputW + 2 * padding) *(py*stride+j) + (px*stride +i)]
		//							* conv.convData_h[l * conv.pointoutputs * 3 * 3 + k * 3 * 3 + j * 3 + i];
		//					}
		//				}
		//			}
		//		}
		//	}
		//	printf("n : %d =================\n", l);
		//}
		//cout << "====ConvOut====" << endl;
		//float* tmpDsc_2 = new float[conv.outputs * *h * *w];
		//memset(tmpDsc_2, 0, sizeof(float) * conv.outputs * *h * *w);
		//checkCudaErrors(cudaMemcpy(tmpDsc_2, *srcData, sizeof(float)* conv.outputs * *h * *w, cudaMemcpyDeviceToHost));//srcData
		//float totalE = 0;
		//for (int l = 0; l < conv.outputs; ++l) {
		//	float error = 0;
		//	for (int py = 0; py < *h; ++py) {
		//		for (int px = 0; px < *w; ++px) {
		//			error += fabsf(tmpOut0[l * *h * *w + py * *w + px] - tmpDsc_2[l * *h * *w + py * *w + px]);
		//			cout << tmpDsc_2[l * *h * *w + py * *w + px] <<"\t" <<tmpOut0[l * *h * *w + py * *w + px] << endl;
		//		}
		//	} 
		//	
		//	printf("error : %f =================\n", error); totalE += error;
		//}
		//cout <<100 *( totalE / (float)(conv.outputs * *h * *w)) << endl;
		//system("pause");



		//cout << "====ConvOut====" << endl;
		//float* tmpDsc_conv = new float[*c * *h * *w];
		//memset(tmpDsc_conv, 0, sizeof(float) * *c * *h * *w);
		//checkCudaErrors(cudaMemcpy(tmpDsc_conv, *srcData, sizeof(float)* *c * *h * *w, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < 1; ++k) {
		//		for (int j = 2; j < 3; ++j) {
		//			for (int i = 2; i < 3; ++i) {
		//				cout << tmpDsc_conv[k * *h * *w + *w * j + i] << "\t";
		//			}
		//			//printf("\n");
		//		} system("pause");
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		//cout << fixed;
		//cout.precision(9);
		//float* tmpOut0 = new float[conv.outputs * *h * *w];
		//memset(tmpOut0, 0, sizeof(float) * conv.outputs * *h * *w);
		////float* tmpPad_h = new float[conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding)];
		////memset(tmpPad_h, 0, sizeof(float) *conv.pointoutputs * (pointwiseOutputW + 2 * padding) * (pointwiseOutputW + 2 * padding));
		//cout << "====ConvOut====" << endl;
		//float* tmpDsc_2 = new float[conv.outputs * *h * *w];
		//memset(tmpDsc_2, 0, sizeof(float) * conv.outputs * *h * *w);
		//checkCudaErrors(cudaMemcpy(tmpDsc_2, *srcData, sizeof(float)* conv.outputs * *h * *w, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < conv.outputs; ++l) {
		//	for (int py = 0; py < *h; ++py) {
		//		for (int px = 0; px < *w; ++px) {
		//			tmpOut0[l * *h * *w + py * *w + px] = conv.convBnScale_h[l] * (tmpDsc_2[l * *h * *w + py * *w + px] - conv.convBnMean_h[l]) / sqrtf(epsilon + conv.convBnVar_h[l]) + conv.convBnBias_h[l];
		//		}
		//	} 
		//}

		//batchNorm_conv(*n, *c, *h, *w, conv, dstTensorDesc, *srcData, dstData);
		resize(1**c**h**w, dstData);
		batchNormalization(*srcData, *dstData, conv.convBnBias_d, conv.convBnScale_d, conv.convBnMean_d, conv.convBnVar_d, (float)epsilon, conv.outputs, *h, *w);
		
		//cout << "====batchOut====" << endl;
		//float* tmpDsc_3 = new float[conv.outputs * *h * *w];
		//memset(tmpDsc_3, 0, sizeof(float) * conv.outputs * *h * *w);
		//checkCudaErrors(cudaMemcpy(tmpDsc_3, *dstData, sizeof(float)* conv.outputs * *h * *w, cudaMemcpyDeviceToHost));//srcData
		//for (int l = 0; l < conv.outputs; ++l) {
		//	float error = 0;
		//	for (int py = 0; py < *h; ++py) {
		//		for (int px = 0; px < *w; ++px) {
		//			error += fabsf(tmpOut0[l * *h * *w + py * *w + px] - tmpDsc_3[l * *h * *w + py * *w + px]);
		//		}
		//	} cout << "error : " << error << endl;
		//}
		//system("pause");
		activationForward(*n, *c, *h, *w, *dstData, srcData);

		//cout << "extraLayer - n, c, h, w : " << *n << "\t" << *c << "\t" << *h << "\t" << *w << endl;
	}

	void boxPredictor(int *n, int *c, int *h, int *w, int original_image_h, int original_image_w, int *anchor_num, int *box_featuremap_size, int* box_index, int* count_layer, const Layer_t<value_type>&conv_loc,
		const Layer_t<value_type>&conv_conf, value_type** anchorShape, value_type** srcData, value_type** locData, value_type** confData, value_type** locData_all, value_type** confData_all)
	{
		const int conv_blocks_num = (*h) / 30 + 1;
		dim3 threads_per_block(30, 30, 1);
		dim3 num_of_blocks_conf(conv_blocks_num * conv_conf.outputs, conv_blocks_num, 1);
		dim3 num_of_blocks_loc(conv_blocks_num * conv_loc.outputs, conv_blocks_num, 1);
		value_type *confData_tmp = NULL;
		
		//cout << " pointf 전" << endl;
		//float* tmDsc22 = new float[*n * *c * *h * *w];
		//memset(tmDsc22, 0, sizeof(float)**n * *c * *h * *w);
		//cudaMemcpy(tmDsc22, *srcData, sizeof(float)**n * *c * *h * *w, cudaMemcpyDeviceToHost);
		//float tmpss = 0;
		//for (int l = 0; l <conv_conf.outputs; ++l) {
		//	for (int j = 0; j < *h; ++j) {
		//		for (int i = 0; i < *w; ++i) {
		//			for (int k = 0; k < *c; ++k) {
		//				tmpss += tmDsc22[k**h**w + j * *w + i] * conv_conf.pointData_h[l**c + k];
		//			}//tmpss += conv_conf.pointBias_h[l];
		//			cout << tmpss << "\t";
		//			tmpss = 0;
		//			
		//		}cout << endl;
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		
		pointSetting(1, conv_loc.outputs, *h, *w, locData);
		pointbias_f(*srcData, *locData, conv_loc.pointData_d, conv_loc.pointBias_d, *h, *c, 1, *h, conv_loc.outputs, 0, 0, conv_blocks_num, threads_per_block, num_of_blocks_loc);

		//cout << " pointf 후" << endl;
		//float* tmDsc33 = new float[*n * conv_loc.outputs * *h * *w];
		//memset(tmDsc33, 0, sizeof(float)**n * conv_loc.outputs * *h * *w);
		//cudaMemcpy(tmDsc33, *locData, sizeof(float)**n * conv_loc.outputs * *h * *w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < *n; ++l) {
		//	for (int k = 8; k < 12; ++k) {
		//		for (int j = 4; j < 5; ++j) {
		//			for (int i = 5; i < 6; ++i) {
		//				cout<<tmDsc33[k**h**w + j * *w + i] <<endl;
		//			}
		//		}
		//	}
		//}
		//system("pause");
		//cout << "====POINT conv전====" << endl;
		//float* tmpDsc_1 = new float[conv_loc.inputs **h **h];
		//memset(tmpDsc_1, 0, sizeof(float) * conv_loc.inputs * *h **h);
		//checkCudaErrors(cudaMemcpy(tmpDsc_1, *srcData, sizeof(float)*conv_loc.inputs * *h* *h, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < conv_loc.outputs; ++l) {
		//	for (int j = 0; j < *h; ++j) {
		//		for (int i = 0; i <*h; ++i) {
		//			float tmpss = 0;
		//			for (int k = 0; k < conv_loc.inputs; ++k) {
		//				tmpss += tmpDsc_1[(k) * *h **h + *h * (j)+(i)] * conv_loc.pointData_h[k + l * conv_loc.inputs];
		//			}//cout << tmpss + conv_conf.pointBias_h[l] <<"\t" <<tmpDsc_11[l * *h * *h + j * *h + i] <<endl;
		//			if (fabsf(tmpss + conv_loc.pointBias_h[l] - tmDsc33[l * *h * *h + j * *h + i]) > 0.001) { cout << "ERROR" << endl; system("pause"); }
		//		}printf("=================\n");
		//	}
		//}
		//system("pause");


		pointSetting(1, conv_conf.outputs, *h, *w, &confData_tmp); //&confData_tmp
		pointbias_f(*srcData, confData_tmp, conv_conf.pointData_d, conv_conf.pointBias_d, *h, *c, 1, *h, conv_conf.outputs, 0, 0, conv_blocks_num, threads_per_block, num_of_blocks_conf);

		//point_f(*srcData, confData_tmp, conv_conf.pointData_d, *h, *c, 1, *h, conv_conf.outputs, 0, 0, conv_blocks_num, threads_per_block, num_of_blocks_conf);
		//float iter = 0;
		//float* tmpDsc_12 = new float[conv_conf.outputs **h **h];
		//memset(tmpDsc_12, 0, sizeof(float) * conv_conf.outputs * *h **h);
		//checkCudaErrors(cudaMemcpy(tmpDsc_12, confData_tmp, sizeof(float)*conv_conf.outputs * *h* *h, cudaMemcpyDeviceToHost));
		//
		//	for (int j = 0; j < *h; ++j) {
		//		for (int i = 0; i <*h; ++i) {
		//			for (int k = 0; k < conv_conf.inputs; ++k) {
		//			cout << tmpDsc_11[(k) * *h **h + *h * (j)+(i)] <<" [[[ -- ]]]" <<conv_conf.pointData_h[k ] << "\n";
		//			iter += tmpDsc_11[(k) * *h **h + *h * (j)+(i)] * conv_conf.pointData_h[k ];
		//			} cout << "값 : " << iter + conv_conf.pointBias_h[0] << endl; system("pause"); iter = 0;
		//	} cout << "=====================" << endl;
		//	
		//}
		//printf("=================\n");
		//system("pause");

		//float* tmpDsc_11 = new float[conv_conf.outputs **h **h];
		//memset(tmpDsc_11, 0, sizeof(float) * conv_conf.outputs * *h **h);
		//checkCudaErrors(cudaMemcpy(tmpDsc_11, confData_tmp, sizeof(float)*conv_conf.outputs * *h* *h, cudaMemcpyDeviceToHost));
		//cout << fixed;
		//cout.precision(3);
		//
		//	for (int j = 0; j < *h; ++j) {
		//		for (int i = 0; i <*w; ++i) {
		//			for (int k = 0; k < 3; ++k) {
		//			cout <<tmpDsc_11[(k) * *h **h + *h * (j)+(i)]<<"  ";
		//		} cout << endl; system("pause");
		//		} cout << "=====================" <<endl;
		//	}
		//printf("=================\n");
		//system("pause");

		//float* softmaxResult = new float[conv_conf.outputs * *h **w];
		//memset(softmaxResult, 0, sizeof(float)*conv_conf.outputs * *h * *w);

		//softmax(tmpDsc_11, softmaxResult, num_anchors, conv_conf.outputs, *h, *w);
		//cout << fixed;
		//cout.precision(4);
		//for (int l = 0; l < *n; ++l) {
		//	for (int k = 0; k < conv_conf.outputs; ++k) {
		//		for (int j = 0; j < *h; ++j) {
		//			for (int i = 0; i <*w; ++i) {
		//				if (softmaxResult[*h**w*k + *h * j + i] > 0.9) {
		//					cout << "anchor : " << " " << l << "   " << "( " << i << " , " << j << " )" << " class : " << k%9 << endl;
		//				}
		//				//cout << softmaxResult[*h**w*k + *h * j + i] << "\t";
		//			} //cout << endl;
		//		}
		//		//printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//checkCudaErrors(cudaMemcpy(confData_tmp, softmaxResult, sizeof(float)*conv_conf.outputs* * h * * w, cudaMemcpyHostToDevice));


		//float* tmpDsc_11 = new float[conv_conf.outputs **h **h];
		//memset(tmpDsc_11, 0, sizeof(float) * conv_conf.outputs * *h **h);
		//checkCudaErrors(cudaMemcpy(tmpDsc_11, confData_tmp, sizeof(float)*conv_conf.outputs * *h* *h, cudaMemcpyDeviceToHost));
		//float* tmDsc2 = new float[*n * conv_loc.outputs * *h * *w];
		//memset(tmDsc2, 0, sizeof(float)**n * conv_loc.outputs * *h * *w);
		//cudaMemcpy(tmDsc2, *locData, sizeof(float)**n * conv_loc.outputs * *h * *w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < *n; ++l) {
		//	for (int k = 0; k < conv_conf.outputs; ++k) {
		//		for (int j = 0; j < *h; ++j) {
		//			for (int i = 0; i <*w; ++i) {
		//				cout << tmpDsc_11[*h**w*k + *w * j + i] << "  ";
		//			} cout << endl;
		//		}system("pause");
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		remove_background(confData_tmp, anchor_num[*box_index], num_classes, *w, threads_per_block, num_of_blocks_conf);
		activationForward_box(*n, conv_conf.outputs, *h, *w, confData_tmp, confData);

		

		


		encode_boxData(*h, *w, original_image_h, original_image_w, anchor_num[*box_index], count_layer, anchorShape, locData);

		clipWindow(locData, anchor_num[*box_index], *h, *w, original_image_h, original_image_w);

		sum_boxes(*locData, *confData, *locData_all, *confData_all, num_classes, box_featuremap_size, anchor_num, *box_index, box_code, box_total);

		//float* tmDsc11 = new float[box_total * (num_classes + 1)];
		//memset(tmDsc11, 0, sizeof(float)* box_total * (num_classes + 1));
		//cudaMemcpy(tmDsc11, *confData_all, sizeof(float)* box_total * (num_classes + 1), cudaMemcpyDeviceToHost);
		//for (int i = 0; i < num_classes + 1; ++i) {
		//	for (int j = 0; j < box_total; ++j) {
		//		if (tmDsc11[i * box_total + j] > 0.90) {
		//			cout << tmDsc11[i * box_total + j] << ",  (x , y) : " << j << " , " << i << endl;
		//		}
		//	}
		//}
		//printf("----------------------\n");
		//system("pause");

		*box_index += 1;
		
		//float* tmDsc2 = new float[*n * conv_loc.outputs * *h * *w];
		//memset(tmDsc2, 0, sizeof(float)**n * conv_loc.outputs * *h * *w);
		//cudaMemcpy(tmDsc2, *locData, sizeof(float)**n * conv_loc.outputs * *h * *w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < *n; ++l) {
		//	for (int k = 0; k < conv_loc.outputs; ++k) {
		//		for (int j = 0; j < *h; ++j) {
		//			for (int i = 0; i <*w; ++i) {
		//				cout << tmDsc2[*h**w*k + *h * j + i] << "\t";
		//			} cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
	}

	void encode_boxData(int h, int w, int original_image_h, int original_image_w, int num_anchors, int* count_layer, value_type** anchorShape, value_type ** locData)
	{
		int channels = box_code * num_anchors;
		dim3 threads_per_block(w, h, 1);
		dim3 num_of_blocks(1, channels, 1);
		
		encode_locData(*locData, num_anchors, *anchorShape, box_code, h, w, original_image_h, original_image_w, *count_layer, threads_per_block, num_of_blocks);

		*count_layer += 1;
		
	}

	void clipWindow(float **locData, int num_anchors, int featuremap_height, int featuremap_width, int original_image_h, int original_image_w)
	{
		dim3 threads_per_block(featuremap_width, featuremap_height, 1);
		dim3 num_of_blocks(1, num_anchors, 1);
		clip_window(*locData, num_anchors, box_code, featuremap_height, featuremap_width, original_image_h, original_image_w, threads_per_block, num_of_blocks);
	}



	int classify_example(const char* fname, const Layer_t<value_type>& conv0, const Layer_t<value_type>& conv1, const Layer_t<value_type>& conv2, const Layer_t<value_type>& conv3,
		const Layer_t<value_type>& conv4, const Layer_t<value_type>& conv5, const Layer_t<value_type>& conv6, const Layer_t<value_type>& conv7, const Layer_t<value_type>& conv8,
		const Layer_t<value_type>& conv9, const Layer_t<value_type>& conv10, const Layer_t<value_type>& conv11, const Layer_t<value_type>& conv12, const Layer_t<value_type>& conv13,
		const Layer_t<value_type>& conv14, const Layer_t<value_type>& conv15, const Layer_t<value_type>& conv16, const Layer_t<value_type>& conv17, const Layer_t<value_type>& box0_loc,
		const Layer_t<value_type>& box0_conf, const Layer_t<value_type>& box1_loc, const Layer_t<value_type>& box1_conf, const Layer_t<value_type>& box2_loc, const Layer_t<value_type>& box2_conf,
		const Layer_t<value_type>& box3_loc, const Layer_t<value_type>& box3_conf, const Layer_t<value_type>& box4_loc, const Layer_t<value_type>& box4_conf,
		const Layer_t<value_type>& box5_loc, const Layer_t<value_type>& box5_conf)
	{
		clock_t start, end;
		double single_start, single_end;
		int n, c, h, w;
		int layer_count = 0;
		int box_index = 0;
		int original_image_h = NULL;
		int original_image_w = NULL;
		int box0_h, box0_w, box1_h, box1_w, box2_h, box2_w, box3_h, box3_w, box4_h, box4_w, box5_h, box5_w;
		value_type *srcData = NULL;
		value_type *dstData = NULL;
		value_type *tmp = NULL;
		value_type *conv11_locData = NULL; value_type *conv11_confData = NULL; value_type *conv13_locData = NULL; value_type *conv13_confData = NULL;
		value_type *conv14_locData = NULL; value_type *conv14_confData = NULL; value_type *conv15_locData = NULL; value_type *conv15_confData = NULL;
		value_type *conv16_locData = NULL; value_type *conv16_confData = NULL; value_type *conv17_locData = NULL; value_type *conv17_confData = NULL;
		value_type *confData_all = NULL; value_type *locData_all = NULL;

		resize(box_total * (num_classes + 1), &confData_all);
		resize(box_total * box_code, &locData_all);

		int *box_featuremap_size = new int[6];
		box_featuremap_size[0] = 19;
		box_featuremap_size[1] = 10;
		box_featuremap_size[2] = 5;
		box_featuremap_size[3] = 3;
		box_featuremap_size[4] = 2;
		box_featuremap_size[5] = 1;

		int *anchor_num = new int[6];
		anchor_num[0] = 3;
		for (int i = 1; i < 6; i++) {
			anchor_num[i] = 6;
		}


		//value_type *inputData = NULL, *outputData = NULL;
		//value_type imgData_h[IMAGE_H*IMAGE_W];
		//float *imgData = new float[1 * IMAGE_H * IMAGE_W * sizeof(float)];
		//memset(imgData, 0, 1 * IMAGE_H * IMAGE_W * sizeof(float));
		//value_type *imgData = NULL;
		checkCudaErrors(cudaMalloc(&srcData, sizeof(value_type)*IMAGE_H*IMAGE_W * IMAGE_C));

		//checkCudaErrors(cudaMallocManaged(&inputData, IMAGE_H*IMAGE_W));
		//checkCudaErrors(cudaMallocManaged(&outputData, 14*14));


		//float* tmpImage = new float[IMAGE_H * IMAGE_W * IMAGE_C];
		//memset(tmpImage, 0, sizeof(float)*IMAGE_H * IMAGE_W * IMAGE_C);
		//checkCudaErrors(cudaMemcpy(tmpImage, srcData, sizeof(float)*IMAGE_H * IMAGE_W * IMAGE_C, cudaMemcpyDeviceToHost));
		//for (int c =2; c < IMAGE_C; c++) {
		//	for (int i = 0; i < IMAGE_H; i++) {
		//	
		//		for (int j = 0; j <IMAGE_W; j++) {
		//			cout << tmpImage[j + i *IMAGE_W + c*IMAGE_H*IMAGE_W] << "\t";
		//			//if (fabsf(tmpImage[j + i * IMAGE_W + c*IMAGE_H*IMAGE_W]) > 1.01)
		//			//{
		//			//	cout << tmpImage[j + i * IMAGE_W + c*IMAGE_H*IMAGE_W] << endl;
		//			//	cout << " normalization error ! " << endl;
		//			//	system("pause");
		//			//}
		//		} cout << endl; system("pause");
		//	}//cout << endl <<"=====================" <<endl;
		//
		//}
		//system("pause");

		//readImage(fname, imgData_h);



		std::cout << "Performing forward propagation ...\n";
	
		
		//float* tmpImage = new float[IMAGE_H * IMAGE_W * IMAGE_C];
		//memset(tmpImage, 0, sizeof(float)*IMAGE_H * IMAGE_W * IMAGE_C);
		//checkCudaErrors(cudaMemcpy(tmpImage, srcData, sizeof(float)*IMAGE_H * IMAGE_W * IMAGE_C, cudaMemcpyDeviceToHost));
		//for (int c = 0; c < 3; c++) {
		//	for (int i = 0; i < IMAGE_H; i++) {
		//		for (int j = 0; j <IMAGE_W; j++) {
		//			if (fabsf(tfImage[i * IMAGE_W * IMAGE_C + j * IMAGE_C + c] - tmpImage[c*IMAGE_H*IMAGE_W + i*IMAGE_H + j]) > 0.001) {
		//				cout << tfImage[i * IMAGE_W * IMAGE_C + j * IMAGE_C + c] << "  " << tmpImage[c*IMAGE_H*IMAGE_W + i*IMAGE_H + j] << endl;
		//			}
		//			
		//		}
		//	}
		//	cout << "----------------------------------------------------------" << endl;
		//}system("pause");

		//ofstream binfile_src(string("srcData.bin"), ios::binary);
		//for (int i = 0; i < IMAGE_C*IMAGE_H*IMAGE_W; i++) {
		//	binfile_src.write(reinterpret_cast<const char*>(&tmpImage[i]), sizeof(float));
		//}
		//cout << "Done" << endl;
		//system("pause");

		//resize(1 * IMAGE_H *IMAGE_W, &srcData);
		//checkCudaErrors(cudaMalloc(&srcData, 3 * IMAGE_H*IMAGE_W * sizeof(value_type)));
		//checkCudaErrors(cudaMemcpy(srcData, imgData, 3 * IMAGE_H*IMAGE_W * sizeof(value_type), cudaMemcpyHostToDevice));
		//delete[] imgData;
		//cout << "memory copy 끝" << endl;
		//system("pause");
		n = 1; c = IMAGE_C; h = IMAGE_H; w = IMAGE_W;
		const int tensorDims = 4;
		int tensorOuputDimA[tensorDims];
		//int num_anchors = 3;

		////-----activation pre-processing-----////
		checkCUDNN(cudnnSetActivationDescriptor(activDesc, CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_PROPAGATE_NAN, 6.0));
		checkCUDNN(cudnnSetActivationDescriptor(activBoxDesc, CUDNN_ACTIVATION_SIGMOID, CUDNN_PROPAGATE_NAN, 0.0));

		//////-----pooling pre-processing-----////

		//const int poolDims = 2;
		//int windowDimA[poolDims] = { 2,2 };
		//int paddingA[poolDims] = { 0,0 };
		//int strideA[poolDims] = { 2,2 };
		//checkCUDNN(cudnnSetPoolingNdDescriptor(poolingDesc,
		//	CUDNN_POOLING_MAX,
		//	CUDNN_PROPAGATE_NAN,
		//	poolDims,
		//	windowDimA,
		//	paddingA,
		//	strideA));
		//setTensorDesc(srcPoolTensorDesc, tensorFormat, dataType, n, c, h, w);
		////const int tensorDims = 4;
		////int tensorOuputDimA[tensorDims] = { n,c,h,w };
		//checkCUDNN(cudnnGetPoolingNdForwardOutputDim(poolingDesc,
		//	srcPoolTensorDesc,
		//	tensorDims,
		//	tensorOuputDimA));
		//n = tensorOuputDimA[0]; c = tensorOuputDimA[1];
		//h = tensorOuputDimA[2]; w = tensorOuputDimA[3];
		//setTensorDesc(dstPoolTensorDesc, tensorFormat, dataType, n, c, h, w);

		//////----softmax pre-processing-----////

		//setTensorDesc(srcSoftmaxTensorDesc, tensorFormat, dataType, 1, 10, 1, 1);
		//setTensorDesc(dstSoftmaxTensorDesc, tensorFormat, dataType, 1, 10, 1, 1);

		value_type* aspect_ratio = new value_type[num_of_featuremaps];
		value_type* aspect_ratio_layer0 = new value_type[(const int)(num_of_featuremaps / 2)];
		value_type *anchorShape0 = NULL;
		value_type *anchorShape1to5 = NULL;

		anchorBox_generator(aspect_ratio_layer0, aspect_ratio, &anchorShape0, &anchorShape1to5);

		
	
		convolutionSetting(&n, &c, &h, &w, conv0, 1, 2, tensorOuputDimA, tensorDims, &dstData);
		//convoluteForward(conv0, 1, 3, 300, 300, algo, workSpace, sizeInBytes, srcData, &dstData);
		//resize(3 * 302 * 302, &tmp);

		start = clock();
		image_resize(fname, &srcData, &original_image_h, &original_image_w, IMAGE_H, IMAGE_W);

		//float *tf_input = new float[300 * 300 * 3];
		//memset(tf_input, 0, sizeof(float)* 300 * 300 * 3);
		//readBinaryFile("test_data/preprocessed_input.bin", 300*300*3, tf_input);
		//float* tmDsc22 = new float[3 * 300 * 300];
		//memset(tmDsc22, 0, sizeof(float) * 3 * 300 * 300);
		//checkCudaErrors(cudaMemcpy(tmDsc22, srcData, sizeof(float)* 3 * 300 * 300, cudaMemcpyDeviceToHost));
		////float tmp111 = 0;
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < 3; ++k) {
		//		for (int j = 0; j < 300; ++j) {
		//			for (int i = 0; i <300; ++i) {
		//				/*if (fabsf(tmDsc22[k * 300 * 300 + j * 300 + i] - tf_input[j * 300 * 3 + i * 3 + k]) > 0.0001) {
		//					cout << tmDsc22[k * 300 * 300 + 300 * j + i] << "  " << tf_input[j * 300 * 3 + i * 3 + k] <<endl;
		//					cout << "( " << i << " , " << j << " , " << k << " )" << endl;
		//				}*/
		//				cout << tmDsc22[k * 300 * 300 + 300 * j + i] << endl;
		//				//tmp111 += tmDsc22[k * 300 * 300 + 300 * j + i];
		//				//cout << tmDsc22[k * 300 *300 + 300 * j + i] << "  ";
		//			} //cout << "avg : " << tmp111/300 << endl; tmp111 = 0; system("pause");
		//			//printf("\n");
		//		}
		//	}
		//	printf("=================\n");
		//}system("pause");



		float * filter = conv0.convData_d;
		convolution(&srcData, &tmp, &dstData, &filter, 300, 300, 3, 32, 3, 1, 2);
		//float* tmDsc22 = new float[c * h * w];
		//memset(tmDsc22, 0, sizeof(float) * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc22, dstData, sizeof(float)* c * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < n; ++l) {
		//	for (int k =10; k < 32; ++k) {
		//		for (int j = 0; j < 150; ++j) {
		//			for (int i = 0; i <150; ++i) {
		//				cout << tmDsc22[k * h * w + w * j + i] << "  ";
		//			}system("pause");
		//			printf("\n");
		//		}
		//		printf("----------------------\n"); system("pause");
		//	}
		//	printf("=================\n");
		//}system("pause");
		resize(c * h * w, &srcData);
		batchNormalization(dstData, srcData, conv0.convBnBias_d, conv0.convBnScale_d, conv0.convBnMean_d, conv0.convBnVar_d, (float)epsilon, conv0.outputs, h, w);
		//batchNorm_conv(n, c, h, w, conv0, srcTensorDesc, dstData, &srcData);
		
		resize(c * h *w, &dstData);
		activationForward(n, c, h, w, srcData, &dstData);
		
		//float* tmDsc22 = new float[c * h * w];
		//memset(tmDsc22, 0, sizeof(float) * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc22, dstData, sizeof(float)* c * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < n; ++l) {
		//	for (int k =10; k < 32; ++k) {
		//		for (int j = 0; j < 150; ++j) {
		//			for (int i = 0; i <150; ++i) {
		//				cout << tmDsc22[k * h * w + w * j + i] << "  ";
		//			}system("pause");
		//			printf("\n");
		//		}
		//		printf("----------------------\n"); system("pause");
		//	}
		//	printf("=================\n");
		//}system("pause");


		//float* test0 = new float[32 * 150 * 150];
		//const char* conv0_out = "test_data/conv0_out.bin";
		//readBinaryFile(conv0_out, 32 * 150 * 150, test0);
		//float* tmDsc22 = new float[c * h * w];
		//memset(tmDsc22, 0, sizeof(float) * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc22, dstData, sizeof(float)* c * h * w, cudaMemcpyDeviceToHost));
		//ofstream binfile_src(string("conv0_out.bin"), ios::binary);
		//for (int i = 0; i < 32*150*150; i++) {
		//	binfile_src.write(reinterpret_cast<const char*>(&tmDsc22[i]), sizeof(float));
		//}
		//cout << "Done" << endl;
		//system("pause");
		
		dsc(&n, &c, &h, &w, 1, 1, conv1, &dstData, &srcData, &tmp);
		//cout << c << " " << h << " " << w << endl; 
		//float* tmDsc222 = new float[c * h * w];
		//memset(tmDsc222, 0, sizeof(float) * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc222, dstData, sizeof(float)* c * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < n; ++l) {
		//		for (int k = 0; k < c; ++k) {
		//			for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc222[k * h * w + w * j + i] << " ";
		//			} 
		//			printf("\n"); system("pause");
		//		}
		//		printf("----------------------\n"); system("pause");
		//	}
		//	printf("=================\n");
		//}system("pause");
		////cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;

		//cout << "dsc1 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//system("pause");
		//float* tmDsc = new float[1 * c * h * w];
		////float* tmDsc3 = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*1 * c * h * w);
		////memset(tmDsc3, 0, sizeof(float)*n * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc, dstData, sizeof(float)* c * h * w, cudaMemcpyDeviceToHost));
		////checkCudaErrors(cudaMemcpy(tmDsc3, dstData, sizeof(float) * 3 * h * w, cudaMemcpyDeviceToHost));
		//cout << fixed;
		//cout.precision(2);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "\t";
		//			}
		//		} system("pause");
		//		/*for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc3[h*w*k + h * j + i] << "\t";
		//			}
		//		} system("pause");*/
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		dsc(&n, &c, &h, &w, 1, 2, conv2, &dstData, &srcData, &tmp);
		//cout << "dsc2 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//system("pause");
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;

		dsc(&n, &c, &h, &w, 1, 1, conv3, &dstData, &srcData, &tmp);
		//cout << "dsc3 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//system("pause");
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "\t";
		//			}
		//		} system("pause");
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		dsc(&n, &c, &h, &w, 1, 2, conv4, &dstData, &srcData, &tmp);
		//cout << "dsc4 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//system("pause");
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "\t";
		//			}
		//		} system("pause");
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		dsc(&n, &c, &h, &w, 1, 1, conv5, &dstData, &srcData, &tmp);
		//cout << "dsc5 후" << endl;
		////cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		////cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "  ";
		//			}cout << endl; system("pause");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		dsc(&n, &c, &h, &w, 1, 2, conv6, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv7, &dstData, &srcData, &tmp);

		dsc(&n, &c, &h, &w, 1, 1, conv8, &dstData, &srcData, &tmp);
		//cout << "dsc8 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "  ";
		//			}cout << endl; system("pause");
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		dsc(&n, &c, &h, &w, 1, 1, conv9, &dstData, &srcData, &tmp);
		//cout << "dsc9 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "  ";
		//			}cout << endl; system("pause");
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		dsc(&n, &c, &h, &w, 1, 1, conv10, &dstData, &srcData, &tmp);
		//cout << "dsc10 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//float* tmDsc = new float[n * c * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "  ";
		//			}cout << endl; system("pause");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		dsc(&n, &c, &h, &w, 1, 1, conv11, &dstData, &srcData, &tmp);
		//cout << "dsc11 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//system("pause");

		//float* tmDsc2 = new float[n * c * h * w];
		//memset(tmDsc2, 0, sizeof(float)*n *c * h * w);
		//cudaMemcpy(tmDsc2, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc2[h*w*k + h * j + i] << "  ";
		//			} cout << endl; system("pause");
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}


		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box0_loc, box0_conf, &anchorShape0,
			&dstData, &conv11_locData, &conv11_confData, &locData_all, &confData_all);
		layer_count = 0;

		//float* tmDsc2 = new float[n * box0_loc.outputs * h * w];
		//memset(tmDsc2, 0, sizeof(float)*n * box0_loc.outputs * h * w);
		//cudaMemcpy(tmDsc2, conv11_locData, sizeof(float)*n * box0_loc.outputs * h * w, cudaMemcpyDeviceToHost);
		////for (int l = 0; l < n; ++l) {
		////	for (int k = 0; k < box0_loc.outputs; ++k) {
		////		for (int j = 0; j < h; ++j) {
		////			for (int i = 0; i <w; ++i) {
		////				cout << tmDsc2[h*w*k + h * j + i] << "\t";
		////			} cout << endl;
		////		} 
		////		printf("----------------------\n");
		////	}
		////	printf("=================\n");
		////}
		////system("pause");
		//float* tmDsc5 = new float[n * box0_conf.outputs * h * w];
		//memset(tmDsc5, 0, sizeof(float)*n * box0_conf.outputs * h * w);
		//cudaMemcpy(tmDsc5, conv11_confData, sizeof(float)*n * box0_conf.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box0_conf.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				if (tmDsc5[h*w*k + h * j + i] > 0.90) {
		//					cout << tmDsc5[h*w*k + h * j + i] << " " << "( x , y , class, anchor) : " << i << " , " << j << " , " << k % (1 + num_classes) <<" , " <<  k / (1 + num_classes) << endl;
		//				}	
		//			} 
		//		}
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//draw_box(fname, tmDsc2, w, box0_loc.outputs, 2, 12, 6);
		//system("pause");
		//draw_box(fname, tmDsc2, w, box0_loc.outputs, 2, 12, 10);
		//system("pause");
		//draw_box(fname, tmDsc2, w, box0_loc.outputs, 2, 7, 6);
		//system("pause");
		//draw_box(fname, tmDsc2, w, box0_loc.outputs, 2, 7, 9);
		//system("pause");
		//draw_box(fname, tmDsc2, w, box0_loc.outputs, 2, 7, 14);
		//system("pause");



		dsc(&n, &c, &h, &w, 1, 2, conv12, &dstData, &srcData, &tmp);
		//cout << "dsc12 후" << endl;
		//cout << "srcData : " << srcData << endl << "dstData : " << dstData << endl;
		//cout << "n : " << n << "c : " << c << "h : " << h << "w : " << w << endl;
		//system("pause");
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		dsc(&n, &c, &h, &w, 1, 1, conv13, &dstData, &srcData, &tmp);

		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box1_loc, box1_conf, &anchorShape1to5,
			&dstData, &conv13_locData, &conv13_confData, &locData_all, &confData_all);
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		//float* tmDsc6 = new float[n * box1_conf.outputs * h * w];
		//memset(tmDsc6, 0, sizeof(float)*n * box1_conf.outputs * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc6, conv13_confData, sizeof(float)*n * box1_conf.outputs * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box1_conf.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				if (tmDsc6[h*w*k + h * j + i] > 0.90) {
		//					cout << tmDsc6[h*w*k + h * j + i] << " " << "( x , y , class, anchor) : " << i << " , " << j << " , " << k % (1 + num_classes) << " , " << k / (1 + num_classes) << endl;
		//				}
		//			}
		//		}
		//	}
		//}
		//system("pause");
		//float* tmDsc = new float[n * box1_loc.outputs * h * w];
		//memset(tmDsc, 0, sizeof(float)*n * box1_loc.outputs * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc, conv13_locData, sizeof(float)*n * box1_loc.outputs * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box1_loc.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc[h*w*k + h * j + i] << "\t";
		//			}cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}

		//for (int i = 0; i < 10; i++) {
		//	for (int j = 0; j < 10; j++) {
		//		cout << "( x , y ) : " << j << " , " << i << endl;
		//		draw_box(fname, tmDsc, w, box1_loc.outputs, 4, i, j);
		//		system("pause");
		//		//cout << "x: " << j << "\t y : " << i << endl;
		//	}
		//}

		//for (int i = 0; i < 6; i++) {
		//	draw_box(fname, tmDsc, w, box1_loc.outputs, i, 4, 3);
		//	system("pause");
		//}
		//for (int i = 0; i < 6; i++) {
		//	draw_box(fname, tmDsc, w, box1_loc.outputs, i, 6, 3);
		//	system("pause");
		//}
		//draw_box(fname, tmDsc, w*h, box1_loc.outputs);
		//cout << c << endl;
		//system("pause");
		//cout << "=================inputdata============" << endl;
		//float* tmDsc_ex1 = new float[n * c * h * w];
		//memset(tmDsc_ex1, 0, sizeof(float)*n * c * h * w);
		//checkCudaErrors(cudaMemcpy(tmDsc_ex1, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost));
		//for (int l = 0; l < 1; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < 1; ++j) {
		//			for (int i = 0; i <1; ++i) {
		//				cout << tmDsc_ex1[h*w*k + 1 * j + i] << "\t";
		//			}cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//cout << "===============pointData==============" << endl;
		//for (int l = 1; l < 2; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < 1; ++j) {
		//			for (int i = 0; i <1; ++i) {
		//				cout << conv14.pointData_h[1*1*c*l + 1*1*k + 1 * j + i] << "\t";
		//			}cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");
		//cout << "==========output========" << endl;
		//for (int l = 0; l < 1; ++l) {
		//	float tmp22 = 0;
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < 1; ++j) {
		//			for (int i = 0; i <1; ++i) {
		//				tmp22 += conv14.pointData_h[1 * 1 * c*(l +1) + 1 * 1 * k + 1 * j + i] * tmDsc_ex1[h * w * k + 1 * j + i];
		//			}
		//		}
		//	} cout << tmp22 << endl;
		//	printf("=================\n");
		//}
		//system("pause");

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv14, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box2_loc, box2_conf, &anchorShape1to5,
			&dstData, &conv14_locData, &conv14_confData, &locData_all, &confData_all);

		//float* tmDsc3 = new float[n * box2_loc.outputs * h * w];
		//memset(tmDsc3, 0, sizeof(float)*n * box2_loc.outputs * h * w);
		//cudaMemcpy(tmDsc3, conv14_locData, sizeof(float)*n * box2_loc.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box2_loc.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc3[h*w*k + h * j + i] << "\t";
		//			} cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}

		//int* boxList = new int[3 * 50];
		//memset(boxList, 0, sizeof(int) * 150);
		//int q = 0;
		//float* tmDsc7 = new float[n * box2_conf.outputs * h * w];
		//memset(tmDsc7, 0, sizeof(float)*n * box2_conf.outputs * h * w);
		//cudaMemcpy(tmDsc7, conv14_confData, sizeof(float)*n * box2_conf.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box2_conf.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc7[h*w*k + h * j + i] << "\t";
		//				//if (tmDsc7[h*w*k + h * j + i] > 0.3) {
		//				//	boxList[q] = k/(num_classes+1);
		//				//	boxList[q + 1] = j;
		//				//	boxList[q + 2] = i;
		//				//	cout << boxList[q] << boxList[q + 1] << boxList[q + 2] << "\t" << k%(num_classes+1) << "\t" <<tmDsc7[h*w*k + h * j + i]  << endl;
		//				//	q += 3;
		//				//	
		//				//}
		//			} cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n"); //cout << "q : " << q << endl;
		//}
		//system("pause");

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv15, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box3_loc, box3_conf, &anchorShape1to5,
			&dstData, &conv15_locData, &conv15_confData, &locData_all, &confData_all);

		//float* tmDsc8 = new float[n * box3_loc.outputs * h * w];
		//memset(tmDsc8, 0, sizeof(float)*n * box3_loc.outputs * h * w);
		//cudaMemcpy(tmDsc8, conv15_locData, sizeof(float)*n * box3_loc.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box3_loc.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc8[h*w*k + h * j + i] << "\t";
		//			} cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//float* tmDsc9 = new float[n * box3_conf.outputs * h * w];
		//memset(tmDsc9, 0, sizeof(float)*n * box3_conf.outputs * h * w);
		//cudaMemcpy(tmDsc9, conv15_confData, sizeof(float)*n * box3_conf.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box3_conf.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc9[h*w*k + h * j + i] << "\t";
		//			} cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		/*draw_box(fname, tmDsc8, w, box3_loc.outputs, 2, 0, 0);
		system("pause");*/

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv16, &dstData, &srcData, &tmp);
		
		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box4_loc, box4_conf, &anchorShape1to5,
			&dstData, &conv16_locData, &conv16_confData, &locData_all, &confData_all);

		//float* tmDsc10 = new float[n * box4_loc.outputs * h * w];
		//memset(tmDsc10, 0, sizeof(float)*n * box4_loc.outputs * h * w);
		//cudaMemcpy(tmDsc10, conv16_locData, sizeof(float)*n * box4_loc.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box4_loc.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc10[h*w*k + h * j + i] << "\t";
		//			}cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//float* tmDsc11 = new float[n * box4_conf.outputs * h * w];
		//memset(tmDsc11, 0, sizeof(float)*n * box4_conf.outputs * h * w);
		//cudaMemcpy(tmDsc11, conv16_confData, sizeof(float)*n * box4_conf.outputs * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < box4_conf.outputs; ++k) {
		//		for (int j = 0; j < h; ++j) {
		//			for (int i = 0; i <w; ++i) {
		//				cout << tmDsc11[h*w*k + h * j + i] << "\t";
		//			}cout << endl;
		//		}
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");


		//draw_box(fname, tmDsc10, w, box4_loc.outputs, 4, 0, 0);
		//system("pause");

		extraLayer(&n, &c, &h, &w, 1, 2, tensorOuputDimA, tensorDims, conv17, &dstData, &srcData, &tmp);
		//cout << " n : " << n << " c : " << c << " h : " << h << " w : " << w << endl;
		//float* tmDsc3 = new float[n * c * h * w];
		//memset(tmDsc3, 0, sizeof(float)*n * c * h * w);
		//cudaMemcpy(tmDsc3, dstData, sizeof(float)*n * c * h * w, cudaMemcpyDeviceToHost);
		//for (int l = 0; l < n; ++l) {
		//	for (int k = 0; k < c; ++k) {
		//		for (int j = 0; j < h; ++j){
		//			for (int i = 0; i <w; ++i){
		//				cout << tmDsc3[ h*w*k + h * j + i] << "\t";
		//			} cout << endl;
		//		} 
		//		printf("----------------------\n");
		//	}
		//	printf("=================\n");
		//}
		//system("pause");

		boxPredictor(&n, &c, &h, &w, original_image_h, original_image_w, anchor_num, box_featuremap_size, &box_index, &layer_count, box5_loc, box5_conf, &anchorShape1to5,
			&dstData, &conv17_locData, &conv17_confData, &locData_all, &confData_all);

		float* tmDsc12 = new float[box_total * (num_classes+1)];
		memset(tmDsc12, 0, sizeof(float)* box_total * (num_classes + 1));
		int* sorted_box_address = new int[box_total];
		memset(sorted_box_address, 0, sizeof(int)* box_total);
		int* sorted_box_class = new int[box_total];
		memset(sorted_box_class, 0, sizeof(int)* box_total);
		
		cudaMemcpy(tmDsc12, confData_all, sizeof(int)* box_total * (num_classes + 1), cudaMemcpyDeviceToHost);
		int sorted_box_index = 0;
		
		for (int i = 0; i < num_classes+1; ++i) {
			for (int j = 0; j < box_total; ++j) {
				if (tmDsc12[i * box_total + j] > 0.50) {
					sorted_box_address[sorted_box_index] = j;
					sorted_box_class[sorted_box_index] = i;
					sorted_box_index += 1;
				}
			}
		}
		float* filtered_box_score = new float[sorted_box_index];
		memset(filtered_box_score, 0, sizeof(float)* sorted_box_index);
		float* sorted_box_score = new float[sorted_box_index];
		memset(sorted_box_score, 0, sizeof(float)* sorted_box_index);
		int* sorted_score_index = new int[sorted_box_index];
		memset(sorted_score_index, 0, sizeof(int)* sorted_box_index);

		for (int i = 0; i < sorted_box_index; i++) {
			sorted_box_score[i] = tmDsc12[sorted_box_class[i] * box_total + sorted_box_address[i]];
			filtered_box_score[i] = sorted_box_score[i];
		}

		qsort(sorted_box_score, sorted_box_index, sizeof(float), descending);

		for (int i = 0; i < sorted_box_index; i++) {
			sorted_score_index[i] = find_index(sorted_box_score, sorted_box_index, filtered_box_score[i]);
			//cout << sorted_box_address[sorted_score_index[i]] << endl;
		}
		
		vector<int> address_index(sorted_box_index);
		vector<float> score_index(sorted_box_index);
		for (int i = 0; i < sorted_box_index; i++){
			score_index[i] = sorted_box_score[i];
			address_index[i] = sorted_box_address[sorted_score_index[i]];
			//cout << address_index[i] << "  " << score_index[i] << endl;
		}


		float* tmDsc13 = new float[box_total * box_code];
		memset(tmDsc13, 0, sizeof(float) * box_total * box_code);
		cudaMemcpy(tmDsc13, locData_all, sizeof(float) * box_total * box_code, cudaMemcpyDeviceToHost);
		//int box_address = 1348;
		//cout << tmDsc13[box_address] <<" , " <<tmDsc13[box_total + box_address] << " , " << tmDsc13[2 * box_total + box_address] << " , " << tmDsc13[3 * box_total + box_address] << endl;
		//system("pause");
		//draw_box(fname, tmDsc13, box_address, box_total);
		//system("pause");
		//for (int i = 0; i < box_code; ++i) {
		//	for (int j = 0; j < box_total; ++j) {
		//		if (tmDsc13[i * box_total + j] > 0.90) {
		//			cout << tmDsc13[i * box_total + j] << ",  (x , y) : " << j << " , " << i << endl;
		//		}
		//	}
		//}
		//system("pause");

		nms(tmDsc13, address_index, 0.6, box_total);

		//for (int i = 0; i < address_index.size(); i++) {
		//	cout << address_index[i] << endl;
		//}system("pause");

		//for (int i = 0; i < address_index.size(); i++) {
		//	draw_box(fname, tmDsc13, address_index[i], box_total);
		//	system("pause");
		//}
		

		//for (int i = 0; i < 6; i++)
		//{
		//	//draw_box(fname, tmDsc12, w*h, box5_loc.outputs, i);
		//	system("pause");
		//}

		

		end = clock();

		//if (sizeInBytes != 0)
		//{
		//	checkCudaErrors(cudaFree(workSpace));
		//}



		std::cout << "\nSingle computing time : " << (end - start) << std::endl;
		//system("pause");


		//cuDNN and cuBLAS library calls are asynchronous w.r.t. the host.
		// Need a device sync here before copying back the results.
		//checkCudaErrors(cudaDeviceSynchronize());

		// Take care of half precision
		Convert<scaling_type> toReal;
		/*checkCudaErrors(cudaMemcpy(result, dstData, max_digits * sizeof(value_type), cudaMemcpyDeviceToHost));
		int id = 123;
		for (int i = 0; i < max_digits; i++)
		{
			if (toReal(result[id]) < toReal(result[i])) id = i;
		}*/

		/*cout << id << endl;*/
		///*std::cout << "Resulting weights from Softmax:" << std::endl;
		//printDeviceVector(n*c*h*w, dstData);*/

		//checkCudaErrors(cudaFree(srcData));
		//checkCudaErrors(cudaFree(dstData));
		return 0;
	}
};

#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
// using 1x1 convolution to emulate gemv in half precision when cuBLAS version <= 7.0
template <>
void network_t<half1>::fullyConnectedForward(const Layer_t<half1>& ip,
	int& n, int& c, int& h, int& w,
	half1* srcData, half1** dstData)
{
	c = c*h*w; h = 1; w = 1;
	network_t<half1>::convoluteForward(ip, n, c, h, w, srcData, dstData);
	c = ip.outputs;
}
#endif

void displayUsage()
{
	printf("mnistCUDNN {<options>}\n");
	printf("help                   : display this help\n");
	printf("device=<int>           : set the device to run the sample\n");
	printf("image=<name>           : classify specific image\n");
}

int main(int argc, char *argv[])
{
	std::string image_path;
	int i1, i2;
	clock_t start, end;
	float single_start, single_end, half_start, half_end;


	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		displayUsage();
		exit(EXIT_WAIVED);
	}

	int version = (int)cudnnGetVersion();
	printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
	printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
	showDevices();

	int device = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		checkCudaErrors(cudaSetDevice(device));
	}
	std::cout << "Using device " << device << std::endl;

	if (checkCmdLineFlag(argc, (const char **)argv, "image"))
	{
		
		char* image_name;
		getCmdLineArgumentString(argc, (const char **)argv, "image", (char **)&image_name);

		network_t<float> mobilenet;
		//convi(input, output, depth_dim, point_dim)
		Layer_t<float> conv0(3, 32, 3, conv0_bin, bnScale0, bnBias0, conv0_mMean, conv0_mVar, argv[0]);
		Layer_t<float> conv1(32, 64, 3, 1, conv1_depth, conv1_point, conv1_dGamma, conv1_dBeta, conv1_pGamma, conv1_pBeta, conv1_dmMean, conv1_dmVar, conv1_pmMean, conv1_pmVar, argv[0]);
		Layer_t<float> conv2(64, 128, 3, 1, conv2_depth, conv2_point, conv2_dGamma, conv2_dBeta, conv2_pGamma, conv2_pBeta, conv2_dmMean, conv2_dmVar, conv2_pmMean, conv2_pmVar, argv[0]);
		Layer_t<float> conv3(128, 128, 3, 1, conv3_depth, conv3_point, conv3_dGamma, conv3_dBeta, conv3_pGamma, conv3_pBeta, conv3_dmMean, conv3_dmVar, conv3_pmMean, conv3_pmVar, argv[0]);
		Layer_t<float> conv4(128, 256, 3, 1, conv4_depth, conv4_point, conv4_dGamma, conv4_dBeta, conv4_pGamma, conv4_pBeta, conv4_dmMean, conv4_dmVar, conv4_pmMean, conv4_pmVar, argv[0]);
		Layer_t<float> conv5(256, 256, 3, 1, conv5_depth, conv5_point, conv5_dGamma, conv5_dBeta, conv5_pGamma, conv5_pBeta, conv5_dmMean, conv5_dmVar, conv5_pmMean, conv5_pmVar, argv[0]);
		Layer_t<float> conv6(256, 512, 3, 1, conv6_depth, conv6_point, conv6_dGamma, conv6_dBeta, conv6_pGamma, conv6_pBeta, conv6_dmMean, conv6_dmVar, conv6_pmMean, conv6_pmVar, argv[0]);
		Layer_t<float> conv7(512, 512, 3, 1, conv7_depth, conv7_point, conv7_dGamma, conv7_dBeta, conv7_pGamma, conv7_pBeta, conv7_dmMean, conv7_dmVar, conv7_pmMean, conv7_pmVar, argv[0]);
		Layer_t<float> conv8(512, 512, 3, 1, conv8_depth, conv8_point, conv8_dGamma, conv8_dBeta, conv8_pGamma, conv8_pBeta, conv8_dmMean, conv8_dmVar, conv8_pmMean, conv8_pmVar, argv[0]);
		Layer_t<float> conv9(512, 512, 3, 1, conv9_depth, conv9_point, conv9_dGamma, conv9_dBeta, conv9_pGamma, conv9_pBeta, conv9_dmMean, conv9_dmVar, conv9_pmMean, conv9_pmVar, argv[0]);
		Layer_t<float> conv10(512, 512, 3, 1, conv10_depth, conv10_point, conv10_dGamma, conv10_dBeta, conv10_pGamma, conv10_pBeta, conv10_dmMean, conv10_dmVar, conv10_pmMean, conv10_pmVar, argv[0]);
		Layer_t<float> conv11(512, 512, 3, 1, conv11_depth, conv11_point, conv11_dGamma, conv11_dBeta, conv11_pGamma, conv11_pBeta, conv11_dmMean, conv11_dmVar, conv11_pmMean, conv11_pmVar, argv[0]);
		Layer_t<float> conv12(512,1024, 3, 1, conv12_depth, conv12_point, conv12_dGamma, conv12_dBeta, conv12_pGamma, conv12_pBeta, conv12_dmMean, conv12_dmVar, conv12_pmMean, conv12_pmVar, argv[0]);
		Layer_t<float> conv13(1024, 1024, 3, 1, conv13_depth, conv13_point, conv13_dGamma, conv13_dBeta, conv13_pGamma, conv13_pBeta, conv13_dmMean, conv13_dmVar, conv13_pmMean, conv13_pmVar, argv[0]);
		Layer_t<float> conv14(1024, 256, 512, 3, 1, conv14_w, conv14_point, conv14_wGamma, conv14_wBeta, conv14_pGamma, conv14_pBeta, conv14_wmMean, conv14_wmVar, conv14_pmMean, conv14_pmVar, argv[0]);
		Layer_t<float> conv15(512, 128, 256, 3, 1, conv15_w, conv15_point, conv15_wGamma, conv15_wBeta, conv15_pGamma, conv15_pBeta, conv15_wmMean, conv15_wmVar, conv15_pmMean, conv15_pmVar, argv[0]);
		Layer_t<float> conv16(256, 128, 256, 3, 1, conv16_w, conv16_point, conv16_wGamma, conv16_wBeta, conv16_pGamma, conv16_pBeta, conv16_wmMean, conv16_wmVar, conv16_pmMean, conv16_pmVar, argv[0]);
		Layer_t<float> conv17(256, 64, 128, 3, 1, conv17_w, conv17_point, conv17_wGamma, conv17_wBeta, conv17_pGamma, conv17_pBeta, conv17_wmMean, conv17_wmVar, conv17_pmMean, conv17_pmVar, argv[0]);
		Layer_t<float> box0_loc(512, 12, 1, box0_loc_w, box0_loc_b, argv[0]);
		Layer_t<float> box0_conf(512, 27, 1, box0_conf_w, box0_conf_b, argv[0]);
		Layer_t<float> box1_loc(1024, 24, 1, box1_loc_w, box1_loc_b, argv[0]);
		Layer_t<float> box1_conf(1024, 54, 1, box1_conf_w, box1_conf_b, argv[0]);
		Layer_t<float> box2_loc(512, 24, 1, box2_loc_w, box2_loc_b, argv[0]);
		Layer_t<float> box2_conf(512, 54, 1, box2_conf_w, box2_conf_b, argv[0]);
		Layer_t<float> box3_loc(256, 24, 1, box3_loc_w, box3_loc_b, argv[0]);
		Layer_t<float> box3_conf(256, 54, 1, box3_conf_w, box3_conf_b, argv[0]);
		Layer_t<float> box4_loc(256, 24, 1, box4_loc_w, box4_loc_b, argv[0]);
		Layer_t<float> box4_conf(256, 54, 1, box4_conf_w, box4_conf_b, argv[0]);
		Layer_t<float> box5_loc(128, 24, 1, box5_loc_w, box5_loc_b, argv[0]);
		Layer_t<float> box5_conf(128, 54, 1, box5_conf_w, box5_conf_b, argv[0]);
		//Layer_t<float> dscConv1(32, 64, 3, 1, depthConv1_bin, pointConv1_bin, depthConv1_bias_bin, pointConv1_bias_bin, bnScale1, bnBias1, argv[0]);
		
		i1 = mobilenet.classify_example(image_name, conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, 
			conv12, conv13, conv14, conv15, conv16, conv17, box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf); 		//conv0, dscConv1
		std::cout << "\nResult of classification: " << i1 << std::endl;

		cudaDeviceReset();

		exit(EXIT_SUCCESS);
	}

	// default behaviour
	if (argc == 1 || (argc == 2) && checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		// check available memory
		struct cudaDeviceProp prop;
		checkCudaErrors(cudaGetDeviceProperties(&prop, device));
		double globalMem = prop.totalGlobalMem / double(1024 * 1024);
		bool low_memory = false;
		if (globalMem < 1536)
		{
			// takes care of 1x1 convolution workaround for fully connected layers
			// when CUDNN_CONVOLUTION_FWD_ALGO_FFT is used
#if !defined(CUDA_VERSION) || (CUDA_VERSION <= 7000)
			low_memory = true;
#endif
		}
		{
			
			std::cout << "\nTesting single precision\n";
			network_t<float> mobilenet;
			Layer_t<float> conv0(3, 32, 3, conv0_bin, bnScale0, bnBias0, conv0_mMean, conv0_mVar, argv[0]); 
			Layer_t<float> conv1(32, 64, 3, 1, conv1_depth, conv1_point, conv1_dGamma, conv1_dBeta, conv1_pGamma, conv1_pBeta, conv1_dmMean, conv1_dmVar, conv1_pmMean, conv1_pmVar, argv[0]);
			Layer_t<float> conv2(64, 128, 3, 1, conv2_depth, conv2_point, conv2_dGamma, conv2_dBeta, conv2_pGamma, conv2_pBeta, conv2_dmMean, conv2_dmVar, conv2_pmMean, conv2_pmVar, argv[0]);
			Layer_t<float> conv3(128, 128, 3, 1, conv3_depth, conv3_point, conv3_dGamma, conv3_dBeta, conv3_pGamma, conv3_pBeta, conv3_dmMean, conv3_dmVar, conv3_pmMean, conv3_pmVar, argv[0]);
			Layer_t<float> conv4(128, 256, 3, 1, conv4_depth, conv4_point, conv4_dGamma, conv4_dBeta, conv4_pGamma, conv4_pBeta, conv4_dmMean, conv4_dmVar, conv4_pmMean, conv4_pmVar, argv[0]);
			Layer_t<float> conv5(256, 256, 3, 1, conv5_depth, conv5_point, conv5_dGamma, conv5_dBeta, conv5_pGamma, conv5_pBeta, conv5_dmMean, conv5_dmVar, conv5_pmMean, conv5_pmVar, argv[0]);
			Layer_t<float> conv6(256, 512, 3, 1, conv6_depth, conv6_point, conv6_dGamma, conv6_dBeta, conv6_pGamma, conv6_pBeta, conv6_dmMean, conv6_dmVar, conv6_pmMean, conv6_pmVar, argv[0]);
			Layer_t<float> conv7(512, 512, 3, 1, conv7_depth, conv7_point, conv7_dGamma, conv7_dBeta, conv7_pGamma, conv7_pBeta, conv7_dmMean, conv7_dmVar, conv7_pmMean, conv7_pmVar, argv[0]);
			Layer_t<float> conv8(512, 512, 3, 1, conv8_depth, conv8_point, conv8_dGamma, conv8_dBeta, conv8_pGamma, conv8_pBeta, conv8_dmMean, conv8_dmVar, conv8_pmMean, conv8_pmVar, argv[0]);
			Layer_t<float> conv9(512, 512, 3, 1, conv9_depth, conv9_point, conv9_dGamma, conv9_dBeta, conv9_pGamma, conv9_pBeta, conv9_dmMean, conv9_dmVar, conv9_pmMean, conv9_pmVar, argv[0]);
			Layer_t<float> conv10(512, 512, 3, 1, conv10_depth, conv10_point, conv10_dGamma, conv10_dBeta, conv10_pGamma, conv10_pBeta, conv10_dmMean, conv10_dmVar, conv10_pmMean, conv10_pmVar, argv[0]);
			Layer_t<float> conv11(512, 512, 3, 1, conv11_depth, conv11_point, conv11_dGamma, conv11_dBeta, conv11_pGamma, conv11_pBeta, conv11_dmMean, conv11_dmVar, conv11_pmMean, conv11_pmVar, argv[0]);
			Layer_t<float> conv12(512, 1024, 3, 1, conv12_depth, conv12_point, conv12_dGamma, conv12_dBeta, conv12_pGamma, conv12_pBeta, conv12_dmMean, conv12_dmVar, conv12_pmMean, conv12_pmVar, argv[0]);
			Layer_t<float> conv13(1024, 1024, 3, 1, conv13_depth, conv13_point, conv13_dGamma, conv13_dBeta, conv13_pGamma, conv13_pBeta, conv13_dmMean, conv13_dmVar, conv13_pmMean, conv13_pmVar, argv[0]);
			Layer_t<float> conv14(1024, 256, 512, 3, 1, conv14_w, conv14_point, conv14_wGamma, conv14_wBeta, conv14_pGamma, conv14_pBeta, conv14_wmMean, conv14_wmVar, conv14_pmMean, conv14_pmVar, argv[0]);
			Layer_t<float> conv15(512, 128, 256, 3, 1, conv15_w, conv15_point, conv15_wGamma, conv15_wBeta, conv15_pGamma, conv15_pBeta, conv15_wmMean, conv15_wmVar, conv15_pmMean, conv15_pmVar, argv[0]);
			Layer_t<float> conv16(256, 128, 256, 3, 1, conv16_w, conv16_point, conv16_wGamma, conv16_wBeta, conv16_pGamma, conv16_pBeta, conv16_wmMean, conv16_wmVar, conv16_pmMean, conv16_pmVar, argv[0]);
			Layer_t<float> conv17(256, 64, 128, 3, 1, conv17_w, conv17_point, conv17_wGamma, conv17_wBeta, conv17_pGamma, conv17_pBeta, conv17_wmMean, conv17_wmVar, conv17_pmMean, conv17_pmVar, argv[0]);
			Layer_t<float> box0_loc(512, 12, 1, box0_loc_w, box0_loc_b, argv[0]);
			Layer_t<float> box0_conf(512, (num_classes + 1)*3, 1, box0_conf_w, box0_conf_b, argv[0]);
			Layer_t<float> box1_loc(1024, 24, 1, box1_loc_w, box1_loc_b, argv[0]);
			Layer_t<float> box1_conf(1024, (num_classes + 1) * 6, 1, box1_conf_w, box1_conf_b, argv[0]);
			Layer_t<float> box2_loc(512, 24, 1, box2_loc_w, box2_loc_b, argv[0]);
			Layer_t<float> box2_conf(512, (num_classes + 1) * 6, 1, box2_conf_w, box2_conf_b, argv[0]);
			Layer_t<float> box3_loc(256, 24, 1, box3_loc_w, box3_loc_b, argv[0]);
			Layer_t<float> box3_conf(256, (num_classes + 1) * 6, 1, box3_conf_w, box3_conf_b, argv[0]);
			Layer_t<float> box4_loc(256, 24, 1, box4_loc_w, box4_loc_b, argv[0]);
			Layer_t<float> box4_conf(256, (num_classes + 1) * 6, 1, box4_conf_w, box4_conf_b, argv[0]);
			Layer_t<float> box5_loc(128, 24, 1, box5_loc_w, box5_loc_b, argv[0]);
			Layer_t<float> box5_conf(128, (num_classes + 1), 1, box5_conf_w, box5_conf_b, argv[0]);
			get_path(image_path, second_image, argv[0]);
			for (int i = 0; i < 10000; i++)
			{
				i2 = mobilenet.classify_example(image_path.c_str(), conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10, conv11, conv12,
					conv13, conv14, conv15, conv16, conv17, box0_loc, box0_conf, box1_loc, box1_conf, box2_loc, box2_conf, box3_loc, box3_conf, box4_loc, box4_conf, box5_loc, box5_conf);
			}
			//conv0, dscConv1, 

			//Layer_t<float> conv0(3, 32, 3, conv0_bin, conv0_bias_bin, bnScale0, bnBias0, argv[0]);
			//Layer_t<float> dscConv1(32, 64, 3, 1, depthConv1_bin, pointConv1_bin, depthConv1_bias_bin, pointConv1_bias_bin, bnScale1, bnBias1, argv[0]);
			//get_path(image_path, first_image, argv[0]);
			//i1 = mnist.classify_example(image_path.c_str(), conv1, ip1);


			//get_path(image_path, third_image, argv[0]);
			// New feature in cuDNN v3: FFT for convolution
			//mnist.setConvolutionAlgorithm(CUDNN_CONVOLUTION_FWD_ALGO_FFT);
			//i3 = mnist.classify_example(image_path.c_str(), conv1, ip1);

			
			std::cout << "\nResult of classification: " << i2 << std::endl;

			std::cout << "\n Test finished" << std::endl;

			//system("pause");
		}

		//{
		//	half_start = clock();
		//	std::cout << "\nTesting half precision (math in single precision)\n";
		//	network_t<half1> mnist;
		//	// Conversion of input weights to half precision is done
		//	// on host using tools from fp16_emu.cpp
		//	Layer_t<half1> conv1(1, 1, 2, conv1_bin, conv1_bias_bin, argv[0], FP16_HOST);

		//	// Conversion of input weights to half precision is done
		//	// on device using cudnnTransformTensor
		//	Layer_t<half1>   ip1(1, 10, 7, ip1_bin, ip1_bias_bin, argv[0], FP16_CUDNN);
		//	// Conversion of input weights to half precision is done
		//	// on device using CUDA kernel from fp16_dev.cu

		//	get_path(image_path, first_image, argv[0]);
		//	i1 = mnist.classify_example(image_path.c_str(), conv1, ip1);

		//	get_path(image_path, second_image, argv[0]);
		//	i2 = mnist.classify_example(image_path.c_str(), conv1, ip1);

		//	get_path(image_path, third_image, argv[0]);
		//	// New feature in cuDNN v3: FFT for convolution
		//	i3 = mnist.classify_example(image_path.c_str(), conv1, ip1);

		//	std::cout << "\nResult of classification: " << i1 << " " << i2 << " " << i3 << std::endl;
		//	if (i1 != 1 || i2 != 3 || i3 != 5)
		//	{
		//		std::cout << "\nTest failed!\n";
		//		system("pause");
		//		FatalError("Prediction mismatch");
		//		
		//	}
		//	else
		//	{
		//		half_end = clock();
		//		std::cout << "\nHalf computing time: " << (half_end - half_start) / CLOCKS_PER_SEC << std::endl;
		//		std::cout << "\nTest passed!\n";
		//	}
		//}
		//cudaDeviceReset();

		exit(EXIT_SUCCESS);
	}

	displayUsage();
	cudaDeviceReset();

	exit(EXIT_WAIVED);
}
