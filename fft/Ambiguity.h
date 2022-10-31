 

#ifndef _DLL_H_
#define _DLL_H_
#pragma once

#pragma warning(disable:4273) //This is because of different functions in header and Ambiguity.cu, so not important
#pragma warning(disable:4267) //It is not important because CUDA functions are size_t - there is no conversion.  Any way these values are much smaller than unsighed long

#if COMPILING _DLL_H_ //BUILDING_DLL
# define DLLIMPORT __declspec (dllexport)
#else /* Not BUILDING_DLL */
# define DLLIMPORT __declspec (dllimport)
#endif /* Not BUILDING_DLL */
#include "cufft.h"

#define MAX_DEVICE_NAME 256

extern "C"
{
	int  DLLIMPORT Initialize(unsigned int BufferSize, unsigned int col, unsigned int row, float doopler_shift, short* name);
	int  DLLIMPORT Run(float* Data_In0, float* Data_In1, float* Data_Out, float amplification, float doopler_zoom, int shift, bool mode, short scale_type, bool remove_symetrics);
	int  DLLIMPORT Release();
}



int Calculate(  int shift, short scale_type, int index);
 
int FFT_forward();
int FFT_bacward();

int CalcCorelateShift(size_t rotation_shift);
int CalcShift(size_t rotation_shift);
int CopyShift(size_t rotation_shift);
int Corelate();
int Magnitude(int shift, short scale_Type, int Col_index);
int StreamSynchronise();
int Synchronise();
float FindMin(float x, float y);

__global__ void CorelateShiftCUDA(cufftComplex* BufX, cufftComplex* BufY, cufftComplex* BufZ, size_t shift, size_t numElements);
__global__ void ShiftCUDA(cufftComplex* In, cufftComplex* Out, size_t shift, size_t numElements);
__global__ void CopyShiftCUDA(cufftComplex* Buf0, cufftComplex* BufX, cufftComplex* BufY, size_t shift, size_t numElements);
__global__ void CopyCUDA(cufftComplex* Inp1, cufftComplex* Inp2, size_t numElements);
__global__ void CorelateCUDA(cufftComplex* InpX, cufftComplex* InpY, cufftComplex* Out, size_t numElements);
__global__ void MagnitudeCUDA(cufftComplex* Inp, float* Out, int cuda_row, int col_index, int cuda_shift, short scale_type);


#endif /* _DLL_H_ */


//Errors
#define CUDA_OK 0
#define CUDA_RUNNING -1
#define CUDA_MEMORY_COPY_ERROR -10
#define CUDA_FFT_ERROR -11
#define CUDA_STREAM_CREATE_ERROR -12
#define CUDA_STREAM_SET_ERROR -13
#define CUDA_FFT_PLAN1D_CREATE_ERROR -14
#define CUDA_FFT_CREATE_ERROR -15
#define CUDA_FFT_EXECUTE_ERROR -16


//ERRORS additionally decorated with ERRORS from drivers_types.h in form of: -XXXX-cudaError
#define CUDA_DEVICE_SYNCHRONISATION_ERROR -1000
#define CUDA_SHIFT_CALCULATION_ERROR -2000
#define CUDA_SHIFT_CORELATE_ERROR -3000
#define CUDA_MAGNITUDE_ERROR -4000
#define CUDA_CORELATE_ERROR -5000
#define CUDA_SHIFT_ERROR -6000
#define CUDA_DEVICE_RESET_ERROR -7000
#define CUDA_STREAM_DESTROY_ERROR -8000
#define CUDA_CUFFT_DESTROY_ERROR -9000
#define CUDA_FREE_ERROR -10000
#define CUDA_MALLOC_ERROR -11000
#define CUDA_SET_DEVICE_ERROR -12000
#define CUDA_GET_DEVICE_ERROR -13000
#define CUDA_STREAM_SYNCHRONISATION_ERROR -14000



