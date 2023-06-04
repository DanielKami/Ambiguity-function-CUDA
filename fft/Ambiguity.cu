// Ambiguity.cpp : Defines the exported functions for the DLL .
// Created by Daniel M. Kaminski 
// Year 2021/2022
// 
// Input: data are float strings from RTL one or two dongles
// Output: Data in map format (x,y) containing Doppler shift X and time delay Y

#define CALCULATE_SQRT_ON_CPU  //it is slightly faster on i5 11400

#include "stdafx.h"
#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ambiguity.h"


#define DEFOULT_CUDA_DEVICE 0
#define BATCH 1
#define RANK 1
#define MAX_SHIFT 400
#define CONSTANT_AMPLIFICATION 200000.0f

//globals
bool openCL_Initiated = false;

//cuda pointers We have a lot of space 6GB
cufftComplex* Cuda_buf0 = nullptr;
cufftComplex* Cuda_bufX = nullptr;
cufftComplex* Cuda_bufY = nullptr;
cufftComplex* Cuda_bufZ = nullptr;
cufftComplex* Cuda_bufW = nullptr;
float* Cuda_ColRow = nullptr;

//Machine pointers to be reduced with time
cufftComplex* dat_inp0 = nullptr;
cufftComplex* dat_inp1 = nullptr;
float* MachinColRow = nullptr;

size_t threadsPerBlock = 1024; //max on rtx3060
size_t blocksPerGrid;
cudaStream_t stream = NULL;
cufftHandle plan;

size_t Nth;
size_t N_corrected;

int Col, Row;
int ColRow;
float Doppler_zoom;
float Doppler_zoom_Col;

size_t sizeR;
size_t sizeN;



int  Initialize(unsigned int BufferSize, unsigned int col, unsigned int row, float doopler_shift, short* name)
{
	cudaError_t cudaStatus;

	//IT mast be divided by 2 because the main program
	Nth = static_cast<size_t>(BufferSize) / 2;   // the size is smaller than 32 bits, so it is ok

	blocksPerGrid = (Nth + threadsPerBlock - 1) / threadsPerBlock;


	Col = col;
	Row = row;
	ColRow = Col * Row;
	N_corrected = sizeof(cufftComplex) * Nth * BATCH;

	//machine (Pc) pointers
	dat_inp0 = new cufftComplex[Nth];
	dat_inp1 = new cufftComplex[Nth];
	MachinColRow = new float[ColRow];

	int count = 0;
	cudaGetDeviceCount(&count);


	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, DEFOULT_CUDA_DEVICE);
	if (cudaStatus != cudaSuccess)
	{
		return CUDA_GET_DEVICE_ERROR;
	}

	//Copy the name to short pointer. With char doesn't work. (Ugly way)
	for (int i = 0; i < MAX_DEVICE_NAME; i++)
		name[i] = prop.name[i];

	// Choose which GPU to run on, change this on a multi-GPU system.
	//0 is default
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		return CUDA_SET_DEVICE_ERROR - cudaStatus;
	}

	if (cufftCreate(&plan) != CUFFT_SUCCESS) {
		return CUDA_FFT_CREATE_ERROR;
	}

	if (cufftPlan1d(&plan, (int)Nth, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		return CUDA_FFT_PLAN1D_CREATE_ERROR;
	}

	 

	//*********************************************************************
	//Memory alocate
	int su = 0;
	cudaMalloc(reinterpret_cast<void**>(&Cuda_buf0), N_corrected);
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufX), N_corrected);
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufY), N_corrected);
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufZ), N_corrected);
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufW), N_corrected);
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_ColRow), sizeof(float) * (ColRow + MAX_SHIFT));
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_MALLOC_ERROR - su;
	}

	if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != CUFFT_SUCCESS) {
		return CUDA_STREAM_CREATE_ERROR;
	}

	if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) {
		return CUDA_STREAM_SET_ERROR;
	}

 
	openCL_Initiated = true;
	return CUDA_OK;
}



//Data_In is in a format Data_In[2*n] - Real values, Data_In[2*n+1] - Imaginary values
int  Run(int* Data_In0, int* Data_In1, float* Data_Out, float amplification, float doppler_zoom, int time_shift, bool mode, short scale_type, bool remove_symetrics)
{

	int err;
	float maxval = 1E-6f;
	float rev;

	if (!openCL_Initiated) 
		return CUDA_RUNNING;

	if (time_shift > MAX_SHIFT) 
		time_shift = MAX_SHIFT;

	
	int HalfCol = Col / 2;
	int shift_cor = time_shift * 2; 
	Doppler_zoom = 1.0f * Nth / doppler_zoom;
	Doppler_zoom_Col = Doppler_zoom / Col;

	 
	//Convert float2 to cufftComplex (a bit annoying)

	if (mode)
	{
#pragma omp parallel for
		for (size_t n = 0; n < Nth; n++)
		{
			size_t j = 2 * n; //For parallel calculations is safer like that
			size_t k = j + 1;

			dat_inp0[n].x = Data_In0[j];
			dat_inp0[n].y = Data_In0[k];

			dat_inp1[n].x = Data_In1[j];
			dat_inp1[n].y = Data_In1[k];
		}
	}
	else
	{
#pragma omp parallel for
		for (size_t n = 0; n < Nth; n++)
		{
			size_t j = 2 * n;
			dat_inp0[n].x = Data_In0[j];
			dat_inp0[n].y = Data_In0[j + 1];
		}
	}


	//First copy reference data to  Cuda_buf0 - reference must be shifted half but this will be later
	if (cudaMemcpyAsync(Cuda_buf0, dat_inp0, N_corrected, cudaMemcpyHostToDevice, stream) != CUFFT_SUCCESS)
	{
		return CUDA_MEMORY_COPY_ERROR;
	}
	err = StreamSynchronise();
	if (err < CUDA_OK)
		return err;

	//Fourier Transform of reference data (shift 0 deg)
	//Input: Cuda_buf0, Outout: Cuda_buf0
	if (cufftExecC2C(plan, Cuda_buf0, Cuda_buf0, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		return CUDA_FFT_ERROR;
	}
	err = Synchronise();
	if (err < CUDA_OK)
		return err;


	//If two dongles, copy the data from second dongle data datZ
	if (mode)
	{
		//Copy data from second dongle dat_inp1
		if (cudaMemcpyAsync(Cuda_bufX, dat_inp1, N_corrected, cudaMemcpyHostToDevice, stream) != CUFFT_SUCCESS)
		{
			return CUDA_MEMORY_COPY_ERROR;
		}
		err = StreamSynchronise();
		if (err < CUDA_OK)
			return err;

		//And do the fft transform in the Cuda_bufX buffer	
		if (cufftExecC2C(plan, Cuda_bufX, Cuda_bufX, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			return CUDA_FFT_ERROR;
		}
		err = Synchronise();
		if (err < CUDA_OK)
			return err;

		//Rotate shift Cuda_buf0  after fft  to the Col/2 (screen midle) output Cuda_bufY
		err = CalcShift((size_t)(Doppler_zoom / 2));
		if (err < CUDA_OK)
			return err;
	}
	else
	{
		//Just copy the fft transformed Cuda_buf0 to the Cuda_bufX and Cuda_bufY(half rotated) it is the same
		err = CopyShift((size_t)(Doppler_zoom / 2));
		if (err < CUDA_OK)
			return err;

	}

	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
    //     This section
	// Input:
	//     Cuda_bufY - fft reference data shifted to half from Cuda_buf0
	//     Cuda_bufX - fft basic  date not shifted  
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Calculate ambiguity
	// 
	//The hardest part of calculations (shift every column) Cuda_bufX in steps of N/Col/rotation_zoom
	for (int n = 0; n < Col; n++)
	{
		err = Calculate(shift_cor, scale_type, n);
		if (err < CUDA_OK)
			return err;
	}
	
	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	
	//Copy from device to machine all the results in one shut (it saves a lot of time) Data_Out contains all shifted lines (the rough image)
	if (cudaMemcpyAsync(Data_Out, Cuda_ColRow, sizeof(float) * (ColRow + time_shift), cudaMemcpyDeviceToHost, stream) != CUFFT_SUCCESS)
	{
		return CUDA_MEMORY_COPY_ERROR;
	}

	//not necesarry
	//err = StreamSynchronise();
	//if (err < CUDA_OK)
	//	return err;
	

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//                                    Signal map postprocesing
	// 
	// 
	// Sometimes  MachinColRow is negative and Radar crashes (solved)
	////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef CALCULATE_SQRT_ON_CPU
#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		Data_Out[n] = sqrtf(Data_Out[n]);

		if (Data_Out[n] > maxval)
		{
			maxval = Data_Out[n];
		}
	}

	if (scale_type == 1)
	{
#pragma omp parallel for
		for (int n = 0; n < ColRow; n++)
		{
			Data_Out[n] = sqrtf(Data_Out[n]);
		}
		maxval = sqrtf(maxval);
	}

	if (scale_type == 2)
	{
#pragma omp parallel for
		for (int n = 0; n < ColRow; n++)
		{
			Data_Out[n] = log2f(Data_Out[n]);
		}
		maxval = log2f(maxval);
	}

#else
    //////   Normalise to max    ///////////////////////////////////////////////////////////////////////////////
#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		if (Data_Out[n] > maxval)
		{
			maxval = Data_Out[n];
		}
	}
#endif

		float TotalMAX = 0;
	maxval = 1.0f / maxval;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////


	/////// Auto signal leveling ///////////////////////////////////////////////////////////////////////////////
#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		TotalMAX += (Data_Out[n] *= maxval);		
	}

	rev = 1;
	if (scale_type == 0)	rev = amplification;
	if (scale_type == 1)	rev = sqrtf(amplification);
	if (scale_type == 2)	rev = log2f(amplification);

	if (TotalMAX <= 10) TotalMAX = 10;
	     TotalMAX = CONSTANT_AMPLIFICATION / TotalMAX * rev;

#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		Data_Out[n] *= TotalMAX;
	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////////
	

	
	///////  Remove symetrics  /////////////////////////////////////////////////////////////////////////////////
	int c1, c2;
	int tmp1, tmp2;
	float min;

	if (remove_symetrics)
	{
#pragma omp parallel for
		for (int C = 0 + 1; C < HalfCol + 1; C++)  
		{
			c1 = (HalfCol + C) * Row;
			c2 = (HalfCol - C) * Row;

			for (int R = 0; R < Row; R++)
			{
				tmp1 = c1 + R;
				tmp2 = c2 + R;
				min = FindMin(Data_Out[tmp1], Data_Out[tmp2]);
				Data_Out[tmp1] -= min;
				Data_Out[tmp2] -= min;
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////


	// Done return corrected image
	return CUDA_OK;
}

 
inline float FindMin(float x, float y)
{
	if (x < y) 
		return (x);

	return (y);
}

//myfft1 - output
//datZ output
int Calculate(int shift, short scale_type, int index)
{
	int err;

	//Input Cuda_bufX, Cuda_bufY, rotation_shift
	//Output: Cuda_bufZ
	err = CalcCorelateShift((size_t)(Doppler_zoom_Col * index));
	if (err < CUDA_OK)
		return err;

	//Painful part
	//Input: Cuda_bufZ
	//Output: Cuda_bufZ
	err = FFT_bacward();
	if (err < CUDA_OK)
		return err;

	//Input: Cuda_bufZ
	//Output: Cuda_bufZ
	err = Magnitude(shift, scale_type, index);
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}

int StreamSynchronise()
{
	int su;
	if ((su = cudaStreamSynchronize(stream)) != cudaSuccess) 
	{
		return CUDA_STREAM_SYNCHRONISATION_ERROR - su;
	}
	return CUDA_OK;
}

int Synchronise()
{
	int su;
	if ((su = cudaDeviceSynchronize()) != cudaSuccess) 
	{
		return CUDA_DEVICE_SYNCHRONISATION_ERROR - su;
	}
	return CUDA_OK;
}
 

int FFT_forward()
{
	//Forward fft
	if (cufftExecC2C(plan, Cuda_bufZ, Cuda_bufZ, CUFFT_FORWARD) != CUFFT_SUCCESS) 
	{
		return CUDA_FFT_EXECUTE_ERROR;
	}

	int err = Synchronise();
	if (err < CUDA_OK) 
		return err;
		
	return CUDA_OK;
}

//In: Cuda_bufZ
//Out: Cuda_bufW - small speed improvement
int FFT_bacward()
{
	//Backward fft
	if (cufftExecC2C(plan, Cuda_bufZ, Cuda_bufW, CUFFT_INVERSE) != CUFFT_SUCCESS) 
	{
		return CUDA_FFT_EXECUTE_ERROR;
	}

	//Synchronisation is crucial
	int err = Synchronise();
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}


int  Release()
{
	cudaError_t cudaStatus;

	if (!openCL_Initiated) 
		return CUDA_OK; //If not initiallised return for safety

	/* Release OpenCL memory objects. */
	

	//***************************************************
	cudaStatus = cudaFree(Cuda_buf0);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}

	cudaStatus = cudaFree(Cuda_bufX);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}
	cudaStatus = cudaFree(Cuda_bufY);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}

	cudaStatus = cudaFree(Cuda_bufZ);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}

	cudaStatus = cudaFree(Cuda_bufW);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}
	cudaStatus = cudaFree(Cuda_ColRow);
	if (cudaStatus != cudaSuccess) {
		return CUDA_FREE_ERROR - cudaStatus;
	}

	int su;
	if ((su = cufftDestroy(plan) ) != cudaSuccess) {
		//fprintf(stderr, "cufftDestroy failed!");
		return CUDA_CUFFT_DESTROY_ERROR - su;
	}

	cudaStatus = cudaStreamDestroy(stream);
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaStreamDestroy failed!");
		return CUDA_STREAM_DESTROY_ERROR - cudaStatus;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		//fprintf(stderr, "cudaDeviceReset failed!");
		return CUDA_DEVICE_RESET_ERROR - cudaStatus;
	}


	//machine pointers
	delete[] dat_inp0;
	delete[] dat_inp1;
	delete[] MachinColRow;

	return    CUDA_OK;
}



///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																											     //
//																									             //
//																												 //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

 
//Input: (Rotation table) Cuda_bufBeta,  Cuda_buf0 
//Output: Cuda_bufX, Cuda_bufY
int CopyShift(size_t rotation_shift)
{

	CopyShiftCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufX, Cuda_bufY, rotation_shift, Nth);

	int su;
	if ((su = cudaGetLastError() )!= cudaSuccess) {
		return CUDA_SHIFT_ERROR - su;
	}

	int err = Synchronise();
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}

//Input: Cuda_buf0 (reference FFT input data), Cuda_bufX (rotation_shifted data)
//Output: Cuda_bufZ
int Corelate()
{
	CorelateCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufY, Cuda_bufZ, Nth);

	int su;
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_SHIFT_ERROR - su;
	}

	int err = Synchronise();
	if (err < CUDA_OK)	
		return err;

	return CUDA_OK;
}


//Possible acceleration
int Magnitude(int shift, short scale_Type, int Col_index)
{
	//Row_corrected = sizeof(cufftComplex) * Row * BATCH;
	//shift_corrected = Row_corrected + sizeof(cufftComplex) * Row * BATCH;
	MagnitudeCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_bufW, Cuda_ColRow, Row, Col_index * Row, shift, scale_Type );//

	int su;
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_SHIFT_ERROR - su;
	}

	//This can be omited because it has no efect on the rotation shift
	//int err = Synchronise();
	//if (err < CUDA_OK) return err;
	//return CUDA_OK;
}


//If calc normal than do normal calculations othervise calculate shift 1/2N for Cuda_buf0
//Input: Cuda_buf0
//Output: Cuda_bufY
int CalcShift(size_t rotation_shift )
{

	ShiftCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufY, rotation_shift, Nth);

	int su;
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_SHIFT_ERROR - su;
	}

	int err = Synchronise();
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}


///NODEFAULTLIB:LIBCMT
//If calc normal than do normal calculations othervise calculate shift 1/2N for Cuda_buf0
//False shift to the half of all columns
//true normal mode shift every column
int CalcCorelateShift(size_t rotation_shift)
{
	CorelateShiftCUDA << <blocksPerGrid, threadsPerBlock >> > ( Cuda_bufX, Cuda_bufY,Cuda_bufZ, rotation_shift, Nth);

	int su;
	if ((su = cudaGetLastError()) != cudaSuccess) {
		return CUDA_SHIFT_ERROR - su;
	}

	int err = Synchronise();
	if (err < CUDA_OK)
		return err;

	return CUDA_OK;
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//																												//
//                                                   CUDA                                                       //
// Functions implemented on device (NVIDIA)                                                                     //                                          //
//                                                                                                              //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * Result goes to  Cuda_bufZ
 *  BufX(Cuda_bufX) is not changed - reference
 *  BufY(Cuda_bufY) is not change it is a reference
 *  BufZ(Cuda_bufZ) output corelated results after BufX rotation
 *  shift phase shift which coresponds to a doppler effect
 *  Out(Buf) result output
 *
 */
__global__ void CorelateShiftCUDA(cufftComplex* BufX, cufftComplex* BufY, cufftComplex* BufZ, size_t shift, size_t numElements) {

	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = i + shift;

	//Just rotate from end to the beginning and convolute
	if (j < numElements) //protection
	{
		BufZ[i].x = BufY[i].x * BufX[j].x + BufY[i].y * BufX[j].y;
		BufZ[i].y =BufY[i].y * BufX[j].x - BufY[i].x * BufX[j].y;
	}
	else
	{
	//	if (i < numElements) //protection
		{
			size_t l = j - numElements;
			BufZ[i].x = BufY[i].x * BufX[l].x + BufY[i].y * BufX[l].y;
			BufZ[i].y = BufY[i].y * BufX[l].x - BufY[i].x * BufX[l].y;
		}
	}
}


/**
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * Computes shift
 * number of elements numElements.
 * Forward shift so Mat[i] = Mat[i+shift] so we shift left more easy calculations without copy and extra temporary buffer
 */
__global__ void ShiftCUDA(cufftComplex* In, cufftComplex* Out, size_t shift, size_t numElements) {

	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = i + shift;

	//Just rotate from end to the beginning
	if (j < numElements)  //To the end
		Out[i] = In[j];
	else
	{
		if (i < numElements) //from the beginning
			Out[i] = In[j - numElements];
	}

}


/**
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * Computes shift
 * number of elements numElements.
 * Forward shift so Mat[i] = Mat[i+shift] so we shift left more easy calculations without copy and extra temp buffer
 * Copy Buf0 to BufX
 * Shift Buf0 and store the shifted results in BufY
 */
__global__ void CopyShiftCUDA(cufftComplex* Buf0, cufftComplex* BufX, cufftComplex* BufY, size_t shift, size_t numElements) {

	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = i + shift;

	//Just copy
	BufX[i] = Buf0[i];

	//Just rotate from end to the beginning
	if (j < numElements)
		BufY[i] = Buf0[j];
	else
	{
		if (i < numElements)
			BufY[i] = Buf0[j - numElements];
	}

}


/**
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * Computes therotation.
 * number of elements numElements.
 * Copy Inp1 to Inp2
 */
__global__ void CopyCUDA(cufftComplex* Inp1, cufftComplex* Inp2, size_t numElements) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		Inp2[i] = Inp1[i];//copy no rotation
	}
}


/**
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * Result goes to  Cuda_bufZ
 *  InpX(Cuda_buf0) is not changed - reference
 *  InpY(Cuda_bufX) is not change here will be rotation_shifted
 *  Out(Cuda_bufZ) result output
 */
__global__ void CorelateCUDA(cufftComplex* InpX, cufftComplex* InpY, cufftComplex* Out, size_t numElements) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		Out[i].x = InpX[i].x * InpY[i].x + InpX[i].y * InpY[i].y;
		Out[i].y = -InpX[i].x * InpY[i].y + InpX[i].y * InpY[i].x;
	}
}


/*
 * CUDA Kernel Device code
 * By Daniel M. Kaminski
 * number of elements numElements.
 *
 *Input: Cuda_bufZ, Cuda_ColRow, Row,    Col_index* Row, shift, scale_Type
 *Output: Cuda_bufZ
 * Input/output Cuda_ColRow (Out) -the results on output are the most important. All transformations are cumulated in this buffer
 */
__global__ void MagnitudeCUDA(cufftComplex* Inp, float* Out, int cuda_row, int col_index, int cuda_shift, short scale_type)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = i + cuda_shift;
	int k = i + col_index;

	if (i < cuda_row) //protection
	{
#ifdef CALCULATE_SQRT_ON_CPU
		Out[k] = (Inp[j].x * Inp[j].x + Inp[j].y * Inp[j].y); 
#else
		Out[k] = sqrtf(Inp[j].x * Inp[j].x + Inp[j].y * Inp[j].y); //if this is performed on CPU is faster ????
		if (scale_type == 1) Out[k] = sqrtf(Out[k]);
		if (scale_type == 2) Out[k] = log2f(Out[k]);
#endif
	}
}