// Ambiguity.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"
#include <stdio.h>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ambiguity.h"
#include <omp.h>


#define BATCH 1
#define RANK 1
#define MAX_SHIFT 200
#define CONSTANT_AMPLIFICATION 200000.0f

//globals
bool openCL_Initiated = false;

//cuda pointers 
cufftComplex* Cuda_buf0 = nullptr;
cufftComplex* Cuda_bufX = nullptr;
cufftComplex* Cuda_bufY = nullptr;
cufftComplex* Cuda_bufZ = nullptr;
float* Cuda_ColRow = nullptr;

//Machine pointers to be reduced with time
cufftComplex* dat_inp0 = nullptr;
cufftComplex* dat_inp1 = nullptr;
float* MachinColRow = nullptr;

size_t threadsPerBlock = 1024;
size_t blocksPerGrid;
cudaStream_t stream = NULL;
cufftHandle plan;

size_t N;
size_t N_corrected;

int Col, Row;
int ColRow;
float Doppler_zoom;
float Doppler_zoom_Col;

size_t sizeR;
size_t sizeN;


//Data_In is in a format Data_In[2*n] - Real values, Data_In[2*n+1] - Imaginary values
int  Run(float* Data_In0, float* Data_In1, float* Data_Out, float amplification, float doppler_zoom, int shift, bool mode, short scale_type, bool remove_symetrics)
{
	//return 0;//test 83fps

	int err;
	float maxval = 1E-6f;
	float rev;

	if (!openCL_Initiated) 
		return -1;

	if (shift > MAX_SHIFT) 
		shift = MAX_SHIFT;

	int HalfColumns = Col / 2;
	int shift_cor = shift * 2;//4 bits size
	Doppler_zoom = 1.0f * N / doppler_zoom;
	Doppler_zoom_Col = Doppler_zoom / Col;



	//Convert float2 to cufftComplex 
	size_t j;
	if (mode)
	{
#pragma omp parallel for
		for (size_t n = 0; n < N; n++)
		{
			j = 2 * n;
			dat_inp0[n].x = Data_In0[j];
			dat_inp0[n].y = Data_In0[j + 1];

			dat_inp1[n].x = Data_In1[j];
			dat_inp1[n].y = Data_In1[j + 1];
		}
	}
	else
	{
#pragma omp parallel for
		for (size_t n = 0; n < N; n++)
		{
			j = 2 * n;
			dat_inp0[n].x = Data_In0[j];
			dat_inp0[n].y = Data_In0[j + 1];
		}
	}


	//First copy reference data to  Cuda_buf0 - reference must be shifted half but this will be later
	if (cudaMemcpyAsync(Cuda_buf0, dat_inp0, N_corrected, cudaMemcpyHostToDevice, stream) != CUFFT_SUCCESS)
	{
		return -301;
	}
	err = StreamSynchronise();
	if (err < 0)
		return err;

	//Fourier Transform of reference data (shift 0 deg)
	//Input: Cuda_buf0, Outout: Cuda_buf0
	if (cufftExecC2C(plan, Cuda_buf0, Cuda_buf0, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		return -303;
	}
	err = Synchronise();
	if (err < 0) 
		return err;


	//If two dongles copy the data datZ-second dongle data
	if (mode)
	{
		if (cudaMemcpyAsync(Cuda_bufX, dat_inp1, N_corrected, cudaMemcpyHostToDevice, stream) != CUFFT_SUCCESS)
		{
			return -304;
		}
		err = StreamSynchronise();
		if (err < 0)
			return err;

		//And do the transform to the Cuda_bufX buffer	
		if (cufftExecC2C(plan, Cuda_bufX, Cuda_bufX, CUFFT_FORWARD) != CUFFT_SUCCESS)
		{
			return -303;
		}
		err = Synchronise();
		if (err < 0)
			return err;

		//Rotate shift Cuda_buf0  after fft  to the Col/2 (screen midle) output Cuda_bufY
		err = CalcShift((int)(Doppler_zoom / 2));
		if (err < 0)
			return err;
	}
	else
	{
		//Just copy the fft transformed Cuda_buf0 to the Cuda_bufX and Cuda_bufY(half rotated) it is the same
		err = CopyShift((int)(Doppler_zoom / 2));
		if (err < 0)
			return err;

	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////
    //     This section
	// Input:
	//     Cuda_bufY - fft reference data shifted to half from Cuda_buf0
	//     Cuda_bufX - fft basic ratated date not shifted yet
	/////////////////////////////////////////////////////////////////////////////////////////////////////////

	//Calculate ambiguity
	//The hardest part of calculations (shift every column) Cuda_bufX in steps of N/Col/rotation_zoom
	for (int n = 0; n < Col; n++)
	{
		err = Calculate(shift_cor, scale_type, n);
		if (err < 0) 
			return err;
	}
	/// <summary>
	/// /////////////////////////////////////////////////////////////////////////////////////////////////////

	//Copy all the results in one shut
	if (cudaMemcpyAsync(Data_Out, Cuda_ColRow, sizeof(float) * (ColRow + shift), cudaMemcpyDeviceToHost, stream) != CUFFT_SUCCESS)
	{
		return -310;
	}
	err = StreamSynchronise();
	if (err < 0)
		return err;

	///////////////////////////////////////////////////////////////////////////////////////////////////////
	//Signal map correction
	// 
	// 
	//It is a problem sometimes MachinColRow is negative and Radar crashes
	float d=0.0001f;
	float TotalMAX = 0;

#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		d = Data_Out[n];

		if (d > maxval)
		{
			maxval = d;
		}
	}


	//Normalise to max
	maxval = 1.0f / maxval;
	 
#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		Data_Out[n] *= maxval;
		TotalMAX += Data_Out[n];
	}

	rev = 1;
	if (scale_type == 0)	rev = amplification;
	if (scale_type == 1)	rev = sqrtf(amplification);
	if (scale_type == 2)	rev = log2f(amplification);


	//Auto signal leveling
	if (TotalMAX <= 10) TotalMAX = 10;
	TotalMAX = CONSTANT_AMPLIFICATION / TotalMAX * rev;

#pragma omp parallel for
	for (int n = 0; n < ColRow; n++)
	{
		Data_Out[n] *= TotalMAX;
	}

 
//////////////////////////////////////////////////////////////////////////////////////////////////////
	//Remove symetrics
	int c1, c2;
	int tmp1, tmp2;
	float min;

	if (remove_symetrics)
	{
#pragma omp parallel for
		for (int C = 0 + 1; C < HalfColumns + 1; C++) //HalfColumns
		{
			c1 = (HalfColumns + C) * Row;
			c2 = (HalfColumns - C) * Row;

			for (int R = 0; R < Row; R++)
			{
				tmp1 = c1 + R;
				tmp2 = c2 + R;
				min = Min(Data_Out[tmp1], Data_Out[tmp2]);
				Data_Out[tmp1] -= min;
				Data_Out[tmp2] -= min;
			}
		}
	}

	return 0;
}

 
float Min(float x, float y)
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
	if (err < 0)
		return err;

	//Painful part
	//Input: Cuda_bufZ
	//Output: Cuda_bufZ
	err = FFT_bacward();
	if (err < 0)
		return err;

	//Input: Cuda_bufZ
	//Output: Cuda_bufZ
	err = Magnitude(shift, scale_type, index);
	if (err < 0)
		return err;

	return 0;
}

int StreamSynchronise()
{
	if (cudaStreamSynchronize(stream) != cudaSuccess) 
	{
		return -403;
	}
	return 0;
}

int Synchronise()
{
	if (cudaDeviceSynchronize() != cudaSuccess) 
	{
		return -403;
	}
	return 0;
}
 

int FFT_forward()
{
	//Forward fft
	if (cufftExecC2C(plan, Cuda_bufZ, Cuda_bufZ, CUFFT_FORWARD) != CUFFT_SUCCESS) 
	{
		return -401;
	}
	int err = Synchronise();
	if (err < 0) return err;
		
	return 0;
}


int FFT_bacward()
{
	//Backward fft
	if (cufftExecC2C(plan, Cuda_bufZ, Cuda_bufZ, CUFFT_INVERSE) != CUFFT_SUCCESS) 
	{
		return -402;
	}
	int err = Synchronise();
	if (err < 0) 
		return err;

	return 0; 
}




int  Initialize(unsigned int BufferSize, unsigned int col, unsigned int row, float doopler_shift, short* name)
{
	cudaError_t cudaStatus;

	//most be divided by 2 because the main program
	N = static_cast<size_t>(BufferSize) / 2;   // the size is smaller than 32 bits

	blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;


	Col = col;
	Row = row;
	ColRow = Col * Row;
	N_corrected = sizeof(cufftComplex) * N * BATCH;

	//machine pointers
	dat_inp0 = new cufftComplex[N];
	dat_inp1 = new cufftComplex[N];
	MachinColRow = new float[ColRow];

	int count = 0;
	cudaGetDeviceCount(&count);
	//fprintf(stderr, "cuda count %d", count);

	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	if (cudaStatus != cudaSuccess) {
		return -10;
	}
 
	//Copy the name to short pointer. With char doesnt work
	for (int i = 0; i < MAX_DEVICE_NAME; i++)
		name[i] = prop.name[i]; 
	// 
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		return -11;
	}

	if (cufftCreate(&plan) != CUFFT_SUCCESS) {
		return -12;
	}

	if (cufftPlan1d(&plan,(int) N, CUFFT_C2C, BATCH) != CUFFT_SUCCESS) {
		return -13;
	}

	//*********************************************************************
	//Memory alocate

	cudaMalloc(reinterpret_cast<void**>(&Cuda_buf0), N_corrected);
	if (cudaGetLastError() != cudaSuccess) {
		return -14;
	}
	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufX), N_corrected);
	if (cudaGetLastError() != cudaSuccess) {
		return -15;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufY), N_corrected);
	if (cudaGetLastError() != cudaSuccess) {
		return -15;
	}
	cudaMalloc(reinterpret_cast<void**>(&Cuda_bufZ), N_corrected);
	if (cudaGetLastError() != cudaSuccess) {
		return -18;
	}

	cudaMalloc(reinterpret_cast<void**>(&Cuda_ColRow), sizeof(float) * (ColRow + MAX_SHIFT));
	if (cudaGetLastError() != cudaSuccess) {
		return -18;
	}

	if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != CUFFT_SUCCESS) {
		return -19;
	}

	if (cufftSetStream(plan, stream) != CUFFT_SUCCESS) {
		return -20;
	}

	openCL_Initiated = true;
	return 0;
}


int  Release()
{
	cudaError_t cudaStatus;

	if (!openCL_Initiated) return 0; //If not initiallised return

	/* Release OpenCL memory objects. */
	

	//***************************************************
	cudaStatus = cudaFree(Cuda_buf0);
	if (cudaStatus != cudaSuccess) {
		return -512;
	}

	cudaStatus = cudaFree(Cuda_bufX);
	if (cudaStatus != cudaSuccess) {
		return -513;
	}
	cudaStatus = cudaFree(Cuda_bufY);
	if (cudaStatus != cudaSuccess) {
		return -513;
	}

	cudaStatus = cudaFree(Cuda_bufZ);
	if (cudaStatus != cudaSuccess) {
		return -515;
	}

	cudaStatus = cudaFree(Cuda_ColRow);
	if (cudaStatus != cudaSuccess) {
		return -515;
	}

	
	if (cufftDestroy(plan) != cudaSuccess) {
		fprintf(stderr, "cufftDestroy failed!");
		return -516;
	}

	cudaStatus = cudaStreamDestroy(stream);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaStreamDestroy failed!");
		return -517;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return -518;
	}


	//machine pointers
	delete[] dat_inp0;
	delete[] dat_inp1;
	delete[] MachinColRow;

	return    0;
}





//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//
//                                        CUDA                                              //
//
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

/**
 * CUDA Kernel Device code
 *
 * Result goes to  Cuda_bufZ
 *  InpX(Cuda_bufY) is not changed - reference
 *  InpY(Cuda_bufX) is not change it is a sorce of rotation the result of rotation goes to Cuda_bufY
 *  Out(Cuda_bufZ) result output
 */
__global__ void CorelateShiftCUDA(cufftComplex* BufX, cufftComplex* BufY,  cufftComplex* BufZ, size_t shift, size_t numElements) {

	size_t i = blockDim.x * blockIdx.x + threadIdx.x;
	size_t j = i + shift;

	//Just rotate from end to the beginning and convolute
	if (j < numElements)
	{
		BufZ[i].x = BufY[i].x * BufX[j].x + BufY[i].y * BufX[j].y;
		BufZ[i].y = -BufY[i].x * BufX[j].y + BufY[i].y * BufX[j].x;
	}
	else
	{
		if (i < numElements)
		{
			// j is >= numElements
			int l = j - numElements;
			BufZ[i].x =  BufY[i].x * BufX[l].x + BufY[i].y * BufX[l].y;
			BufZ[i].y = -BufY[i].x * BufX[l].y + BufY[i].y * BufX[l].x;
		}
	}
}

/**
 * CUDA Kernel Device code
 *
 * Computes shift
 * number of elements numElements.
 * Forward shift so Mat[i] = Mat[i+shift] so we shift left more easy calculations without copy and extra temp buffer
 */
__global__ void ShiftCUDA(cufftComplex* In, cufftComplex* Out, size_t shift, size_t numElements) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	size_t j = i + shift;

	//Just rotate from end to the beginning
	if (j < numElements)
		Out[i] = In[j];
	else
	{
		if (i < numElements)
			Out[i] = In[j - numElements];
	}

}

/**
 * CUDA Kernel Device code
 *
 * Computes shift
 * number of elements numElements.
 * Forward shift so Mat[i] = Mat[i+shift] so we shift left more easy calculations without copy and extra temp buffer
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
 *
 * Computes therotation.
 * number of elements numElements.
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
 *
 * Result goes to  Cuda_bufZ
 *  InpX(Cuda_buf0) is not changed - reference
 *  InpY(Cuda_bufX) is not change here will be rotation_shifted
 *  Out(Cuda_bufZ) result output 
 */
__global__ void CorelateCUDA(cufftComplex* InpX, cufftComplex* InpY, cufftComplex* Out, size_t numElements) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements)
	{
		Out[i].x =  InpX[i].x * InpY[i].x + InpX[i].y * InpY[i].y;
		Out[i].y = -InpX[i].x * InpY[i].y + InpX[i].y * InpY[i].x;		
	}
}

/**
 * CUDA Kernel Device code
 *
 * number of elements numElements.
 */
//                                        (Cuda_bufZ, Cuda_ColRow, Row,    Col_index* Row, shift, scale_Type);

__global__ void MagnitudeCUDA(cufftComplex* Inp, float* Out, int cuda_row, int col_index, int cuda_shift, short scale_type )
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = i + cuda_shift;
	int k = i + col_index;

	if (i < cuda_row)
	{
		Out[k] = sqrtf(Inp[j].x * Inp[j].x + Inp[j].y * Inp[j].y);

		if (scale_type == 1) Out[k] = sqrtf(Out[k]);
		if (scale_type == 2) Out[k] = log2f(Out[k]);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//
//                                                                                          //
//
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

 
//Input: (Rotation table) Cuda_bufBeta,  Cuda_buf0 
//Output: Cuda_bufX, Cuda_bufY
int CopyShift(size_t rotation_shift)
{

	CopyShiftCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufX, Cuda_bufY, rotation_shift, N);

	if (cudaGetLastError() != cudaSuccess) {
		return -803;
	}

	int err = Synchronise();
	if (err < 0)
		return err;

	return 0;
}

//Input: Cuda_buf0 (reference FFT input data), Cuda_bufX (rotation_shifted data)
//Output: Cuda_bufZ
int Corelate()
{
	CorelateCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufY, Cuda_bufZ, N);

	if (cudaGetLastError() != cudaSuccess) {
		return -802;
	}

	int err = Synchronise();
	if (err < 0)		return err;

	return 0;
}



//Possible acceleration
int Magnitude(int shift, short scale_Type, int Col_index)
{
	//Row_corrected = sizeof(cufftComplex) * Row * BATCH;
	//shift_corrected = Row_corrected + sizeof(cufftComplex) * Row * BATCH;
	MagnitudeCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_bufZ, Cuda_ColRow, Row, Col_index * Row, shift, scale_Type );

	if (cudaGetLastError() != cudaSuccess) {
		return -803;
	}

	int err = Synchronise();
	if (err < 0) return err;
	return 0;
}
//If calc normal than do normal calculations othervise calculate shift 1/2N for Cuda_buf0
//Input: Cuda_buf0
//Output: Cuda_bufY
int CalcShift(size_t rotation_shift )
{

	ShiftCUDA << <blocksPerGrid, threadsPerBlock >> > (Cuda_buf0, Cuda_bufY, rotation_shift, N);

	if (cudaGetLastError() != cudaSuccess) {
		return -803;
	}

	int err = Synchronise();
	if (err < 0) 
		return err;

	return 0;
}

//If calc normal than do normal calculations othervise calculate shift 1/2N for Cuda_buf0
//False shift to the half of all columns
//true normal mode shift every column
int CalcCorelateShift(size_t rotation_shift)
{
	CorelateShiftCUDA << <blocksPerGrid, threadsPerBlock >> > ( Cuda_bufX, Cuda_bufY,Cuda_bufZ, rotation_shift, N);

	if (cudaGetLastError() != cudaSuccess) 
	{
		return -803;
	}

	int err = Synchronise();
	if (err < 0)
		return err;

	return 0;
}