 

#ifndef _DLL_H_
#define _DLL_H_
#pragma once
#if BUILDING_DLL
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
#endif /* _DLL_H_ */


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
float Min(float x, float y);