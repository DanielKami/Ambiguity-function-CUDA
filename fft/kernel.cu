//#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
// dllmain.cpp : Defines the entry point for the DLL application.
#include "stdafx.h"
#include "Ambiguity.h"

BOOL APIENTRY Main(HMODULE hModule,
	DWORD  ul_reason_for_call,
	LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

