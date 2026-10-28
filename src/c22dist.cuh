#pragma once
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "c22_features.cuh"

__global__ void computeCatch22Features(
    const float *d_timeSeries, int windowLength, int numWindows,
    const double *splineA, const double *splineAtA, const double *splineM, int splineBreaks1,
    double *d_workspacePool, c22::C22WorkspaceLayout layout,
    int *d_nextWindow, float *d_features);

void normalizeFeatures(float *d_features, int numWindows);
