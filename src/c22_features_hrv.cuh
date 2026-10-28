#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline double MD_hrv_classic_pnn40(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        double pnn40 = 0.0;
        for (int i = 1; i < n; ++i)
            if (fabs(y[i] - y[i - 1]) * 1000.0 > 40.0)
                pnn40 += 1.0;
        return pnn40 / (double)(n - 1);
    }

}
