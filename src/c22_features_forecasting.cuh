#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline double FC_LocalSimple_mean_stderr(const double *y, int n, int L, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        if (n <= L)
            return qnan();
        int m = n - L;
        double *res = ws.dA;
        for (int i = 0; i < m; ++i)
        {
            double s = 0.0;
            for (int j = 0; j < L; ++j)
                s += y[i + j];
            res[i] = y[i + L] - s / (double)L;
        }
        return stddev(res, m);
    }
    __host__ __device__ __noinline__ inline double FC_LocalSimple_mean3_stderr(const double *y, int n, C22Workspace ws) { return FC_LocalSimple_mean_stderr(y, n, 3, ws); }

    __host__ __device__ __noinline__ inline double FC_LocalSimple_mean_tauresrat(const double *y, int n, int L, C22Workspace ws, int yFirstZero = -1)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        if (n <= L)
            return qnan();
        int m = n - L;
        double *res = ws.dA;
        for (int i = 0; i < m; ++i)
        {
            double s = 0.0;
            for (int j = 0; j < L; ++j)
                s += y[i + j];
            res[i] = y[i + L] - s / (double)L;
        }
        double resAC1stZ = (double)firstZeroACF(res, m, m, ws.fft);
        double yAC1stZ = (double)(yFirstZero >= 0 ? yFirstZero : firstZeroACF(y, n, n, ws.fft));
        return resAC1stZ / yAC1stZ;
    }
    __host__ __device__ __noinline__ inline double FC_LocalSimple_mean1_tauresrat(const double *y, int n, C22Workspace ws) { return FC_LocalSimple_mean_tauresrat(y, n, 1, ws); }

}
