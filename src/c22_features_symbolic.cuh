#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline void sbCoarsegrainQuantile3(const double *y, int n, int *labels, double *sortScratch)
    {
        for (int i = 0; i < n; ++i)
            sortScratch[i] = y[i];
        heapSort(sortScratch, n);
        double th[4];
        for (int i = 0; i < 4; ++i)
            th[i] = quantileSorted(sortScratch, n, (double)i / 3.0);
        th[0] -= 1.0;
        for (int j = 0; j < n; ++j)
        {
            labels[j] = 3;
            for (int i = 0; i < 3; ++i)
            {
                if (y[j] > th[i] && y[j] <= th[i + 1])
                {
                    labels[j] = i + 1;
                    break;
                }
            }
        }
    }

    __host__ __device__ __noinline__ inline double SB_BinaryStats_mean_longstretch1(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        double yMean = mean(y, n);
        int maxstretch1 = 0, last1 = 0;
        for (int i = 0; i < n - 1; ++i)
        {
            int bin = (y[i] - yMean <= 0.0) ? 0 : 1;
            if (bin == 0 || i == n - 2)
            {
                int stretch1 = i - last1;
                if (stretch1 > maxstretch1)
                    maxstretch1 = stretch1;
                last1 = i;
            }
        }
        return (double)maxstretch1;
    }

    __host__ __device__ __noinline__ inline double SB_BinaryStats_diff_longstretch0(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        int maxstretch0 = 0, last1 = 0;
        for (int i = 0; i < n - 1; ++i)
        {
            double d = y[i + 1] - y[i];
            int bin = (d < 0.0) ? 0 : 1;
            if (bin == 1 || i == n - 2)
            {
                int stretch0 = i - last1;
                if (stretch0 > maxstretch0)
                    maxstretch0 = stretch0;
                last1 = i;
            }
        }
        return (double)maxstretch0;
    }

    __host__ __device__ __noinline__ inline double SB_MotifThree_quantile_hh(const double *y, int n, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        int *labels = ws.iA;
        double *sortScratch = ws.dA;
        sbCoarsegrainQuantile3(y, n, labels, sortScratch);

        double count2[3][3] = {};
        for (int p = 0; p < n - 1; ++p)
        {
            int si = labels[p] - 1, sj = labels[p + 1] - 1;
            if (si >= 0 && si < 3 && sj >= 0 && sj < 3)
                count2[si][sj] += 1.0;
        }
        double out2[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                out2[i][j] = count2[i][j] / (double)(n - 1);

        double hh = 0.0;
        for (int i = 0; i < 3; ++i)
            hh += f_entropy(out2[i], 3);
        return hh;
    }

    __host__ __device__ __noinline__ inline double transitionMatrixCore(const double *y, int n, int firstZero, C22Workspace ws)
    {
        bool constant = true;
        for (int i = 0; i < n; ++i)
        {
            if (isNaN(y[i]))
                return qnan();
            if (y[i] != y[0])
                constant = false;
        }
        if (constant)
            return qnan();

        int tau = firstZero >= 0 ? firstZero : firstZeroACF(y, n, n, ws.fft);
        if (tau < 1)
            tau = 1;
        int nDown = (n - 1) / tau + 1;
        if (nDown < 2)
            return qnan();

        double *yDown = ws.dA;
        for (int i = 0; i < nDown; ++i)
            yDown[i] = y[i * tau];

        int *labels = ws.iA;
        double *sortScratch = ws.dB;
        sbCoarsegrainQuantile3(yDown, nDown, labels, sortScratch);

        double T[3][3] = {};
        for (int j = 0; j < nDown - 1; ++j)
            T[labels[j] - 1][labels[j + 1] - 1] += 1.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T[i][j] /= (double)(nDown - 1);

        double col[3][3];
        for (int k = 0; k < 3; ++k)
            for (int i = 0; i < 3; ++i)
                col[k][i] = T[i][k];

        double COV[3][3];
        for (int i = 0; i < 3; ++i)
        {
            for (int j = i; j < 3; ++j)
            {
                double meanI = mean(col[i], 3), meanJ = mean(col[j], 3);
                double c = 0.0;
                for (int k = 0; k < 3; ++k)
                    c += (col[i][k] - meanI) * (col[j][k] - meanJ);
                c /= 2.0;
                COV[i][j] = c;
                COV[j][i] = c;
            }
        }
        double sumdiag = 0.0;
        for (int i = 0; i < 3; ++i)
            sumdiag += COV[i][i];
        return sumdiag;
    }

    __host__ __device__ __noinline__ inline double SB_TransitionMatrix_3ac_sumdiagcov(const double *y, int n, C22Workspace ws)
    {
        return transitionMatrixCore(y, n, -1, ws);
    }

}
