#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline void solveSpline5x5(const double *AtA25, const double *b5, double *x5)
    {
        double M[5][5];
        double bb[5];
        for (int i = 0; i < 5; ++i)
        {
            for (int j = 0; j < 5; ++j)
                M[i][j] = AtA25[i * 5 + j];
            bb[i] = b5[i];
        }
        for (int i = 0; i < 5; ++i)
        {
            for (int j = i + 1; j < 5; ++j)
            {
                double factor = M[j][i] / M[i][i];
                bb[j] -= factor * bb[i];
                for (int k = i; k < 5; ++k)
                    M[j][k] -= factor * M[i][k];
            }
        }
        for (int i = 4; i >= 0; --i)
        {
            double s = bb[i];
            for (int j = i + 1; j < 5; ++j)
                s -= x5[j] * M[i][j];
            x5[i] = s / M[i][i];
        }
    }

    __host__ __device__ __noinline__ inline void pdSplineDetrend(const double *y, int n, const double *A, const double *AtA, const double *M, int breaks1, double *ySub)
    {
        double Atb[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < 5; ++c)
                Atb[c] += A[i * 5 + c] * y[i];

        double x[5];
        solveSpline5x5(AtA, Atb, x);

        double coefsSplineFlat[8];
        for (int j = 0; j < 8; ++j)
        {
            double s = 0.0;
            for (int i = 0; i < 5; ++i)
                s += M[j * 5 + i] * x[i];
            coefsSplineFlat[j] = s;
        }

        for (int i = 0; i < n; ++i)
        {
            int half = (i < breaks1) ? 0 : 1;
            double xloc = (double)(i - (half ? breaks1 : 0));
            double v = coefsSplineFlat[half + 0 * 2];
            for (int k = 1; k < 4; ++k)
                v = v * xloc + coefsSplineFlat[half + k * 2];
            ySub[i] = y[i] - v;
        }
    }

#ifndef C22_PDACF_FFT_MIN_N
#define C22_PDACF_FFT_MIN_N 800
#endif

    __host__ __device__ __noinline__ inline void pdAcf(const double *ySub, int n, int acmax, double *acf, double *fftBuf)
    {
        if (n < C22_PDACF_FFT_MIN_N)
        {
            for (int tau = 1; tau <= acmax; ++tau)
            {
                int m = n - tau;
                double acc = 0.0;
                for (int i = 0; i < m; ++i)
                    acc += ySub[i] * ySub[i + tau];
                acf[tau - 1] = acc / (double)m;
            }
            return;
        }
        autocorrSumsFFT(ySub, n, false, fftBuf);
        for (int tau = 1; tau <= acmax; ++tau)
            acf[tau - 1] = fftBuf[tau] / (double)(n - tau);
    }

    __host__ __device__ __noinline__ inline int PD_PeriodicityWang_th0_01(const double *y, int n, const double *splineA, const double *splineAtA, const double *splineM, int breaks1, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return 0;
        const double th = 0.01;

        double *ySub = ws.dA;
        pdSplineDetrend(y, n, splineA, splineAtA, splineM, breaks1, ySub);

        int acmax = (int)ceil((double)n / 3.0);
        if (acmax < 3)
            return 0;

        double *acf = ws.dB;
        pdAcf(ySub, n, acmax, acf, ws.fft);

        int *troughs = ws.iA;
        int nTroughs = 0, jT = -1, out = 0;
        for (int i = 1; i <= acmax - 2; ++i)
        {
            double slopeIn = acf[i] - acf[i - 1];
            double slopeOut = acf[i + 1] - acf[i];
            if (slopeIn < 0.0 && slopeOut > 0.0)
            {
                troughs[nTroughs] = i;
                nTroughs++;
            }
            else if (slopeIn > 0.0 && slopeOut < 0.0)
            {
                int iPeak = i;
                double thePeak = acf[iPeak];
                while (jT + 1 < nTroughs && troughs[jT + 1] < iPeak)
                    jT++;
                if (jT == -1)
                    continue;
                int iTrough = troughs[jT];
                double theTrough = acf[iTrough];
                if (thePeak - theTrough < th)
                    continue;
                if (thePeak < 0.0)
                    continue;
                out = iPeak;
                break;
            }
        }
        return out;
    }

}
