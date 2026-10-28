#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline int spWelchCompute(const double *y, int n, double *Sw, double *w, double *fftBuf)
    {
        const double PI = 3.14159265359;
        int NFFT = nextPow2(n);
        int Nout = NFFT / 2 + 1;
        double m = mean(y, n);
        double scale = 1.0 / ((double)n * 2.0 * PI);

        for (int i = 0; i < NFFT; ++i)
            fftBuf[i] = (i < n) ? y[i] - m : 0.0;
        realFFTInPlace(fftBuf, NFFT);

        for (int l = 0; l < Nout; ++l)
        {
            double re = fftBuf[2 * l], im = fftBuf[2 * l + 1];
            double Pxx = (re * re + im * im) * scale;
            if (l > 0 && l < Nout - 1)
                Pxx *= 2.0;
            Sw[l] = Pxx;
            w[l] = 2.0 * PI * (double)l / (double)NFFT;
        }
        return Nout;
    }

    __host__ __device__ __noinline__ inline double spAreaFromWelch(const double *Sw, const double *w, int Nout)
    {
        for (int i = 0; i < Nout; ++i)
            if (isInf(Sw[i]))
                return 0.0;
        double dw = w[1] - w[0];
        double area = 0.0;
        for (int i = 0; i < Nout / 5; ++i)
            area += Sw[i];
        return area * dw;
    }

    __host__ __device__ __noinline__ inline double spCentroidFromWelch(const double *Sw, const double *w, int Nout)
    {
        for (int i = 0; i < Nout; ++i)
            if (isInf(Sw[i]))
                return 0.0;
        double csSLast = 0.0;
        for (int i = 0; i < Nout; ++i)
            csSLast += Sw[i];
        double thresh = csSLast * 0.5;

        double running = 0.0, centroid = 0.0;
        for (int i = 0; i < Nout; ++i)
        {
            running += Sw[i];
            if (running > thresh)
            {
                centroid = w[i];
                break;
            }
        }
        return centroid;
    }

    __host__ __device__ __noinline__ inline double SP_Summaries_welch_rect_area_5_1(const double *y, int n, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        double *Sw = ws.noA, *w = ws.noB;
        int Nout = spWelchCompute(y, n, Sw, w, ws.fft);
        return spAreaFromWelch(Sw, w, Nout);
    }

    __host__ __device__ __noinline__ inline double SP_Summaries_welch_rect_centroid(const double *y, int n, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        double *Sw = ws.noA, *w = ws.noB;
        int Nout = spWelchCompute(y, n, Sw, w, ws.fft);
        return spCentroidFromWelch(Sw, w, Nout);
    }

    __host__ __device__ __noinline__ inline void SP_Summaries_welch_rect_both(const double *y, int n, double *outArea, double *outCentroid, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
        {
            if (isNaN(y[i]))
            {
                *outArea = qnan();
                *outCentroid = qnan();
                return;
            }
        }
        double *Sw = ws.noA, *w = ws.noB;
        int Nout = spWelchCompute(y, n, Sw, w, ws.fft);
        *outArea = spAreaFromWelch(Sw, w, Nout);
        *outCentroid = spCentroidFromWelch(Sw, w, Nout);
    }

}
