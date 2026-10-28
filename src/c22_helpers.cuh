#pragma once

#ifdef __CUDACC__
#include "cuda_runtime.h"
#else
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __noinline__
#define __noinline__
#endif
#endif

#include <cmath>
#include "c22_fft.cuh"
#ifndef __CUDACC__
#include <limits>
#endif

namespace c22
{
    struct C22Workspace
    {
        double *persistN;
        double *dA;
        double *dB;
        int *iA;
        int *iB;
        double *tdA;
        double *tdB;
        int *tiA;
        int *tiB;
        double *noA;
        double *noB;
        double *fft;
    };

    struct C22WorkspaceLayout
    {
        int nSize;
        int threshSize;
        int noutSize;
        int fftSize;
        long long slotsPerThread;
    };

    __host__ __device__ inline int outlierThreshBound(int n)
    {
        double inc = 0.01;
        double maxPossible = sqrt((double)(n > 1 ? n - 1 : 1));
        return (int)ceil(maxPossible / inc) + 4;
    }

    __host__ __device__ inline int nextPow2(int n)
    {
        int p = 1;
        while (p < n)
            p *= 2;
        return p;
    }

    __host__ __device__ inline C22WorkspaceLayout computeC22WorkspaceLayout(int n)
    {
        C22WorkspaceLayout L;
        L.nSize = n + 1;
        L.threshSize = outlierThreshBound(n);
        L.noutSize = nextPow2(n) / 2 + 1;
        L.fftSize = acfFFTLength(n) + 2;

        L.slotsPerThread = 5LL * L.nSize + 4LL * L.threshSize + 2LL * L.noutSize + L.fftSize;
        return L;
    }

    __host__ __device__ inline C22Workspace makeC22Workspace(double *threadPool, const C22WorkspaceLayout &L)
    {
        C22Workspace ws;
        double *p = threadPool;
        ws.persistN = p;
        p += L.nSize;
        ws.dA = p;
        p += L.nSize;
        ws.dB = p;
        p += L.nSize;
        ws.iA = (int *)p;
        p += L.nSize;
        ws.iB = (int *)p;
        p += L.nSize;
        ws.tdA = p;
        p += L.threshSize;
        ws.tdB = p;
        p += L.threshSize;
        ws.tiA = (int *)p;
        p += L.threshSize;
        ws.tiB = (int *)p;
        p += L.threshSize;
        ws.noA = p;
        p += L.noutSize;
        ws.noB = p;
        p += L.noutSize;
        ws.fft = p;
        p += L.fftSize;
        return ws;
    }

    __host__ __device__ inline double qnan()
    {
#ifdef __CUDACC__
        return nan("");
#else
        return std::numeric_limits<double>::quiet_NaN();
#endif
    }

    __host__ __device__ inline bool isNaN(double x) { return x != x; }
    __host__ __device__ inline bool isInf(double x) { return x > 1.7976931348623157e308 || x < -1.7976931348623157e308; }

    __host__ __device__ inline double mean(const double *a, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += a[i];
        return s / (double)n;
    }

    __host__ __device__ inline double minv(const double *a, int n)
    {
        double m = a[0];
        for (int i = 1; i < n; ++i)
            if (a[i] < m)
                m = a[i];
        return m;
    }

    __host__ __device__ inline double maxv(const double *a, int n)
    {
        double m = a[0];
        for (int i = 1; i < n; ++i)
            if (a[i] > m)
                m = a[i];
        return m;
    }

    __host__ __device__ inline double stddev(const double *a, int n)
    {
        double m = mean(a, n);
        double s = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double d = a[i] - m;
            s += d * d;
        }
        return sqrt(s / (double)(n - 1));
    }

    __host__ __device__ inline void zscore(const double *a, int n, double *out)
    {
        double m = mean(a, n);
        double sd = stddev(a, n);
        for (int i = 0; i < n; ++i)
            out[i] = (a[i] - m) / sd;
    }

    __host__ __device__ inline void diff(const double *a, int n, double *out)
    {
        for (int i = 1; i < n; ++i)
            out[i - 1] = a[i] - a[i - 1];
    }

    __host__ __device__ inline double dot(const double *x, const double *y, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += x[i] * y[i];
        return s;
    }

    __host__ __device__ inline void heapifyDown(double *a, int n, int i)
    {
        for (;;)
        {
            int largest = i, l = 2 * i + 1, r = 2 * i + 2;
            if (l < n && a[l] > a[largest])
                largest = l;
            if (r < n && a[r] > a[largest])
                largest = r;
            if (largest == i)
                break;
            double t = a[i];
            a[i] = a[largest];
            a[largest] = t;
            i = largest;
        }
    }

    __host__ __device__ inline void heapSort(double *a, int n)
    {
        for (int i = n / 2 - 1; i >= 0; --i)
            heapifyDown(a, n, i);
        for (int i = n - 1; i > 0; --i)
        {
            double t = a[0];
            a[0] = a[i];
            a[i] = t;
            heapifyDown(a, i, 0);
        }
    }

    __host__ __device__ inline double quantileSorted(const double *sorted, int n, double quant)
    {
        double q = 0.5 / (double)n;
        if (quant < q)
            return sorted[0];
        if (quant > (1.0 - q))
            return sorted[n - 1];
        double quantIdx = (double)n * quant - 0.5;
        int idxLeft = (int)floor(quantIdx);
        int idxRight = (int)ceil(quantIdx);
        if (idxRight == idxLeft)
            return sorted[idxLeft];
        return sorted[idxLeft] + (quantIdx - idxLeft) * (sorted[idxRight] - sorted[idxLeft]) / (double)(idxRight - idxLeft);
    }

    __host__ __device__ inline double medianSorted(const double *sorted, int n)
    {
        if (n % 2 == 1)
            return sorted[n / 2];
        int m1 = n / 2, m2 = m1 - 1;
        return (sorted[m1] + sorted[m2]) / 2.0;
    }

    __host__ __device__ inline double median(const double *a, int n, double *scratch)
    {
        for (int i = 0; i < n; ++i)
            scratch[i] = a[i];
        heapSort(scratch, n);
        return medianSorted(scratch, n);
    }

    __host__ __device__ inline double f_entropy(const double *a, int n)
    {
        double f = 0.0;
        for (int i = 0; i < n; ++i)
            if (a[i] > 0.0)
                f += a[i] * log(a[i]);
        return -f;
    }

    __host__ __device__ inline void linreg(int n, const double *x, const double *y, double *m, double *b)
    {
        double sumx = 0.0, sumx2 = 0.0, sumxy = 0.0, sumy = 0.0;
        for (int i = 0; i < n; ++i)
        {
            sumx += x[i];
            sumx2 += x[i] * x[i];
            sumxy += x[i] * y[i];
            sumy += y[i];
        }
        double denom = (double)n * sumx2 - sumx * sumx;
        if (denom == 0.0)
        {
            *m = 0.0;
            *b = 0.0;
            return;
        }
        *m = ((double)n * sumxy - sumx * sumy) / denom;
        *b = (sumy * sumx2 - sumx * sumxy) / denom;
    }

    __host__ __device__ inline double norm2(const double *a, int n)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            s += a[i] * a[i];
        return sqrt(s);
    }

    __host__ __device__ inline double acfBiasedAt(const double *y, int n, double m, double denom, int lag)
    {
        double s = 0.0;
        for (int i = 0; i < n - lag; ++i)
            s += (y[i] - m) * (y[i + lag] - m);
        return s / denom;
    }

    __host__ __device__ inline double acfDenom(const double *y, int n, double m)
    {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
        {
            double d = y[i] - m;
            s += d * d;
        }
        return s;
    }

#ifndef C22_ACF_DIRECT_LAGS
#define C22_ACF_DIRECT_LAGS 128
#endif

    __host__ __device__ inline int firstZeroACF(const double *y, int n, int maxtau, double *fftBuf)
    {
        double m = mean(y, n);
        double denom = acfDenom(y, n, m);
        int k = 0;
        while (k < maxtau && k < C22_ACF_DIRECT_LAGS)
        {
            double acfk = acfBiasedAt(y, n, m, denom, k);
            if (!(acfk > 0.0))
                return k;
            k++;
        }
        if (k >= maxtau)
            return k;

        autocorrSumsFFT(y, n, true, fftBuf);
        for (; k < maxtau && k < n; ++k)
            if (!(fftBuf[k] / fftBuf[0] > 0.0))
                return k;
        return k;
    }

    struct AcfStats
    {
        int firstZero;
        double f1ecac;
        int firstMin;
    };

    __host__ __device__ inline bool scanAcfStats(const double *acf, int avail, int n, AcfStats &st)
    {
        bool zeroDone = false, ecacDone = false, minDone = false;
        for (int k = 0; k < avail; ++k)
            if (!(acf[k] > 0.0))
            {
                st.firstZero = k;
                zeroDone = true;
                break;
            }
        if (!zeroDone && avail >= n)
        {
            st.firstZero = n;
            zeroDone = true;
        }

        const double thresh = 1.0 / exp(1.0);
        for (int i = 0; i < n - 2 && i + 1 < avail; ++i)
            if (acf[i + 1] < thresh)
            {
                st.f1ecac = (double)i + (thresh - acf[i]) / (acf[i + 1] - acf[i]);
                ecacDone = true;
                break;
            }
        if (!ecacDone && avail >= n - 1)
        {
            st.f1ecac = (double)n;
            ecacDone = true;
        }

        for (int i = 1; i < n - 1 && i + 1 < avail; ++i)
            if (acf[i] < acf[i - 1] && acf[i] < acf[i + 1])
            {
                st.firstMin = i;
                minDone = true;
                break;
            }
        if (!minDone && avail >= n)
        {
            st.firstMin = n;
            minDone = true;
        }
        return zeroDone && ecacDone && minDone;
    }

    __host__ __device__ inline AcfStats acfStats(const double *y, int n, double *acf, double *fftBuf)
    {
        AcfStats st;
        double m = mean(y, n);
        double denom = acfDenom(y, n, m);
        int direct = n < C22_ACF_DIRECT_LAGS ? n : C22_ACF_DIRECT_LAGS;
        for (int k = 0; k < direct; ++k)
            acf[k] = acfBiasedAt(y, n, m, denom, k);
        if (scanAcfStats(acf, direct, n, st))
            return st;

        autocorrSumsFFT(y, n, true, fftBuf);
        for (int k = direct; k < n; ++k)
            acf[k] = fftBuf[k] / fftBuf[0];
        scanAcfStats(acf, n, n, st);
        return st;
    }

    __host__ __device__ inline int firstMinACF(const double *y, int n)
    {
        double m = mean(y, n);
        double denom = acfDenom(y, n, m);
        double prev = acfBiasedAt(y, n, m, denom, 0);
        double cur = (n > 1) ? acfBiasedAt(y, n, m, denom, 1) : prev;
        for (int i = 1; i < n - 1; ++i)
        {
            double next = acfBiasedAt(y, n, m, denom, i + 1);
            if (cur < prev && cur < next)
                return i;
            prev = cur;
            cur = next;
        }
        return n;
    }

    __host__ __device__ inline double f1ecac(const double *y, int n)
    {
        double m = mean(y, n);
        double denom = acfDenom(y, n, m);
        double thresh = 1.0 / exp(1.0);
        double acfPrev = acfBiasedAt(y, n, m, denom, 0);
        for (int i = 0; i < n - 2; ++i)
        {
            double acfCur = acfBiasedAt(y, n, m, denom, i + 1);
            if (acfCur < thresh)
            {
                double slope = acfCur - acfPrev;
                double dy = thresh - acfPrev;
                return (double)i + dy / slope;
            }
            acfPrev = acfCur;
        }
        return (double)n;
    }

    __host__ __device__ inline double autocorrLagPearson(const double *y, int n, int lag)
    {
        int m = n - lag;
        double meanX = 0.0, meanY = 0.0;
        for (int i = 0; i < m; ++i)
        {
            meanX += y[i];
            meanY += y[i + lag];
        }
        meanX /= m;
        meanY /= m;
        double nom = 0.0, dX = 0.0, dY = 0.0;
        for (int i = 0; i < m; ++i)
        {
            double dx = y[i] - meanX, dy = y[i + lag] - meanY;
            nom += dx * dy;
            dX += dx * dx;
            dY += dy * dy;
        }
        return nom / sqrt(dX * dY);
    }

    __host__ __device__ inline void bitAdd(int *bit, int n, int i)
    {
        for (; i <= n; i += i & (-i))
            bit[i]++;
    }

    __host__ __device__ inline int bitSelect(const int *bit, int n, int logn, int k)
    {
        int pos = 0;
        for (int pw = logn; pw > 0; pw >>= 1)
        {
            if (pos + pw <= n && bit[pos + pw] < k)
            {
                pos += pw;
                k -= bit[pos];
            }
        }
        return pos + 1;
    }

    __host__ __device__ inline int numBinsAuto(const double *y, int n)
    {
        double mx = maxv(y, n), mn = minv(y, n);
        double sd = stddev(y, n);
        if (sd < 0.001)
            return 0;
        double width = 3.5 * sd / pow((double)n, 1.0 / 3.0);
        return (int)ceil((mx - mn) / width);
    }

    __host__ __device__ inline void histcounts(const double *y, int n, int nBins, int *counts, double *edges)
    {
        double mn = minv(y, n), mx = maxv(y, n);
        double step = (mx - mn) / (double)nBins;
        for (int i = 0; i < nBins; ++i)
            counts[i] = 0;
        for (int i = 0; i < n; ++i)
        {
            int b = (step > 0.0) ? (int)((y[i] - mn) / step) : 0;
            if (b < 0)
                b = 0;
            if (b >= nBins)
                b = nBins - 1;
            counts[b]++;
        }
        for (int i = 0; i <= nBins; ++i)
            edges[i] = (double)i * step + mn;
    }

}
