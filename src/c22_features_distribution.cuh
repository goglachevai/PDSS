#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline double dnHistogramMode(const double *y, int n, int nBins)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        int counts[10];
        double edges[11];
        histcounts(y, n, nBins, counts, edges);

        double maxCount = 0.0;
        int numMaxs = 1;
        double out = 0.0;
        for (int i = 0; i < nBins; ++i)
        {
            if ((double)counts[i] > maxCount)
            {
                maxCount = (double)counts[i];
                numMaxs = 1;
                out = (edges[i] + edges[i + 1]) * 0.5;
            }
            else if ((double)counts[i] == maxCount)
            {
                numMaxs += 1;
                out += (edges[i] + edges[i + 1]) * 0.5;
            }
        }
        return out / (double)numMaxs;
    }

    __host__ __device__ __noinline__ inline double DN_HistogramMode_5(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        return dnHistogramMode(y, n, 5);
    }
    __host__ __device__ __noinline__ inline double DN_HistogramMode_10(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        return dnHistogramMode(y, n, 10);
    }

    __host__ __device__ __noinline__ inline double dnOutlierInclude(const double *y, int n, double sign, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();

        const double inc = 0.01;
        int tot = 0;
        double *z = ws.dA;
        bool constantFlag = true;
        for (int i = 0; i < n; ++i)
        {
            if (y[i] != y[0])
                constantFlag = false;
            z[i] = sign * y[i];
            if (z[i] >= 0.0)
                tot += 1;
        }
        if (constantFlag)
            return 0.0;

        double maxVal = maxv(z, n);
        if (maxVal < inc)
            return 0.0;

        int nThresh = (int)(maxVal / inc) + 1;
        int nThreshCap = outlierThreshBound(n);
        if (nThresh > nThreshCap)
            nThresh = nThreshCap;
        if (nThresh < 1)
            nThresh = 1;

        int *head = ws.tiA;
        int *nxt = ws.iA;
        int *cnt = ws.tiB;
        for (int j = 0; j < nThresh; ++j)
        {
            head[j] = -1;
            cnt[j] = 0;
        }

        for (int i = 0; i < n; ++i)
        {
            double v = z[i];
            double q = v / inc;
            int j;
            if (q >= (double)(nThresh - 1))
                j = nThresh - 1;
            else if (q < 0.0)
                j = -1;
            else
                j = (int)q;
            while (j + 1 < nThresh && v >= (double)(j + 1) * inc)
                j++;
            while (j >= 0 && v < (double)j * inc)
                j--;
            if (j >= 0)
            {
                nxt[i] = head[j];
                head[j] = i;
                cnt[j]++;
            }
        }

        int running = 0;
        for (int j = nThresh - 1; j >= 0; --j)
        {
            running += cnt[j];
            cnt[j] = running;
        }

        int mj = 0, fbi = nThresh - 1;
        for (int j = 0; j < nThresh; ++j)
            if ((cnt[j] - 1) * 100.0 / (double)tot > 2.0)
                mj = j;
        for (int j = nThresh - 1; j >= 0; --j)
            if (cnt[j] - 1 == 0)
                fbi = j;
        int trimLimit = (mj < fbi) ? mj : fbi;
        if (trimLimit < 0)
            trimLimit = 0;
        if (trimLimit >= nThreshCap)
            trimLimit = nThreshCap - 1;

        int *bit = ws.iB;
        for (int i = 0; i <= n; ++i)
            bit[i] = 0;
        int logn = 1;
        while ((logn << 1) <= n)
            logn <<= 1;

        double *msDti4 = ws.tdA;
        int m = 0;
        double denom = (double)n / 2.0;

        for (int j = nThresh - 1; j >= 0; --j)
        {
            for (int i = head[j]; i != -1; i = nxt[i])
            {
                bitAdd(bit, n, i + 1);
                m++;
            }
            if (j > trimLimit)
                continue;
            if (m == 0)
            {
                msDti4[j] = qnan();
                continue;
            }

            double med;
            if (m % 2 == 1)
            {
                med = (double)bitSelect(bit, n, logn, m / 2 + 1);
            }
            else
            {
                double lo = (double)bitSelect(bit, n, logn, m / 2);
                double hi = (double)bitSelect(bit, n, logn, m / 2 + 1);
                med = (hi + lo) / 2.0;
            }
            msDti4[j] = med / denom - 1.0;
        }

        double *scratch = ws.tdB;
        return median(msDti4, trimLimit + 1, scratch);
    }

    __host__ __device__ __noinline__ inline double DN_OutlierInclude_p_001_mdrmd(const double *y, int n, C22Workspace ws) { return dnOutlierInclude(y, n, 1.0, ws); }
    __host__ __device__ __noinline__ inline double DN_OutlierInclude_n_001_mdrmd(const double *y, int n, C22Workspace ws) { return dnOutlierInclude(y, n, -1.0, ws); }

}
