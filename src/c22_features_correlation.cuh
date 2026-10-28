#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline double CO_f1ecac_feat(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return 0.0;
        return f1ecac(y, n);
    }

    __host__ __device__ __noinline__ inline double CO_FirstMin_ac_feat(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return 0.0;
        return (double)firstMinACF(y, n);
    }

    __host__ __device__ __noinline__ inline double CO_trev_1_num(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        double s = 0.0;
        for (int i = 0; i < n - 1; ++i)
        {
            double d = y[i + 1] - y[i];
            s += d * d * d;
        }
        return s / (double)(n - 1);
    }

    __host__ __device__ __noinline__ inline double CO_HistogramAMI_even_2_5(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        const int tau = 2;
        int m = n - tau;
        if (m <= 0)
            return qnan();

        double maxValue = maxv(y, n), minValue = minv(y, n);
        double binStep = (maxValue - minValue + 0.2) / 5.0;
        double edges[6];
        for (int i = 0; i < 6; ++i)
            edges[i] = minValue + binStep * (double)i - 0.1;

        double pij[5][5];
        for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b)
                pij[a][b] = 0.0;

        for (int i = 0; i < m; ++i)
        {
            int bin1 = 0;
            for (int j = 0; j < 6; ++j)
                if (y[i] < edges[j])
                {
                    bin1 = j;
                    break;
                }
            int bin2 = 0;
            for (int j = 0; j < 6; ++j)
                if (y[i + tau] < edges[j])
                {
                    bin2 = j;
                    break;
                }
            int b1 = bin1 - 1, b2 = bin2 - 1;
            if (b1 < 0)
                b1 = 0;
            if (b1 > 4)
                b1 = 4;
            if (b2 < 0)
                b2 = 0;
            if (b2 > 4)
                b2 = 4;
            pij[b2][b1] += 1.0;
        }

        for (int a = 0; a < 5; ++a)
            for (int b = 0; b < 5; ++b)
                pij[a][b] /= (double)m;

        double pi[5] = {0, 0, 0, 0, 0}, pj[5] = {0, 0, 0, 0, 0};
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
            {
                pi[i] += pij[i][j];
                pj[j] += pij[i][j];
            }

        double ami = 0.0;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                if (pij[i][j] > 0.0)
                    ami += pij[i][j] * log(pij[i][j] / (pj[j] * pi[i]));
        return ami;
    }

    __host__ __device__ __noinline__ inline double IN_AutoMutualInfoStats_40_gaussian_fmmi(const double *y, int n, C22Workspace ws)
    {
        (void)ws;
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        int tau = 40;
        int maxTau = (n + 1) / 2;
        if (tau > maxTau)
            tau = maxTau;
        if (tau < 3)
            return (double)tau;

        double ac = autocorrLagPearson(y, n, 1);
        double amiPrev = -0.5 * log(1.0 - ac * ac);
        ac = autocorrLagPearson(y, n, 2);
        double amiCur = -0.5 * log(1.0 - ac * ac);

        double fmmi = (double)tau;
        for (int i = 1; i < tau - 1; ++i)
        {
            ac = autocorrLagPearson(y, n, i + 2);
            double amiNext = -0.5 * log(1.0 - ac * ac);
            if (amiCur < amiPrev && amiCur < amiNext)
            {
                fmmi = (double)i;
                break;
            }
            amiPrev = amiCur;
            amiCur = amiNext;
        }
        return fmmi;
    }

    __host__ __device__ __noinline__ inline double embed2DistCore(const double *y, int n, int tau, C22Workspace ws)
    {
        int cap = n / 10;
        if (tau > cap)
            tau = cap;
        if (tau < 0)
            tau = 0;

        int dn = n - tau - 1;
        if (dn <= 0)
            return 0.0;

        double *d = ws.dA;
        for (int i = 0; i < dn; ++i)
        {
            double d1 = y[i + 1] - y[i];
            double d2 = y[i + tau] - y[i + tau + 1];
            d[i] = sqrt(d1 * d1 + d2 * d2);
            if (isNaN(d[i]))
                return qnan();
        }

        double l = mean(d, dn);
        int nBins = numBinsAuto(d, dn);
        if (nBins == 0)
            return 0.0;
        if (nBins > 200)
            nBins = 200;

        int counts[200];
        double edges[201];
        histcounts(d, dn, nBins, counts, edges);

        double out = 0.0;
        for (int i = 0; i < nBins; ++i)
        {
            double histNorm = (double)counts[i] / (double)dn;
            double expf = exp(-(edges[i] + edges[i + 1]) * 0.5 / l) / l;
            if (expf < 0.0)
                expf = 0.0;
            out += fabs(histNorm - expf);
        }
        return out / (double)nBins;
    }

    __host__ __device__ __noinline__ inline double CO_Embed2_Dist_tau_d_expfit_meandiff(const double *y, int n, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();
        return embed2DistCore(y, n, firstZeroACF(y, n, n, ws.fft), ws);
    }

}
