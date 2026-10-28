#pragma once
#include "c22_helpers.cuh"

namespace c22
{
    __host__ __device__ __noinline__ inline double scFluctAnal(const double *y, int n, int lag, bool isDFA, C22Workspace ws)
    {
        for (int i = 0; i < n; ++i)
            if (isNaN(y[i]))
                return qnan();

        double linLow = log(5.0);
        double linHigh = log((double)(n / 2));
        const int nTauSteps = 50;
        double tauStep = (linHigh - linLow) / (double)(nTauSteps - 1);

        int tau[50];
        for (int i = 0; i < nTauSteps; ++i)
            tau[i] = (int)round(exp(linLow + i * tauStep));

        int nTau = nTauSteps;
        for (int i = 0; i < nTauSteps - 1; ++i)
        {
            while (i < nTau - 1 && tau[i] == tau[i + 1])
            {
                for (int j = i + 1; j < nTauSteps - 1; ++j)
                    tau[j] = tau[j + 1];
                nTau -= 1;
            }
        }
        if (nTau < 12)
            return 0.0;

        int sizeCS = n / lag;
        if (sizeCS < 1)
            return 0.0;
        double *yCS = ws.dA;
        yCS[0] = y[0];
        for (int i = 0; i < sizeCS - 1; ++i)
            yCS[i + 1] = yCS[i] + y[(i + 1) * lag];

        double F[50];
        for (int i = 0; i < nTau; ++i)
        {
            int t = tau[i];
            int nBuffer = sizeCS / t;
            double sumx = 0.0, sumx2 = 0.0;
            for (int k = 0; k < t; ++k)
            {
                double xv = (double)(k + 1);
                sumx += xv;
                sumx2 += xv * xv;
            }
            double denom = (double)t * sumx2 - sumx * sumx;
            bool singular = (denom == 0.0);

            double Fi = 0.0;
            for (int j = 0; j < nBuffer; ++j)
            {
                const double *w = yCS + (size_t)j * t;
                double sumxy = 0.0, sumy = 0.0;
                for (int k = 0; k < t; ++k)
                {
                    double xv = (double)(k + 1);
                    sumxy += xv * w[k];
                    sumy += w[k];
                }
                double m = 0.0, b = 0.0;
                if (!singular)
                {
                    m = ((double)t * sumxy - sumx * sumy) / denom;
                    b = (sumy * sumx2 - sumx * sumxy) / denom;
                }

                if (isDFA)
                {
                    for (int k = 0; k < t; ++k)
                    {
                        double r = w[k] - (m * (k + 1) + b);
                        Fi += r * r;
                    }
                }
                else
                {
                    double r = w[0] - (m * 1.0 + b);
                    double mx = r, mn = r;
                    for (int k = 1; k < t; ++k)
                    {
                        r = w[k] - (m * (k + 1) + b);
                        if (r > mx)
                            mx = r;
                        if (r < mn)
                            mn = r;
                    }
                    double d = mx - mn;
                    Fi += d * d;
                }
            }
            F[i] = isDFA ? sqrt(Fi / ((double)nBuffer * t)) : sqrt(Fi / (double)nBuffer);
        }

        double logtt[50], logFF[50];
        int ntt = nTau;
        for (int i = 0; i < nTau; ++i)
        {
            logtt[i] = log((double)tau[i]);
            logFF[i] = log(F[i]);
        }

        const int minPoints = 6;
        int nsserr = ntt - 2 * minPoints + 1;
        if (nsserr <= 0)
            return 0.0;
        double sserr[50];
        double buffer[50];
        for (int i = minPoints; i < ntt - minPoints + 1; ++i)
        {
            double m1 = 0.0, b1 = 0.0, m2 = 0.0, b2 = 0.0;
            sserr[i - minPoints] = 0.0;
            linreg(i, logtt, logFF, &m1, &b1);
            linreg(ntt - i + 1, logtt + i - 1, logFF + i - 1, &m2, &b2);
            for (int j = 0; j < i; ++j)
                buffer[j] = logtt[j] * m1 + b1 - logFF[j];
            sserr[i - minPoints] += norm2(buffer, i);
            for (int j = 0; j < ntt - i + 1; ++j)
                buffer[j] = logtt[j + i - 1] * m2 + b2 - logFF[j + i - 1];
            sserr[i - minPoints] += norm2(buffer, ntt - i + 1);
        }

        double firstMinInd = 0.0;
        double minimum = minv(sserr, nsserr);
        for (int i = 0; i < nsserr; ++i)
            if (sserr[i] == minimum)
            {
                firstMinInd = (double)(i + minPoints - 1);
                break;
            }

        return (firstMinInd + 1.0) / (double)ntt;
    }

    __host__ __device__ __noinline__ inline double SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(const double *y, int n, C22Workspace ws) { return scFluctAnal(y, n, 2, true, ws); }
    __host__ __device__ __noinline__ inline double SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(const double *y, int n, C22Workspace ws) { return scFluctAnal(y, n, 1, false, ws); }

}
