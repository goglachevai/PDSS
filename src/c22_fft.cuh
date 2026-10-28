#pragma once

namespace c22
{
    __host__ __device__ inline void fftInPlace(double *z, int H)
    {
        const double PI = 3.14159265359;
        for (int i = 1, j = 0; i < H; ++i)
        {
            int bit = H >> 1;
            for (; j & bit; bit >>= 1)
                j ^= bit;
            j ^= bit;
            if (i < j)
            {
                double tr = z[2 * i], ti = z[2 * i + 1];
                z[2 * i] = z[2 * j];
                z[2 * i + 1] = z[2 * j + 1];
                z[2 * j] = tr;
                z[2 * j + 1] = ti;
            }
        }
        for (int len = 2; len <= H; len <<= 1)
        {
            int half = len >> 1;
            double ang = -2.0 * PI / (double)len;
            double wlr = cos(ang), wli = sin(ang);
            double wr = 1.0, wi = 0.0;
            for (int k = 0; k < half; ++k)
            {
                for (int s = k; s < H; s += len)
                {
                    int a = 2 * s, b = 2 * (s + half);
                    double tr = z[b] * wr - z[b + 1] * wi;
                    double ti = z[b] * wi + z[b + 1] * wr;
                    z[b] = z[a] - tr;
                    z[b + 1] = z[a + 1] - ti;
                    z[a] += tr;
                    z[a + 1] += ti;
                }
                double t = wr * wlr - wi * wli;
                wi = wr * wli + wi * wlr;
                wr = t;
            }
        }
    }

    __host__ __device__ inline void realFFTInPlace(double *buf, int L)
    {
        const double PI = 3.14159265359;
        int H = L / 2;
        fftInPlace(buf, H);

        double z0r = buf[0], z0i = buf[1];
        buf[0] = z0r + z0i;
        buf[1] = 0.0;
        buf[2 * H] = z0r - z0i;
        buf[2 * H + 1] = 0.0;

        double ang = -2.0 * PI / (double)L;
        double wlr = cos(ang), wli = sin(ang);
        double wr = wlr, wi = wli;
        for (int k = 1; k <= H / 2; ++k)
        {
            double ar = buf[2 * k], ai = buf[2 * k + 1];
            double br = buf[2 * (H - k)], bi = -buf[2 * (H - k) + 1];
            double er = 0.5 * (ar + br), ei = 0.5 * (ai + bi);
            double orr = 0.5 * (ai - bi), oi = -0.5 * (ar - br);
            double tr = wr * orr - wi * oi, ti = wr * oi + wi * orr;
            buf[2 * k] = er + tr;
            buf[2 * k + 1] = ei + ti;
            buf[2 * (H - k)] = er - tr;
            buf[2 * (H - k) + 1] = -(ei - ti);
            double t = wr * wlr - wi * wli;
            wi = wr * wli + wi * wlr;
            wr = t;
        }
    }

    __host__ __device__ inline int acfFFTLength(int n)
    {
        int p = 1;
        while (p < 2 * n)
            p *= 2;
        return p;
    }

    __host__ __device__ inline void autocorrSumsFFT(const double *x, int n, bool subtractMean, double *buf)
    {
        int L = acfFFTLength(n);
        double mu = 0.0;
        if (subtractMean)
        {
            for (int i = 0; i < n; ++i)
                mu += x[i];
            mu /= (double)n;
        }
        for (int i = 0; i < L; ++i)
            buf[i] = (i < n) ? x[i] - mu : 0.0;
        realFFTInPlace(buf, L);

        int H = L / 2;
        for (int k = 0; k <= H; ++k)
        {
            double re = buf[2 * k], im = buf[2 * k + 1];
            buf[k] = re * re + im * im;
        }
        for (int k = 1; k < H; ++k)
            buf[L - k] = buf[k];

        realFFTInPlace(buf, L);
        for (int t = 0; t < n; ++t)
            buf[t] = buf[2 * t] / (double)L;
    }

}
