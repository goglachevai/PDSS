#include "c22_spline.cuh"
#include <algorithm>
#include <cmath>

namespace
{
    constexpr int kDeg = 3;
    constexpr int kNBreaks = 3;
    constexpr int kNSpline = 4;
    constexpr int kPieces = 2;
    constexpr int kPiecesExt = 8;
    constexpr int kCoefRows = kNSpline * kPiecesExt;

    int iLimit(int x, int lim) { return x < lim ? x : lim; }

    void icumsum(const int *a, int n, int *b)
    {
        b[0] = a[0];
        for (int i = 1; i < n; ++i)
            b[i] = a[i] + b[i - 1];
    }
}

SplineDesign buildSplineDesign(int n)
{
    SplineDesign out;
    out.n = n;

    int breaks[kNBreaks] = {0, (int)std::floor((double)n / 2.0) - 1, n - 1};
    out.breaks1 = breaks[1];

    int h0[2] = {breaks[1] - breaks[0], breaks[2] - breaks[1]};
    int hCopy[4] = {h0[0], h0[1], h0[0], h0[1]};

    int hl[kDeg] = {hCopy[kDeg - 0], hCopy[kDeg - 1], hCopy[kDeg - 2]};
    int hlCS[kDeg];
    icumsum(hl, kDeg, hlCS);
    int bl[kDeg];
    for (int i = 0; i < kDeg; ++i)
        bl[i] = breaks[0] - hlCS[i];

    int hr[kDeg] = {hCopy[0], hCopy[1], hCopy[2]};
    int hrCS[kDeg];
    icumsum(hr, kDeg, hrCS);
    int br[kDeg];
    for (int i = 0; i < kDeg; ++i)
        br[i] = breaks[2] + hrCS[i];

    int breaksExt[3 * kDeg];
    for (int i = 0; i < kDeg; ++i)
    {
        breaksExt[i] = bl[kDeg - 1 - i];
        breaksExt[i + 3] = breaks[i];
        breaksExt[i + 6] = br[i];
    }
    int hExt[kPiecesExt];
    for (int i = 0; i < kPiecesExt; ++i)
        hExt[i] = breaksExt[i + 1] - breaksExt[i];

    double coefs[kCoefRows][kNSpline + 1] = {};
    for (int i = 0; i < kCoefRows; i += kNSpline)
        coefs[i][0] = 1.0;

    int ii[kDeg + 1][kPiecesExt];
    for (int i = 0; i < kPiecesExt; ++i)
        for (int r = 0; r <= kDeg; ++r)
            ii[r][i] = iLimit(r + i, kPiecesExt - 1);

    double H[kCoefRows];
    for (int i = 0; i < kCoefRows; ++i)
    {
        int iiFlat = ii[i % kNSpline][i / kNSpline];
        H[i] = (double)hExt[iiFlat];
    }

    double Q[kNSpline][kPiecesExt];
    for (int k = 1; k < kNSpline; ++k)
    {
        for (int j = 0; j < k; ++j)
            for (int l = 0; l < kCoefRows; ++l)
                coefs[l][j] *= H[l] / (double)(k - j);

        for (int l = 0; l < kCoefRows; ++l)
        {
            double s = 0.0;
            for (int m = 0; m < kNSpline; ++m)
                s += coefs[l][m];
            Q[l % kNSpline][l / kNSpline] = s;
        }

        for (int l = 0; l < kPiecesExt; ++l)
            for (int m = 1; m < kNSpline; ++m)
                Q[m][l] += Q[m - 1][l];

        for (int l = 0; l < kCoefRows; ++l)
        {
            if (l % kNSpline == 0)
                coefs[l][k] = 0.0;
            else
                coefs[l][k] = Q[l % kNSpline - 1][l / kNSpline];
        }

        double fmax[kCoefRows];
        for (int i = 0; i < kPiecesExt; ++i)
            for (int j = 0; j < kNSpline; ++j)
                fmax[i * kNSpline + j] = Q[kNSpline - 1][i];

        for (int j = 0; j <= k; ++j)
            for (int l = 0; l < kCoefRows; ++l)
                coefs[l][j] /= fmax[l];

        for (int i = 0; i < kCoefRows - kDeg; ++i)
            for (int j = 0; j <= k; ++j)
                coefs[i][j] -= coefs[kDeg + i][j];

        for (int i = 0; i < kCoefRows; i += kNSpline)
            coefs[i][k] = 0.0;
    }

    double scale[kCoefRows];
    for (int i = 0; i < kCoefRows; ++i)
        scale[i] = 1.0;
    for (int k = 0; k < kNSpline - 1; ++k)
    {
        for (int i = 0; i < kCoefRows; ++i)
            scale[i] /= H[i];
        for (int i = 0; i < kCoefRows; ++i)
            coefs[i][(kNSpline - 1) - (k + 1)] *= scale[i];
    }

    int jj[kNSpline][kPieces];
    for (int i = 0; i < kNSpline; ++i)
        for (int j = 0; j < kPieces; ++j)
            jj[i][j] = (i == 0) ? kNSpline * (1 + j) : kDeg;
    for (int i = 1; i < kNSpline; ++i)
        for (int j = 0; j < kPieces; ++j)
            jj[i][j] += jj[i - 1][j];

    double coefsOut[kNSpline * kPieces][kNSpline];
    for (int i = 0; i < kNSpline * kPieces; ++i)
    {
        int jjFlat = jj[i % kNSpline][i / kNSpline] - 1;
        for (int j = 0; j < kNSpline; ++j)
            coefsOut[i][j] = coefs[jjFlat][j];
    }

    out.M.assign(8 * 5, 0.0);
    for (int i = 0; i < kNSpline * kNSpline * kPieces; ++i)
    {
        int CRow = i % kNSpline + (i / kNSpline) % 2;
        int CCol = i / kNSpline;
        int coefRow = i % (kNSpline * 2);
        int coefCol = i / (kNSpline * 2);
        out.M[CCol * 5 + CRow] = coefsOut[coefRow][coefCol];
    }

    std::vector<int> xsB((size_t)n * kNSpline), indexB((size_t)n * kNSpline);
    int breakInd = 1;
    for (int i = 0; i < n; ++i)
    {
        if (i >= breaks[breakInd] && breakInd < kNBreaks - 1)
            breakInd += 1;
        for (int j = 0; j < kNSpline; ++j)
        {
            xsB[i * kNSpline + j] = i - breaks[breakInd - 1];
            indexB[i * kNSpline + j] = j + (breakInd - 1) * kNSpline;
        }
    }

    std::vector<double> vB((size_t)n * kNSpline);
    for (size_t i = 0; i < vB.size(); ++i)
        vB[i] = coefsOut[indexB[i]][0];
    for (int i = 1; i < kNSpline; ++i)
        for (size_t j = 0; j < vB.size(); ++j)
            vB[j] = vB[j] * xsB[j] + coefsOut[indexB[j]][i];

    out.A.assign((size_t)n * (kNSpline + 1), 0.0);
    breakInd = 0;
    for (int i = 0; i < n * kNSpline; ++i)
    {
        if (i / kNSpline >= breaks[1])
            breakInd = 1;
        out.A[(i % kNSpline) + breakInd + (i / kNSpline) * (kNSpline + 1)] = vB[i];
    }

    out.AtA.assign(25, 0.0);
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < 5; ++c)
        {
            double s = 0.0;
            for (int i = 0; i < n; ++i)
                s += out.A[i * 5 + r] * out.A[i * 5 + c];
            out.AtA[r * 5 + c] = s;
        }

    return out;
}
