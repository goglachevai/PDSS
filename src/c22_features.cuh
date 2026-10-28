#pragma once
#include "c22_helpers.cuh"
#include "c22_features_distribution.cuh"
#include "c22_features_hrv.cuh"
#include "c22_features_symbolic.cuh"
#include "c22_features_correlation.cuh"
#include "c22_features_forecasting.cuh"
#include "c22_features_spectral.cuh"
#include "c22_features_scaling.cuh"
#include "c22_features_periodicity.cuh"

#define C22_NUM_FEATURES 22

namespace c22
{
    __host__ __device__ __noinline__ inline void computeCatch22Vector(
        int n,
        const double *splineA, const double *splineAtA, const double *splineM, int splineBreaks1,
        double out[C22_NUM_FEATURES], C22Workspace ws)
    {
        if (n < 10)
        {
            for (int i = 0; i < C22_NUM_FEATURES; ++i)
                out[i] = qnan();
            return;
        }

        double *y = ws.persistN;
        zscore(y, n, y);

        bool hasNaN = false;
        for (int i = 0; i < n; ++i)
            hasNaN |= isNaN(y[i]);
        AcfStats acf = acfStats(y, n, ws.dB, ws.fft);

        out[0] = DN_HistogramMode_5(y, n, ws);
        out[1] = DN_HistogramMode_10(y, n, ws);
        out[2] = DN_OutlierInclude_p_001_mdrmd(y, n, ws);
        out[3] = DN_OutlierInclude_n_001_mdrmd(y, n, ws);
        out[4] = hasNaN ? 0.0 : acf.f1ecac;
        out[5] = hasNaN ? 0.0 : (double)acf.firstMin;
        SP_Summaries_welch_rect_both(y, n, &out[6], &out[7], ws);
        out[8] = FC_LocalSimple_mean3_stderr(y, n, ws);
        out[9] = FC_LocalSimple_mean_tauresrat(y, n, 1, ws, acf.firstZero);
        out[10] = MD_hrv_classic_pnn40(y, n, ws);
        out[11] = SB_BinaryStats_mean_longstretch1(y, n, ws);
        out[12] = SB_BinaryStats_diff_longstretch0(y, n, ws);
        out[13] = SB_MotifThree_quantile_hh(y, n, ws);
        out[14] = CO_HistogramAMI_even_2_5(y, n, ws);
        out[15] = CO_trev_1_num(y, n, ws);
        out[16] = IN_AutoMutualInfoStats_40_gaussian_fmmi(y, n, ws);
        out[17] = transitionMatrixCore(y, n, acf.firstZero, ws);
        out[18] = (double)PD_PeriodicityWang_th0_01(y, n, splineA, splineAtA, splineM, splineBreaks1, ws);
        out[19] = hasNaN ? qnan() : embed2DistCore(y, n, acf.firstZero, ws);
        out[20] = SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1(y, n, ws);
        out[21] = SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1(y, n, ws);
    }

}
