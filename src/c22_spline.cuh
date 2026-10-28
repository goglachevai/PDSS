#pragma once
#include <vector>

struct SplineDesign
{
    int n = 0;
    int breaks1 = 0;
    std::vector<double> A;
    std::vector<double> AtA;
    std::vector<double> M;
};

SplineDesign buildSplineDesign(int n);
