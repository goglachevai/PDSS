#pragma once
#include <vector>

struct SnippetSelection
{
    std::vector<int> snippets;
    std::vector<int> labels;
    std::vector<float> fracs;
    std::vector<float> profiles;
};

SnippetSelection selectSnippets(const float *d_seriesFeatures, int numWindows, int candidateStep, int numSnippets);
