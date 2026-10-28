#include "snippets.cuh"
#include "c22dist.cuh"
#include <algorithm>
#include <limits>

static const int AREA_THREADS = 256;

__device__ inline float c22Distance(const float *a, const float *b)
{
    float sumSq = 0.0f;
    for (int f = 0; f < C22_NUM_FEATURES; ++f)
    {
        float d = a[f] - b[f];
        sumSq += d * d;
    }
    return sqrtf(sumSq);
}

__global__ void candidateAreas(const float *d_features, int numWindows, int candidateStep, int firstCandidate,
                               const float *d_curve, double *d_areas)
{
    __shared__ double partial[AREA_THREADS];

    int candidate = firstCandidate + blockIdx.x;
    const float *src = d_features + (size_t)candidate * candidateStep * C22_NUM_FEATURES;
    float a[C22_NUM_FEATURES];
    for (int f = 0; f < C22_NUM_FEATURES; ++f)
        a[f] = src[f];

    float sum = 0.0f;
    for (int j = threadIdx.x; j < numWindows; j += blockDim.x)
    {
        float d = c22Distance(a, d_features + (size_t)j * C22_NUM_FEATURES);
        float c = d_curve[j];
        sum += (d < c) ? d : c;
    }
    partial[threadIdx.x] = (double)sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            partial[threadIdx.x] += partial[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_areas[candidate] = partial[0];
}

__global__ void applySnippet(const float *d_features, int numWindows, int snippetWindow, int round,
                             float *d_curve, int *d_labels, float *d_profile)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numWindows)
        return;

    float d = c22Distance(d_features + (size_t)snippetWindow * C22_NUM_FEATURES,
                          d_features + (size_t)j * C22_NUM_FEATURES);
    d_profile[j] = d;
    if (d < d_curve[j])
    {
        d_curve[j] = d;
        d_labels[j] = round;
    }
}

SnippetSelection selectSnippets(const float *d_seriesFeatures, int numWindows, int candidateStep, int numSnippets)
{
    int numCandidates = (numWindows - 1) / candidateStep + 1;

    SnippetSelection result;
    result.snippets.resize(numSnippets);
    result.labels.resize(numWindows);
    result.fracs.resize(numSnippets);
    result.profiles.resize((size_t)numSnippets * numWindows);

    float *d_curve, *d_profile;
    int *d_labels;
    double *d_areas;
    cudaMalloc(&d_curve, (size_t)numWindows * sizeof(float));
    cudaMalloc(&d_profile, (size_t)numWindows * sizeof(float));
    cudaMalloc(&d_labels, (size_t)numWindows * sizeof(int));
    cudaMalloc(&d_areas, (size_t)numCandidates * sizeof(double));

    std::vector<float> initialCurve(numWindows, std::numeric_limits<float>::infinity());
    cudaMemcpy(d_curve, initialCurve.data(), (size_t)numWindows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_labels, 0, (size_t)numWindows * sizeof(int));

    const long long pairsPerLaunch = 1LL << 30;
    int batch = (int)std::max(1LL, std::min((long long)numCandidates, pairsPerLaunch / numWindows));

    std::vector<double> areas(numCandidates);
    for (int round = 0; round < numSnippets; round++)
    {
        for (int first = 0; first < numCandidates; first += batch)
        {
            int blocks = std::min(batch, numCandidates - first);
            candidateAreas<<<blocks, AREA_THREADS>>>(d_seriesFeatures, numWindows, candidateStep, first, d_curve, d_areas);
        }
        cudaMemcpy(areas.data(), d_areas, (size_t)numCandidates * sizeof(double), cudaMemcpyDeviceToHost);

        int best = 0;
        double bestArea = std::numeric_limits<double>::infinity();
        for (int c = 0; c < numCandidates; c++)
        {
            if (areas[c] < bestArea)
            {
                bestArea = areas[c];
                best = c;
            }
        }
        result.snippets[round] = best;

        int threads = 256;
        applySnippet<<<(numWindows + threads - 1) / threads, threads>>>(d_seriesFeatures, numWindows, best * candidateStep, round,
                                                                    d_curve, d_labels, d_profile);
        cudaMemcpy(result.profiles.data() + (size_t)round * numWindows, d_profile,
                   (size_t)numWindows * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaMemcpy(result.labels.data(), d_labels, (size_t)numWindows * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_curve);
    cudaFree(d_profile);
    cudaFree(d_labels);
    cudaFree(d_areas);

    for (int snippet = 0; snippet < numSnippets; snippet++)
    {
        float count = 0;
        for (int j = 0; j < numWindows; j++)
        {
            if (result.labels[j] == snippet)
                count++;
        }
        result.fracs[snippet] = count / numWindows;
    }

    return result;
}
