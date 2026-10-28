#include "c22dist.cuh"

struct WindowJob
{
    const float *timeSeries;
    int windowLength;
    const double *splineA, *splineAtA, *splineM;
    int splineBreaks1;
    double *threadPool;
    c22::C22WorkspaceLayout layout;
    float *features;
};

__device__ __noinline__ void computeWindowFeatures(const WindowJob &job, int w)
{
    c22::C22Workspace ws = c22::makeC22Workspace(job.threadPool, job.layout);
    for (int i = 0; i < job.windowLength; ++i) ws.persistN[i] = (double)job.timeSeries[w + i];

    double feat[C22_NUM_FEATURES];
    c22::computeCatch22Vector(job.windowLength, job.splineA, job.splineAtA, job.splineM, job.splineBreaks1, feat, ws);

    float *out = job.features + (size_t)w * C22_NUM_FEATURES;
    for (int f = 0; f < C22_NUM_FEATURES; ++f) out[f] = (float)feat[f];
}

__global__ void computeCatch22Features(
    const float *d_timeSeries, int windowLength, int numWindows,
    const double *splineA, const double *splineAtA, const double *splineM, int splineBreaks1,
    double *d_workspacePool, c22::C22WorkspaceLayout layout,
    int *d_nextWindow, float *d_features)
{
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    WindowJob job = {d_timeSeries, windowLength, splineA, splineAtA, splineM, splineBreaks1,
                     d_workspacePool + (size_t)slot * layout.slotsPerThread, layout, d_features};
    for (;;)
    {
        int w = atomicAdd(d_nextWindow, 1);
        if (w >= numWindows) return;
        computeWindowFeatures(job, w);
    }
}

static const int NORM_THREADS = 256;

__global__ void featureRanges(const float *d_features, int numWindows, float *d_min, float *d_max)
{
    __shared__ float lo[NORM_THREADS], hi[NORM_THREADS];
    int f = blockIdx.x;
    float mn = INFINITY, mx = -INFINITY;
    for (int w = threadIdx.x; w < numWindows; w += blockDim.x)
    {
        float v = d_features[(size_t)w * C22_NUM_FEATURES + f];
        if (isfinite(v))
        {
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
    }
    lo[threadIdx.x] = mn;
    hi[threadIdx.x] = mx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            lo[threadIdx.x] = fminf(lo[threadIdx.x], lo[threadIdx.x + s]);
            hi[threadIdx.x] = fmaxf(hi[threadIdx.x], hi[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        d_min[f] = lo[0];
        d_max[f] = hi[0];
    }
}

__global__ void rescaleFeatures(float *d_features, size_t count, const float *d_min, const float *d_max)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int f = (int)(i % C22_NUM_FEATURES);
    float v = d_features[i];
    if (d_max[f] > d_min[f])
        d_features[i] = (v - d_min[f]) / (d_max[f] - d_min[f]);
    else if (!isnan(v))
        d_features[i] = 0.0f;
}

void normalizeFeatures(float *d_features, int numWindows)
{
    float *d_range;
    cudaMalloc(&d_range, 2 * C22_NUM_FEATURES * sizeof(float));
    featureRanges<<<C22_NUM_FEATURES, NORM_THREADS>>>(d_features, numWindows, d_range, d_range + C22_NUM_FEATURES);

    size_t count = (size_t)numWindows * C22_NUM_FEATURES;
    int threads = 256;
    rescaleFeatures<<<(unsigned)((count + threads - 1) / threads), threads>>>(d_features, count, d_range, d_range + C22_NUM_FEATURES);
    cudaFree(d_range);
}
