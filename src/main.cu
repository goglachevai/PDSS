#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <chrono>

#include "c22dist.cuh"
#include "c22_spline.cuh"
#include "snippets.cuh"

namespace fs = std::filesystem;

static void start_timer(cudaEvent_t *start)
{
    cudaEventCreate(start);
    cudaEventRecord(*start);
}

static float stop_timer(cudaEvent_t *start, cudaEvent_t *stop)
{
    cudaEventCreate(stop);
    cudaEventRecord(*stop);
    cudaEventSynchronize(*stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, *start, *stop);
    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
    return milliseconds;
}

int main(int argc, char *argv[])
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file> <time_series_length> <segment_length> <num_snippets> [candidate_step]" << std::endl;
        std::cerr << "  candidate_step: stride between candidate segment starts (default: segment_length, i.e. non-overlapping)." << std::endl;
        std::cerr << "                  Use a value < segment_length to let candidates overlap." << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputDir = argv[2];
    int timeSeriesLength = std::stoi(argv[3]);
    int segmentLength = std::stoi(argv[4]);
    int numSnippets = std::stoi(argv[5]);
    int candidateStep = (argc == 7) ? std::stoi(argv[6]) : segmentLength;

    if (segmentLength <= 0 || numSnippets <= 0 || timeSeriesLength <= 0 || candidateStep <= 0)
    {
        std::cerr << "Error: segment_length, num_snippets, time_series_length, and candidate_step must be positive integers." << std::endl;
        return 1;
    }
    if (timeSeriesLength < segmentLength)
    {
        std::cerr << "Error: time_series_length must be at least segment_length." << std::endl;
        return 1;
    }

    std::ifstream input(inputFile);
    if (!input.is_open())
    {
        std::cerr << "Error: Could not open input file " << inputFile << std::endl;
        return 1;
    }

    try
    {
        fs::create_directories(outputDir);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: Could not create output directory " << outputDir << ": " << e.what() << std::endl;
        return 1;
    }

    fs::path indiciesFilePath = fs::path(outputDir) / "indicies.txt";
    fs::path snippetsFilePath = fs::path(outputDir) / "snippets.txt";
    fs::path fracsFilePath = fs::path(outputDir) / "fracs.txt";
    fs::path labelsFilePath = fs::path(outputDir) / "labels.txt";
    fs::path profilesFilePath = fs::path(outputDir) / "profiles.txt";

    std::ofstream indiciesFile(indiciesFilePath.string());
    std::ofstream snippetsFile(snippetsFilePath.string());
    std::ofstream fracsFile(fracsFilePath.string());
    std::ofstream labelsFile(labelsFilePath.string());
    std::ofstream profilesFile(profilesFilePath.string());
    if (!indiciesFile.is_open() || !snippetsFile.is_open() || !fracsFile.is_open() ||
        !labelsFile.is_open() || !profilesFile.is_open())
    {
        std::cerr << "Error: Could not open one or more output files in " << outputDir << std::endl;
        return 1;
    }

    std::ofstream logFile("log.txt", std::ios::app);
    if (!logFile.is_open())
    {
        std::cerr << "Error: Could not open log file log.txt" << std::endl;
        return 1;
    }

    float *timeSeriesData = new float[timeSeriesLength];
    int index = 0;
    std::string line;
    while (std::getline(input, line) && index < timeSeriesLength)
    {
        std::istringstream iss(line);
        float value;
        while (iss >> value && index < timeSeriesLength)
        {
            timeSeriesData[index++] = value;
        }
    }

    int numWindows = timeSeriesLength - segmentLength + 1;

    float *d_timeSeriesData;
    cudaMalloc(&d_timeSeriesData, timeSeriesLength * sizeof(float));
    cudaMemcpy(d_timeSeriesData, timeSeriesData, timeSeriesLength * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float total_time = 0, featureTime = 0, selectionTime = 0;

    SplineDesign spline = buildSplineDesign(segmentLength);
    double *d_splineA, *d_splineAtA, *d_splineM;
    cudaMalloc(&d_splineA, spline.A.size() * sizeof(double));
    cudaMalloc(&d_splineAtA, spline.AtA.size() * sizeof(double));
    cudaMalloc(&d_splineM, spline.M.size() * sizeof(double));
    cudaMemcpy(d_splineA, spline.A.data(), spline.A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_splineAtA, spline.AtA.data(), spline.AtA.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_splineM, spline.M.data(), spline.M.size() * sizeof(double), cudaMemcpyHostToDevice);

    float *d_seriesFeatures = nullptr;
    cudaError_t featErr = cudaMalloc(&d_seriesFeatures, (size_t)numWindows * C22_NUM_FEATURES * sizeof(float));
    if (featErr != cudaSuccess)
    {
        std::cerr << "Error: could not allocate the catch22 feature matrix ("
                  << (double)((size_t)numWindows * C22_NUM_FEATURES * sizeof(float)) / (1024.0 * 1024.0 * 1024.0)
                  << " GB requested): " << cudaGetErrorString(featErr) << std::endl;
        return 1;
    }

    const int threads = 128;
    c22::C22WorkspaceLayout wsLayout = c22::computeC22WorkspaceLayout(segmentLength);
    size_t workspaceBytesPerBlock = (size_t)wsLayout.slotsPerThread * sizeof(double) * threads;

    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    int blocksPerProcessor = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerProcessor, computeCatch22Features, threads, 0);
    size_t freeBytes = 0, totalBytes = 0;
    cudaMemGetInfo(&freeBytes, &totalBytes);

    size_t blocksForWindows = ((size_t)numWindows + threads - 1) / threads;
    size_t blocksResident = (size_t)deviceProp.multiProcessorCount * std::max(1, blocksPerProcessor);
    size_t blocksInMemory = (freeBytes / 2) / workspaceBytesPerBlock;
    int blocks = (int)std::max<size_t>(1, std::min({blocksForWindows, blocksResident, blocksInMemory}));

    double *d_workspacePool = nullptr;
    cudaError_t wsErr = cudaMalloc(&d_workspacePool, (size_t)blocks * workspaceBytesPerBlock);
    if (wsErr != cudaSuccess)
    {
        std::cerr << "Error: could not allocate the catch22 workspace pool ("
                  << (double)((size_t)blocks * workspaceBytesPerBlock) / (1024.0 * 1024.0 * 1024.0)
                  << " GB requested): " << cudaGetErrorString(wsErr) << std::endl;
        return 1;
    }

    int *d_nextWindow;
    cudaMalloc(&d_nextWindow, sizeof(int));
    cudaMemset(d_nextWindow, 0, sizeof(int));

    start_timer(&start);
    computeCatch22Features<<<blocks, threads>>>(d_timeSeriesData, segmentLength, numWindows,
                                                d_splineA, d_splineAtA, d_splineM, spline.breaks1,
                                                d_workspacePool, wsLayout,
                                                d_nextWindow, d_seriesFeatures);
    normalizeFeatures(d_seriesFeatures, numWindows);
    featureTime = stop_timer(&start, &stop);
    total_time += featureTime;

    cudaFree(d_nextWindow);
    cudaFree(d_workspacePool);
    cudaFree(d_splineA);
    cudaFree(d_splineAtA);
    cudaFree(d_splineM);
    cudaFree(d_timeSeriesData);

    start_timer(&start);
    SnippetSelection selection = selectSnippets(d_seriesFeatures, numWindows, candidateStep, numSnippets);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    selectionTime = stop_timer(&start, &stop);
    total_time += selectionTime;

    cudaFree(d_seriesFeatures);

    for (int i = 0; i < numSnippets; i++)
    {
        int snippetStart = selection.snippets[i] * candidateStep;
        indiciesFile << snippetStart << " ";
        fracsFile << selection.fracs[i] << " ";
        for (int j = 0; j < numWindows; j++)
            profilesFile << selection.profiles[(size_t)i * numWindows + j] << " ";
        profilesFile << std::endl;
        for (int j = 0; j < segmentLength; j++)
            snippetsFile << timeSeriesData[snippetStart + j] << " ";
        snippetsFile << std::endl;
    }

    for (int j = 0; j < numWindows; j++)
        labelsFile << selection.labels[j] << " ";
    labelsFile << std::endl;

    logFile << inputFile << " " << total_time << std::endl;

    std::cout << "PDSS phases: features " << featureTime << " ms, selection " << selectionTime << " ms" << std::endl;
    std::cout << "PDSS total runtime: " << total_time << " ms" << std::endl;

    delete[] timeSeriesData;

    input.close();
    logFile.close();
    indiciesFile.close();
    snippetsFile.close();
    fracsFile.close();
    labelsFile.close();
    profilesFile.close();

    return 0;
}
