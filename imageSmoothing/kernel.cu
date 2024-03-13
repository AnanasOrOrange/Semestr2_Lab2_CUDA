#include "cuda_runtime.h"					
#include "device_launch_parameters.h"		
#include <iostream>
#include <chrono>

#define M 1920
#define N 1080

#define BLOCK_SIZE 16

//#define OUT

#define CHECK_ERR()														
void check() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\M", __FILE__, __LINE__, cudaGetErrorString(err));
        system("pause");
        exit(1);
    }
}

struct Pixel {
    int r = 0;
    int g = 0;
    int b = 0;
};

bool isBorer2d(int i, int j) {
    return i == 0 || i == M || j == 0 || j == N;
}

void CPU_AoS(Pixel** AoS, Pixel** smAoS) {
    for (int i = 1; i < M - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            smAoS[i][j].r = (AoS[i + 1][j].r + AoS[i + 1][j + 1].r + AoS[i][j + 1].r + AoS[i - 1][j + 1].r + AoS[i - 1][j].r + AoS[i - 1][j - 1].r + AoS[i][j - 1].r + AoS[i + 1][j - 1].r) / 8;
            smAoS[i][j].g = (AoS[i + 1][j].g + AoS[i + 1][j + 1].g + AoS[i][j + 1].g + AoS[i - 1][j + 1].g + AoS[i - 1][j].g + AoS[i - 1][j - 1].g + AoS[i][j - 1].g + AoS[i + 1][j - 1].g) / 8;
            smAoS[i][j].b = (AoS[i + 1][j].b + AoS[i + 1][j + 1].b + AoS[i][j + 1].b + AoS[i - 1][j + 1].b + AoS[i - 1][j].b + AoS[i - 1][j - 1].b + AoS[i][j - 1].b + AoS[i + 1][j - 1].b) / 8;
        }
    }
    // borders
    for (int j = 0; j < N; j++) {
        smAoS[0][j].r = AoS[0][j].r;
        smAoS[0][j].g = AoS[0][j].g;
        smAoS[0][j].b = AoS[0][j].b;

        smAoS[M-1][j].r = AoS[M-1][j].r;
        smAoS[M-1][j].g = AoS[M-1][j].g;
        smAoS[M-1][j].b = AoS[M-1][j].b;
    }
    for (int i = 1; i < M - 1; i++) {
        smAoS[i][0].r = AoS[i][0].r;
        smAoS[i][0].g = AoS[i][0].g;
        smAoS[i][0].b = AoS[i][0].b;

        smAoS[i][N-1].r = AoS[i][N-1].r;
        smAoS[i][N-1].g = AoS[i][N-1].g;
        smAoS[i][N-1].b = AoS[i][N-1].b;
    }

}

void CPU_SoA(int** r, int** g, int** b, int** smr, int** smg, int** smb) {
    for (int i = 1; i < M - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            smr[i][j] = (r[i + 1][j] + r[i + 1][j + 1] + r[i][j + 1] + r[i - 1][j + 1] + r[i - 1][j] + r[i - 1][j - 1] + r[i][j - 1] + r[i + 1][j - 1]) / 8;
            smg[i][j] = (g[i + 1][j] + g[i + 1][j + 1] + g[i][j + 1] + g[i - 1][j + 1] + g[i - 1][j] + g[i - 1][j - 1] + g[i][j - 1] + g[i + 1][j - 1]) / 8;
            smb[i][j]=  (b[i + 1][j] + b[i + 1][j + 1] + b[i][j + 1] + b[i - 1][j + 1] + b[i - 1][j] + b[i - 1][j - 1] + b[i][j - 1] + b[i + 1][j - 1]) / 8;
        }
    }
    // borders
    for (int j = 0; j < N; j++) {
        smr[0][j] = r[0][j];
        smg[0][j] = g[0][j];
        smb[0][j] = b[0][j];

        smr[M-1][j] = r[M-1][j];
        smg[M-1][j] = g[M-1][j];
        smb[M-1][j] = b[M-1][j];
    }
    for (int i = 1; i < M - 1; i++) {
        smr[i][0] = r[i][0];
        smg[i][0] = g[i][0];
        smb[i][0] = b[i][0];

        smr[i][N-1] = r[i][N-1];
        smg[i][N-1] = g[i][N-1];
        smb[i][N-1] = b[i][N-1];
    }
}

bool isEquel(Pixel** AoS, int** r, int** g, int** b) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (r[i][j] != AoS[i][j].r || g[i][j] != AoS[i][j].g || b[i][j] != AoS[i][j].b) {
                return false;
            }
        }
    }
    return true;
}

bool isEquelAoS(Pixel** AoS_1, Pixel** AoS_2) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (AoS_1[i][j].r != AoS_2[i][j].r || AoS_1[i][j].g != AoS_2[i][j].g || AoS_1[i][j].b != AoS_2[i][j].b) {
                return false;
            }
        }
    }
    return true;
}

bool isEquelSoA(int** r1, int** g1, int** b1, int** r2, int** g2, int** b2) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (r1[i][j] != r2[i][j] || g1[i][j] != g2[i][j] || b1[i][j] != b2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

#ifdef OUT
void printAoS(Pixel** AoS) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << AoS[i][j].r << "," << AoS[i][j].g << "," << AoS[i][j].b << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printSoA(int** r, int** g, int** b) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << r[i][j] << "," << g[i][j] << "," << b[i][j] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
#endif

void copyAoS_2dTo1d(Pixel** AoS, Pixel* AoS_1d) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            AoS_1d[i * N + j].r = AoS[i][j].r;
            AoS_1d[i * N + j].g = AoS[i][j].g;
            AoS_1d[i * N + j].b = AoS[i][j].b;
        }
    }
}

void copyAoS_1dTo2d(Pixel** AoS, Pixel* AoS_1d) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            AoS[i][j].r = AoS_1d[i * N + j].r;
            AoS[i][j].g = AoS_1d[i * N + j].g;
            AoS[i][j].b = AoS_1d[i * N + j].b;
        }
    }
}

void copySoA_2dTo1d(int** r, int** g, int** b, int* r_1d, int* g_1d, int* b_1d) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            r_1d[i * N + j] = r[i][j];
            g_1d[i * N + j] = g[i][j];
            b_1d[i * N + j] = b[i][j];
        }
    }
}

void copySoA_1dTo2d(int** r, int** g, int** b, int* r_1d, int* g_1d, int* b_1d) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
             r[i][j] = r_1d[i * N + j];
             g[i][j] = g_1d[i * N + j];
             b[i][j] = b_1d[i * N + j];
        }
    }
}

__global__
void GPU_AoS(Pixel* AoS, Pixel* smAoS) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i == 0 || i == M - 1 || j == 0 || j == N - 1) && (i < M && j < N)) {
        smAoS[i * N + j].r = AoS[i * N + j].r;
        smAoS[i * N + j].g = AoS[i * N + j].g;
        smAoS[i * N + j].b = AoS[i * N + j].b;
    }
    else if (i < M - 1 && j < N - 1) {
        smAoS[i * N + j].r = (AoS[(i + 1) * N + j].r + AoS[(i + 1) * N + j + 1].r + AoS[i * N + j + 1].r + AoS[(i - 1) * N + j + 1].r + AoS[(i - 1) * N + j].r + AoS[(i - 1) * N + j - 1].r + AoS[i * N + j - 1].r + AoS[(i + 1) * N + j - 1].r) / 8;
        smAoS[i * N + j].g = (AoS[(i + 1) * N + j].g + AoS[(i + 1) * N + j + 1].g + AoS[i * N + j + 1].g + AoS[(i - 1) * N + j + 1].g + AoS[(i - 1) * N + j].g + AoS[(i - 1) * N + j - 1].g + AoS[i * N + j - 1].g + AoS[(i + 1) * N + j - 1].g) / 8;
        smAoS[i * N + j].b = (AoS[(i + 1) * N + j].b + AoS[(i + 1) * N + j + 1].b + AoS[i * N + j + 1].b + AoS[(i - 1) * N + j + 1].b + AoS[(i - 1) * N + j].b + AoS[(i - 1) * N + j - 1].b + AoS[i * N + j - 1].b + AoS[(i + 1) * N + j - 1].b) / 8;
    }
}

__global__
void GPU_SoA(int* r, int* g, int* b, int* smr, int* smg, int* smb) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i == 0 || i == M - 1 || j == 0 || j == N - 1) && (i < M && j < N)) {
        smr[i * N + j] = r[i * N + j];
        smg[i * N + j] = g[i * N + j];
        smb[i * N + j] = b[i * N + j];
    }
    else if (i < M - 1 && j < N - 1) {
        smr[i * N + j] = (r[(i + 1) * N + j] + r[(i + 1) * N + j + 1] + r[i * N + j + 1] + r[(i - 1) * N + j + 1] + r[(i - 1) * N + j] + r[(i - 1) * N + j - 1] + r[i * N + j - 1] + r[(i + 1) * N + j - 1]) / 8;
        smg[i * N + j] = (g[(i + 1) * N + j] + g[(i + 1) * N + j + 1] + g[i * N + j + 1] + g[(i - 1) * N + j + 1] + g[(i - 1) * N + j] + g[(i - 1) * N + j - 1] + g[i * N + j - 1] + g[(i + 1) * N + j - 1]) / 8;
        smb[i * N + j] = (b[(i + 1) * N + j] + b[(i + 1) * N + j + 1] + b[i * N + j + 1] + b[(i - 1) * N + j + 1] + b[(i - 1) * N + j] + b[(i - 1) * N + j - 1] + b[i * N + j - 1] + b[(i + 1) * N + j - 1]) / 8;
    }
}

__global__  // леть r и smr переименововать
void Stream_SoA(int* r, int* smr) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ((i == 0 || i == M - 1 || j == 0 || j == N - 1) && (i < M && j < N)) {
        smr[i * N + j] = r[i * N + j];
    }
    else if (i < M - 1 && j < N - 1) {
        smr[i * N + j] = (r[(i + 1) * N + j] + r[(i + 1) * N + j + 1] + r[i * N + j + 1] + r[(i - 1) * N + j + 1] + r[(i - 1) * N + j] + r[(i - 1) * N + j - 1] + r[i * N + j - 1] + r[(i + 1) * N + j - 1]) / 8;
    }
}

int main()
{
    // Array of structures

    Pixel** AoS = new Pixel * [M];
    for (int i = 0; i < M; i++) {
        AoS[i] = new Pixel[N];
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            AoS[i][j].r = rand() % 256;
            AoS[i][j].g = rand() % 256;
            AoS[i][j].b = rand() % 256;
        }
    }

    // for checking result
    Pixel** chAoS = new Pixel * [M];
    for (int i = 0; i < M; i++) {
        chAoS[i] = new Pixel[N];
    }


    // Structure of arrays

    int** Red = new int* [M];
    int** Green = new int* [M];
    int** Blue = new int* [M];

    for (int i = 0; i < M; i++) {
        Red[i] = new int[N];
        Green[i] = new int[N];
        Blue[i] = new int[N];
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            Red[i][j] = AoS[i][j].r;
            Green[i][j] = AoS[i][j].g;
            Blue[i][j] = AoS[i][j].b;
        }
    }

    // for checking result
    int** chRed = new int* [M];
    int** chGreen = new int* [M];
    int** chBlue = new int* [M];

    for (int i = 0; i < M; i++) {
        chRed[i] = new int[N];
        chGreen[i] = new int[N];
        chBlue[i] = new int[N];
    }

#ifdef OUT
    std::cout << "The initial image:" << std::endl;
    printAoS(AoS);
    printSoA(Red, Green, Blue);
    std::cout << std::endl;
#endif

    // CPU both

    auto startCPU = std::chrono::steady_clock::now();
    CPU_AoS(AoS, chAoS);
    auto endCPU = std::chrono::steady_clock::now();
    std::cout << "CPU AoS time\t= " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << "\tmillisec, (1e-3 sec)" << std::endl;

    startCPU = std::chrono::steady_clock::now();
    CPU_SoA(Red, Green, Blue, chRed, chGreen, chBlue);
    endCPU = std::chrono::steady_clock::now();
    std::cout << "CPU SaO time\t= " << std::chrono::duration_cast<std::chrono::milliseconds>(endCPU - startCPU).count() << "\tmillisec, (1e-3 sec)" << std::endl;

#ifdef OUT
    std::cout << "After CPU smoothing:" << std::endl;
    printAoS(chAoS);
    printSoA(chRed, chGreen, chBlue);
    std::cout << std::endl;
#endif

    // CUDA AoS

    Pixel** resAoS = new Pixel * [M];
    for (int i = 0; i < M; i++) {
        resAoS[i] = new Pixel[N];
    }

    cudaEvent_t startGPU, stopGPU;

    Pixel* hostAoS = new Pixel[M * N];
    Pixel* devAoS, *devSmAoS;

    cudaMalloc((void**)&devAoS, M * N * sizeof(Pixel)); CHECK_ERR();
    cudaMalloc((void**)&devSmAoS, M * N * sizeof(Pixel)); CHECK_ERR();

    copyAoS_2dTo1d(AoS, hostAoS);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / BLOCK_SIZE + 1, M / BLOCK_SIZE + 1);

    cudaEventCreate(&startGPU);
    cudaEventCreate(&stopGPU);
    cudaEventRecord(startGPU);

    cudaMemcpy(devAoS, hostAoS, M * N * sizeof(Pixel), cudaMemcpyHostToDevice); CHECK_ERR();

    GPU_AoS << < dimGrid, dimBlock >> > (devAoS, devSmAoS); CHECK_ERR();

    cudaMemcpy(hostAoS, devSmAoS, M* N * sizeof(Pixel), cudaMemcpyDeviceToHost); CHECK_ERR();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    float timeCUDA = 0;
    cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);
    
    copyAoS_1dTo2d(resAoS, hostAoS);

    std::cout << "GPU AoS time\t= " << timeCUDA << "\tmillisec, (1e-3 sec)" << std::endl;


    // CUDA SoA

    int** resRed = new int* [M];
    int** resGreen = new int* [M];
    int** resBlue = new int* [M];

    for (int i = 0; i < M; i++) {
        resRed[i] = new int[N];
        resGreen[i] = new int[N];
        resBlue[i] = new int[N];
    }

    int* hostRed = new int[M * N];
    int* hostGreen = new int[M * N];
    int* hostBlue = new int[M * N];

    int* devRed;
    int* devGreen;
    int* devBlue;

    int* devSmRed;
    int* devSmGreen;
    int* devSmBlue;

    cudaMalloc((void**)&devRed, M * N * sizeof(int)); CHECK_ERR();
    cudaMalloc((void**)&devGreen, M * N * sizeof(int)); CHECK_ERR();
    cudaMalloc((void**)&devBlue, M * N * sizeof(int)); CHECK_ERR();

    cudaMalloc((void**)&devSmRed, M* N * sizeof(int)); CHECK_ERR();
    cudaMalloc((void**)&devSmGreen, M* N * sizeof(int)); CHECK_ERR();
    cudaMalloc((void**)&devSmBlue, M* N * sizeof(int)); CHECK_ERR();

    copySoA_2dTo1d(Red, Green, Blue, hostRed, hostGreen, hostBlue);

    cudaEventRecord(startGPU);

    cudaMemcpy(devRed, hostRed, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
    cudaMemcpy(devGreen, hostGreen, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();
    cudaMemcpy(devBlue, hostBlue, M * N * sizeof(int), cudaMemcpyHostToDevice); CHECK_ERR();


    GPU_SoA << < dimGrid, dimBlock >> > (devRed, devGreen, devBlue, devSmRed, devSmGreen, devSmBlue); CHECK_ERR();

    cudaMemcpy(hostRed, devSmRed, M * N * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();
    cudaMemcpy(hostGreen, devSmGreen, M * N * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();
    cudaMemcpy(hostBlue, devSmBlue, M * N * sizeof(int), cudaMemcpyDeviceToHost); CHECK_ERR();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    timeCUDA = 0;
    cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

    copySoA_1dTo2d(resRed, resGreen, resBlue, hostRed, hostGreen, hostBlue);

    std::cout << "GPU SoA time\t= " << timeCUDA << "\tmillisec, (1e-3 sec)" << std::endl;

    if (!isEquel(resAoS, resRed, resGreen, resBlue)) {
        std::cout << "GPU AoS != GPU SoA" << std::endl;
    }
    if (!isEquelAoS(resAoS, chAoS)) {
        std::cout << "CPU AoS != GPU AoS" << std::endl;
    }
    if (!isEquelSoA(resRed, resGreen, resBlue, chRed, chGreen, chBlue)) {
        std::cout << "CPU SoA != GPU SoA" << std::endl;
    }

#ifdef OUT
    std::cout << "After GPU smoothing:" << std::endl;
    printAoS(resAoS);
    printSoA(resRed, resGreen, resBlue);
    std::cout << std::endl;
#endif

    // CUDA Stream (SoA)

    copySoA_2dTo1d(Red, Green, Blue, hostRed, hostGreen, hostBlue);

    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    cudaEventRecord(startGPU);

    cudaMemcpyAsync(devRed, hostRed, M * N * sizeof(int), cudaMemcpyHostToDevice, stream1); CHECK_ERR();
    cudaMemcpyAsync(devGreen, hostGreen, M * N * sizeof(int), cudaMemcpyHostToDevice, stream2); CHECK_ERR();
    cudaMemcpyAsync(devBlue, hostBlue, M * N * sizeof(int), cudaMemcpyHostToDevice, stream3); CHECK_ERR();


    Stream_SoA << < dimGrid, dimBlock, 0, stream1 >> > (devRed, devSmRed); CHECK_ERR();
    Stream_SoA << < dimGrid, dimBlock, 0, stream2 >> > (devGreen,devSmGreen); CHECK_ERR();
    Stream_SoA << < dimGrid, dimBlock, 0, stream3 >> > (devBlue, devSmBlue); CHECK_ERR();

    cudaMemcpyAsync(hostRed, devSmRed, M * N * sizeof(int), cudaMemcpyDeviceToHost, stream1); CHECK_ERR();
    cudaMemcpyAsync(hostGreen, devSmGreen, M * N * sizeof(int), cudaMemcpyDeviceToHost, stream2); CHECK_ERR();
    cudaMemcpyAsync(hostBlue, devSmBlue, M * N * sizeof(int), cudaMemcpyDeviceToHost, stream3); CHECK_ERR();

    cudaEventRecord(stopGPU);
    cudaEventSynchronize(stopGPU);
    timeCUDA = 0;
    cudaEventElapsedTime(&timeCUDA, startGPU, stopGPU);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    copySoA_1dTo2d(resRed, resGreen, resBlue, hostRed, hostGreen, hostBlue);

    std::cout << "GPU Stream time\t= " << timeCUDA << "\tmillisec, (1e-3 sec)" << std::endl;

    if (!isEquel(resAoS, resRed, resGreen, resBlue)) {
        std::cout << "GPU AoS != GPU SoA" << std::endl;
    }
    if (!isEquelAoS(resAoS, chAoS)) {
        std::cout << "CPU AoS != GPU AoS" << std::endl;
    }
    if (!isEquelSoA(resRed, resGreen, resBlue, chRed, chGreen, chBlue)) {
        std::cout << "CPU SoA != GPU SoA" << std::endl;
    }

#ifdef OUT
    std::cout << "After GPU smoothing:" << std::endl;
    printAoS(resAoS);
    printSoA(resRed, resGreen, resBlue);
    std::cout << std::endl;
#endif
}