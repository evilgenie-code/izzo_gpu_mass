#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _CRT_SECURE_NO_WARNINGS

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h> 

#include <time.h>

using namespace std;

double asinh(double x) { return log(x + sqrt(x * x + 1)); };
double acosh(double x) { return log(x + sqrt(x * x - 1)); };

__device__ void vers(const double* V_in, double* Ver_out)
{
    double v_mod = 0;
    int i;

    for (i = 0; i < 3; i++)
    {
        v_mod += V_in[i] * V_in[i];
    }

    double sqrtv_mod = sqrt(v_mod);

    for (i = 0; i < 3; i++)
    {
        Ver_out[i] = V_in[i] / sqrtv_mod;
    }
}
__device__ void vett(const double* vet1, const double* vet2, double* prod)
{
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

void vett_cpu(const double* vet1, const double* vet2, double* prod)
{
    prod[0] = (vet1[1] * vet2[2] - vet1[2] * vet2[1]);
    prod[1] = (vet1[2] * vet2[0] - vet1[0] * vet2[2]);
    prod[2] = (vet1[0] * vet2[1] - vet1[1] * vet2[0]);
}

__device__ double x2tof(const double& x, const double& s, const double& c, const int lw, int N) {

    double am, a, alfa, beta;
    //%Subfunction that evaluates the time of flight as a function of x
    am = s / 2;
    a = am / (1 - x * x);

    if (x < 1)//ellipse
    {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acos(x);
    }

    else  //hyperbola
    {
        alfa = 2 * acosh(x);
        beta = 2 * asinh(sqrt((s - c) / (-2 * a)));
        if (lw) beta = -beta;
    }

    if (a > 0)
    {
        return (a * sqrt(a) * ((alfa - sin(alfa)) - (beta - sin(beta)) + N * 2 * acos(-1.0)));
    }
    else
    {
        return (-a * sqrt(-a) * ((sinh(alfa) - alfa) - (sinh(beta) - beta)));
    }

}

__global__ void lambert(const float* r0, const float* rk, float *dt, int revs, float mu, int *lww, const int branch,  // INPUT,
    float* v1, float* v2)                                                                        // OUTPUT
{

    int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;

    int ThreadIndex = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;

    int tid = blockIndex * blockDim.x * blockDim.y * blockDim.z + ThreadIndex;
	
    double r1[3], r2[3], r2Vers[3];
    double	V, T, r2Mod = 0.0,    // R2 module
        dotProd = 0.0, // dot product
        c,		        // non-dimensional chord
        s,		        // non dimesnional semi-perimeter
        am,		        // minimum energy ellipse semi major axis
        lambda,	        //lambda parameter defined in Battin's Book
        x, x1, x2, y1, y2, xNew = 0, yNew, err = 1, alfa, beta, psi, eta, eta2, sigma1, vr1, vt1, vt2, vr2, R = 0.0;
    int iterate = 0, i, leftbranch = 0, off = 3;
    const double tolerance = 1e-15;
    double ihDum[3], ih[3], dum[3];

    double t = dt[tid];
    int lw = lww[tid];
	
    double a, p, theta;

    for (i = 0; i < 3; i++)
    {
        r1[i] = r0[tid * off + i];
        r2[i] = rk[tid * off + i];
        R += r1[i] * r1[i];
    }

    R = sqrt(R);
    V = sqrt(mu / R);
    T = R / V;

    t /= T;

    for (i = 0; i < 3; i++)
    {
        r1[i] /= R;
        r2[i] /= R;
        r2Mod += r2[i] * r2[i];
    }

    r2Mod = sqrt(r2Mod);

    for (i = 0; i < 3; i++)
        dotProd += (r1[i] * r2[i]);

    theta = acos(dotProd / r2Mod);
    vett(r1, r2, ihDum);

    if (lw < 0) {

        if (ihDum[2] >= 0.0) lw = 0;
        else lw = 1;

    }

    if (lw) theta = 2 * acos(-1.0) - theta;

    c = sqrt(1.0 + r2Mod * (r2Mod - 2.0 * cos(theta)));
    s = (1.0 + r2Mod + c) / 2.0;
    am = s / 2.0;
    lambda = sqrt(r2Mod) * cos(theta / 2.0) / s;

    double inn1, inn2;

    if (revs == 0)
    {
        x1 = log(0.4767);
        x2 = log(1.5233);
        y1 = log(x2tof(-.5233, s, c, lw, revs)) - log(t);
        y2 = log(x2tof(.5233, s, c, lw, revs)) - log(t);

        err = 1;
        iterate = 0;
        // Newton iterations
        while ((err > tolerance) && (y1 != y2))
        {
            iterate++;
            xNew = (x1 * y2 - y1 * x2) / (y2 - y1);
            yNew = log(x2tof(expf(xNew) - 1.0, s, c, lw, revs)) - logf(t);
            x1 = x2;
            y1 = y2;
            x2 = xNew;
            y2 = yNew;
            err = fabs(x1 - xNew);
        }

        x = expf(xNew) - 1;
    }
    else
    {
        if (leftbranch == 1)   // left branch
        {
            inn1 = -0.5234;
            inn2 = -0.2234;
        }
        else			   // right branch
        {
            inn1 = 0.7234;
            inn2 = 0.5234;
        }

        x1 = tan(inn1 * acos(-1.0) / 2);
        x2 = tan(inn2 * acos(-1.0) / 2);
        y1 = x2tof(inn1, s, c, lw, revs) - t;
        y2 = x2tof(inn2, s, c, lw, revs) - t;

        int imax = 30;
        // Newton Iteration
        while ((err > tolerance) && (y1 != y2) && iterate < imax)
        {
            iterate++;
            xNew = (x1 * y2 - y1 * x2) / (y2 - y1);
            yNew = x2tof(atan(xNew) * 2 / acos(-1.0), s, c, lw, revs) - t;
            x1 = x2;
            y1 = y2;
            x2 = xNew;
            y2 = yNew;
            err = abs(x1 - xNew);
        }

        x = atan(xNew) * 2 / acos(-1.0);

        iterate = iterate == imax ? iterate - 1 : iterate;
    }

    a = am / (1 - x * x);		    // solution semimajor axis
    // psi evaluation
    if (x < 1)                         // ellipse
    {
        beta = 2 * asin(sqrt((s - c) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acos(x);
        psi = (alfa - beta) / 2;
        eta2 = 2 * a * pow(sin(psi), 2) / s;
        eta = sqrt(eta2);
    }
    else       // hyperbola
    {
        beta = 2 * asinh(sqrt((c - s) / (2 * a)));
        if (lw) beta = -beta;
        alfa = 2 * acosh(x);
        psi = (alfa - beta) / 2;
        eta2 = -2 * a * pow(sinh(psi), 2) / s;
        eta = sqrt(eta2);
    }

    p = (r2Mod / (am * eta2)) * pow(sin(theta / 2), 2);
    sigma1 = (1 / (eta * sqrt(am))) * (2 * lambda * am - (lambda + x * eta));
    vers(ihDum, ih);

    if (lw)
    {
        for (i = 0; i < 3; i++)
            ih[i] = -ih[i];
    }

    vr1 = sigma1;
    vt1 = sqrt(p);
    vett(ih, r1, dum);

    for (i = 0; i < 3; i++)
        v1[tid * off + i] = vr1 * r1[i] + vt1 * dum[i];

    vt2 = vt1 / r2Mod;
    vr2 = -vr1 + (vt1 - vt2) / tan(theta / 2);

    vers(r2, r2Vers);
    vett(ih, r2Vers, dum);
    for (i = 0; i < 3; i++)
        v2[tid * off + i] = vr2 * r2[i] / r2Mod + vt2 * dum[i];

    for (i = 0; i < 3; i++)
    {
        v1[tid * off + i] *= V;
        v2[tid * off + i] *= V;
    }
}

double** reading_data(char* name_file, double** DATA, char ADD[][1000], int ignore, int& SIZE1, int SIZE2) {

    SIZE1 = 1;
    int i = 0, j;
    DATA = (double**)malloc(SIZE1 * sizeof(double*));

    char line[1000], * tok, * next_token = NULL;

    
        ifstream file(name_file);
        while (file.getline(line, 1000) && ignore) {
            ignore--;
            strcpy(ADD[i], line);
            i++;
        }
        i = SIZE1 - 1;
        do {

            DATA = (double**)realloc(DATA, ++SIZE1 * sizeof(double*));
            DATA[i] = (double*)malloc(SIZE2 * sizeof(double));

            for (char* tok = strtok_s(line, " ", &next_token), j = 0; tok; tok = strtok_s(NULL, " ", &next_token)) {

                DATA[i][j] = atof(tok);
                j++;
                if (j == SIZE2) break;
            }

            i++;
        	//if (i==3000) break;      	

        } while (file.getline(line, 1000));
        file.close();
    

    return DATA;
}

void writing_data(char* name_file, double** DATA, int SIZE1, int SIZE2, char ADD[][1000], int add) {

    int i, j;
    FILE* fileout;
    fileout = fopen(name_file, "w");

    for (i = 0; i < add; i++) fprintf(fileout, "%s\n", ADD[i]);

    for (i = 0; i < SIZE1 - 1; i++) {
        for (j = 0; j < SIZE2; j++)
            fprintf(fileout, "%26.16e", DATA[i][j]);
        fprintf(fileout, "\n");
    }
    fclose(fileout);
}

int div_up(int x, int y)
{
    return (x - 1) / y + 1;
}

int main() {

    cudaSetDevice(1);
    int threads = 384;
    int blocs;
    int leght = 3;
	
    printf("1. start program\n");

    double AU = 1.49597870691e8;
    double fMSun = 1.32712440018e11;             // km^3/sec^2

    double UnitR = AU;
    double UnitV = sqrt(fMSun / UnitR);          // km/sec
    double UnitT = (UnitR / UnitV) / 86400;      // day

    double mu = 1.;							// гравитационная постоянная
	int nrev = 0;							// число витков
    int lw = -1;

    double dv1[3], dv2[3], dV1, dV2;
  
    char name_file[] = { "data1.txt" };
    char name_file2[] = { "data1_izzo_cpu.txt" };
    double** DATA = NULL;
    int i, k, SIZE1, SIZE2 = 29;
    char boof[2][1000];
    double R0[3] = { 0.0,0.0,0.0 };

    printf("2. reading file \n");

    DATA = reading_data(name_file, DATA, boof, 2, SIZE1, SIZE2);

    printf("3. finish reading file\n");
    printf("4. count tasks %i \n", SIZE1);
    printf("5. start calculate \n");

    blocs = div_up(SIZE1, threads);
    const int countTasks = SIZE1;
 
	
    printf("6. Blocs = %i, Threads = %i \n", blocs, threads);

    int sizeCudaVariableBig = (SIZE1 - 1) * 3;
    int sizeCudaVariableSmall = (SIZE1 - 1);
	
    float* r0 = new float[sizeCudaVariableBig];
    float* r1 = new float[sizeCudaVariableBig];
    float* v1 = new float[sizeCudaVariableBig];
    float* v2 = new float[sizeCudaVariableBig];
    float* dt = new float[sizeCudaVariableSmall];
    int* lww = new int[sizeCudaVariableSmall];
    int lenght = 3;
    int off = 3;

    float* dev_r0, * dev_r1, * dev_v1, * dev_v2, * dev_dt;
    int* dev_lw;

    for (int n = 0; n < SIZE1 - 1; n++)
    {
        dt[n] = DATA[n][14] / UnitT;

        vett_cpu(&DATA[n][0], &DATA[n][6], R0);
        if (R0[2] >= 0.0) lww[n] = 0;
        else lww[n] = 1;

        for (int m = 0; m < lenght; m++)
        {
            r0[n * off + m] = DATA[n][m];
            r1[n * off + m] = DATA[n][6 + m];
        }
    }

    cudaMalloc((void**)&dev_r0, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_r1, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_v1, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_v2, sizeCudaVariableBig * sizeof(float));
    cudaMalloc((void**)&dev_dt, sizeCudaVariableSmall * sizeof(float));
    cudaMalloc((void**)&dev_lw, sizeCudaVariableSmall * sizeof(int));

    cudaMemcpy(dev_r0, r0, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r1, r1, sizeCudaVariableBig * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dt, dt, sizeCudaVariableSmall * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_lw, lww, sizeCudaVariableSmall * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    float gpuTime = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
	
    lambert << <blocs, threads >> > (dev_r0, dev_r1, dev_dt, 1, 1., dev_lw, 0, dev_v1, dev_v2);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&gpuTime, start, stop);

    printf("7. Time one on GPU = %16.10e miliseconds\n", (gpuTime / (SIZE1 - 1)));
    printf("7.1. Time on GPU = %16.10e miliseconds\n", gpuTime);
	
    cudaMemcpy(v1, dev_v1, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v2, dev_v2, sizeCudaVariableBig * sizeof(float), cudaMemcpyDeviceToHost);
	
    printf("8. finish calculate \n");

    for (int n = 0; n < SIZE1 - 1; n++)
    {
        for (int m = 0; m < lenght; m++)
        {
            r0[n * off + m] = DATA[n][m];
            r1[n * off + m] = DATA[n][6 + m];
        }
    }

    for (i = 0; i < SIZE1 - 1; i++) {

        dV1 = 0; dV2 = 0;
        for (k = 0; k < 3; k++) {
            dv1[k] = DATA[i][18 + k] - v1[i * off + k];
            dv2[k] = DATA[i][21 + k] - v2[i * off + k];
            dV1 += dv1[k] * dv1[k];
            dV2 += dv2[k] * dv2[k];
        }

        dV1 = sqrt(dV1) * UnitV;
        dV2 = sqrt(dV2) * UnitV;

        DATA[i][SIZE2 - 2] = DATA[i][15] - dV1;
        DATA[i][SIZE2 - 1] = DATA[i][16] - dV2;

        DATA[i][15] = dV1;
        DATA[i][16] = dV2;
        DATA[i][17] = DATA[i][15] + DATA[i][16];

    }

    printf("9. start write data \n");

    writing_data(name_file2, DATA, SIZE1, SIZE2, boof, 2);

    printf("10. finish write data \n");

    cudaFree(dev_r0);
    cudaFree(dev_r1);
    cudaFree(dev_v1);
    cudaFree(dev_v2);
    cudaFree(dev_lw);
    cudaFree(dev_dt);
	

    printf("11. finish program");
	
    return 0;
}