#include <cuda.h>
#include <bits/stdc++.h>

#define MAXIMUM_DIGITS 1
#define ARRAY_SIZE 10 
#define NUM_OF_PROCESSORS 4
#define NUM_OF_BUCKETS 10

__device__ int getBucket(int d_num, int d_level, int d_maxLevel=MAXIMUM_DIGITS)
{
    int i, powerOfTen=1;
    for(i=0; i<d_maxLevel-d_level-1; i++)
    {
        powerOfTen *= 10;
    }
    return (d_num/powerOfTen)%10;
}

__global__ void buildLocalHistogram(int *d_localHistogram, int *d_arr, int d_size, int d_level, int d_numOfBuckets, int d_numOfProcessors)
{
    int id =  blockIdx.x*blockDim.x+threadIdx.x;
    int i = id;
    while(i<d_size)
    {
        *(d_localHistogram + id*d_numOfBuckets + getBucket(d_arr[i], d_level)) += 1;
        i += d_numOfProcessors;
    }
}

__global__ void buildHistogram(int *d_histogram,int *d_localHistogram, int d_numOfBuckets, int d_numOfProcessors)
{
    int id = blockIdx.x*blockDim.x+threadIdx.x;

    for(int i=0;i<d_numOfProcessors;i++)
    {
        d_histogram[id] += *(d_localHistogram + i*d_numOfBuckets + id);
    }    
}

__global__ void prefixSum(int *d_out, int *d_in, int n)
{
    extern __shared__ float temp[];
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int offset = 1;

    temp[2*id] = d_in[2*id]; 
    temp[2*id+1] = d_in[2*id+1];

    for (int d=n>>1; d>0; d >>= 1)                   
    { 
        __syncthreads();
        if (id < d)
        {
            int ai = offset*(2*id+1)-1;
            int bi = offset*(2*id+2)-1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    if (id == 0) 
    { 
        temp[n-1] = 0; 
    }
    
    for (int d=1; d<n; d*= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (id < d)                     
        {
            int ai = offset*(2*id+1)-1;
            int bi = offset*(2*id+2)-1;
         
           
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t; 
        }
    }
    __syncthreads();
    d_out[2*id] = temp[2*id];
    d_out[2*id+1] = temp[2*id+1]; 
}

void paradis(int *h_arr, int size, int level, int numOfBuckets, int numOfProcessors)
{
    int *d_arr;
    //Allocate memory for array on device
    cudaMalloc((void **)&d_arr, size*sizeof(int));
    //Copy array from host to device
    cudaMemcpy((void *)d_arr, (void *)h_arr, size*sizeof(int), cudaMemcpyHostToDevice);
    
    //
    int *h_histogram, *h_localHistogram, *h_gh;   
    h_histogram = (int *)malloc(sizeof(int)*numOfBuckets);
    // h_localHistogram = (int *)malloc(sizeof(int)*numOfBuckets*numOfProcessors);
    h_gh = (int *)malloc(sizeof(int)*numOfBuckets);

    int *d_histogram,  *d_localHistogram, *d_gh, *d_gt;
 
    cudaMalloc((void **)&d_histogram, numOfBuckets*sizeof(int));
    cudaMalloc((void **)&d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int));
    cudaMalloc((void **)&d_gh, numOfBuckets*sizeof(int));
    cudaMalloc((void **)&d_gt, numOfBuckets*sizeof(int));
    
    cudaMemset(d_histogram, 0, numOfBuckets*sizeof(int));
    cudaMemset(d_localHistogram, 0, numOfBuckets*numOfProcessors*sizeof(int));

    buildLocalHistogram<<<1, numOfProcessors>>>(d_localHistogram, d_arr, size, level, numOfBuckets, numOfProcessors);
    buildHistogram<<<1, numOfBuckets>>>(d_histogram, d_localHistogram, numOfBuckets, numOfProcessors);

    prefixSum<<1, numOfBuckets>>(d_gt, d_histogram, numOfBuckets);    


    cudaMemcpy(h_gh, d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
    // cudaMemcpy(h_localHistogram, d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_histogram, d_histogram, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost); 

    for(int i=0;i<numOfBuckets;i++)
    {
        printf("%d: hist=%d cumu=%d\n", i, h_histogram[i], h_gh[i]);
    }

    // for(int i=0;i<numOfProcessors;i++)
    // {
    //     for(int j=0;j<numOfBuckets;j++)
    //     {
    //         printf("%d\t", *(h_localHistogram + i*numOfBuckets + j));    
    //     }    
    //     printf("\n");
    // }
    

    //Copy sorted array from device to host
    cudaMemcpy((void *)h_arr, (void *)d_arr, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

int main()
{
    int *h_arr;
    int i;

    //Allocate memory for array on host                
    h_arr = (int *)malloc(ARRAY_SIZE*sizeof(int));

    //Initialize array elements with random values    
    for(i=0; i<ARRAY_SIZE; i++)
    {
        h_arr[i] = abs(rand()%((int)pow(10, MAXIMUM_DIGITS)));
        printf("%d ", h_arr[i]);    
    }

    //Call sort function
    paradis(h_arr, ARRAY_SIZE, 0, NUM_OF_BUCKETS, NUM_OF_PROCESSORS);

    //Print the sorted array
    printf("\nSorted Array :\n");    
    for(i=0; i<ARRAY_SIZE; i++)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    //Free memory
    free(h_arr);

    return 0;
}
