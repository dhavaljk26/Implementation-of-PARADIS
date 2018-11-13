#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

#define MAXIMUM_DIGITS 1
#define ARRAY_SIZE 10 
#define NUM_OF_PROCESSORS 4
#define NUM_OF_BUCKETS 10

#define TRUE 1
#define FALSE 0

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

__global__ void prefixSum(int *d_out, int*d_out2, int *d_in, int n)
{
    // extern __shared__ float temp[]; 
    // int id = threadIdx.x;
    // int pout = 0, pin = 1;
    
    // temp[pout*n + id] = (id > 0) ? d_in[id-1] : 0;
    // __syncthreads();
    
    // for (int offset=1; offset<n; offset*=2)
    // {
    //     pout = 1 - pout; 
    //     pin = 1 - pout;
    //     if (id >= offset)
    //         temp[pout*n+id] += temp[pin*n+id - offset];
    //     else
    //         temp[pout*n+id] = temp[pin*n+id];
    //     __syncthreads();
    // }

    // d_out[id] = temp[pout*n+id];

    d_out[0] = 0;
    d_out2[0] = d_in[0];
    for(int i=1;i<n;i++)
    {
        d_out[i] = d_out[i-1] + d_in[i-1];
        d_out2[i] = d_out2[i-1] + d_in[i];
    }
    
}

int allBucketsAreEmpty(int *h_gh,int *h_gt, int *d_gh,int *d_gt, int numOfBuckets)
{
    cudaMemcpy(h_gh, d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_gt, d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);  

    for(int i=0;i<numOfBuckets;i++)
    {
        if(h_gh[i]!=h_gt[i])
            return  FALSE;
    }
    return TRUE;
}

__global__ void partitionForPermutation(int *d_ph,int *d_pt,int *d_gh,int *d_gt,int numOfBuckets,int numOfProcessors)
{
    int bucket = blockIdx.y*blockDim.y + threadIdx.y;
    int processor = blockIdx.x*blockDim.x + threadIdx.x;
    int interval = (d_gt[bucket] - d_gh[bucket])/numOfProcessors;

    *(d_ph + processor*numOfBuckets + bucket) = d_gh[bucket] + interval*processor;
    *(d_pt + processor*numOfBuckets + bucket) = d_gh[bucket] + interval*(processor+1);    

    if(processor == numOfProcessors-1)
        *(d_pt + processor*numOfBuckets + bucket) = d_gt[bucket];
}

__global__ void paradisPermute(int *d_arr, int size, int *d_gh, int *d_gt, int *d_ph, int *d_pt, 
                                int d_level, int numOfBuckets, int numOfProcessors)
{
    int p = blockIdx.x*blockDim.x+threadIdx.x;
    int head, v, k, temp;   

    for(int i=0; i<numOfBuckets; i++)
    {
        head = *(d_ph + p*numOfBuckets + i);

        while( head < *(d_pt + p*numOfBuckets + i) )
        {
            v = d_arr[head];
            k = getBucket(v, d_level);
            
            while(k!=i && *(d_ph+ p*numOfBuckets + k) < *(d_pt + p*numOfBuckets + k))
            {
                //swap
                temp = v;
                v = d_arr[*(d_ph + p*numOfBuckets + k)];
                d_arr[*(d_ph + p*numOfBuckets + k)] = temp;

                *(d_ph + p*numOfBuckets + k) += 1;
                k = getBucket(v, d_level);
            }

            if(k == i)
            {
                d_arr[head++] = d_arr[*(d_ph + p*numOfBuckets + i)];
                d_arr[*(d_ph + p*numOfBuckets + i)] = v;
                *(d_ph + p*numOfBuckets + i) += 1;
            }
            else
            {
                d_arr[head++] = v;
            }
        }
    }
} 

__global__ void paradisRepair(int i, int *d_arr, int size, int *d_gh, int *d_gt, int *d_ph, int *d_pt, 
    int d_level, int numOfBuckets, int numOfProcessors)
{
    i = blockIdx.x*blockDim.x+threadIdx.x;

    int head, tail, v, w;   

    tail = d_gt[i];

    for(int p=0; p<numOfProcessors; p++)
    {
        head = *(d_ph + p*numOfBuckets + i);

        while( head < *(d_pt + p*numOfBuckets + i) && head < tail)
        {
            v = d_arr[head++];
            
            if(getBucket(v,d_level) != i)
            {
                while(head < tail)
                {
                    w = d_arr[--tail];
                    if(getBucket(w,d_level) == i)
                    {
                        d_arr[head-1] = w;
                        d_arr[tail] = v;
                        break;
                    }
                }
            }
        }
    }

    d_gh[i] = tail;
}    

__global__ void partitionForRepair(int *d_ph, int *d_pt, int *d_rh, int *d_rt, int numOfBuckets, int numOfProcessors)
{
    int average=0, sum, *C, temp, p;
    C = (int *)malloc(numOfBuckets*sizeof(int));

    for(int j=0; j<numOfBuckets; j++)
    {
        temp = 0;
        for(int i=0; i<numOfProcessors; i++)
        {
            temp += *(d_pt + i*numOfBuckets + j) - *(d_ph + i*numOfBuckets + j);
        }
        
        C[j] = temp;
        average += temp;
    }  
       
    average = average/numOfProcessors;

    sum=0;
    p=0;
    d_rh[p] = 0;

    for(int i=0; i<numOfBuckets; i++)
    {
        sum += C[i];

        if(sum > average)
        {
            d_rt[p++] = i;
            d_rh[p] = i+1;
            sum=0;
        }
    }

    d_rt[p++] = numOfBuckets-1;

    for(; p<numOfProcessors; p++)
    {
        d_rh[p] = d_rt[p] = 0;
    }
}

void paradis(int *h_arr, int size, int level, int numOfBuckets, int numOfProcessors)
{
    printf("--------------------------------------------------------------------------------\n");
    printf("Calling first=%d size=%d level=%d\n", *h_arr, size, level);
    printf("--------\n");
    int *d_arr;
    //Allocate memory for array on device
    cudaMalloc((void **)&d_arr, size*sizeof(int));
    //Copy array from host to device
    cudaMemcpy((void *)d_arr, (void *)h_arr, size*sizeof(int), cudaMemcpyHostToDevice);
    
    //
    int *h_histogram, /**h_localHistogram,*/ *h_gh, *h_gt, *h_ph, *h_pt/*, *h_rh, *h_rt*/;   
    h_histogram = (int *)malloc(sizeof(int)*numOfBuckets);
    // h_localHistogram = (int *)malloc(sizeof(int)*numOfBuckets*numOfProcessors);
    h_gh = (int *)malloc(sizeof(int)*numOfBuckets);
    h_gt = (int *)malloc(sizeof(int)*numOfBuckets);
    h_ph = (int *)malloc(sizeof(int)*numOfBuckets*numOfProcessors);
    h_pt = (int *)malloc(sizeof(int)*numOfBuckets*numOfProcessors);
    // h_rh = (int *)malloc(sizeof(int)*numOfProcessors);
    // h_rt = (int *)malloc(sizeof(int)*numOfProcessors);
    
    int *d_histogram,  *d_localHistogram, *d_gh, *d_gt, *d_ph, *d_pt, *d_rh, *d_rt;
 
    cudaMalloc((void **)&d_histogram, numOfBuckets*sizeof(int));
    cudaMalloc((void **)&d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int));
    cudaMalloc((void **)&d_gh, numOfBuckets*sizeof(int));
    cudaMalloc((void **)&d_gt, numOfBuckets*sizeof(int));
    cudaMalloc((void **)&d_ph, numOfBuckets*numOfProcessors*sizeof(int));
    cudaMalloc((void **)&d_pt, numOfBuckets*numOfProcessors*sizeof(int));
    cudaMalloc((void **)&d_rh, numOfProcessors*sizeof(int));
    cudaMalloc((void **)&d_rt, numOfProcessors*sizeof(int));

    cudaMemset(d_histogram, 0, numOfBuckets*sizeof(int));
    cudaMemset(d_localHistogram, 0, numOfBuckets*numOfProcessors*sizeof(int));

    //STEP 1
    buildLocalHistogram<<<1, numOfProcessors>>>(d_localHistogram, d_arr, size, level, numOfBuckets, numOfProcessors);
    buildHistogram<<<1, numOfBuckets>>>(d_histogram, d_localHistogram, numOfBuckets, numOfProcessors);

    //STEP 2
    prefixSum<<<1, 1>>>(d_gh, d_gt, d_histogram, numOfBuckets);    

    

    //STEP 3
    int numIter = 0;
    while(! allBucketsAreEmpty(h_gh, h_gt, d_gh, d_gt, numOfBuckets))
    {
        numIter++;
        printf("***********************Num Iter = %d**************************\n",numIter);

        dim3 DimGrid(1, 1, 1);
        dim3 DimBlock(numOfProcessors, numOfBuckets, 1);
        partitionForPermutation<<<DimGrid, DimBlock>>>(d_ph, d_pt, d_gh, d_gt, numOfBuckets, numOfProcessors);

        paradisPermute<<<1, numOfProcessors>>>(d_arr, size, d_gh, d_gt, d_ph, d_pt, level, numOfBuckets, numOfProcessors);    

        {
            // TEMP
            cudaMemcpy(h_ph, d_ph, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_pt, d_pt, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
    
            cudaMemcpy(h_gh, d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_gt, d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
    
            // cudaMemcpy(h_localHistogram, d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_histogram, d_histogram, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost); 
    
            printf("\nAfter permute");
            for(int i=0;i<numOfBuckets;i++)
            {
                printf("\n%d: hist=%d c1=%d c2=%d", i, h_histogram[i], h_gh[i], h_gt[i]);
            }
            
            printf("\n");
            for(int i=0;i<numOfProcessors;i++)
            {
                for(int j=0;j<numOfBuckets;j++)
                {
                    printf("[%d,%d]\t", *(h_ph + i*numOfBuckets + j), *(h_pt + i*numOfBuckets + j));    
                }    
                printf("\n");
            }
            cudaMemcpy((void *)h_arr, (void *)d_arr, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

            printf("\nSorted Array after permute at level %d\n:", level);    
            for(int i=0; i<size; i++)
            {
                printf("%d ", h_arr[i]);
            }
            //TEMP
        }
        paradisRepair<<<1, numOfBuckets>>>(0, d_arr, size, d_gh, d_gt, d_ph, d_pt, level, numOfBuckets, numOfProcessors);

        {
            // TEMP
            cudaMemcpy(h_ph, d_ph, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_pt, d_pt, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
    
            cudaMemcpy(h_gh, d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_gt, d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
    
            // cudaMemcpy(h_localHistogram, d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
            cudaMemcpy(h_histogram, d_histogram, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost); 
    
            printf("\nAfter repair\n");
            for(int i=0;i<numOfBuckets;i++)
            {
                printf("\n%d: hist=%d c1=%d c2=%d", i, h_histogram[i], h_gh[i], h_gt[i]);
            }
            
            printf("\n");
            for(int i=0;i<numOfProcessors;i++)
            {
                for(int j=0;j<numOfBuckets;j++)
                {
                    printf("[%d,%d]\t", *(h_ph + i*numOfBuckets + j), *(h_pt + i*numOfBuckets + j));    
                }    
                printf("\n");
            }

            printf("\nSorted Array after repair at level %d\n:", level);    
            for(int i=0; i<size; i++)
            {
                printf("%d ", h_arr[i]);
            }
            //TEMP
        }
    }
    
    printf("Num Iter = %d\n",numIter);

    cudaMemcpy(h_ph, d_ph, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_pt, d_pt, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    

    cudaMemcpy(h_gh, d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_gt, d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);    

    // cudaMemcpy(h_localHistogram, d_localHistogram, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);    
    cudaMemcpy(h_histogram, d_histogram, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost); 

    for(int i=0;i<numOfBuckets;i++)
    {
        printf("\n%d: hist=%d c1=%d c2=%d", i, h_histogram[i], h_gh[i], h_gt[i]);
    }
    
    printf("\n");
    for(int i=0;i<numOfProcessors;i++)
    {
        for(int j=0;j<numOfBuckets;j++)
        {
            printf("[%d,%d]\t", *(h_ph + i*numOfBuckets + j), *(h_pt + i*numOfBuckets + j));    
        }    
        printf("\n");
    }

    cudaMemcpy((void *)h_arr, (void *)d_arr, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

    printf("\nSorted Array after step 3 at level %d\n:", level);    
    for(int i=0; i<size; i++)
    {
        printf("%d ", h_arr[i]);
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("\n");

    //STEP 4
    if (level < MAXIMUM_DIGITS - 1)
    {
        for(int i=0; i<numOfBuckets; i++)
        {
            int index;
            if(i)
            {
                index = h_gt[i-1];
            }    
            else
            {    
                index = 0;
            }   
            size = h_gt[i] - index;    

            if(size)
                paradis(h_arr + index, size, level+1, NUM_OF_BUCKETS, NUM_OF_PROCESSORS);    
        }    
    }

    //Copy sorted array from device to host
    // cudaMemcpy((void *)h_arr, (void *)d_arr, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost);

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
        h_arr[i] = abs(rand()*rand()%((int)pow(10, MAXIMUM_DIGITS)));
        printf("%d ", h_arr[i]);    
    }

    //Call sort function
    paradis(h_arr, ARRAY_SIZE, 0, NUM_OF_BUCKETS, NUM_OF_PROCESSORS);

    //Print the sorted array
    printf("\n***********Sorted Array*********** :\n");    
    for(i=0; i<ARRAY_SIZE; i++)
    {
        printf("%d ", h_arr[i]);
    }
    printf("\n");
    //Free memory
    free(h_arr);

    return 0;
}