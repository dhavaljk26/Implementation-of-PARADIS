#include <cuda.h>
#include <bits/stdc++.h>

using namespace std;

#define MAXIMUM_DIGITS 10
#define ARRAY_SIZE 10 
#define NUM_OF_PROCESSORS 2
#define NUM_OF_BUCKETS 10

void printArr(int *d_arr, int size)
{
    int *t_arr = (int *)malloc(size*sizeof(int));
    cudaMemcpy((void *)t_arr, (void *)d_arr, size*sizeof(int), cudaMemcpyDeviceToHost);

    printf("[DEVICE] Array: ");
    for(int i=0; i<size; i++)
    {
        printf("%d ", t_arr[i]);
    }
    printf("\n");
    
}

void printAux(int *d_histogram, int *d_gh, int *d_gt, int *d_ph, int *d_pt, int numOfBuckets, int numOfProcessors)
{
    int *t_histogram = (int *)malloc(numOfBuckets*sizeof(int));
    cudaMemcpy((void *)t_histogram, (void *)d_histogram, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);

    int *t_gh = (int *)malloc(numOfBuckets*sizeof(int));
    cudaMemcpy((void *)t_gh, (void *)d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);

    int *t_gt = (int *)malloc(numOfBuckets*sizeof(int));
    cudaMemcpy((void *)t_gt, (void *)d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);

    printf("[DEVICE] Histogram: \nIndex\tHist\tgh\tgt\n");
    for(int i=0; i<numOfBuckets; i++)
    {
        printf("[%d]\t%d\t%d\t%d\n", i, t_histogram[i], t_gh[i], t_gt[i]);
    }
    printf("\n[DEVICE] (ph,pt) \n");
    
    int *t_ph = (int *)malloc(numOfBuckets*numOfProcessors*sizeof(int));
    cudaMemcpy((void *)t_ph, (void *)d_ph, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);
    
    int *t_pt = (int *)malloc(numOfBuckets*numOfProcessors*sizeof(int));
    cudaMemcpy((void *)t_pt, (void *)d_pt, numOfBuckets*numOfProcessors*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i=0; i<numOfBuckets; i++)
        printf("\t%d", i);

    for(int i=0;i<numOfProcessors;i++)
    {
        printf("\n%d\t", i);
        for(int j=0;j<numOfBuckets;j++)
        {
            printf("(%d,%d)\t", *(t_ph + i*numOfBuckets + j), *(t_pt + i*numOfBuckets + j));    
        }    
    }    
    printf("\n\n");
}

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
    d_out[0] = 0;
    d_out2[0] = d_in[0];
    for(int i=1;i<n;i++)
    {
        d_out[i] = d_out[i-1] + d_in[i-1];
        d_out2[i] = d_out2[i-1] + d_in[i];
    }
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
                while(head <= tail)
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

int allBucketsAreEmpty(int *h_gh, int *h_gt, int numOfBuckets)
{
    for(int i=0; i<numOfBuckets; i++)
    {
        if(h_gh[i]!=h_gt[i])
        {
            return 0;       
        }
    }
    return 1;
}

void paradisUtil(int *d_arr, int size, int level, int numOfBuckets, int numOfProcessors)
{
    int intSize = sizeof(int);
    int *h_gh, *h_gt;
    int *d_histogram, *d_localHistogram, *d_gh, *d_gt, *d_ph, *d_pt;

    h_gh = (int *)malloc(numOfBuckets*sizeof(int));
    h_gt = (int *)malloc(numOfBuckets*sizeof(int));

    cudaMalloc((void **)&d_histogram        , numOfBuckets*intSize);
    cudaMalloc((void **)&d_localHistogram   , numOfBuckets*numOfProcessors*intSize);
    cudaMalloc((void **)&d_gh               , numOfBuckets*intSize);
    cudaMalloc((void **)&d_gt               , numOfBuckets*intSize);
    cudaMalloc((void **)&d_ph               , numOfBuckets*numOfProcessors*intSize);
    cudaMalloc((void **)&d_pt               , numOfBuckets*numOfProcessors*intSize);
    
    cudaMemset((void *)d_histogram      , 0, numOfBuckets*intSize);
    cudaMemset((void *)d_localHistogram , 0, numOfBuckets*numOfProcessors*intSize);
    
    //STEP 1
    buildLocalHistogram<<<1, numOfProcessors>>>(d_localHistogram, d_arr, size, level, numOfBuckets, numOfProcessors);
    buildHistogram<<<1, numOfBuckets>>>(d_histogram, d_localHistogram, numOfBuckets, numOfProcessors);
    
    //STEP 2
    prefixSum<<<1, 1>>>(d_gh, d_gt, d_histogram, numOfBuckets);   

    //STEP 3
    int iteration=0;
    do 
    {
        // printf("Iteration %d\n", iteration++);
        dim3 DimGrid(1, 1, 1);
        dim3 DimBlock(numOfProcessors, numOfBuckets, 1);

        partitionForPermutation<<<DimGrid, DimBlock>>>(d_ph, d_pt, d_gh, d_gt, numOfBuckets, numOfProcessors);
        paradisPermute<<<1, numOfProcessors>>>(d_arr, size, d_gh, d_gt, d_ph, d_pt, level, numOfBuckets, numOfProcessors);

        // printf("\n***After Permute***\n");
        // printAux(d_histogram, d_gh, d_gt, d_ph, d_pt, numOfBuckets, numOfProcessors);
        // printArr(d_arr, size);

        paradisRepair<<<1, numOfBuckets>>>(0, d_arr, size, d_gh, d_gt, d_ph, d_pt, level, numOfBuckets, numOfProcessors);

        // printf("\n***After Repair***\n");
        // printAux(d_histogram, d_gh, d_gt, d_ph, d_pt, numOfBuckets, numOfProcessors);
        // printArr(d_arr, size);

        
        cudaMemcpy((void *)h_gh, (void *)d_gh, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy((void *)h_gt, (void *)d_gt, numOfBuckets*sizeof(int), cudaMemcpyDeviceToHost);

    } while(!allBucketsAreEmpty(h_gh, h_gt, numOfBuckets));  

    //STEP 4

    if(level < MAXIMUM_DIGITS - 1)
    {
        for(int i=0; i<numOfBuckets; i++)
        {
            int offset, bucketSize;

            if(i)
            {
                offset = h_gt[i-1];
                bucketSize = h_gt[i] - h_gt[i-1];
            }
            else
            {
                offset = 0;
                bucketSize = h_gt[i];
            }

            if(bucketSize)
            {
                //printf("----------\nCall paradisUtil: firstIndex=%d, size=%d, level=%d\n", offset, bucketSize, level+1);
                paradisUtil(d_arr+offset, bucketSize, level+1, numOfBuckets, numOfProcessors);
            }    
        }
    }
}


void paradis(int *h_arr, int size, int numOfBuckets, int numOfProcessors)
{
    int *d_arr;
    
    cudaMalloc((void **)&d_arr, size*sizeof(int));
    cudaMemcpy((void *)d_arr, (void *)h_arr, size*sizeof(int), cudaMemcpyHostToDevice);
    
    //printf("----------\nCall paradisUtil: firstIndex=%d, size=%d, level=%d\n", 0, size, 0);
    paradisUtil(d_arr, size, 0, numOfBuckets, numOfProcessors);   
    
    cudaMemcpy((void *)h_arr, (void *)d_arr, size*sizeof(int), cudaMemcpyDeviceToHost);
}

int main()
{
    freopen("out.txt", "w", stdout);
    freopen("in.txt", "r", stdin);

    int *h_arr;
    int i, size;

    scanf("%d", &size);

    //Allocate memory for array on host                
    h_arr = (int *)malloc(size*sizeof(int));

    //Initialize array elements with random values   
    // printf("[HOST] Input array: ");
    for(i=0; i<size; i++)
    {
        scanf("%d", h_arr+i);
        // h_arr[i] = abs(rand()*rand()%((int)pow(10, MAXIMUM_DIGITS)));
        // printf("%d ", h_arr[i]);    
    }
    // printf("\n\n");

    //Call sort function
    auto t1 = chrono::steady_clock::now();

    paradis(h_arr, size, NUM_OF_BUCKETS, NUM_OF_PROCESSORS);

    auto t2 = chrono::steady_clock::now();
    auto time_span = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();

    printf("\nTime : %ld ms\n", time_span);

    //Print the sorted array
    // printf("\n[HOST] Sorted Array : ");    
    for(i=0; i<size; i++)
    {
        // printf("%d ", h_arr[i]);
        
        if(i)
        {
            if(h_arr[i] < h_arr[i-1])
                printf("\nError\n");
        }
    }
    // printf("\n");
    //Free memory
    free(h_arr);

    return 0;
}