#include <stdio.h>
#include <torch/script.h>
#include "cuda_ops.h"

#define MINPDIM 16
#define BATCH 8

namespace pg {

__global__ void convert_csc_kernel(int *indptr, const int *dst, int node_num,
                                   int edge_num) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x + 1;
  if (idx < edge_num) {
    if (dst[idx] != dst[idx - 1]) {
      indptr[dst[idx]] = idx;
    }
  }
  if (idx == 1) {
    indptr[0] = 0;
    indptr[node_num] = edge_num;
  }
}

__global__ void bisage_avgaggr_kernel(float *output, const unsigned char *input,
                                      const int *src, const int *dst,
                                      const int *indptr, int dim, int edge_num,
                                      int node_num) {
  int packed_dim = (dim + 7) / 8;
  // const int len = indptr[blockIdx.x+1]-indptr[blockIdx.x];
  // __shared__ int feat[len][packed_dim];
  __shared__ unsigned char feat[8][MINPDIM];
  // for (int i=threadIdx.y;
  // i<indptr[blockIdx.x+1]-indptr[blockIdx.x];i+=blockDim.y){
  //     for(int j=threadIdx.x; j<packed_dim; j+=blockDim.x){
  //         feat[i*packed_dim + j] = input[src[i]*packed_dim+j];
  //     }
  // }
  // unsigned char mask = 0x01;
  int nid = blockIdx.x + threadIdx.z * blockDim.x;
  int t = (dim - threadIdx.y + 7) / 8;
  float result = 0.0;
  // int nid = threadIdx.x;
  if (nid < node_num) {
    if (threadIdx.x < (dim + 7) / 8) {
      for (int i = 0; i < (dim + 7) / 8; i += MINPDIM) {
        int start = indptr[nid];
        int end = indptr[nid + 1];
        for (int j = start; j < end; j += 8) {
          if (j + threadIdx.y < end) {
            feat[threadIdx.y][threadIdx.x] =
                input[src[j + threadIdx.y] * packed_dim + i + threadIdx.x];
          }
          __syncthreads();

          for (int k = 0; k < 8 && j + k < end; k++) {
            result +=
                (float)((feat[k][threadIdx.x] >> (7 - threadIdx.y)) & 0x01);
          }
        }
        // __syncthreads();
        if ((threadIdx.x + i < t)) {
          if (threadIdx.y < dim % 8) {
            // printf("%d %f->%d\n", threadIdx.x, result, threadIdx.x*dim +
            // threadIdx.y*t  + (threadIdx.x+blockIdx.y*blockDim.x));
            output[nid * dim + threadIdx.y * t + threadIdx.x + i] =
                result * 2 / (end - start) - 1;
          } else {
            // printf("%d %f->%d\n", threadIdx.x, result, threadIdx.x*dim +
            // threadIdx.y*t  + (threadIdx.x+blockIdx.y*blockDim.x));
            output[nid * dim + threadIdx.y * t + dim % 8 + threadIdx.x + i] =
                result * 2 / (end - start) - 1;
          }
        }
        result = 0;
      }
      // free(result);
      // (dim - threadIdx.y + 7) / 8
    }
  }
  // __syncthreads();
}

__global__ void bisage_div_kernel(float *arr, const int *degree, int node_num,
                                  int dim) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < node_num;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < dim; j++) {
      arr[i * dim + j] = arr[i * dim + j] * 2 / degree[i] - 1;
    }
  }
}

void bisage_meanaggr(float *output, const unsigned char *input, const int *src,
                     const int *dst, int dim, int node_num, int edge_num) {
  cudaMemset(output, 0.0, node_num * dim * sizeof(float));
  int *indptr = NULL;
  cudaMalloc((void **)&indptr, (1 + node_num) * sizeof(int));
  // cudaMemset(indptr, 0, 1*sizeof(int));
  // cudaMemset(indptr+node_num, edge_num, 1*sizeof(int));
  dim3 cvtgrid((edge_num + 1023) / 1024);
  dim3 cvtblock(1024);
  convert_csc_kernel<<<cvtgrid, cvtblock>>>(indptr, dst, node_num, edge_num);
  cudaDeviceSynchronize();
  // int *host_indptr = (int *)malloc((node_num+1)*sizeof(int));
  // cudaMemcpy(host_indptr,indptr,(node_num+1)*sizeof(int),cudaMemcpyDeviceToHost);

  // for (int i=0;i<=node_num; i++){
  //     printf("%d, ", host_indptr[i]);
  // }
  // printf("\n");
  if (dim >= MINPDIM * 8) {
    dim3 aggrgrid((node_num + BATCH - 1) / BATCH);
    dim3 aggrblock(MINPDIM, 8, BATCH);
    bisage_avgaggr_kernel<<<aggrgrid, aggrblock>>>(
        output, input, src, dst, indptr, dim, edge_num, node_num);
  }

  cudaFree(indptr);
  // if (dim >= 128){
  //     dim3 aggrgrid(node_num, (dim+127)/128);
  //     dim3 aggrblock(16, 8);
  //     bisage_sumaggr_kernel<<<aggrgrid, aggrblock>>>(output, input, src, dst,
  //     indptr, dim, edge_num, node_num);
  // }

  // float *host_output = (float *)malloc(node_num*dim*sizeof(float));
  // cudaMemcpy(host_output,output,node_num*dim*sizeof(float),cudaMemcpyDeviceToHost);

  // for (int i=0;i<node_num; i++){
  //     printf("%d, ", host_output[i*dim]);
  // }

  // int stride=5;
  // int dd = (dim+7)/8;
  // if (dd > 128){dd = 128;}
  // int bx = 128/dd;
  // dim3 aggrgrid(((edge_num+bx-1)/bx+stride-1)/stride);
  // dim3 aggrblock(bx, 8, dd);
  // // printf("__%d\n", (edge_num+63)/64);
  // for (int i=0;i<stride;i++){
  //     bisage_sumaggr_kernel<<<aggrgrid, aggrblock>>>(output, input, src, dst,
  //     dim, edge_num, stride, i);
  // }
  // cudaDeviceSynchronize();

  // dim3 divgrid((node_num+1023)/1024);
  // dim3 divblock(1024);
  // bisage_div_kernel<<<divgrid, divblock>>>(output, degree, node_num, dim);
}

void meanaggr(torch::Tensor &output, const torch::Tensor &input,
              const torch::Tensor &src, const torch::Tensor &dst, int64_t dim,
              int64_t node_num, int64_t edge_num) {
  bisage_meanaggr((float *)output.data_ptr(),
                  (const unsigned char *)input.data_ptr(),
                  (const int *)src.data_ptr(), (const int *)dst.data_ptr(), dim,
                  node_num, edge_num);
}

}  // namespace pg