#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cmath>
#include <cstdint>

#include <vector>
#include <stdexcept>

#define TILE_SIZE 16

namespace mat_mul{

    __global__ void linear_input_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ error_der,
                                            const float* __restrict__ grad_out, 
                                            float* __restrict__ C, 
                                            int M, int N, int K) {
        int row = threadIdx.y + blockIdx.y * TILE_SIZE;
        int col = threadIdx.x + blockIdx.x * TILE_SIZE;       
        if(row < M && col < K){                               
            float sum = 0.0f;
            float a_value = A[row * K + col];
            for(int i = 0; i < N; ++i){
                float b_value = B[col * N + i];
                float grad_value = grad_out[row * N + i];
                sum += (b_value - error_der[(__float2int_rn(a_value) + 128) * 256 + (__float2int_rn(b_value)+ 128)]) * grad_value;
            }
            C[row * K + col] = sum;
        }
    }



    torch::Tensor linear_input(torch::Tensor A, torch::Tensor B, torch::Tensor error_der, torch::Tensor grad_out) {
        const int M = A.size(0);
        const int K = A.size(1);
        const int N = B.size(1);
        auto C = torch::zeros({M, K}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        linear_input_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            error_der.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            C.data_ptr<float>(), 
            M, N, K
        );
        return C;
    }

    __global__ void linear_weight_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ error_der,
                                            const float* __restrict__ grad_out, 
                                            float* __restrict__ C, 
                                            int M, int N, int K) {
        int row = threadIdx.x + blockIdx.x * TILE_SIZE;
        int col = threadIdx.y + blockIdx.y * TILE_SIZE;
        if(row < K && col < N){
            float sum = 0.0f;
            float b_value = B[row * N + col];
            for(int i = 0; i < M; ++i){
                float a_value = A[i * K + row];
                float grad_value = grad_out[i * N + col];
                sum += (a_value  - error_der[(__float2int_rn(a_value) + 128) * 256 + (__float2int_rn(b_value)+ 128)]) * grad_value;
            }
            C[row * N + col] += sum;
        }
    }


    torch::Tensor linear_weight(torch::Tensor A, torch::Tensor B, torch::Tensor error_der, torch::Tensor grad) {
        const int M = A.size(0);
        const int K = A.size(1);
        const int N = B.size(1);
        auto C = torch::zeros({K, N}, torch::TensorOptions().device(B.device()).dtype(B.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        linear_weight_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            error_der.data_ptr<float>(),
            grad.data_ptr<float>(),
            C.data_ptr<float>(), 
            M, N, K
        );
        return C;
    }

    __global__ void derivate_input_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ res_matrix_const,
                                            const float* __restrict__ grad_out, 
                                            float* __restrict__ C, 
                                            int M, int N, int K,
                                            int weight_zp,
                                            int bit_width,
                                            bool sign_unsign) {
        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int t = blockIdx.x * TILE_SIZE;
        int batch = blockIdx.z;

        A += batch * M * K;
        C += batch * M * K;
        grad_out += batch * N * M;

        // int res_dim = std::pow(2, bit_width);
        // int res_offeset = std::pow(2, bit_width - 1);
        int res_dim = 1 << bit_width;
        int res_offeset = 1 << (bit_width - 1);

        float sum = {0.0};
        int tiledRow = row;
        int tiledCol = t + threadIdx.x;
        
        if (tiledRow < M && tiledCol < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        for(int c = 0; c < N; c+=TILE_SIZE){
            
            tiledRow = t + threadIdx.y;
            tiledCol = c + threadIdx.x;
            if (tiledRow < K && tiledCol < N) {
                tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            
            #pragma unroll 8
            float a_val_sign = tile_A[threadIdx.y][threadIdx.x];
            for(int h = 0; h < TILE_SIZE; ++h){
                float grad_value = 0.0f;
                if(c + h < N){
                    grad_value = grad_out[(c + h) * M + row];
                }
                float b_val_sign = tile_B[threadIdx.x][h];
                if(sign_unsign){
                    sum += res_dim * (a_val_sign - res_matrix_const[(__float2int_rn(a_val_sign) + res_offeset) * res_dim + (__float2int_rn(b_val_sign) + res_offeset)]) * grad_value;
                }else{
                    sum += ((b_val_sign + weight_zp)  - res_matrix_const[(__float2int_rn(a_val_sign)) * res_dim + (__float2int_rn(b_val_sign))]) * grad_value;
                }
            }
            
            __syncthreads();
        }
        if (row < M && t + threadIdx.x < K) {
            C[row * K + t + + threadIdx.x] = sum;
        } 
    }



    torch::Tensor derivate_input(torch::Tensor A, torch::Tensor B, torch::Tensor res, torch::Tensor grad_out, int64_t weight_zp, int64_t bit_width, bool sign_unsign) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, K}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

        derivate_input_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            res.data_ptr<float>(),
            grad_out.data_ptr<float>(),
            C.data_ptr<float>(), 
            M, N, K,
            weight_zp,
            bit_width,
            sign_unsign
        );
        return C;
    }


    __global__ void derivate_weight_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ diff_matrix_const,
                                            const float* __restrict__ grad_out, 
                                            float* __restrict__ C, 
                                            int M, int N, int K, 
                                            int batch_size,
                                            int bit_width,
                                            int act_zp,
                                            bool sign_unsign) {

        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        // int res_dim = std::pow(2, bit_width);
        // int res_offeset = std::pow(2, bit_width - 1);
        int res_dim = 1 << bit_width;
        int res_offeset = 1 << (bit_width - 1);

        float sum[TILE_SIZE] = {0.0};
        
        
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {            
            for(int i = 0; i < batch_size; ++i){
                int tiledRow = row;
                int tiledCol = t * TILE_SIZE + threadIdx.x;

                
                if (tiledRow < M && tiledCol < K) {
                    tile_A[threadIdx.y][threadIdx.x] = A[i * M * K + tiledRow * K + tiledCol];
                } else {
                    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
                }

                
                tiledRow = t * TILE_SIZE + threadIdx.y;
                tiledCol = col;

                if (tiledRow < K && tiledCol < N) {
                    tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
                } else {
                    tile_B[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();
                float grad_value = grad_out[i * N * M + col * M + row];
                
                #pragma unroll 8
                for (int k = 0; k < TILE_SIZE; ++k) {
                    float a_val_sign = tile_A[threadIdx.y][k];
                    float b_val_sign = tile_B[k][threadIdx.x];
                    if(sign_unsign){
                        sum[k] += (a_val_sign - diff_matrix_const[(__float2int_rn(a_val_sign) + res_offeset) * res_dim + (__float2int_rn(b_val_sign) + res_offeset)]) * grad_value;
                    }else{ 
                        
                        sum[k] += ((a_val_sign + act_zp) - diff_matrix_const[(__float2int_rn(a_val_sign)) * res_dim + (__float2int_rn(b_val_sign))]) * grad_value;
                    }
                }
                __syncthreads();
            }
            if (row < M && col < N) {
                for(int i = 0; i < TILE_SIZE; ++i){
                    if(t * TILE_SIZE + i < K){
                        C[col * M * K + row * K + t * TILE_SIZE + i] = sum[i];
                    } 
                }  
            }
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum[i] = 0.0f;
            }
        }
    }


    torch::Tensor derivate_weight(torch::Tensor A, torch::Tensor B, torch::Tensor res, torch::Tensor grad, int64_t act_zp, int64_t bit_width, bool sign_unsign) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({N, M, K}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        derivate_weight_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            res.data_ptr<float>(),
            grad.data_ptr<float>(),
            C.data_ptr<float>(), 
            M, N, K, batch_size,
            bit_width,
            act_zp,
            sign_unsign
        );
        return C;
    }


    // NEED TO ENSURE:
    // BM % TM == 0
    // BN % TN == 0
    // (BM * BK) % THREADS_PER_BLOCK == 0
    // (BK * BN) % THREADS_PER_BLOCK == 0


    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void matmul_kernel(
        const uint8_t* __restrict__ A,          // Input matrix A (quantized activations, [Batch, M, K])
        const uint8_t* __restrict__ B,          // Input matrix B (quantized weights, [K, N])
        const uint16_t* __restrict__ res_matrix,// Precomputed Lookup Table (LUT) for quantized multiplication
        float* __restrict__ C,                  // Output matrix C (dequantized FP32, [Batch, M, N])
        int M, int N, int K,                    // Matrix dimensions
        float act_scale, float act_min,         // Dequantization scale and zero-point offset for A
        float weight_scale, float weight_min,   // Dequantization scale and zero-point offset for B
        int bit_width,                          // Quantization bit-width (used for LUT indexing)
        bool sign_unsign                        // True: signed quantization; False: unsigned asymmetric quantization
    ) {

        constexpr int THREADS_PER_BLOCK = (BM / TM) * (BN / TN);
        static_assert(BM % TM == 0 && BN % TN == 0, "tile must be divisible by TM/TN");
        static_assert(BM * BK >= THREADS_PER_BLOCK && (BM * BK) % THREADS_PER_BLOCK == 0,
                      "degenerate cooperative load of A: BM*BK must be >= and a multiple of the number of threads");
        static_assert(BK * BN >= THREADS_PER_BLOCK && (BK * BN) % THREADS_PER_BLOCK == 0,
                      "degenerate cooperative load of B: BK*BN must be >= and a multiple of the number of threads");
        static_assert(THREADS_PER_BLOCK <= 1024, "too many threads per block");

        __shared__ uint8_t s_A[BM][BK + 1];
        __shared__ uint8_t s_B[BK][BN + 1];

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;
        
        const int tid_m = threadIdx.y;
        const int tid_n = threadIdx.x; 


        const int c_row_start = blockIdx.y * BM;
        const int c_col_start = blockIdx.x * BN;
        const int batch = blockIdx.z;

        A += (size_t)batch * M * K;
        C += (size_t)batch * M * N;

        const int res_offset = 1 << (bit_width - 1);


        int sum[TM][TN] = {0};
        

        int act_acc[TM] = {0};
        int weight_acc[TN] = {0};


        for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {


            const int load_elem_per_thread_A = (BM * BK) / THREADS_PER_BLOCK;
            #pragma unroll
            for (int i = 0; i < load_elem_per_thread_A; ++i) {
                int linear_idx = tid * load_elem_per_thread_A + i;
                int r = linear_idx / BK; 
                int c = linear_idx % BK; 
                int global_r = c_row_start + r;
                int global_c = bk_idx + c;

                if (global_r < M && global_c < K) {
                    s_A[r][c] = A[global_r * K + global_c];
                } else {
                    s_A[r][c] = 0;
                }
            }


            const int load_elem_per_thread_B = (BK * BN) / THREADS_PER_BLOCK;
            #pragma unroll
            for (int i = 0; i < load_elem_per_thread_B; ++i) {
                int linear_idx = tid * load_elem_per_thread_B + i;
                int r = linear_idx / BN; 
                int c = linear_idx % BN; 
                int global_r = bk_idx + r;
                int global_c = c_col_start + c;

                if (global_r < K && global_c < N) {
                    s_B[r][c] = B[global_r * N + global_c];
                } else {
                    s_B[r][c] = 0;
                }
            }

            __syncthreads();

            #pragma unroll
            for (int dot_k = 0; dot_k < BK; ++dot_k) {

                if (bk_idx + dot_k >= K) break;

                int reg_A[TM];
                int reg_B[TN];

                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    reg_A[m] = s_A[tid_m * TM + m][dot_k];
                    if (!sign_unsign) act_acc[m] += reg_A[m];
                }

                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    reg_B[n] = s_B[dot_k][tid_n * TN + n];
                    if (!sign_unsign) weight_acc[n] += reg_B[n];
                }

                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    #pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        int index;
                        if (sign_unsign) {
                            int a_idx = (int8_t)reg_A[m] + res_offset;
                            int b_idx = (int8_t)reg_B[n] + res_offset;
                            index = (a_idx << bit_width) + b_idx;
                        } else {
                            index = (reg_A[m] << bit_width) + reg_B[n];
                        }
                        sum[m][n] += __ldg(&res_matrix[index]);
                    }
                }
            }

            __syncthreads();
        }


        const float scale_prod = act_scale * weight_scale;
        const float base_offset = act_min * weight_min * (float)K;

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int row = c_row_start + tid_m * TM + m;
            if (row < M) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    int col = c_col_start + tid_n * TN + n;
                    if (col < N) {
                        if (sign_unsign) {
                            C[row * N + col] = scale_prod * (float)sum[m][n];
                        } else {
                            C[row * N + col] = scale_prod * (
                                (float)sum[m][n] + 
                                (float)act_acc[m] * weight_min + 
                                (float)weight_acc[n] * act_min + 
                                base_offset
                            );
                        }
                    }
                }
            }
        }
    }


    torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor res, 
                            double act_scale, double act_min, 
                            double weight_scale, double weight_min, 
                            int64_t bit_width, bool sign_unsign) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));

        if (N >= 32 && M >= 32 && (M > 64 || N > 64)) {
            dim3 blockDim(32 / 2, 32 / 2);
            dim3 gridDim((N + 31) / 32, (M + 31) / 32, batch_size);
            matmul_kernel<32, 32, 16, 2, 2><<<gridDim, blockDim>>>(
                A.data_ptr<uint8_t>(), 
            B.data_ptr<uint8_t>(),
            res.data_ptr<uint16_t>(),
            C.data_ptr<float>(), 
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign);
        } else {
            dim3 blockDim(16 / 2, 16 / 2);
            dim3 gridDim((N + 15) / 16, (M + 15) / 16, batch_size);
            matmul_kernel<16, 16, 16, 2, 2><<<gridDim, blockDim>>>(
                A.data_ptr<uint8_t>(), 
            B.data_ptr<uint8_t>(),
            res.data_ptr<uint16_t>(),
            C.data_ptr<float>(), 
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign);
        }
        return C;
    }

    template <int BM, int BN, int BK, int TM, int TN>
    __global__ void matmul_no_error_cuda_kernel(
        const uint8_t* __restrict__ A,          // Input matrix A (quantized activations, [Batch, M, K])
        const uint8_t* __restrict__ B,          // Input matrix B (quantized weights, [K, N])
        float* __restrict__ C,                  // Output matrix C (dequantized FP32, [Batch, M, N])
        int M, int N, int K,                    // Matrix dimensions
        float act_scale, float act_min,         // Dequantization scale and zero-point offset for A
        float weight_scale, float weight_min,   // Dequantization scale and zero-point offset for B
        int bit_width,                          // Quantization bit-width
        bool sign_unsign,                       // True: signed quantization; False: unsigned asymmetric quantization
        int shift_bits                          // Bit-shift factor for value reconstruction
    ) {

        constexpr int THREADS_PER_BLOCK = (BM / TM) * (BN / TN);
        static_assert(BM % TM == 0 && BN % TN == 0, "tile must be divisible by TM/TN");
        static_assert(BM * BK >= THREADS_PER_BLOCK && (BM * BK) % THREADS_PER_BLOCK == 0,
                      "degenerate cooperative load of A: BM*BK must be >= and a multiple of the number of threads");
        static_assert(BK * BN >= THREADS_PER_BLOCK && (BK * BN) % THREADS_PER_BLOCK == 0,
                      "degenerate cooperative load of B: BK*BN must be >= and a multiple of the number of threads");
        static_assert(THREADS_PER_BLOCK <= 1024, "too many threads per block");

        __shared__ uint8_t s_A[BM][BK + 1];
        __shared__ uint8_t s_B[BK][BN + 1];

        const int tid = threadIdx.y * blockDim.x + threadIdx.x;

        const int tid_m = threadIdx.y; // Row index of the thread's register tile (0 .. 15)
        const int tid_n = threadIdx.x; // Column index of the thread's register tile (0 .. 15)

        const int c_row_start = blockIdx.y * BM;
        const int c_col_start = blockIdx.x * BN;
        const int batch = blockIdx.z;

        A += (size_t)batch * M * K;
        C += (size_t)batch * M * N;

        int sum[TM][TN] = {0};
        int act_acc[TM] = {0};
        int weight_acc[TN] = {0};
        const int shift_mult = exp2f((int)shift_bits);

        for (int bk_idx = 0; bk_idx < K; bk_idx += BK) {

            const int load_elem_per_thread_A = (BM * BK) / THREADS_PER_BLOCK;
            #pragma unroll
            for (int i = 0; i < load_elem_per_thread_A; ++i) {
                int linear_idx = tid * load_elem_per_thread_A + i;
                int r = linear_idx / BK; // Local row in s_A (0 .. BM-1)
                int c = linear_idx % BK; // Local col in s_A (0 .. BK-1)
                int global_r = c_row_start + r;
                int global_c = bk_idx + c;

                if (global_r < M && global_c < K) {
                    s_A[r][c] = A[global_r * K + global_c];
                } else {
                    s_A[r][c] = 0;
                }
            }

            const int load_elem_per_thread_B = (BK * BN) / THREADS_PER_BLOCK;
            #pragma unroll
            for (int i = 0; i < load_elem_per_thread_B; ++i) {
                int linear_idx = tid * load_elem_per_thread_B + i;
                int r = linear_idx / BN; // Local row in s_B (0 .. BK-1)
                int c = linear_idx % BN; // Local col in s_B (0 .. BN-1)
                int global_r = bk_idx + r;
                int global_c = c_col_start + c;

                if (global_r < K && global_c < N) {
                    s_B[r][c] = B[global_r * N + global_c];
                } else {
                    s_B[r][c] = 0;
                }
            }

            __syncthreads();

            #pragma unroll
            for (int dot_k = 0; dot_k < BK; ++dot_k) {

                float reg_A[TM];
                float reg_B[TN];

                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    const float a_val_sign = s_A[tid_m * TM + m][dot_k];
                    reg_A[m] = a_val_sign * shift_mult;
                    act_acc[m] += reg_A[m];
                }

                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const float b_val_sign = s_B[dot_k][tid_n * TN + n];
                    reg_B[n] = b_val_sign * shift_mult;
                    weight_acc[n] += reg_B[n];
                }

                #pragma unroll
                for (int m = 0; m < TM; ++m) {
                    #pragma unroll
                    for (int n = 0; n < TN; ++n) {
                        sum[m][n] += reg_A[m] * reg_B[n];
                    }
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int row = c_row_start + tid_m * TM + m;
            if (row < M) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    int col = c_col_start + tid_n * TN + n;
                    if (col < N) {
                        if (sign_unsign) {
                            C[row * N + col] = sum[m][n] * act_scale * weight_scale;
                        } else {
                            C[row * N + col] = act_scale * weight_scale * (
                                sum[m][n] +
                                act_acc[m] * weight_min +
                                weight_acc[n] * act_min +
                                act_min * weight_min * K
                            );
                        }
                    }
                }
            }
        }
    }

    torch::Tensor matmul_no_error_cuda(torch::Tensor A, torch::Tensor B, double act_scale, double act_min, double weight_scale, double weight_min, int64_t bit_width, bool sign_unsign, int64_t shift_bits) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kFloat32));
        
        if (N >= 32 && M >= 32 && (M > 64 || N > 64)) {
            dim3 blockDim(32 / 2, 32 / 2);
            dim3 gridDim((N + 31) / 32, (M + 31) / 32, batch_size);
            matmul_no_error_cuda_kernel<32, 32, 16, 2, 2><<<gridDim, blockDim>>>(
                A.data_ptr<uint8_t>(), 
            B.data_ptr<uint8_t>(),
            C.data_ptr<float>(), 
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign,
            shift_bits);
        } else {
            dim3 blockDim(16 / 2, 16 / 2);
            dim3 gridDim((N + 15) / 16, (M + 15) / 16, batch_size);
            matmul_no_error_cuda_kernel<16, 16, 16, 2, 2><<<gridDim, blockDim>>>(
                A.data_ptr<uint8_t>(), 
            B.data_ptr<uint8_t>(),
            C.data_ptr<float>(), 
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign,
            shift_bits);
        }
        return C;
    }
    
    
    __global__ void matmul_stats_kernel(const uint8_t* __restrict__ A,
                                     const uint8_t* __restrict__ B,
                                     const uint16_t* __restrict__ res_matrix,
                                     float* __restrict__ C,
                                     float* __restrict__ heat_map, 
                                     int M, int N, int K,
                                     float act_scale, float act_min,
                                     float weight_scale, float weight_min,
                                     int bit_width, bool sign_unsign, bool approx) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;   
        int batch = blockIdx.z;

        A += batch * M * K;
        C += batch * M * N;

        int sum = 0.0f;
        int activations_sum = 0.0f;
        int weights_sum = 0.0f;

        const int res_dim = 1 << bit_width;

        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            int tiledRow = row;
            int tiledCol = t * TILE_SIZE + threadIdx.x;

            if (tiledRow < M && tiledCol < K) {
                tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
            } else {
                tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            tiledRow = t * TILE_SIZE + threadIdx.y;
            tiledCol = col;

            if (tiledRow < K && tiledCol < N) {
                tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = t * TILE_SIZE + k;
                
                if (global_k >= K) continue;

                float a_val_sign = tile_A[threadIdx.y][k];
                float b_val_sign = tile_B[k][threadIdx.x];
                activations_sum += a_val_sign;
                weights_sum += b_val_sign;

                int a_idx = __float2int_rn(a_val_sign);
                int b_idx = __float2int_rn(b_val_sign);

                if (approx) {
                    if (a_idx < res_dim && b_idx < res_dim) {
                        sum += res_matrix[a_idx * res_dim + b_idx];
                    }
                } else {
                    sum += a_val_sign * b_val_sign;
                }

                if (heat_map != nullptr && row < M && col < N) {
                    if (a_idx < res_dim && b_idx < res_dim) {
                        long long idx = (long long)col * res_dim * res_dim +
                                        (long long)a_idx * res_dim + b_idx;
                        atomicAdd(&heat_map[idx], 1.0f);
                    }
                }
            }
            __syncthreads();
        }

        if (row < M && col < N) {
            C[row * N + col] = act_scale * weight_scale *
                (sum + activations_sum * weight_min + weights_sum * act_min +
                act_min * weight_min * K);
        }
    }

    torch::Tensor matmul_stats(torch::Tensor A, torch::Tensor B, torch::Tensor res,
                                torch::Tensor heat_map, double act_scale,
                                double act_min, double weight_scale,
                                double weight_min, int64_t bit_width,
                                bool sign_unsign, bool approx) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, N},
                            torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);
        uint16_t* res_ptr = nullptr;
        if (res.defined()) {
            res_ptr = res.data_ptr<uint16_t>();
        }
        matmul_stats_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<uint8_t>(),
            B.data_ptr<uint8_t>(),
            res_ptr,
            C.data_ptr<float>(),
            heat_map.data_ptr<float>(),
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign,
            approx
        );
        return C;
    }



    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(mat_mul, m) {
        m.def("linear_input(Tensor a, Tensor b, Tensor c, Tensor d) -> (Tensor)");
        m.def("linear_weight(Tensor a, Tensor b, Tensor c, Tensor d) -> (Tensor)");
        m.def("derivate_input(Tensor a, Tensor b, Tensor c, Tensor d, int weight_zp, int bit_width, bool signed) -> (Tensor)");
        m.def("derivate_weight(Tensor a, Tensor b, Tensor c, Tensor d, int act_zp, int bit_width, bool signed) -> (Tensor)");
        m.def("matmul_cuda(Tensor a, Tensor b, Tensor c, float act_scale, float act_min, float weight_scale, float weight_min,int bit_width, bool signed) -> (Tensor)");
        m.def("matmul_no_error_cuda(Tensor a, Tensor b, float act_scale, float act_min, float weight_scale, float weight_min, int bit_width, bool signed, int shifts_bits) -> (Tensor)");
        m.def("matmul_stats(Tensor a, Tensor b, Tensor c, Tensor heat_map, float act_scale, float act_min, float weight_scale, float weight_min,int bit_width, bool signed, bool approx) -> (Tensor)");
    }

    TORCH_LIBRARY_IMPL(mat_mul, CUDA, m) {
        m.impl("linear_input", &linear_input);
        m.impl("linear_weight", &linear_weight);
        m.impl("derivate_input", &derivate_input);
        m.impl("derivate_weight", &derivate_weight);
        m.impl("matmul_cuda", &matmul_cuda);
        m.impl("matmul_no_error_cuda", &matmul_no_error_cuda);
        m.impl("matmul_stats", &matmul_stats);
    }
}
