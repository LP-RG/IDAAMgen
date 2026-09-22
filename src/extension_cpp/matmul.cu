#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cmath>

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
        cudaDeviceSynchronize();
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
        cudaDeviceSynchronize();
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

        int res_dim = std::pow(2, bit_width);
        int res_offeset = std::pow(2, bit_width - 1);

        float sum = {0.0};
        int tiledRow = row;
        int tiledCol = t + threadIdx.x;
        // Caricamento coalescente di A
        if (tiledRow < M && tiledCol < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Loop per i blocchi di TILE_SIZE
        for(int c = 0; c < N; c+=TILE_SIZE){
            // Caricamento coalescente di B
            tiledRow = t + threadIdx.y;
            tiledCol = c + threadIdx.x;
            if (tiledRow < K && tiledCol < N) {
                tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Unrolling aggressivo e accesso alla memoria costante
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
        cudaDeviceSynchronize();
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

        int res_dim = std::pow(2, bit_width);
        int res_offeset = std::pow(2, bit_width - 1);

        float sum[TILE_SIZE] = {0.0};
        
        // Loop per i blocchi di TILE_SIZE
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {            
            for(int i = 0; i < batch_size; ++i){
                int tiledRow = row;
                int tiledCol = t * TILE_SIZE + threadIdx.x;

                // Caricamento coalescente di A
                if (tiledRow < M && tiledCol < K) {
                    tile_A[threadIdx.y][threadIdx.x] = A[i * M * K + tiledRow * K + tiledCol];
                } else {
                    tile_A[threadIdx.y][threadIdx.x] = 0.0f;
                }

                // Caricamento coalescente di B
                tiledRow = t * TILE_SIZE + threadIdx.y;
                tiledCol = col;

                if (tiledRow < K && tiledCol < N) {
                    tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
                } else {
                    tile_B[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();
                float grad_value = grad_out[i * N * M + col * M + row];
                // Unrolling aggressivo e accesso alla memoria costante
                #pragma unroll 8
                for (int k = 0; k < TILE_SIZE; ++k) {
                    float a_val_sign = tile_A[threadIdx.y][k];
                    float b_val_sign = tile_B[k][threadIdx.x];
                    if(sign_unsign){
                        sum[k] += (a_val_sign - diff_matrix_const[(__float2int_rn(a_val_sign) + res_offeset) * res_dim + (__float2int_rn(b_val_sign) + res_offeset)]) * grad_value;
                    }else{ 
                        //Understand how to treat a_val_sign                     
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
        cudaDeviceSynchronize();
        return C;
    }

    __global__ void matmul_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            const float* __restrict__ res_matrix,
                                            float* __restrict__ C, 
                                            int M,
                                            int N, 
                                            int K, 
                                            float act_scale, 
                                            float act_min, 
                                            float weight_scale, 
                                            float weight_min,
                                            int bit_width,
                                            bool sign_unsign) {

        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        int batch = blockIdx.z;

        A += batch * M * K;
        C += batch * M * N;

        int res_dim = std::pow(2, bit_width);
        int res_offeset = std::pow(2, bit_width - 1);

        float activations_sum = 0.0f;
        float weights_sum = 0.0f;
        float sum = 0.0f;

        // Loop per i blocchi di TILE_SIZE
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            int tiledRow = row;
            int tiledCol = t * TILE_SIZE + threadIdx.x;

            // Caricamento coalescente di A
            if (tiledRow < M && tiledCol < K) {
                tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
            } else {
                tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Caricamento coalescente di B
            tiledRow = t * TILE_SIZE + threadIdx.y;
            tiledCol = col;

            if (tiledRow < K && tiledCol < N) {
                tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Unrolling aggressivo e accesso alla memoria costante
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                float a_val_sign = tile_A[threadIdx.y][k];
                float b_val_sign = tile_B[k][threadIdx.x];

                if(sign_unsign){
                    sum += res_matrix[(__float2int_rn(a_val_sign ) + res_offeset) * res_dim + (__float2int_rn(b_val_sign) + res_offeset)];
                }else{
                    activations_sum += a_val_sign;
                    weights_sum += b_val_sign;
                    sum += res_matrix[(__float2int_rn(a_val_sign)) * res_dim + (__float2int_rn(b_val_sign))];
                }
                
            }

            __syncthreads();
        }

        // Scrittura coalescente del risultato
        if (row < M && col < N) {
            if(sign_unsign){
                C[row * N + col] =  act_scale * weight_scale * sum;
            }
            else{
                C[row * N + col] = act_scale * weight_scale * (sum + activations_sum * weight_min + weights_sum * act_min + act_min * weight_min * K);
            }
        }
    }


    torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor res, double act_scale, double act_min, double weight_scale, double weight_min, int64_t bit_width, bool sign_unsign) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

        matmul_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            res.data_ptr<float>(),
            C.data_ptr<float>(), 
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign
        );
        cudaDeviceSynchronize();
        return C;
    }

    __global__ void matmul_no_error_cuda_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ C, 
                                            float* __restrict__ heat_map, 
                                            int M,
                                            int N, 
                                            int K, 
                                            float act_scale, 
                                            float act_min, 
                                            float weight_scale, 
                                            float weight_min,
                                            int bit_width,
                                            bool sign_unsign,
                                            int shift_bits) {
        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];

        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;
        int batch = blockIdx.z;

        A += batch * M * K;
        C += batch * M * N;

        float sum = 0.0f;
        float activations_sum = 0.0f;
        float weights_sum = 0.0f;
        float shift_mult = exp2f((float)shift_bits); 
        // Loop per i blocchi di TILE_SIZE
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
            int tiledRow = row;
            int tiledCol = t * TILE_SIZE + threadIdx.x;

            // Caricamento coalescente di A
            if (tiledRow < M && tiledCol < K) {
                tile_A[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledCol];
            } else {
                tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            // Caricamento coalescente di B
            tiledRow = t * TILE_SIZE + threadIdx.y;
            tiledCol = col;

            if (tiledRow < K && tiledCol < N) {
                tile_B[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
            } else {
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();

            // Unrolling aggressivo e accesso alla memoria costante
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                float a_val_sign = tile_A[threadIdx.y][k];
                float b_val_sign = tile_B[k][threadIdx.x];
                float a_val_recon = a_val_sign * shift_mult;
                float b_val_recon = b_val_sign * shift_mult;

                activations_sum += a_val_recon;
                weights_sum += b_val_recon;
                sum += a_val_recon * b_val_recon;
                if(heat_map != nullptr){
                    int heat_map_dim = std::pow(2, bit_width);
                    heat_map[col * heat_map_dim * heat_map_dim + (__float2int_rn(a_val_sign)) * heat_map_dim + (__float2int_rn(b_val_sign))] += 1.0f;
                }
            }

            __syncthreads();
        }

        // Scrittura coalescente del risultato
        if (row < M && col < N) {
            if(sign_unsign){
                C[row * N + col] = sum * act_scale * weight_scale;
            }
            else{
                C[row * N + col] = act_scale * weight_scale * (sum + activations_sum * weight_min + weights_sum * act_min + act_min * weight_min * K);
            }
        }
    }


    torch::Tensor matmul_no_error_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor heat_map, double act_scale, double act_min, double weight_scale, double weight_min, int64_t bit_width, bool sign_unsign, int64_t shift_bits) {
        const int batch_size = A.size(0);
        const int M = A.size(1);
        const int K = A.size(2);
        const int N = B.size(1);
        auto C = torch::zeros({batch_size, M, N}, torch::TensorOptions().device(A.device()).dtype(A.dtype()));
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);
        float* heat_map_data = nullptr;
        if(heat_map.defined()){
            heat_map_data = heat_map.data_ptr<float>();
        }
        matmul_no_error_cuda_kernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(), 
            B.data_ptr<float>(),
            C.data_ptr<float>(), 
            heat_map_data,
            M, N, K,
            static_cast<float>(act_scale),
            static_cast<float>(act_min),
            static_cast<float>(weight_scale),
            static_cast<float>(weight_min),
            bit_width,
            sign_unsign,
            shift_bits
        );
        cudaDeviceSynchronize();
        return C;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

    TORCH_LIBRARY(mat_mul, m) {
        m.def("linear_input(Tensor a, Tensor b, Tensor c, Tensor d) -> (Tensor)");
        m.def("linear_weight(Tensor a, Tensor b, Tensor c, Tensor d) -> (Tensor)");
        m.def("derivate_input(Tensor a, Tensor b, Tensor c, Tensor d, int weight_zp, int bit_width, bool signed) -> (Tensor)");
        m.def("derivate_weight(Tensor a, Tensor b, Tensor c, Tensor d, int act_zp, int bit_width, bool signed) -> (Tensor)");
        m.def("matmul_cuda(Tensor a, Tensor b, Tensor c, float act_scale, float act_min, float weight_scale, float weight_min,int bit_width, bool signed) -> (Tensor)");
        m.def("matmul_no_error_cuda(Tensor a, Tensor b, Tensor heat_map, float act_scale, float act_min, float weight_scale, float weight_min, int bit_width, bool signed, int shifts_bits) -> (Tensor)");
    }

    TORCH_LIBRARY_IMPL(mat_mul, CUDA, m) {
        m.impl("linear_input", &linear_input);
        m.impl("linear_weight", &linear_weight);
        m.impl("derivate_input", &derivate_input);
        m.impl("derivate_weight", &derivate_weight);
        m.impl("matmul_cuda", &matmul_cuda);
        m.impl("matmul_no_error_cuda", &matmul_no_error_cuda);
    }
}

