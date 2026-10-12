// Estos kernels se usan tanto en el video como en la validación.
// KernelParams está definido en cada archivo .cu.

#pragma once
#include <cuda_fp16.h>

// Aumentamos el amortiguamiento cerca de los bordes.
__device__ __forceinline__ float border_absorption_dev(int x, int y, const KernelParams& p) {
    constexpr int band = 32;
    const int dist = min(min(x, y), min(p.width - 1 - x, p.height - 1 - y));
    if (dist >= band) {
        return p.damping;
    }
    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return p.damping + p.edge_damping * t * t;
}

// Cada hilo actualiza una celda de la malla.
__global__ void simulate_step_kernel(const __half* __restrict__ previous,
                                     const __half* __restrict__ current,
                                     __half* __restrict__ next,
                                     KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= p.width || y >= p.height) return;

    const int idx = y * p.width + x;

    if (x == 0 || y == 0 || x == p.width - 1 || y == p.height - 1) {
        next[idx] = __float2half_rn(0.0f);
        return;
    }

    // Leemos half y pasamos a float para calcular la ecuación.
    const float c = __half2float(current[idx]);
    const float laplacian = __half2float(current[idx - 1]) + __half2float(current[idx + 1]) +
                            __half2float(current[idx - p.width]) + __half2float(current[idx + p.width]) -
                            4.0f * c;

    const float velocity = c - __half2float(previous[idx]);
    const float local_damping = border_absorption_dev(x, y, p);

    // El resultado vuelve a half al guardarlo; aquí se pierde precisión.
    next[idx] = __float2half_rn(2.0f * c - __half2float(previous[idx]) + p.c2 * laplacian - local_damping * velocity);
}

// Cada hilo calcula un píxel del video.
__global__ void render_frame_kernel(const __half* __restrict__ height,
                                    unsigned char* __restrict__ image,
                                    KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    // En los bordes usamos el vecino que queda dentro de la malla.
    const int xm = max(0, x - 1);
    const int xp = min(p.width - 1, x + 1);
    const int ym = max(0, y - 1);
    const int yp = min(p.height - 1, y + 1);

    const float h = __half2float(height[y * p.width + x]);
    const float dx = __half2float(height[y * p.width + xm]) - __half2float(height[y * p.width + xp]);
    const float dy = __half2float(height[ym * p.width + x]) - __half2float(height[yp * p.width + x]);

    // La pendiente de las alturas se usa para calcular la iluminación.
    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;
    const float inv_len = 1.0f / sqrtf(nx * nx + ny * ny + nz * nz);

    const float diffuse = fmaxf(0.0f, (nx * p.lx + ny * p.ly + nz * p.lz) * inv_len);
    const float wave = fminf(fmaxf(0.5f + 1.8f * h, 0.0f), 1.0f);
    const float specular = powf(fmaxf(0.0f, diffuse), 24.0f);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;

    const unsigned char gray = static_cast<unsigned char>(fminf(fmaxf(intensity, 0.0f), 255.0f));

    // Repetimos el gris en los tres canales que espera OpenCV.
    const int o = 3 * (y * p.width + x);
    image[o + 0] = gray;
    image[o + 1] = gray;
    image[o + 2] = gray;
}
