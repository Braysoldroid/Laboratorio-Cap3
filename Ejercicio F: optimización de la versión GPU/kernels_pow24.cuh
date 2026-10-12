// Kernels compartidos por el programa y su validador.
// KernelParams se define antes de incluir este archivo.
#pragma once

// Amortiguamos más la onda cerca de los bordes.
__device__ __forceinline__ float border_absorption_dev(int x, int y, const KernelParams& p) {
    constexpr int band = 32;
    const int dist = min(min(x, y), min(p.width - 1 - x, p.height - 1 - y));
    if (dist >= band) {
        return p.damping;
    }
    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return p.damping + p.edge_damping * t * t;
}

// La simulación sigue igual que en D, con las alturas en float.
__global__ void simulate_step_kernel(const float* __restrict__ previous,
                                     const float* __restrict__ current,
                                     float* __restrict__ next,
                                     KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= p.width || y >= p.height) return;

    const int idx = y * p.width + x;

    if (x == 0 || y == 0 || x == p.width - 1 || y == p.height - 1) {
        next[idx] = 0.0f;
        return;
    }

    const float c = current[idx];
    const float laplacian = current[idx - 1] + current[idx + 1] +
                            current[idx - p.width] + current[idx + p.width] -
                            4.0f * c;

    const float velocity = c - previous[idx];
    const float local_damping = border_absorption_dev(x, y, p);

    next[idx] = 2.0f * c - previous[idx] + p.c2 * laplacian - local_damping * velocity;
}

// Cada hilo dibuja un píxel.
__global__ void render_frame_kernel(const float* __restrict__ height,
                                    unsigned char* __restrict__ image,
                                    KernelParams p) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.width || y >= p.height) return;

    const int xm = max(0, x - 1);
    const int xp = min(p.width - 1, x + 1);
    const int ym = max(0, y - 1);
    const int yp = min(p.height - 1, y + 1);

    const float h = height[y * p.width + x];
    const float dx = height[y * p.width + xm] - height[y * p.width + xp];
    const float dy = height[ym * p.width + x] - height[yp * p.width + x];

    const float nx = 2.8f * dx;
    const float ny = 2.8f * dy;
    const float nz = 1.0f;
    const float inv_len = 1.0f / sqrtf(nx * nx + ny * ny + nz * nz);

    const float diffuse = fmaxf(0.0f, (nx * p.lx + ny * p.ly + nz * p.lz) * inv_len);
    const float wave = fminf(fmaxf(0.5f + 1.8f * h, 0.0f), 1.0f);
    // Como el exponente es 24, usamos d^16 * d^8 en lugar de powf.
    const float d = fmaxf(0.0f, diffuse);
    const float d2 = d * d;
    const float d4 = d2 * d2;
    const float d8 = d4 * d4;
    const float d16 = d8 * d8;
    const float specular = d16 * d8;

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;

    const unsigned char gray = static_cast<unsigned char>(fminf(fmaxf(intensity, 0.0f), 255.0f));

    // Los tres canales llevan el mismo valor de gris.
    const int o = 3 * (y * p.width + x);
    image[o + 0] = gray;
    image[o + 1] = gray;
    image[o + 2] = gray;
}
