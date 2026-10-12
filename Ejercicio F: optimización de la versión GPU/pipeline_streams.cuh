#pragma once
// El programa y el validador usan la misma simulación y las mismas copias.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#define CUDA_CHECK(call) do { \
    const cudaError_t error = (call); \
    if (error != cudaSuccess) throw std::runtime_error( \
        std::string("CUDA: ") + cudaGetErrorString(error) + \
        " en " + __FILE__ + ":" + std::to_string(__LINE__)); \
} while (0)

namespace drop_streams {

// Parámetros y cálculos conservados de D, incluyendo powf y bloques 16x16.
struct Config {
    int width = 640;
    int height = 640;
    double seconds = 30.0;
    int fps = 30;
    int steps_per_frame = 1;
    float wave_speed = 0.45f;
    float damping = 0.006f;
    float edge_damping = 0.035f;
    float drop_radius = 18.0f;
    float drop_strength = 1.0f;
    std::string output = "output/drop_simulation_streams.mp4";
};

struct KernelParams {
    int width;
    int height;
    float c2;
    float damping;
    float edge_damping;
    float lx, ly, lz;
};

int index_of(int x, int y, int width) {
    return y * width + x;
}

void add_drop(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
    const float cx = 0.5f * static_cast<float>(cfg.width - 1);
    const float cy = 0.5f * static_cast<float>(cfg.height - 1);
    const float sigma2 = cfg.drop_radius * cfg.drop_radius;

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const float dx = static_cast<float>(x) - cx;
            const float dy = static_cast<float>(y) - cy;
            const float r2 = dx * dx + dy * dy;
            const float pulse = cfg.drop_strength * std::exp(-r2 / (2.0f * sigma2));
            const int idx = index_of(x, y, cfg.width);
            current[idx] += pulse;
            previous[idx] -= 0.35f * pulse;
        }
    }
}

__device__ __forceinline__ float border_absorption_dev(int x, int y, const KernelParams& p) {
    constexpr int band = 32;
    const int dist = min(min(x, y), min(p.width - 1 - x, p.height - 1 - y));
    if (dist >= band) {
        return p.damping;
    }
    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return p.damping + p.edge_damping * t * t;
}

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
    const float specular = powf(fmaxf(0.0f, diffuse), 24.0f);

    float intensity = 35.0f + 120.0f * wave;
    intensity *= 0.60f + 0.65f * diffuse;
    intensity += 130.0f * specular;

    const unsigned char gray = static_cast<unsigned char>(fminf(fmaxf(intensity, 0.0f), 255.0f));

    const int o = 3 * (y * p.width + x);
    image[o + 0] = gray;
    image[o + 1] = gray;
    image[o + 2] = gray;
}

struct FrameTimes {
    float simulate = 0.0f;
    float render = 0.0f;
    float d2h = 0.0f;
};

class AsyncSimulation {
public:
    explicit AsyncSimulation(const Config& cfg, bool validation = false)
        : cfg_(cfg), validation_(validation),
          cells_(static_cast<std::size_t>(cfg.width) * cfg.height),
          field_bytes_(cells_ * sizeof(float)), image_bytes_(cells_ * 3),
          block_(16, 16), grid_((cfg.width + 15) / 16, (cfg.height + 15) / 16) {
        if (cfg.width < 2 || cfg.height < 2 || cfg.steps_per_frame < 1) {
            throw std::runtime_error("Configuracion de simulacion invalida");
        }
        try {
            CUDA_CHECK(cudaFree(0));
            CUDA_CHECK(cudaStreamCreateWithFlags(&compute_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking));
            CUDA_CHECK(cudaMalloc(&previous_, field_bytes_));
            CUDA_CHECK(cudaMalloc(&current_, field_bytes_));
            CUDA_CHECK(cudaMalloc(&next_, field_bytes_));
            CUDA_CHECK(cudaEventCreate(&h2d_start_));
            CUDA_CHECK(cudaEventCreate(&h2d_end_));

            for (Slot& slot : slots_) {
                CUDA_CHECK(cudaMalloc(&slot.device_image, image_bytes_));
                CUDA_CHECK(cudaMallocHost(&slot.host_image, image_bytes_));
                CUDA_CHECK(cudaEventCreate(&slot.sim_start));
                CUDA_CHECK(cudaEventCreate(&slot.sim_end));
                CUDA_CHECK(cudaEventCreate(&slot.render_end));
                CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
                CUDA_CHECK(cudaEventCreate(&slot.copy_start));
                CUDA_CHECK(cudaEventCreate(&slot.copy_end));
                CUDA_CHECK(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming));
                if (validation_) {
                    CUDA_CHECK(cudaMalloc(&slot.device_heights, field_bytes_));
                    CUDA_CHECK(cudaMallocHost(&slot.host_heights, field_bytes_));
                }
            }

            std::vector<float> previous(cells_, 0.0f), current(cells_, 0.0f);
            add_drop(current, previous, cfg_);
            CUDA_CHECK(cudaMallocHost(&initial_previous_, field_bytes_));
            CUDA_CHECK(cudaMallocHost(&initial_current_, field_bytes_));
            std::memcpy(initial_previous_, previous.data(), field_bytes_);
            std::memcpy(initial_current_, current.data(), field_bytes_);
            CUDA_CHECK(cudaEventRecord(h2d_start_, compute_));
            CUDA_CHECK(cudaMemcpyAsync(previous_, initial_previous_, field_bytes_,
                                       cudaMemcpyHostToDevice, compute_));
            CUDA_CHECK(cudaMemcpyAsync(current_, initial_current_, field_bytes_,
                                       cudaMemcpyHostToDevice, compute_));
            CUDA_CHECK(cudaEventRecord(h2d_end_, compute_));
            CUDA_CHECK(cudaMemsetAsync(next_, 0, field_bytes_, compute_));
            CUDA_CHECK(cudaStreamSynchronize(compute_));
            CUDA_CHECK(cudaEventElapsedTime(&initial_h2d_ms_, h2d_start_, h2d_end_));
            CUDA_CHECK(cudaFreeHost(initial_previous_));
            initial_previous_ = nullptr;
            CUDA_CHECK(cudaFreeHost(initial_current_));
            initial_current_ = nullptr;

            const cv::Vec3f light = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));
            params_.width = cfg.width;
            params_.height = cfg.height;
            params_.c2 = cfg.wave_speed * cfg.wave_speed;
            params_.damping = cfg.damping;
            params_.edge_damping = cfg.edge_damping;
            params_.lx = light[0];
            params_.ly = light[1];
            params_.lz = light[2];
        } catch (...) {
            cleanup();
            throw;
        }
    }

    ~AsyncSimulation() { cleanup(); }
    AsyncSimulation(const AsyncSimulation&) = delete;
    AsyncSimulation& operator=(const AsyncSimulation&) = delete;

    void enqueue(int frame, bool capture_heights = false) {
        if (frame != next_frame_ || (capture_heights && !validation_)) {
            throw std::runtime_error("Cuadro fuera de orden o captura no habilitada");
        }
        Slot& slot = slots_[frame % 2];
        if (slot.in_use) throw std::runtime_error("El buffer todavia esta en uso");
        slot.frame = frame;
        slot.in_use = true;
        slot.waited = false;
        slot.has_heights = capture_heights;

        // Todos los pasos van en este stream porque cada uno depende del anterior.
        CUDA_CHECK(cudaEventRecord(slot.sim_start, compute_));
        for (int step = 0; step < cfg_.steps_per_frame; ++step) {
            simulate_step_kernel<<<grid_, block_, 0, compute_>>>(previous_, current_, next_, params_);
            CUDA_CHECK(cudaGetLastError());
            float* temp = previous_;
            previous_ = current_;
            current_ = next_;
            next_ = temp;
        }
        CUDA_CHECK(cudaEventRecord(slot.sim_end, compute_));
        render_frame_kernel<<<grid_, block_, 0, compute_>>>(current_, slot.device_image, params_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(slot.render_end, compute_));

        // El validador guarda una foto de la malla antes del siguiente paso.
        // Copiar directamente current_ desde otro stream provocaría una carrera.
        if (capture_heights) {
            CUDA_CHECK(cudaMemcpyAsync(slot.device_heights, current_, field_bytes_,
                                       cudaMemcpyDeviceToDevice, compute_));
        }
        CUDA_CHECK(cudaEventRecord(slot.ready, compute_));

        // La copia espera solo los datos de este cuadro, no toda la GPU.
        CUDA_CHECK(cudaStreamWaitEvent(copy_, slot.ready, 0));
        CUDA_CHECK(cudaEventRecord(slot.copy_start, copy_));
        CUDA_CHECK(cudaMemcpyAsync(slot.host_image, slot.device_image, image_bytes_,
                                   cudaMemcpyDeviceToHost, copy_));
        CUDA_CHECK(cudaEventRecord(slot.copy_end, copy_));
        if (capture_heights) {
            CUDA_CHECK(cudaMemcpyAsync(slot.host_heights, slot.device_heights, field_bytes_,
                                       cudaMemcpyDeviceToHost, copy_));
        }
        CUDA_CHECK(cudaEventRecord(slot.done, copy_));
        ++next_frame_;
    }

    void wait(int frame) {
        Slot& slot = get_slot(frame);
        CUDA_CHECK(cudaEventSynchronize(slot.done));
        slot.waited = true;
    }

    const unsigned char* image(int frame) const {
        return readable_slot(frame).host_image;
    }

    const float* heights(int frame) const {
        const Slot& slot = readable_slot(frame);
        if (!slot.has_heights) throw std::runtime_error("Este cuadro no tiene captura de alturas");
        return slot.host_heights;
    }

    FrameTimes times(int frame) const {
        const Slot& slot = readable_slot(frame);
        FrameTimes t;
        CUDA_CHECK(cudaEventElapsedTime(&t.simulate, slot.sim_start, slot.sim_end));
        CUDA_CHECK(cudaEventElapsedTime(&t.render, slot.sim_end, slot.render_end));
        CUDA_CHECK(cudaEventElapsedTime(&t.d2h, slot.copy_start, slot.copy_end));
        return t;
    }

    void release(int frame) {
        readable_slot(frame);
        get_slot(frame).in_use = false;
    }

    std::size_t image_bytes() const { return image_bytes_; }
    float initial_h2d_ms() const { return initial_h2d_ms_; }

private:
    struct Slot {
        unsigned char *device_image = nullptr, *host_image = nullptr;
        float *device_heights = nullptr, *host_heights = nullptr;
        cudaEvent_t sim_start = nullptr, sim_end = nullptr, render_end = nullptr;
        cudaEvent_t ready = nullptr, copy_start = nullptr, copy_end = nullptr, done = nullptr;
        int frame = -1;
        bool in_use = false, waited = false, has_heights = false;
    };

    Slot& get_slot(int frame) {
        if (frame < 0) throw std::runtime_error("Numero de cuadro invalido");
        Slot& slot = slots_[frame % 2];
        if (!slot.in_use || slot.frame != frame) throw std::runtime_error("Cuadro no disponible");
        return slot;
    }

    const Slot& readable_slot(int frame) const {
        if (frame < 0) throw std::runtime_error("Numero de cuadro invalido");
        const Slot& slot = slots_[frame % 2];
        if (!slot.in_use || slot.frame != frame || !slot.waited) {
            throw std::runtime_error("Hay que esperar la copia antes de leer el cuadro");
        }
        return slot;
    }

    void cleanup() noexcept {
        // Terminamos el trabajo pendiente antes de liberar sus buffers.
        if (compute_) cudaStreamSynchronize(compute_);
        if (copy_) cudaStreamSynchronize(copy_);
        for (Slot& slot : slots_) {
            if (slot.device_image) cudaFree(slot.device_image);
            if (slot.host_image) cudaFreeHost(slot.host_image);
            if (slot.device_heights) cudaFree(slot.device_heights);
            if (slot.host_heights) cudaFreeHost(slot.host_heights);
            if (slot.sim_start) cudaEventDestroy(slot.sim_start);
            if (slot.sim_end) cudaEventDestroy(slot.sim_end);
            if (slot.render_end) cudaEventDestroy(slot.render_end);
            if (slot.ready) cudaEventDestroy(slot.ready);
            if (slot.copy_start) cudaEventDestroy(slot.copy_start);
            if (slot.copy_end) cudaEventDestroy(slot.copy_end);
            if (slot.done) cudaEventDestroy(slot.done);
        }
        if (previous_) cudaFree(previous_);
        if (current_) cudaFree(current_);
        if (next_) cudaFree(next_);
        if (initial_previous_) cudaFreeHost(initial_previous_);
        if (initial_current_) cudaFreeHost(initial_current_);
        if (h2d_start_) cudaEventDestroy(h2d_start_);
        if (h2d_end_) cudaEventDestroy(h2d_end_);
        if (compute_) cudaStreamDestroy(compute_);
        if (copy_) cudaStreamDestroy(copy_);
    }

    Config cfg_;
    bool validation_;
    std::size_t cells_, field_bytes_, image_bytes_;
    dim3 block_, grid_;
    KernelParams params_{};
    float *previous_ = nullptr, *current_ = nullptr, *next_ = nullptr;
    float *initial_previous_ = nullptr, *initial_current_ = nullptr;
    cudaStream_t compute_ = nullptr, copy_ = nullptr;
    cudaEvent_t h2d_start_ = nullptr, h2d_end_ = nullptr;
    Slot slots_[2];
    int next_frame_ = 0;
    float initial_h2d_ms_ = 0.0f;
};

} // namespace drop_streams
