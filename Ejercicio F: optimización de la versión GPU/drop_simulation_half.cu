// Derivado de D/E en el commit dbe0b55. Solo se evalúa almacenamiento FP16.
// Modelo físico aproximado:
// ecuación de onda en 2D con amortiguamiento
//
// EXPERIMENTO F: mallas FP16; aritmética de kernels FP32.
// Único cambio evaluado: almacenamiento half; sin shared memory, pinned memory ni streams.
//
// Plataforma: Jetson (GCC 7.5.0, CUDA 10.2). nvcc 10.2 solo admite hasta C++14.


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <experimental/filesystem>   // GCC 7.5: filesystem aún es experimental (enlazar con -lstdc++fs)
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            throw std::runtime_error(std::string("CUDA error en ") + __FILE__ +   \
                                     ":" + std::to_string(__LINE__) + " -> " +    \
                                     cudaGetErrorString(err__));                  \
        }                                                                         \
    } while (0)

namespace {

// Estructura inicial. Parámetros de la simulación y del video.
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
    std::string output = "output/drop_simulation_half.mp4";
};

// Parámetros planos (POD) que se pasan por valor a los kernels.
struct KernelParams {
    int width;
    int height;
    float c2;            // wave_speed^2
    float damping;
    float edge_damping;
    float lx, ly, lz;    // dirección de la luz, ya normalizada en el host
};

int index_of(int x, int y, int width) {
    return y * width + x;
}

// La perturbación inicial se calcula en CPU (se ejecuta una sola vez)
// y se copia a la GPU junto con los demás datos iniciales.
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

// ----------------------------------------------------------------------------
// Absorción de bordes (versión device, idéntica a la de CPU)
// ----------------------------------------------------------------------------
#include "kernels_half.cuh"

}  // namespace

int main() {
    try {
        const Config cfg{};

        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
        const std::size_t bytes_field = cells * sizeof(__half);
        const std::size_t bytes_image = cells * 3 * sizeof(unsigned char);

        namespace fs = std::experimental::filesystem;
        fs::path output_path(cfg.output);
        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }

        // Estado inicial en host
        std::vector<float> h_previous(cells, 0.0f);
        std::vector<float> h_current(cells, 0.0f);
        add_drop(h_current, h_previous, cfg);

        // Inicializa el contexto CUDA antes de medir (evita contar su arranque)
        CUDA_CHECK(cudaFree(0));

        // Reserva de memoria en GPU
        __half *d_previous = nullptr, *d_current = nullptr, *d_next = nullptr;
        unsigned char* d_image = nullptr;
        CUDA_CHECK(cudaMalloc(&d_previous, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_current, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_next, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_image, bytes_image));

        // Copia Host -> Device (una sola vez)
        std::vector<__half> initial_previous(cells), initial_current(cells);
        for (std::size_t i = 0; i < cells; ++i) {
            initial_previous[i] = __float2half_rn(h_previous[i]);
            initial_current[i] = __float2half_rn(h_current[i]);
        }
        CUDA_CHECK(cudaMemcpy(d_previous, initial_previous.data(), bytes_field, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_current, initial_current.data(), bytes_field, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_next, 0, bytes_field));

        // Parámetros de kernel
        const cv::Vec3f light = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));
        KernelParams params{};
        params.width = cfg.width;
        params.height = cfg.height;
        params.c2 = cfg.wave_speed * cfg.wave_speed;
        params.damping = cfg.damping;
        params.edge_damping = cfg.edge_damping;
        params.lx = light[0];
        params.ly = light[1];
        params.lz = light[2];

        // Configuración de bloques e hilos (grid 2D, un hilo por celda/píxel)
        const dim3 block(16, 16);
        const dim3 grid((cfg.width + block.x - 1) / block.x,
                        (cfg.height + block.y - 1) / block.y);

        // Escritor de video (flujo existente, sin cambios)
        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height));
        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        // Imagen en host reutilizada en todos los cuadros (continua, CV_8UC3)
        cv::Mat image(cfg.height, cfg.width, CV_8UC3);

        // Eventos CUDA para medir por componente
        cudaEvent_t e0, e1, e2, e3;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2));
        CUDA_CHECK(cudaEventCreate(&e3));

        double ms_sim = 0.0, ms_render = 0.0, ms_d2h = 0.0, ms_host = 0.0;

        const auto start = std::chrono::steady_clock::now();

        for (int frame = 0; frame < total_frames; ++frame) {
            // Simulación (kernel 1)
            CUDA_CHECK(cudaEventRecord(e0));
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step_kernel<<<grid, block>>>(d_previous, d_current, d_next, params);
                CUDA_CHECK(cudaGetLastError());

                // Rotación de buffers (solo punteros), equivalente a los swap de CPU:
                // previous <- current, current <- next, next <- previous(viejo)
                __half* tmp = d_previous;
                d_previous = d_current;
                d_current = d_next;
                d_next = tmp;
            }
            CUDA_CHECK(cudaEventRecord(e1));

            // Render (kernel 2)
            render_frame_kernel<<<grid, block>>>(d_current, d_image, params);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(e2));

            // Copia Device -> Host del cuadro
            CUDA_CHECK(cudaMemcpy(image.data, d_image, bytes_image, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaEventRecord(e3));
            CUDA_CHECK(cudaEventSynchronize(e3));

            float t_sim = 0.f, t_render = 0.f, t_d2h = 0.f;
            CUDA_CHECK(cudaEventElapsedTime(&t_sim, e0, e1));
            CUDA_CHECK(cudaEventElapsedTime(&t_render, e1, e2));
            CUDA_CHECK(cudaEventElapsedTime(&t_d2h, e2, e3));
            ms_sim += t_sim;
            ms_render += t_render;
            ms_d2h += t_d2h;

            // Parte en CPU: texto + escritura del video
            const auto h0 = std::chrono::steady_clock::now();
            cv::putText(image,
                        "GPU float16 | frame " + std::to_string(frame),
                        cv::Point(18, 32),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.65,
                        cv::Scalar(235, 235, 235),
                        1,
                        cv::LINE_AA);
            writer.write(image);
            const auto h1 = std::chrono::steady_clock::now();
            ms_host += std::chrono::duration<double, std::milli>(h1 - h0).count();

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double simulated_steps = static_cast<double>(total_frames) * cfg.steps_per_frame;

        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << simulated_steps << '\n';
        std::cout << "Rendimiento: " << simulated_steps / elapsed << " pasos/s\n";

        std::cout << "\n--- Desglose por cuadro (ms, promedio) ---\n";
        std::cout << "Kernel simulate_step : " << ms_sim / total_frames << '\n';
        std::cout << "Kernel render_frame  : " << ms_render / total_frames << '\n';
        std::cout << "Copia D2H            : " << ms_d2h / total_frames << '\n';
        std::cout << "putText + write (CPU): " << ms_host / total_frames << '\n';

        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaEventDestroy(e2));
        CUDA_CHECK(cudaEventDestroy(e3));
        CUDA_CHECK(cudaFree(d_previous));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_image));
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
