// Comparamos el renderizado con multiplicaciones contra la referencia CPU original.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

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

// Conservamos las tolerancias de E.
constexpr double TOL_ALTURA_MAX = 1e-4;
constexpr double TOL_IMG_MAX = 2.0;

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
};

struct KernelParams {
    int width;
    int height;
    float c2;
    float damping;
    float edge_damping;
    float lx, ly, lz;
};

inline int index_of(int x, int y, int width) { return y * width + x; }

inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

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

float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }
    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

// La CPU conserva la ecuación original en float.
void simulate_step_cpu(const std::vector<float>& previous,
                       const std::vector<float>& current,
                       std::vector<float>& next,
                       const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;
    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            const float laplacian =
                current[idx - 1] + current[idx + 1] +
                current[idx - cfg.width] + current[idx + cfg.width] -
                4.0f * current[idx];
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);
            next[idx] = 2.0f * current[idx] - previous[idx] +
                        c2 * laplacian -
                        local_damping * velocity;
        }
    }
}

// La referencia CPU sigue usando pow; no lleva el cambio de la GPU.
cv::Mat render_frame_cpu(const std::vector<float>& height, const Config& cfg) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);
    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f));

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            const float dx = height[index_of(xm, y, cfg.width)] - height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] - height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal = cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f));

            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            const float wave = clampf(0.5f + 1.8f * height[index_of(x, y, cfg.width)], 0.0f, 1.0f);
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(clampf(intensity, 0.0f, 255.0f));
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }
    return image;
}

#include "kernels_pow24.cuh"

struct Metrics {
    double h_mae = 0.0;
    double h_max = 0.0;
    double h_amp = 0.0;
    double img_mae = 0.0;
    double img_max = 0.0;
    double img_pct = 0.0;
};

// Revisamos que las alturas sigan coincidiendo.
Metrics compare_heights(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    Metrics m;
    double sum = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            throw std::runtime_error("Valor no finito en la malla, celda " + std::to_string(i));
        }
        const double e = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpu[i]));
        sum += e;
        m.h_max = std::max(m.h_max, e);
        m.h_amp = std::max(m.h_amp, static_cast<double>(std::fabs(cpu[i])));
    }
    m.h_mae = sum / static_cast<double>(cpu.size());
    return m;
}

// Comparamos un canal porque los tres tienen el mismo gris.
void compare_images(const cv::Mat& cpu_img, const cv::Mat& gpu_img, Metrics& m, cv::Mat& diff) {
    cv::Mat c0, g0;
    cv::extractChannel(cpu_img, c0, 0);
    cv::extractChannel(gpu_img, g0, 0);
    cv::absdiff(c0, g0, diff);
    m.img_mae = cv::mean(diff)[0];
    double maxv = 0.0;
    cv::minMaxLoc(diff, nullptr, &maxv);
    m.img_max = maxv;
    m.img_pct = 100.0 * static_cast<double>(cv::countNonZero(diff)) /
                static_cast<double>(diff.total());
}

void save_comparison(const cv::Mat& cpu_img, const cv::Mat& gpu_img, const cv::Mat& diff,
                     int frame, const std::string& dir) {
    cv::Mat c0, g0, amp, panel;
    cv::extractChannel(cpu_img, c0, 0);
    cv::extractChannel(gpu_img, g0, 0);
    // Amplificamos la diferencia para verla en la imagen.
    diff.convertTo(amp, CV_8U, 40.0);
    cv::hconcat(std::vector<cv::Mat>{c0, g0, amp}, panel);

    char name[256];
    std::snprintf(name, sizeof(name), "%s/comparacion_f%04d.png", dir.c_str(), frame);
    cv::imwrite(name, panel);
}

}

int main(int argc, char** argv) {
    try {
        const Config cfg{};
        int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        if (argc > 1) total_frames = std::max(1, std::atoi(argv[1]));

        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);
        const std::size_t bytes_field = cells * sizeof(float);
        const std::size_t bytes_image = cells * 3;

        namespace fs = std::experimental::filesystem;
        const std::string out_dir = "validacion_pow24";
        fs::create_directories(out_dir);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
        std::cout << "Cuadros: " << total_frames << " | Tolerancias: altura max < " << TOL_ALTURA_MAX
                  << ", imagen max <= " << TOL_IMG_MAX << " niveles\n\n";

        std::vector<float> previous(cells, 0.0f), current(cells, 0.0f), next(cells, 0.0f);
        add_drop(current, previous, cfg);

        float *d_previous = nullptr, *d_current = nullptr, *d_next = nullptr;
        unsigned char* d_image = nullptr;
        CUDA_CHECK(cudaMalloc(&d_previous, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_current, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_next, bytes_field));
        CUDA_CHECK(cudaMalloc(&d_image, bytes_image));
        CUDA_CHECK(cudaMemcpy(d_previous, previous.data(), bytes_field, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_current, current.data(), bytes_field, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_next, 0, bytes_field));

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

        const dim3 block(16, 16);
        const dim3 grid((cfg.width + block.x - 1) / block.x, (cfg.height + block.y - 1) / block.y);

        std::vector<float> gpu_heights(cells, 0.0f);
        cv::Mat gpu_img(cfg.height, cfg.width, CV_8UC3);

        // Usamos los mismos cuadros de control que E.
        const std::set<int> save_frames = {0, 100, 300, 600, total_frames - 1};
        const int check_every = 30;

        std::ofstream csv(out_dir + "/metricas.csv");
        csv << "frame,h_mae,h_max,h_amp,h_max_relativo,img_mae,img_max,img_pct_difieren\n";
        csv << std::scientific << std::setprecision(6);

        Metrics worst;
        bool pass = true;

        std::cout << std::setw(7) << "frame" << std::setw(14) << "h_MAE" << std::setw(14) << "h_MAX"
                  << std::setw(12) << "img_MAE" << std::setw(10) << "img_MAX" << std::setw(12)
                  << "%difieren" << '\n';

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step_cpu(previous, current, next, cfg);
                previous.swap(current);
                current.swap(next);

                simulate_step_kernel<<<grid, block>>>(d_previous, d_current, d_next, params);
                CUDA_CHECK(cudaGetLastError());
                float* tmp = d_previous;
                d_previous = d_current;
                d_current = d_next;
                d_next = tmp;
            }

            const bool do_save = save_frames.count(frame) > 0;
            if (!(frame % check_every == 0 || frame == total_frames - 1 || do_save)) continue;

            CUDA_CHECK(cudaMemcpy(gpu_heights.data(), d_current, bytes_field, cudaMemcpyDeviceToHost));
            Metrics m = compare_heights(current, gpu_heights);

            render_frame_kernel<<<grid, block>>>(d_current, d_image, params);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(gpu_img.data, d_image, bytes_image, cudaMemcpyDeviceToHost));
            const cv::Mat cpu_img = render_frame_cpu(current, cfg);

            cv::Mat diff;
            compare_images(cpu_img, gpu_img, m, diff);

            const double rel = (m.h_amp > 0.0) ? m.h_max / m.h_amp : 0.0;
            csv << frame << ',' << m.h_mae << ',' << m.h_max << ',' << m.h_amp << ',' << rel << ','
                << m.img_mae << ',' << m.img_max << ',' << m.img_pct << '\n';

            std::cout << std::setw(7) << frame << std::scientific << std::setprecision(3)
                      << std::setw(14) << m.h_mae << std::setw(14) << m.h_max << std::fixed
                      << std::setprecision(4) << std::setw(12) << m.img_mae << std::setprecision(0)
                      << std::setw(10) << m.img_max << std::setprecision(3) << std::setw(12)
                      << m.img_pct << '\n';

            worst.h_mae = std::max(worst.h_mae, m.h_mae);
            worst.h_max = std::max(worst.h_max, m.h_max);
            worst.img_mae = std::max(worst.img_mae, m.img_mae);
            worst.img_max = std::max(worst.img_max, m.img_max);
            worst.img_pct = std::max(worst.img_pct, m.img_pct);

            // El cambio debe cumplir ambas tolerancias.
            if (m.h_max >= TOL_ALTURA_MAX || m.img_max > TOL_IMG_MAX) pass = false;
            if (do_save) save_comparison(cpu_img, gpu_img, diff, frame, out_dir);
        }

        std::cout << "\n=== Peor caso en todos los cuadros de control ===\n"
                  << std::scientific << std::setprecision(3)
                  << "Altura  : MAE max = " << worst.h_mae << " | error abs. maximo = " << worst.h_max << '\n'
                  << std::fixed << std::setprecision(4)
                  << "Imagen  : MAE max = " << worst.img_mae << " | diferencia max = " << worst.img_max
                  << " niveles | % pixeles que difieren (max) = " << worst.img_pct << "\n"
                  << "Resultado: " << (pass ? "VALIDACION CORRECTA" : "FUERA DE TOLERANCIA") << '\n'
                  << "Archivos : " << out_dir << "/metricas.csv y " << out_dir << "/comparacion_f*.png\n";

        CUDA_CHECK(cudaFree(d_previous));
        CUDA_CHECK(cudaFree(d_current));
        CUDA_CHECK(cudaFree(d_next));
        CUDA_CHECK(cudaFree(d_image));
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
