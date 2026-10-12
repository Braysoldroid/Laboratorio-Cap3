// Comparamos los cuadros que salen del mismo pipeline usado para el video.
#include "pipeline_streams.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <experimental/filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
using drop_streams::Config;
constexpr double TOL_ALTURA_MAX = 1e-4;
constexpr double TOL_IMG_MAX = 2.0;

// Referencia CPU de E, sin cambiar los cálculos.
inline int index_of(int x, int y, int width) { return y * width + x; }

inline float clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }

void add_drop_cpu(std::vector<float>& current, std::vector<float>& previous, const Config& cfg) {
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


// Medimos las diferencias sin agregar el texto del video.
struct Metrics {
    double h_mae = 0.0;
    double h_max = 0.0;
    double h_amp = 0.0;
    double img_mae = 0.0;
    double img_max = 0.0;
    double img_pct = 0.0;
};

Metrics compare_heights(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    Metrics m;
    double sum = 0.0;
    for (std::size_t i = 0; i < cpu.size(); ++i) {
        if (!std::isfinite(cpu[i]) || !std::isfinite(gpu[i])) {
            m.h_mae = m.h_max = std::numeric_limits<double>::infinity();
            return m;
        }
        const double e = std::fabs(static_cast<double>(cpu[i]) - static_cast<double>(gpu[i]));
        sum += e;
        m.h_max = std::max(m.h_max, e);
        m.h_amp = std::max(m.h_amp, static_cast<double>(std::fabs(cpu[i])));
    }
    m.h_mae = sum / static_cast<double>(cpu.size());
    return m;
}

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
    diff.convertTo(amp, CV_8U, 40.0);
    cv::hconcat(std::vector<cv::Mat>{c0, g0, amp}, panel);

    char name[256];
    std::snprintf(name, sizeof(name), "%s/comparacion_f%04d.png", dir.c_str(), frame);
    if (!cv::imwrite(name, panel)) throw std::runtime_error("No se pudo guardar la comparacion");
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Config cfg{};
        int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        if (argc > 1) total_frames = std::max(1, std::atoi(argv[1]));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * cfg.height;
        const std::string out_dir = "validacion_streams";
        std::experimental::filesystem::create_directories(out_dir);

        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
        std::cout << "Cuadros: " << total_frames << " | Tolerancias: altura max < "
                  << TOL_ALTURA_MAX << ", imagen max <= " << TOL_IMG_MAX << " niveles\n\n";

        std::vector<float> previous(cells, 0.0f), current(cells, 0.0f), next(cells, 0.0f);
        add_drop_cpu(current, previous, cfg);
        drop_streams::AsyncSimulation gpu(cfg, true);
        std::vector<float> gpu_heights(cells, 0.0f);
        cv::Mat gpu_img(cfg.height, cfg.width, CV_8UC3);
        const std::set<int> save_frames = {0, 100, 300, 600, total_frames - 1};
        const auto check_frame = [&](int frame) {
            return frame % 30 == 0 || frame == total_frames - 1 || save_frames.count(frame) > 0;
        };

        std::ofstream csv(out_dir + "/metricas.csv");
        if (!csv) throw std::runtime_error("No se pudo abrir metricas.csv");
        csv << "frame,h_mae,h_max,h_amp,h_max_relativo,img_mae,img_max,img_pct_difieren\n";
        csv << std::scientific << std::setprecision(6);
        Metrics worst;
        bool pass = true;
        std::cout << std::setw(7) << "frame" << std::setw(14) << "h_MAE"
                  << std::setw(14) << "h_MAX" << std::setw(12) << "img_MAE"
                  << std::setw(10) << "img_MAX" << std::setw(12) << "%difieren" << '\n';

        gpu.enqueue(0, check_frame(0));
        if (total_frames > 1) gpu.enqueue(1, check_frame(1));

        for (int frame = 0; frame < total_frames; ++frame) {
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step_cpu(previous, current, next, cfg);
                previous.swap(current);
                current.swap(next);
            }
            gpu.wait(frame);

            if (check_frame(frame)) {
                // La captura pertenece a este cuadro, aunque la GPU ya vaya adelantada.
                std::memcpy(gpu_heights.data(), gpu.heights(frame), cells * sizeof(float));
                std::memcpy(gpu_img.data, gpu.image(frame), gpu.image_bytes());
                Metrics m = compare_heights(current, gpu_heights);
                const cv::Mat cpu_img = render_frame_cpu(current, cfg);
                cv::Mat diff;
                compare_images(cpu_img, gpu_img, m, diff);

                const double relative = m.h_amp > 0.0 ? m.h_max / m.h_amp : 0.0;
                csv << frame << ',' << m.h_mae << ',' << m.h_max << ',' << m.h_amp << ','
                    << relative << ',' << m.img_mae << ',' << m.img_max << ',' << m.img_pct << '\n';
                std::cout << std::setw(7) << frame << std::scientific << std::setprecision(3)
                          << std::setw(14) << m.h_mae << std::setw(14) << m.h_max
                          << std::fixed << std::setprecision(4) << std::setw(12) << m.img_mae
                          << std::setprecision(0) << std::setw(10) << m.img_max
                          << std::setprecision(3) << std::setw(12) << m.img_pct << '\n';

                worst.h_mae = std::max(worst.h_mae, m.h_mae);
                worst.h_max = std::max(worst.h_max, m.h_max);
                worst.img_mae = std::max(worst.img_mae, m.img_mae);
                worst.img_max = std::max(worst.img_max, m.img_max);
                worst.img_pct = std::max(worst.img_pct, m.img_pct);
                if (!std::isfinite(m.h_max) || m.h_max >= TOL_ALTURA_MAX ||
                    m.img_max > TOL_IMG_MAX) pass = false;
                if (save_frames.count(frame) > 0) save_comparison(cpu_img, gpu_img, diff, frame, out_dir);
            }

            gpu.release(frame);
            if (frame + 2 < total_frames) gpu.enqueue(frame + 2, check_frame(frame + 2));
        }
        std::cout << "\n=== Peor caso en todos los cuadros de control ===\n"
                  << std::scientific << std::setprecision(3)
                  << "Altura  : MAE max = " << worst.h_mae << " | error abs. maximo = " << worst.h_max << '\n'
                  << std::fixed << std::setprecision(4)
                  << "Imagen  : MAE max = " << worst.img_mae << " | diferencia max = " << worst.img_max
                  << " niveles | % pixeles que difieren (max) = " << worst.img_pct << '\n'
                  << "Resultado: " << (pass ? "VALIDACION CORRECTA" : "FUERA DE TOLERANCIA") << '\n'
                  << "Archivos : " << out_dir << "/metricas.csv y " << out_dir << "/comparacion_f*.png\n";
        return pass ? 0 : 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
