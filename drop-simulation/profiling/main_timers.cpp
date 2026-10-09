#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

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
    std::string output = "output/drop_simulation.mp4";
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

float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32;
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band);
    return cfg.damping + cfg.edge_damping * t * t;
}

void simulate_step(const std::vector<float>& previous,
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

cv::Mat render_frame(const std::vector<float>& height, const Config& cfg, int frame_number) {
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
            const float wave = std::clamp(0.5f + 1.8f * height[index_of(x, y, cfg.width)], 0.0f, 1.0f);
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(std::clamp(intensity, 0.0f, 255.0f));
            image.at<cv::Vec3b>(y, x) = cv::Vec3b(gray, gray, gray);
        }
    }

    cv::putText(image,
                "CPU float32 | frame " + std::to_string(frame_number),
                cv::Point(18, 32),
                cv::FONT_HERSHEY_SIMPLEX,
                0.65,
                cv::Scalar(235, 235, 235),
                1,
                cv::LINE_AA);

    return image;
}

}  // namespace

int main() {
    try {
        const Config cfg;
        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);

        std::filesystem::path output_path(cfg.output);
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::vector<float> previous(cells, 0.0f);
        std::vector<float> current(cells, 0.0f);
        std::vector<float> next(cells, 0.0f);

        // --- Instrumentación manual: acumuladores de tiempo (segundos) por función ---
        using clk = std::chrono::steady_clock;
        const auto seconds_between = [](clk::time_point a, clk::time_point b) {
            return std::chrono::duration<double>(b - a).count();
        };
        double t_add_drop = 0.0;
        double t_simulate = 0.0;
        double t_render = 0.0;
        double t_write = 0.0;

        const auto ta = clk::now();
        add_drop(current, previous, cfg);
        t_add_drop = seconds_between(ta, clk::now());

        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height));

        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        const auto start = std::chrono::steady_clock::now();

        for (int frame = 0; frame < total_frames; ++frame) {
            const auto t0 = clk::now();
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step(previous, current, next, cfg);
                previous.swap(current);
                current.swap(next);
            }
            const auto t1 = clk::now();

            const cv::Mat image = render_frame(current, cfg, frame);
            const auto t2 = clk::now();

            writer.write(image);
            const auto t3 = clk::now();

            t_simulate += seconds_between(t0, t1);
            t_render += seconds_between(t1, t2);
            t_write += seconds_between(t2, t3);

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

        // --- Reporte de la instrumentación manual ---
        const double t_other = elapsed - (t_simulate + t_render + t_write);
        const auto row = [&](const char* name, double t) {
            std::cout << std::left << std::setw(16) << name << std::right << std::fixed
                      << std::setprecision(4) << std::setw(10) << t << " s"
                      << std::setprecision(2) << std::setw(9) << 100.0 * t / elapsed << " %"
                      << std::setprecision(3) << std::setw(10) << 1000.0 * t / total_frames
                      << " ms/frame\n";
        };
        std::cout << "\n--- Perfil manual (bucle principal) ---\n";
        row("simulate_step", t_simulate);
        row("render_frame", t_render);
        row("writer.write", t_write);
        row("otros (print)", t_other);
        row("TOTAL bucle", elapsed);
        std::cout << "(add_drop, fuera del bucle: " << std::setprecision(6) << t_add_drop << " s)\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
