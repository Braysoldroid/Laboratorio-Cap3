// Probamos streams sobre la GPU base de D.
#include "pipeline_streams.cuh"

#include <chrono>
#include <cstring>
#include <experimental/filesystem>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

int main() {
    try {
        using namespace drop_streams;
        const Config cfg{};
        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        namespace fs = std::experimental::filesystem;
        fs::create_directories(fs::path(cfg.output).parent_path());

        AsyncSimulation gpu(cfg);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
        std::cout << "Streams: computo y copia | Buffers de imagen: 2\n";

        cv::VideoWriter writer(cfg.output, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                               static_cast<double>(cfg.fps), cv::Size(cfg.width, cfg.height));
        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el video: " + cfg.output);
        }
        cv::Mat image(cfg.height, cfg.width, CV_8UC3);
        double ms_sim = 0.0, ms_render = 0.0, ms_d2h = 0.0;
        double ms_host_copy = 0.0, ms_host = 0.0;

        // El total incluye llenar y vaciar los dos buffers, como parte del bucle.
        const auto start = std::chrono::steady_clock::now();
        gpu.enqueue(0);
        if (total_frames > 1) gpu.enqueue(1);

        for (int frame = 0; frame < total_frames; ++frame) {
            gpu.wait(frame);
            const FrameTimes times = gpu.times(frame);
            ms_sim += times.simulate;
            ms_render += times.render;
            ms_d2h += times.d2h;

            // En Tegra X1 la memoria pinned no tiene cache de CPU.
            // OpenCV trabaja sobre una Mat normal después de esta copia.
            const auto h0 = std::chrono::steady_clock::now();
            std::memcpy(image.data, gpu.image(frame), gpu.image_bytes());
            const auto h1 = std::chrono::steady_clock::now();
            ms_host_copy += std::chrono::duration<double, std::milli>(h1 - h0).count();

            cv::putText(image, "GPU float32 | frame " + std::to_string(frame),
                        cv::Point(18, 32), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(235, 235, 235), 1, cv::LINE_AA);
            writer.write(image);
            const auto h2 = std::chrono::steady_clock::now();
            ms_host += std::chrono::duration<double, std::milli>(h2 - h1).count();

            // Ya terminamos de leer este buffer; ahora se puede reutilizar.
            gpu.release(frame);
            if (frame + 2 < total_frames) gpu.enqueue(frame + 2);

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double steps = static_cast<double>(total_frames) * cfg.steps_per_frame;
        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << steps << '\n';
        std::cout << "Rendimiento: " << steps / elapsed << " pasos/s\n";
        std::cout << "Copia H2D inicial (fuera del bucle): " << gpu.initial_h2d_ms() << " ms\n";
        std::cout << "\n--- Desglose por cuadro (ms, promedio) ---\n";
        std::cout << "Kernel simulate_step : " << ms_sim / total_frames << '\n';
        std::cout << "Kernel render_frame  : " << ms_render / total_frames << '\n';
        std::cout << "Copia D2H            : " << ms_d2h / total_frames << '\n';
        std::cout << "Copia pinned -> Mat  : " << ms_host_copy / total_frames << '\n';
        std::cout << "putText + write (CPU): " << ms_host / total_frames << '\n';
        std::cout << "Los componentes se pueden solapar; su suma no es el tiempo total.\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
