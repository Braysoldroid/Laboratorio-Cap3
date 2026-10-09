// Modelo físico aproximado:
// ecuación de onda en 2D con amortiguamiento

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace {

// Estructura inicial. Parámetros de la simulación y del video.
struct Config {
    int width = 640;                                    // Ancho
    int height = 640;                                   // Alto
    double seconds = 30.0;                              // Duración del vídeo
    int fps = 30;                                       // Frames por segundo
    int steps_per_frame = 1;                            // Pasos por frame
    float wave_speed = 0.45f;                           // Velocidad de la ola, controla propagación
    float damping = 0.006f;                             // Amortiguación
    float edge_damping = 0.035f;                        // Borde de amortiguación
    float drop_radius = 18.0f;                          // Radio de caída
    float drop_strength = 1.0f;                         // Fuerza de caída
    std::string output = "output/drop_simulation.mp4";
};

// Indice del mapa de alturas
int index_of(int x, int y, int width) {
    return y * width + x;
}

// Función que inicia la perturbación inicial de la gota
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

// Función que calcula la absorción de los bordes
float border_absorption(int x, int y, const Config& cfg) {
    constexpr int band = 32; // Ancho de la franja absorbente (px)
    const int dist = std::min({x, y, cfg.width - 1 - x, cfg.height - 1 - y});
    // Fuera de la franja, solo hay fricción normal
    if (dist >= band) {
        return cfg.damping;
    }

    const float t = 1.0f - static_cast<float>(dist) / static_cast<float>(band); // 0 al entrar en la franja, 1 en el borde
    return cfg.damping + cfg.edge_damping * t * t; // Fricción que crece
}

// Función que actualiza la función de onda 2D
void simulate_step(const std::vector<float>& previous,
                   const std::vector<float>& current,
                   std::vector<float>& next,
                   const Config& cfg) {
    const float c2 = cfg.wave_speed * cfg.wave_speed;

    std::fill(next.begin(), next.end(), 0.0f);

    for (int y = 1; y < cfg.height - 1; ++y) {
        for (int x = 1; x < cfg.width - 1; ++x) {
            const int idx = index_of(x, y, cfg.width);
            // El laplaciano compara cada celda son sus 4 vecinas. 
            const float laplacian =
                current[idx - 1] + current[idx + 1] + // Izquierda + derecha
                current[idx - cfg.width] + current[idx + cfg.width] - // Arriba + abajo -
                4.0f * current[idx]; // 4 veces la celda actual
            
            // Se calcula la velocidad como la diferencia entre la celdad actual y la anterior
            const float velocity = current[idx] - previous[idx];
            const float local_damping = border_absorption(x, y, cfg);
            
            // Se calcula la celda siguiente como:
            //2·current - previous + c²·laplaciano - fricción.
            next[idx] = 2.0f * current[idx] - previous[idx] + // Inercia
                        c2 * laplacian - // Fuerza de las celdas vecinas
                        local_damping * velocity; // Fricción
        }
    }
}

// Función que renderiza cada cuadro del video
// 
cv::Mat render_frame(const std::vector<float>& height, const Config& cfg, int frame_number) {
    cv::Mat image(cfg.height, cfg.width, CV_8UC3);

    // Dirección hacia la luz
    const cv::Vec3f light_dir = cv::normalize(cv::Vec3f(-0.35f, -0.55f, 0.76f)); // Arriba a la izquierda y muy de frente hacia el espectador

    for (int y = 0; y < cfg.height; ++y) {
        for (int x = 0; x < cfg.width; ++x) {

            // Coordenadas de las vecinas
            const int xm = std::max(0, x - 1);
            const int xp = std::min(cfg.width - 1, x + 1);
            const int ym = std::max(0, y - 1);
            const int yp = std::min(cfg.height - 1, y + 1);

            // Pendiente de la superficie
            const float dx = height[index_of(xm, y, cfg.width)] - height[index_of(xp, y, cfg.width)];
            const float dy = height[index_of(x, ym, cfg.width)] - height[index_of(x, yp, cfg.width)];
            const cv::Vec3f normal = cv::normalize(cv::Vec3f(2.8f * dx, 2.8f * dy, 1.0f)); // Vector normal a la superficie
            
            // Iluinación difusa para relieve
            const float diffuse = std::max(0.0f, normal.dot(light_dir));
            // Brillo según la altura
            const float wave = std::clamp(0.5f + 1.8f * height[index_of(x, y, cfg.width)], 0.0f, 1.0f);
            // Reflejo especular para las zonas que están directo al rayo de luz
            const float specular = std::pow(std::max(0.0f, diffuse), 24.0f);

            float intensity = 35.0f + 120.0f * wave;
            intensity *= 0.60f + 0.65f * diffuse;
            intensity += 130.0f * specular;

            const auto gray = static_cast<unsigned char>(std::clamp(intensity, 0.0f, 255.0f)); // Escala de grises
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

//Función principal
int main() {
    try {
        const Config cfg;

        // 30*30 = 900 imágenes en todo el vídeo
        const int total_frames = static_cast<int>(std::round(cfg.seconds * cfg.fps));
        // 640*640 = 409.600 tamaño de cada vector/mapa de alturas
        const std::size_t cells = static_cast<std::size_t>(cfg.width) * static_cast<std::size_t>(cfg.height);

        std::filesystem::path output_path(cfg.output); // Crea la carpeta output
        if (output_path.has_parent_path()) {
            std::filesystem::create_directories(output_path.parent_path());
        }

        std::vector<float> previous(cells, 0.0f); // Vector de alturas previo
        std::vector<float> current(cells, 0.0f); // Vector de alturas actual
        std::vector<float> next(cells, 0.0f); // Vector de alturas siguiente

        add_drop(current, previous, cfg); // Perturbación inicial de la gota
        
        // Se llama al escritor de video
        cv::VideoWriter writer(
            cfg.output,
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            static_cast<double>(cfg.fps),
            cv::Size(cfg.width, cfg.height));

        // Mensaje de error por si falla el escritor
        if (!writer.isOpened()) {
            throw std::runtime_error("No se pudo abrir el archivo de salida: " + cfg.output);
        }

        const auto start = std::chrono::steady_clock::now(); // Inicio de todo el ciclo

        // Bucle for para escribir y renderizar cada uno de los 900 frames
        for (int frame = 0; frame < total_frames; ++frame) {

            // Bucle 
            for (int step = 0; step < cfg.steps_per_frame; ++step) {
                simulate_step(previous, current, next, cfg);
                // Se rotan los vectores.
                previous.swap(current); // Lo actual pasa a ser anterior (t->t-1)
                current.swap(next); // Lo siguiente, ahora es lo actual
            }

            writer.write(render_frame(current, cfg, frame)); // Toma el vector actual y lo escribe en el frame actual del video

            if (frame % std::max(1, total_frames / 10) == 0) {
                std::cout << "Frame " << frame << " / " << total_frames << '\n';
            }
        }

        // Se extraen parámetros y estadísticas
        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(end - start).count();
        const double simulated_steps = static_cast<double>(total_frames) * cfg.steps_per_frame;

        // Impresión de las estadísticas luego de finalizar el renderizado del video
        std::cout << "Video generado: " << cfg.output << '\n';
        std::cout << "Tiempo: " << elapsed << " s\n";
        std::cout << "Pasos simulados: " << simulated_steps << '\n';
        std::cout << "Rendimiento: " << simulated_steps / elapsed << " pasos/s\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
