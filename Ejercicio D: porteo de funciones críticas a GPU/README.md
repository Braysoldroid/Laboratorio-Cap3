# Compilación y ejecución de la versión GPU (Jetson)

Entorno: Jetson, GCC 7.5.0, CUDA 10.2, OpenCV (JetPack).

## 1. Ubicarse en la carpeta

La carpeta tiene espacios y dos puntos, así que va entre comillas:

```bash
cd ~/grupo-5/lab-cap3/Laboratorio-Cap3/"Ejercicio D:porteo de funciones críticas a GPU"
```

## 2. Compilar

```bash
nvcc -O2 -std=c++14 -arch=sm_53 drop_simulation_gpu.cu -o drop_gpu \
     `pkg-config --cflags --libs opencv4` -lstdc++fs
```

Notas:

- `-std=c++14`: nvcc 10.2 no admite C++17.
- `-lstdc++fs`: necesario para `<experimental/filesystem>` en GCC 7.5.
- Si `pkg-config` no encuentra `opencv4`, cambia `opencv4` por `opencv`.



## 3. Ejecutar

```bash
./drop_gpu
```

Salida esperada:

- Progreso por frames.
- Tiempo total, pasos simulados y pasos/s.
- Desglose por cuadro (ms): `simulate_step`, `render_frame`, copia D2H, `putText` y `writer.write`.
- Video generado en `output/drop_simulation_gpu.mp4`.

## 4. Repetir 5 veces y guardar resultados

```bash
for i in 1 2 3 4 5; do
    ./drop_gpu | tee resultado_gpu_$i.txt
done
```

Reporta la media de las ejecuciones. Si la primera es notablemente distinta, descártala por calentamiento.


## 8. Resultados
| Métrica            | CPU             | GPU                                             |
|--------------------|-----------------|-------------------------------------------------|
| Tiempo total       | 10.288 s        | 32.548 s                                        |
| Frames procesados  | 900             | 900                                             |
| Rendimiento        | 87.5 pasos/s    | 27.65 pasos/s                                   |
| `render_frame`     | 8.900 ms/frame  | 8.973 ms/frame                                  |
| `simulate_step`    | 1.597 ms/frame  | 3.320 ms/frame                                  |
| Escritura de video | 0.934 ms/frame  | Incluida en `putText + write`: 20.984 ms/frame |
| Copia GPU → CPU    | No aplica       | 2.315 ms/frame                                  |