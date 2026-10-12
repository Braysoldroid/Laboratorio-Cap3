# Optimización de la versión GPU

El ejercicio pide aplicar al menos tres técnicas, medir cada etapa con un perfilador y comprobar la precisión frente a CPU. Primero hicimos corridas normales y después perfilamos cada variante con **nvprof**. La tabla siguiente usa las ejecuciones perfiladas.


## Compilación y ejecución

Desde esta carpeta en la Jetson, con CUDA 10.2 y OpenCV 4. Cambiar `variante` por `half`, `pow24`, `bloques` o `streams`.

```bash
variante=streams
nvcc -O2 -std=c++14 -arch=sm_53 drop_simulation_${variante}.cu -o gpu_${variante} $(pkg-config --cflags --libs opencv4) -lstdc++fs
nvcc -O2 -std=c++14 -arch=sm_53 validar_${variante}_vs_cpu.cu -o validar_${variante} $(pkg-config --cflags --libs opencv4) -lstdc++fs
./validar_${variante}
./gpu_${variante}
sudo /usr/local/cuda/bin/nvprof ./gpu_${variante} 2>&1 | tee perfil_${variante}_nvprof.txt
```

Para la referencia CPU:

```bash
g++ -O3 -std=c++17 cpu_base_jetson.cpp -o cpu_base $(pkg-config --cflags --libs opencv4) -lstdc++fs
./cpu_base
```

La GPU base y sus instrucciones están en el ejercicio D.



## Resultados

| Versión | Tiempo con nvprof (s) | simulate_step (ms) | render_frame (ms) | Validación |
|---|---:|---:|---:|---|
| GPU base | 14.5739 | 0.42109 | 0.78982 | Correcta |
| Half | 15.9121 | 0.34821 | 0.81610 | Falló |
| Multiplicaciones | 17.1872 | 0.42424 | 0.56008 | Correcta |
| Bloques 32x8 | 14.6505 | 0.39917 | 0.78883 | Correcta |
| Streams y copias asíncronas | 14.6155 | 0.422475 | 0.78996 | Correcta |

El tiempo total es el del bucle durante el perfilado. Los kernels son promedios por llamada de nvprof. Streams usa la media de dos perfiles, de 14.5917 y 14.6393 s; las demás versiones tienen uno.

Multiplicaciones redujo render **29.09%** y bloques redujo simulate_step **5.21%**, pero ninguna variante redujo el total de estas ejecuciones perfiladas. Half se descartó por precisión. Streams quedó cerca de la base.

Los archivos `perfil_*_nvprof*.txt` contienen las mediciones completas de kernels y copias. Las corridas sin perfilador están en `resultado_*_actual_*.txt`; los errores de validación están en `resultado_validacion_*.txt` y `validacion_*/metricas.csv`.
