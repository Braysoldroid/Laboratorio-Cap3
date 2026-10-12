# Laboratorio 5 — Reporte final

Reporte final del laboratorio de simulación de una gota mediante la ecuación de onda 2D con amortiguamiento. Se documenta el programa base en CPU, su perfilado, el porteo a GPU, la validación de resultados, las optimizaciones aplicadas y el análisis comparativo de rendimiento.

## Contenido

1. [Programa base](#1-programa-base)
2. [Resultados de perfilado CPU](#2-resultados-de-perfilado-cpu)
3. [Programa porteado](#3-programa-porteado)
4. [Validación](#4-validación)
5. [Resultados de optimización](#5-resultados-de-optimización)
6. [Análisis](#6-análisis)
7. [Conclusiones](#7-conclusiones)

---

## 1. Programa base

El programa base es una simulación de ondas en la superficie del agua escrita en C++, este programa consta de diversas funciones que crean la perturbación inicial, calculan una ecuación de onda en 2D con amortiguamiento, se aplica una amortiguación para atenuar rebotes, convierten el mapa de alturas a una imagen en escala de grises simulando iluminación sobre la superficie para una mejor apreciación del relieve, escribe 900 frames y al final de todo ese proceso físico y matemático se muestran resultados de ejecución y rendimiento en consola, finalizando con la exportación de un vídeo en MP4 usando OpenCV. Todo se ejecuta en CPU con números float de 32 bits siendo solo una versión base pensada para compararla después con otras implementaciones.

El detalle de la construcción, ejecución y análisis del código se encuentra en:

- `README.md`: Ejercicio A, construcción y ejecución del código base.
- `README_Ejercicio_B.md`: Ejercicio B, análisis del código y diagrama de flujo.

### Parámetros de la simulación

| Parámetro          | Valor          |
|--------------------|----------------|
| Resolución         | 640 × 640 px   |
| Duración           | 30 s           |
| FPS                | 30             |
| Total de cuadros   | 900            |
| Pasos por cuadro   | 1              |
| Celdas por malla   | 409 600        |

### Resultados de la ejecución base

| Métrica                     | Valor            |
|-----------------------------|------------------|
| Tiempo total de ejecución   | 17.0499 s        |
| Pasos simulados             | 900              |
| Rendimiento                 | 52.7864 pasos/s  |

---

## 2. Resultados de perfilado CPU

El programa tardó en promedio 10.288 s en generar los 900 cuadros (11.43 ms por cuadro, 87.5 pasos/s). Ese tiempo se concentra en pocas funciones:

| Función | ms/cuadro | % (temporizadores) | % (`perf`) |
|---|---|---|---|
| `render_frame` | 8.900 | 77.9 % | ~74 % |
| `simulate_step` | 1.597 | 14.0 % | ~15 % |
| `writer.write` y resto | 0.934 | 8.2 % | ~11 % |
| `add_drop` (una vez) | — | 0.05 % | — |

`render_frame` domina con claridad: casi 4 de cada 5 segundos de ejecución se gastan en dibujar los cuadros, no en simular la física. Dentro de esa función, el cálculo del reflejo especular, `std::pow(diffuse, 24.0f)`, aporta por sí solo cerca del 21 % del tiempo total del programa.

`simulate_step` es mucho más barata a pesar de recorrer la misma malla, porque cada celda solo requiere sumas, restas y multiplicaciones, sin funciones matemáticas costosas.

### Confiabilidad de la medición

Las dos herramientas miden de forma distinta (`perf` por muestreo estadístico, los temporizadores por tiempo absoluto) y aun así coinciden con diferencias de 4 puntos porcentuales o menos. Las pequeñas diferencias se explican por cómo cada una reparte el tiempo: `perf` asigna `powf` a la librería `libm` y no a `render_frame`, y agrupa en "resto" funciones que los temporizadores no miden por separado.

`perf` tampoco alteró el comportamiento del programa: la ejecución con muestreo tardó 10.40 s contra 10.29 s sin él, alrededor de 1 % más. Las muestras atribuidas a OpenBLAS (7.19 %) se excluyeron del análisis, porque corresponden a hilos que OpenCV carga al iniciar y no a trabajo del programa.

### Potencial de aceleración

Según la ley de Amdahl, la parte que no se acelera limita la mejora total:

- Acelerando solo `render_frame` (77.9 %), el techo teórico es de unas **4.5 veces**.
- Acelerando `render_frame` y `simulate_step` (91.9 % en conjunto), el techo sube a unas **12 veces**.

Estos valores son cotas superiores: suponen que esas funciones tardan cero y no consideran el costo de mover datos entre CPU y GPU.

`render_frame` es la candidata principal para portar a GPU: es la más costosa y cada píxel se calcula de forma independiente de los demás, lo que permite asignar un hilo por píxel. Conviene portar también `simulate_step`, que tiene la misma estructura por celda. Así, además de sumar su 14 %, las mallas de alturas pueden quedarse siempre en la GPU, y solo la imagen final tiene que volver a la CPU en cada cuadro.

No conviene portar `writer.write`, porque es la codificación de video de OpenCV, que no tiene ruta en CUDA, ni `add_drop`, que se ejecuta una sola vez y es despreciable.

## 3. Programa porteado

El perfilado en CPU mostró que `render_frame` (~78 %) y `simulate_step` (~14 %) concentran más del 90 % del tiempo, y que ambas comparten una estructura muy favorable para la GPU: recorren la malla de 640 × 640 celdas y el resultado de cada celda o píxel depende solo de sus vecinas, no del orden en que se procesan. Por eso el diseño sigue una idea central: **reemplazar los dos ciclos `for` anidados de cada función por un kernel donde cada hilo de la GPU se encarga de un solo elemento**.

| Función en CPU | En la versión GPU | Dónde se ejecuta |
|---|---|---|
| `simulate_step` | `simulate_step_kernel` (1 hilo = 1 celda) | GPU |
| `border_absorption` | `border_absorption_dev`, función `__device__` dentro del kernel | GPU |
| `render_frame` | `render_frame_kernel` (1 hilo = 1 píxel) | GPU |
| `add_drop` | Sin cambios; se ejecuta una vez y se copia el resultado | CPU |
| `putText` y `writer.write` | Sin cambios; OpenCV no tiene ruta CUDA para codificar video | CPU |

### Las mallas viven en la GPU

El perfilado recomendaba portar también `simulate_step`, aunque pesara menos, para que los datos no tuvieran que ir y venir. Así quedó el flujo de memoria:

1. Al inicio, `add_drop` calcula la perturbación en la CPU y las mallas `previous` y `current` se copian **una sola vez** a la GPU.
2. En cada cuadro, la simulación y el render se encadenan dentro de la GPU: el render lee directamente la malla que acaba de producir la simulación.
3. Lo único que vuelve a la CPU en cada cuadro es la imagen terminada (640 × 640 × 3 bytes), que es lo que necesita el escritor de video.

La rotación de las tres mallas al final de cada paso se hace intercambiando punteros, igual que los `swap` de la versión CPU, sin copiar ningún dato.

### Organización de los hilos

Ambos kernels usan la misma configuración: bloques de 16 × 16 hilos (256 por bloque) organizados en una cuadrícula 2D de 40 × 40 bloques, lo que da exactamente un hilo por cada una de las 409 600 celdas. Cada hilo calcula su posición `(x, y)` en la malla a partir de su índice de bloque y de hilo.

### Kernel de simulación

`simulate_step_kernel` aplica la misma ecuación de onda que la versión CPU, celda por celda: el laplaciano con las 4 vecinas, la velocidad como diferencia entre el paso actual y el anterior, y el amortiguamiento. Hay tres decisiones de diseño relevantes:

- **Bordes.** En la CPU, `std::fill` pone toda la malla en cero y luego solo se calcula el interior. En la GPU, los hilos del borde escriben el cero directamente, lo que evita una pasada extra sobre la memoria.
- **Absorción de bordes.** `border_absorption` no se porta como kernel aparte. Se convierte en una función `__device__` marcada para expandirse en línea, de modo que cada hilo la evalúa sin costo de llamada.
- **Sincronización entre pasos.** Cada paso necesita la malla completa del paso anterior. En lugar de sincronizar hilos manualmente, se lanza un kernel por paso: la GPU ejecuta en orden los kernels lanzados en el mismo flujo, así que un paso no empieza hasta que el anterior termina.

Los punteros de entrada se declaran `const __restrict__`, lo que le indica al compilador que esas mallas solo se leen y no se solapan con la de salida, y le permite usar la caché de solo lectura de la GPU.

### Kernel de render

`render_frame_kernel` es el porteo de la función más costosa. Cada hilo lee la altura de su píxel y las de sus 4 vecinas, calcula la pendiente, la normal de la superficie y la iluminación difusa y especular, y escribe el nivel de gris resultante. Para trasladarlo a la GPU hubo que resolver tres cosas:

- **Sin OpenCV dentro del kernel.** `cv::normalize` y `cv::Vec3f` no existen en la GPU, así que la normal se normaliza a mano con `sqrtf` y un inverso de la longitud.
- **Cálculos repetidos fuera del kernel.** La dirección de la luz es la misma para todos los píxeles, por lo que se normaliza una sola vez en la CPU y se pasa al kernel ya lista.
- **Formato listo para OpenCV.** El kernel escribe directamente en el formato de 3 canales (BGR) de un `cv::Mat`, así que la imagen se copia de la GPU al `cv::Mat` del host sin ninguna conversión intermedia.

El `powf` que el perfilado señaló como responsable de ~21 % del tiempo se mantiene igual, ahora ejecutado en paralelo por todos los hilos.

### Parámetros de los kernels

Los kernels no pueden recibir la estructura `Config` original porque contiene un `std::string`, que no existe en la GPU. Por eso se creó `KernelParams`, una estructura plana con solo los valores que usan los kernels (dimensiones, constantes físicas y dirección de la luz), que se pasa por valor en cada lanzamiento.

### Medición

Para obtener el desglose por componente, el programa usa eventos CUDA, que marcan tiempos dentro de la propia GPU, alrededor del kernel de simulación, del kernel de render y de la copia de la imagen. La parte en CPU (`putText` y `writer.write`) se mide con `std::chrono`. Antes de empezar a medir se inicializa el contexto CUDA con `cudaFree(0)`, para que el costo de arranque de la GPU no se cuente en el primer cuadro.

| Métrica            | CPU             | GPU                                             |
|--------------------|-----------------|-------------------------------------------------|
| Tiempo total       | 10.288 s        | 32.548 s                                        |
| Frames procesados  | 900             | 900                                             |
| Rendimiento        | 87.5 pasos/s    | 27.65 pasos/s                                   |
| `render_frame`     | 8.900 ms/frame  | 8.973 ms/frame                                  |
| `simulate_step`    | 1.597 ms/frame  | 3.320 ms/frame                                  |
| Escritura de video | 0.934 ms/frame  | Incluida en `putText + write`: 20.984 ms/frame |
| Copia GPU → CPU    | No aplica       | 2.315 ms/frame                                  |

---

## 4. Validación

Antes de optimizar, se verificó que la versión GPU produce los mismos resultados que la versión CPU. Para eso se escribió un programa de validación aparte, `validar_gpu_vs_cpu.cu`, que no genera video: ejecuta ambas versiones lado a lado, partiendo del mismo estado inicial, y las avanza un paso a la vez. En ciertos cuadros de control (cada 30 cuadros, más los cuadros 0, 100, 300, 600 y el último) detiene la comparación y mide las diferencias.

Se descartó comparar los videos generados, porque la compresión MP4 es con pérdida: dos simulaciones idénticas podrían producir cuadros ligeramente distintos solo por el códec. En su lugar se compararon los datos crudos, antes de cualquier compresión, con tres técnicas complementarias.

**1. Comparación numérica de la malla de alturas.** La malla de la GPU se copia al host y se compara celda por celda con la de la CPU, sobre las 409 600 celdas. Se calculan el error absoluto medio y el error absoluto máximo.

**2. Comparación del cuadro renderizado.** Ambos renders reciben la misma malla y sus imágenes se comparan píxel a píxel. Se miden la diferencia media, la diferencia máxima en niveles de gris y el porcentaje de píxeles que difieren. Al usar la misma malla de entrada, el error del render queda aislado del error de la simulación.

**3. Comparación visual.** En los cuadros 0, 100, 300, 600 y el último se guarda una imagen con tres paneles: el cuadro de la CPU, el de la GPU y su diferencia amplificada 40 veces, para que incluso una diferencia de 1 nivel de gris sea visible.

### Resultados

| Métrica | Peor caso (32 cuadros de control) | Tolerancia | Estado |
|---|---|---|---|
| Error absoluto medio de alturas | 0 | — | Cumple |
| Error absoluto máximo de alturas | 0 | < 1e-4 | Cumple |
| Diferencia media de imagen | ≈ 0 niveles | — | Cumple |
| Diferencia máxima de imagen | 1 nivel de gris | ≤ 2 | Cumple |
| Píxeles que difieren | 0.0012 % (unos 5 de 409 600) | — | Cumple |

El programa de validación terminó con el veredicto `VALIDACION CORRECTA`: la versión GPU es equivalente a la versión CPU.

### Interpretación de las diferencias

**La malla de alturas es idéntica bit a bit.** El error es exactamente cero en todos los cuadros de control, no solo pequeño. La actualización de la onda usa únicamente sumas, restas y multiplicaciones, y el kernel las evalúa en el mismo orden que la versión CPU. Con las mismas operaciones, en el mismo orden y sobre los mismos datos, el redondeo de punto flotante también es el mismo.

**La imagen difiere en, como máximo, 1 nivel de gris en unos pocos píxeles.** Como las alturas son idénticas, esa diferencia solo puede venir del render. Este usa funciones matemáticas (`powf`, `sqrtf` y la normalización de vectores) cuyas implementaciones en la GPU y en la librería de la CPU pueden diferir en el último bit de precisión. Normalmente eso no cambia nada, pero si la intensidad de un píxel cae justo en el límite entre dos enteros (por ejemplo, 127.9999 en una versión y 128.0001 en la otra), la conversión a `unsigned char` lo lleva a niveles distintos. Por eso las diferencias son raras, aisladas y nunca mayores a 1 nivel.

**El error no se acumula.** El porcentaje de píxeles que difieren se mantiene cerca de 0.001 % durante los 900 cuadros, sin tendencia a crecer. Esto es coherente con lo anterior: como la simulación no tiene error, cada cuadro se renderiza a partir de la misma malla en ambas versiones, y las diferencias del render no se arrastran de un cuadro al siguiente.

---

## 5. Resultados de optimización

### 5.1 Optimización 1

**Descripción:**

<!-- Qué se cambió y por qué debería mejorar el rendimiento -->

**Resultados:**

| Métrica                     | Valor |
|-----------------------------|-------|
| Tiempo total de ejecución   |       |
| Rendimiento (pasos/s)       |       |
| Error máximo                |       |

### 5.2 Optimización 2

**Descripción:**

<!-- Qué se cambió y por qué debería mejorar el rendimiento -->

**Resultados:**

| Métrica                     | Valor |
|-----------------------------|-------|
| Tiempo total de ejecución   |       |
| Rendimiento (pasos/s)       |       |
| Error máximo                |       |

### 5.3 Optimización 3

**Descripción:**

<!-- Qué se cambió y por qué debería mejorar el rendimiento -->

**Resultados:**

| Métrica                     | Valor |
|-----------------------------|-------|
| Tiempo total de ejecución   |       |
| Rendimiento (pasos/s)       |       |
| Error máximo                |       |

---

## 6. Análisis

### Comparación de versiones

| Versión          | Tiempo total | Pasos/s  | Speedup | Error máximo   |
|------------------|--------------|----------|---------|----------------|
| CPU base         | 17.0499 s    | 52.7864  | 1.00x   | — (referencia) |
| GPU base         |              |          |         |                |
| Optimización 1   |              |          |         |                |
| Optimización 2   |              |          |         |                |
| Optimización 3   |              |          |         |                |

El speedup se calcula respecto a la versión CPU base:

```
speedup = tiempo_CPU_base / tiempo_versión
```

### Discusión

<!-- Analiza la evolución del rendimiento entre versiones, qué optimización tuvo mayor impacto, dónde quedó el cuello de botella (cómputo, memoria, transferencias, renderizado o codificación del video) y si el error se mantuvo dentro de lo aceptable -->

---

## 7. Conclusiones

<!-- Escribe aquí tus conclusiones -->
