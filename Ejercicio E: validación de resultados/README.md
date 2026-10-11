# Ejercicio E: Validación de resultados CPU vs GPU

## 1. Compilación y ejecución

Archivo del programa de validación: `validar_gpu_vs_cpu.cu`.

### 1.1 Ubicarse en la carpeta

La ruta contiene espacios y dos puntos, por lo que debe ir entre comillas:

```bash
cd ~/grupo-5/lab-cap3/Laboratorio-Cap3/"Ejercicio E: validación de resultados"
```

### 1.2 Compilar

```bash
nvcc -O2 -std=c++14 -arch=sm_53 validar_gpu_vs_cpu.cu -o validar \
     `pkg-config --cflags --libs opencv4` -lstdc++fs
```

Notas:

- `-std=c++14`: nvcc 10.2 no admite C++17.
- `-arch=sm_53`: arquitectura de la Jetson Nano.
- `-lstdc++fs`: necesario para `<experimental/filesystem>` en GCC 7.5.
- Si `pkg-config` no encuentra `opencv4`, usar `opencv`.

### 1.3 Ejecutar

```bash
./validar          # validación completa: 900 cuadros
./validar 300      # prueba rápida: primeros 300 cuadros
```

### 1.4 Archivos generados

| Archivo | Contenido |
|---|---|
| `validacion/metricas.csv` | Métricas por cuadro de control |
| `validacion/comparacion_fXXXX.png` | Imagen con tres paneles: CPU, GPU y diferencia amplificada ×40 |

El programa termina con código `0` si la validación pasa y `2` si algún valor excede la tolerancia.

## 2. Métodos de comparación

El programa ejecuta la simulación en CPU y en GPU **en paralelo**, partiendo del mismo estado inicial (la perturbación de la gota se calcula en CPU y se copia a la GPU). En cada cuadro avanza ambas versiones un paso y, en los cuadros de control, compara los resultados.

**Cuadros de control:** cada 30 cuadros, el último cuadro y los cuadros 0, 100, 300 y 600 (estos para las imágenes).

### 2.1 Comparación numérica de la malla de alturas

Se copia la malla de la GPU al host y se compara celda por celda con la de CPU:

- **Error absoluto medio (MAE):** promedio de `|h_cpu - h_gpu|` sobre las 409 600 celdas.
- **Error absoluto máximo:** mayor valor de `|h_cpu - h_gpu|` en toda la malla.

### 2.2 Comparación del frame renderizado

Se renderiza el mismo frame en CPU y en GPU (en escala de grises, 0 a 255) y se compara píxel a píxel:

- **Diferencia media (MAE de imagen)** en niveles de gris.
- **Diferencia máxima** en niveles de gris.
- **Porcentaje de píxeles** que difieren en al menos 1 nivel.

### 2.3 Comparación visual

Para los frames 0, 100, 300, 600 y el último se guarda una imagen con tres paneles:

1. Frame generado por la CPU.
2. Frame generado por la GPU.
3. Diferencia absoluta amplificada ×40, para que incluso 1 nivel de gris sea visible.

### 2.4 Tolerancias

| Magnitud | Tolerancia | Justificación |
|---|---|---|
| Error absoluto máximo de alturas | < 1e-4 | Margen amplio para el error de redondeo de `float32` acumulado en 900 pasos |
| Diferencia máxima de imagen | ≤ 2 niveles de gris | Diferencias de 1 a 2 niveles son imperceptibles y esperables al truncar a 8 bits |

## 3. Resultados

### 3.1 Resumen

| Métrica | Resultado (peor caso en los cuadros de control) | Tolerancia | Estado |
|---|---|---|---|
| Error absoluto medio de alturas | 0 | n/a | Cumple |
| Error absoluto máximo de alturas | 0 | < 1e-4 | Cumple |
| Diferencia media de imagen | 0.0000 niveles | n/a | Cumple |
| Diferencia máxima de imagen | 1 nivel de gris | ≤ 2 | Cumple |
| Píxeles que difieren | 0.0012 % como máximo (unos 5 de 409 600) | n/a | Cumple |

**Veredicto del programa: `VALIDACION CORRECTA`.**

### 3.2 Muestra de la salida por cuadro

| Cuadro | h_MAE | h_MAX | img_MAE | img_MAX | % difieren |
|---|---|---|---|---|---|
| 0 | 0 | 0 | 0.0000 | 1 | 0.000 |
| 30 | 0 | 0 | 0.0000 | 0 | 0.000 |
| 150 | 0 | 0 | 0.0000 | 1 | 0.000 |
| 300 | 0 | 0 | 0.0000 | 1 | 0.000 |
| 450 | 0 | 0 | 0.0000 | 1 | 0.001 |
| 600 | 0 | 0 | 0.0000 | 1 | 0.000 |
| 750 | 0 | 0 | 0.0000 | 1 | 0.001 |
| 870 | 0 | 0 | 0.0000 | 1 | 0.001 |
| 899 | 0 | 0 | 0.0000 | 1 | 0.000 |

La tabla completa (32 cuadros de control) está en `validacion/metricas.csv`.


## 4. Explicación de las diferencias observadas

**Malla de alturas: sin diferencias.** El error es exactamente 0 en todos los cuadros de control. La actualización de la onda solo usa sumas, restas y multiplicaciones. CPU y GPU las evalúan en el mismo orden, por lo que el resultado es idéntico bit a bit.

**Imagen: diferencias de como máximo 1 nivel de gris.** Como las alturas son idénticas, la diferencia proviene del cálculo del render, que usa funciones matemáticas (`powf`, `sqrtf`, normalización de vectores). Sus implementaciones en la GPU y en la librería de la CPU pueden diferir en el último bit de precisión. Al truncar el valor a `unsigned char`, una diferencia mínima puede cruzar un umbral y cambiar el píxel en 1 nivel. Es lo esperable y es visualmente imperceptible.

**Evolución en el tiempo.** La proporción de píxeles que difieren se mantiene en torno a 0.001 % durante los 900 cuadros, sin tendencia a crecer, por lo que no hay acumulación de error.


