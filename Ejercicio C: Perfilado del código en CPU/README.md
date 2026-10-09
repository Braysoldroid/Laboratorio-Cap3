# Ejercicio C: Perfilado del código en CPU

---

## 1. Herramientas utilizadas

| Herramienta | Cómo mide | Por qué se eligió |
|---|---|---|
| `perf` | Muestreo: interrumpe el programa 999 veces por segundo y registra qué función se ejecuta | No modifica el código y da el porcentaje por función directamente |
| Temporizadores manuales | `std::chrono::steady_clock` alrededor de cada llamada, acumulando el tiempo | Dan tiempos absolutos y ms/frame de cada función |

Se descartó **Callgrind** porque simula cada instrucción y haría el programa decenas de veces más lento. Se usa 999 Hz en vez de 1000 Hz para que el muestreo no se sincronice con eventos periódicos del sistema.

---

## 2. Metodología

**Entorno:** g++ 15.2.0 (`-O3`, C++17), Meson 1.10.1, OpenCV 4.10.0, perf 7.0.14, CPU híbrida (núcleos P y E).

**Compilación para `perf`** (build aparte, con símbolos y sin inlinear funciones llamadas una vez, para que `render_frame` aparezca por nombre):

```bash
meson setup build-perf -Dcpp_args="-g -fno-omit-frame-pointer -fno-inline-functions-called-once"
meson compile -C build-perf
```

`simulate_step` sigue inlineado en `main`, pero como `add_drop` es despreciable, lo que queda en `main` equivale a `simulate_step`.

**CPU híbrida:** la primera medición, sin restricciones, cayó casi toda en los núcleos E y tardó **31.79 s**. Fijada a los núcleos P tardó **10.40 s**. Por eso todas las mediciones se hicieron con:

```bash
taskset -c "$(cat /sys/devices/cpu_core/cpus)" <comando>
```

**Temporizadores:** `profiling/main_timers.cpp` es una copia de `src/main.cpp` (el original no se modificó) que mide `simulate_step`, `render_frame`, `writer.write` y `add_drop`. Agrega 4 lecturas del reloj por cuadro, un costo despreciable.

```bash
g++ -std=c++17 -O3 profiling/main_timers.cpp -o build-timers/drop_timers $(pkg-config --cflags --libs opencv4)
```

---

## 3. Resultados con `perf`

```bash
taskset -c "$(cat /sys/devices/cpu_core/cpus)" perf record -F 999 -g -o perf-pcore.data ./build-perf/drop_simulation
perf report -i perf-pcore.data --no-children --sort dso,symbol --stdio -g none --percent-limit 0.5
```

| Símbolo | % de muestras |
|---|---|
| `render_frame` | 47.56 % |
| `__powf_fma` (libm) + `powf@plt` | 21.49 % |
| `main` (incluye `simulate_step`) | 13.81 % |
| OpenBLAS | 7.19 % |
| Resto (video, `putText`, kernel…) | ~10 % |

- El `powf` viene de `std::pow(diffuse, 24.0f)` en `render_frame`, único uso de `pow` en el programa. Con él, `render_frame` suma **69.05 %**.
- OpenBLAS no es trabajo del programa: probablemente son hilos de espera que OpenCV carga al iniciar. Descontándolo: `render_frame` ≈ **74 %**, `simulate_step` ≈ **15 %**, resto ≈ **11 %**.

---

## 4. Resultados con temporizadores

Promedio de 4 ejecuciones fijadas a núcleos P, sin `perf` (desviación del total: ~1 %):

| Función | Tiempo (s) | % del total | ms/frame |
|---|---|---|---|
| `render_frame` | 8.010 | **77.9 %** | 8.900 |
| `simulate_step` | 1.437 | **14.0 %** | 1.597 |
| `writer.write` | 0.841 | **8.2 %** | 0.934 |
| `add_drop` (una vez) | 0.005 | 0.05 % | — |
| **Total** | **10.288** | 100 % | 11.431 |

Rendimiento promedio: **87.5 pasos/s**.

---

## 5. Comparación entre herramientas

| Función | `perf` | Temporizadores |
|---|---|---|
| `render_frame` | ~74 % | 77.9 % |
| `simulate_step` | ~15 % | 14.0 % |
| `writer.write` y resto | ~11 % | 8.2 % |

Las dos herramientas coinciden con diferencias de unos 4 puntos porcentuales o menos. Además, la ejecución con `perf` tardó 10.40 s contra 10.29 s sin él (≈ 1 % más), así que `perf` casi no altera la medición.

---

## 6. Conclusiones

**Funciones críticas:**

1. `render_frame`: ~78 % (8.9 ms por cuadro); el `powf` solo aporta ~21 % del tiempo total.
2. `simulate_step`: ~14 % (1.6 ms por cuadro).
3. `writer.write`: ~8 % (0.93 ms por cuadro).

**Portar primero: `render_frame`.** Es la más pesada, cada píxel se calcula de forma independiente y tiene operaciones costosas por píxel (`powf`, dos normalizaciones). Conviene portar también `simulate_step`, para que las mallas se queden en la GPU y solo baje la imagen final en cada cuadro.

**Techo de aceleración (Amdahl, sin contar copias):** portando solo `render_frame`, ~4.5×; portando `render_frame` y `simulate_step`, ~12×.

**No conviene portar:**

- `writer.write`: es la codificación de video en CPU (OpenCV/FFmpeg), sin ruta CUDA, y pesa solo ~8 %. Obliga a que cada imagen regrese al host.
- `add_drop`: se ejecuta una vez y tarda ~5 ms (0.05 %).
- `putText` y las impresiones por consola: despreciables.
- `border_absorption`: no se porta aparte, se integra como función `__device__` dentro del kernel de `simulate_step`.

---

## 7. Archivos relacionados

- `profiling/main_timers.cpp`: versión con temporizadores manuales.
