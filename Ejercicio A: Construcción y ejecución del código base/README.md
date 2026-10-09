## Ejercicio A: Construcción y ejecución del código base

### Pasos

1. Ingresar al directorio del proyecto:

   ```bash
   cd drop-simulation
   ```

2. Construir el código base:

   ```bash
   meson setup build
   meson compile -C build
   ```

3. Ejecutar el programa:

   ```bash
   ./build/drop_simulation
   ```

4. Verificar que se genere el archivo:

   ```
   output/drop_simulation.mp4
   ```

5. Obtener la resolución, duración y FPS del video generado con `ffprobe`:

   ```bash
   ffprobe -v error -select_streams v:0 \
     -show_entries stream=width,height,r_frame_rate,nb_frames \
     -show_entries format=duration \
     -of default=noprint_wrappers=1 output/drop_simulation.mp4
   ```

### Salida obtenida

Ejecución del programa:

```
$ ./build/drop_simulation
Frame 0 / 900
Frame 90 / 900
Frame 180 / 900
Frame 270 / 900
Frame 360 / 900
Frame 450 / 900
Frame 540 / 900
Frame 630 / 900
Frame 720 / 900
Frame 810 / 900
Video generado: output/drop_simulation.mp4
Tiempo: 17.0499 s
Pasos simulados: 900
Rendimiento: 52.7864 pasos/s
```

Inspección del video con `ffprobe`:

```
width=640
height=640
r_frame_rate=30/1
nb_frames=900
duration=30.000000
```

### Resultados

| Métrica                         | Valor                     |
|---------------------------------|---------------------------|
| Tiempo total de ejecución       | 17.0499 s                 |
| Cantidad de pasos simulados     | 900                       |
| Rendimiento                     | 52.7864 pasos/s           |
| Resolución                      | 640 × 640 px              |
| Duración                        | 30 s                      |
| FPS                             | 30                        |
| Total de cuadros (`nb_frames`)  | 900                       |

### Observaciones

- Los valores obtenidos con `ffprobe` coinciden con la configuración del programa (`Config`): `width = 640`, `height = 640`, `seconds = 30.0` y `fps = 30`.
- La cantidad de pasos simulados corresponde a `total_frames × steps_per_frame = 900 × 1 = 900`.
- El tiempo medido abarca todo el bucle principal: la simulación (`simulate_step`), el renderizado de cada cuadro (`render_frame`) y la codificación del video (`writer.write`). Por lo tanto, el rendimiento reportado en pasos/s refleja el costo del proceso completo y no únicamente el de la simulación numérica.
