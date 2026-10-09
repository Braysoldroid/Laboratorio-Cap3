# Drop Simulation

Prototipo CPU en C++ y OpenCV para simular la caida de una gota sobre un estanque
usando una ecuacion de onda 2D amortiguada.

El objetivo es que esta version sea la referencia inicial del laboratorio antes de
portar el calculo principal a GPU y aplicar optimizaciones como coalescing,
shared memory, matematica aproximada y precision de 16 bits.

## Compilar

```bash
meson setup build
meson compile -C build
```

## Ejecutar

```bash
./build/drop_simulation
```

Por defecto genera:

```text
output/drop_simulation.mp4
```

## Parametros actuales

```text
width             640
height            640
seconds           30
fps               30
steps_per_frame   1
wave_speed        0.45
damping           0.006
edge_damping      0.035
drop_radius       18
drop_strength     1.0
output            output/drop_simulation.mp4
```

Estos valores estan definidos en `src/main.cpp`, dentro de `Config`.

## Idea fisica

Cada celda guarda la altura de la superficie del agua. La actualizacion usa un
stencil de 5 puntos:

```text
laplaciano = izquierda + derecha + arriba + abajo - 4 * centro

h_next = 2 * h_current - h_previous
         + c^2 * dt^2 * laplaciano
         - damping * (h_current - h_previous)
```

Para visualizar, se calculan normales aproximadas con diferencias finitas y se
aplica iluminacion simple. Esto produce un efecto de agua sin requerir un motor
3D.
