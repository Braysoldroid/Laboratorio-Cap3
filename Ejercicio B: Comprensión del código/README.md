# Ejercicio B: Análisis del código base

**Curso:** Computación Heterogénea
**Autor:** Brayan Solís

Este ejercicio consiste en identificar los componentes principales del programa `drop_simulation.cpp`, elaborar un diagrama de flujo del programa y explicar qué hace y qué modelo físico aproxima.

---

## 1. Componentes del programa

### 1.1 Estructura de configuración de la simulación

**Ubicación en el código:** `struct Config`

Estructura inicial que declara parámetros de la simulación y del video.
Se define la resolución 640x640 de alto y ancho. Duración del video en 30 segundos y la misma cantidad
de frames por segundo (fps). Se define la velocidad de la ola la cual sirve para controlar la propagación
de la ola en la función de simulate_step para la solución de la función de onda. Se define además valores de
amortiguación, borde de amortiguación y parámetros de caída que sirven en la función de perturbación de la gota.

### 1.2 Función que inicializa la perturbación de la gota

**Ubicación en el código:** `add_drop()`



### 1.3 Función que calcula la absorción en los bordes

**Ubicación en el código:** `border_absorption()`

<!-- Escribe aquí tu explicación -->

### 1.4 Función que actualiza la ecuación de onda

**Ubicación en el código:** `simulate_step()`

<!-- Escribe aquí tu explicación -->

### 1.5 Función que renderiza cada cuadro del video

**Ubicación en el código:** `render_frame()`

<!-- Escribe aquí tu explicación -->

### 1.6 Función principal y ciclo de simulación

**Ubicación en el código:** `main()`

<!-- Escribe aquí tu explicación -->

---

## 2. Diagrama de flujo

![Diagrama de flujo del programa](docs/flujo_simulacion_gota.png)

El archivo editable se encuentra en `docs/flujo_simulacion_gota.drawio`.

### Correspondencia con los elementos solicitados

| Elemento solicitado                   | Ubicación en el diagrama                                                   |
|---------------------------------------|----------------------------------------------------------------------------|
| Inicialización de parámetros          | Banda 1 · Inicialización: carga de `Config`, `total_frames` y `cells`      |
| Reserva de memoria                    | Banda 1 · Inicialización: vectores `previous`, `current` y `next`          |
| Perturbación inicial de la gota       | Banda 1 · Inicialización: llamada a `add_drop()`                           |
| Ciclo de simulación                   | Banda 3 · Bucle de simulación: decisiones de `frame` y `step`              |
| Actualización de la malla             | Banda 3 · Bucle de simulación: `simulate_step()` y rotación de buffers     |
| Renderizado del cuadro                | Banda 4 · Renderizado y escritura: `render_frame()`                        |
| Escritura del video                   | Banda 4 · Renderizado y escritura: `writer.write()`                        |
| Cálculo de métricas de rendimiento    | Banda 5 · Estadísticas y cierre: `elapsed`, pasos simulados y rendimiento  |

El diagrama incluye además la configuración del escritor de video con su validación (Banda 2) y la ruta de manejo de errores.

---

## 3. Explicación del programa y del modelo físico

### 3.1 ¿Qué hace el programa?

<!-- Escribe aquí tu explicación -->

### 3.2 ¿Qué modelo físico aproxima?

<!-- Escribe aquí tu explicación -->
