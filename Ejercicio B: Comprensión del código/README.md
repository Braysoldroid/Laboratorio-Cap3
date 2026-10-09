# Ejercicio B: Análisis del código base

Este ejercicio consiste en identificar los componentes principales del programa `drop_simulation.cpp`, elaborar un diagrama de flujo del programa y explicar qué hace y qué modelo físico aproxima.

---

## 1. Componentes del programa

### 1.1 Estructura de configuración de la simulación

**Ubicación en el código:** `struct Config`

Estructura inicial que declara parámetros de la simulación y del video.
Se define la resolución 640x640 de alto y ancho. Duración del video en 30 segundos y la misma cantidad
de frames por segundo (fps). También se definen parámetros sobre la física como la velocidad de la ola la cual sirve para controlar la propagación
de la ola en la función de simulate_step para la solución de la función de onda, además de valores de
fricción, tamaño y fuerza de la gota.

### 1.2 Función que inicializa la perturbación de la gota

**Ubicación en el código:** `add_drop()`

Crea el estado inicial del flujo. Una montaña gaussiana en el centro del agua con un impulso inicial; este se
ejecuta solo una vez en el main y a partir de ese punto sucede la evolución de esa gota con las demás funciones.

### 1.3 Función que calcula la absorción en los bordes

**Ubicación en el código:** `border_absorption()`

Función que calcula la fricción de cada celda. Mientras más cerca del borde, mayor fricción. Esto para apagar las olas
al llegar al límite en vez de rebotar. Sin embargo, con el código fuente original se ve en el video que la absorción no
es suficiente para eliminar el rebote ya que sobre los últimos 5 segundos se notan reflexiones.

### 1.4 Función que actualiza la ecuación de onda

**Ubicación en el código:** `simulate_step()`

Avanza la física del movimiento un paso. Cada celda se mueve dependiendo de su propia inercia, el efecto de sus vecinas
(izquierda, derecha, arriba y abajo) y la fricción. Usando los vectores de estado previo y actual se calcula el siguiente.

### 1.5 Función que renderiza cada cuadro del video

**Ubicación en el código:** `render_frame()`

Convierte el mapa de alturas en una imagen. Calcula la inclinación de la superficie y la ilumina con una luz virtual, logrando
que crestas y zonas de frente a la luz se vean claras mientras que valles y zonas ocultas como sombras y oscuridad. Escala de grises.

### 1.6 Función principal y ciclo de simulación

**Ubicación en el código:** `main()`

Acá sucede el flujo principal que inicializa parámetros, la gota, y repite 900 veces el ciclo de simulación, rotar vectores, renderizar
y escribir frames. Cuando terminan los bucles, se extraen parámetros de tiempo y rendimiento y se muestran al usuario.

---

## 2. Diagrama de flujo

El archivo editable se encuentra en `Diagrama de flujo del programa.pdf`.

## 3. Explicación del programa y del modelo físico

### 3.1 ¿Qué hace el programa?

En síntesis, el programa crea una gota (add_drop) y la mueve con funciones de física y simulación (border_absorption y simulate_step).
Luego, se dibuja el efecto (render_frame) y el flujo en Main repite el ciclo frame a frame.

### 3.2 ¿Qué modelo físico aproxima?

Aproxima la ecuación de onda 2D lineal con amortiguamiento resuelta con diferencias finitas. Conocido como efecto ripple, 
describe cómo una acción inicial genera una serie de consecuencias sucesivas que se extienden a su entorno. El código captura 
bien la idea principal sobre una perturbación local que se propaga en anillos, se debilita en función de su expansión y eventualmente
se apaga al perder su energía. Separa física y renderizado por lo que creo que facilita modificar cada lógica por separado.
