# Auralis — Visión

## Misión

Construir una inteligencia artificial propia desde los fundamentos y hacerla evolucionar de forma continua mediante ingeniería, investigación y experimentación reproducible, con la IA participando activamente en el diseño y desarrollo de las siguientes generaciones de Auralis.

Auralis nace con una ambición deliberadamente grande: intentar alcanzar capacidades de frontera y, cuando los resultados lo permitan, superar sistemas existentes en dimensiones concretas de inteligencia, eficiencia, razonamiento, memoria, autonomía científica o coste computacional.

Esa ambición no es una garantía. Es el norte del proyecto.

## Identidad

Auralis no debe convertirse en una interfaz alrededor de otro modelo ni en una colección de llamadas a APIs externas presentada como una IA propia.

Podrá usar herramientas, modelos externos o datasets cuando eso sea útil para investigación o desarrollo, pero el sistema central que llamamos **Auralis** debe ser entrenable, ejecutable, medible y evolucionable como tecnología propia.

> Auralis no es un wrapper. Auralis es el sistema que estamos construyendo.

## IA creada por IA

“Creada por IA” no significa generar código una vez y declararlo terminado.

Significa establecer un proceso de desarrollo donde sistemas de IA puedan participar de forma creciente en:

- diseño de arquitectura;
- generación de código;
- revisión de código;
- creación de tests;
- búsqueda de errores;
- diseño de datasets;
- formulación de hipótesis;
- creación de experimentos;
- análisis de resultados;
- optimización de kernels;
- diseño de evaluaciones;
- documentación;
- propuestas de nuevas generaciones de Auralis.

La autoridad final de una mejora no será la opinión del sistema que la propone, sino la evidencia.

## Bucle de mejora

```text
Auralis / IA investigadora
        ↓
formula una hipótesis
        ↓
propone arquitectura o código
        ↓
crea un experimento aislado
        ↓
ejecuta tests y entrenamiento
        ↓
mide calidad + coste + estabilidad
        ↓
compara contra el baseline
        ↓
revisión y decisión explícita
        ↓
integración o descarte
```

Este mecanismo permite que la IA participe en su propia evolución sin confundir autoafirmación con progreso real.

## Principios permanentes

### 1. Evidencia sobre narrativa

Toda afirmación fuerte sobre capacidades debe poder apoyarse en resultados medidos.

### 2. Capacidad por cómputo

El objetivo no es tener el modelo más grande, sino obtener la mayor capacidad posible por unidad de cómputo, memoria, energía, datos y coste.

### 3. Construir antes de escalar

Primero debe existir un sistema pequeño que funcione correctamente. Después se optimiza. Después se escala.

### 4. Arquitectura abierta a evolución

Ninguna arquitectura actual es sagrada. El transformer es el punto de partida de Auralis, no su destino obligatorio.

### 5. Reproducibilidad

Los experimentos importantes deben registrar configuración, seed, datos, versión de código, métricas y resultado.

### 6. Baselines obligatorios

No existe una “mejora” sin una referencia contra la que compararla.

### 7. Modularidad

Tokenización, modelo, memoria, optimizador, entrenamiento, inferencia, agentes y herramientas deben poder evolucionar sin obligar a reescribir todo el sistema.

### 8. Observabilidad

Auralis debe poder ser inspeccionado: pérdidas, gradientes, rendimiento, memoria, decisiones de agente, llamadas a herramientas y resultados de experimentos.

### 9. Robustez antes de autonomía

Más autonomía solo tiene sentido cuando existen tests, límites, trazabilidad y mecanismos de recuperación suficientes.

### 10. Aprender de cualquier fuente sin depender de una sola

Auralis puede estudiar investigación pública, código, papers, datasets y otros modelos, pero su continuidad tecnológica no debe depender de un único proveedor externo.

## Qué significa “más potente”

Auralis no utilizará una única cifra para definir inteligencia. El progreso se evaluará como un vector de capacidades, entre ellas:

- calidad de lenguaje;
- razonamiento;
- programación;
- matemáticas;
- memoria;
- planificación;
- uso de herramientas;
- multimodalidad;
- aprendizaje de conocimiento nuevo;
- velocidad de inferencia;
- eficiencia de entrenamiento;
- coste por tarea;
- robustez;
- capacidad de investigación.

Por tanto, “llegar a ser la IA más potente” se traduce técnicamente en una misión más concreta: **acercarse al frente de Pareto de capacidad, eficiencia y autonomía científica, y empujarlo mediante innovación propia**.

## Estrategia frente a laboratorios con más recursos

Auralis no puede asumir que vencerá mediante fuerza bruta a organizaciones con enormes clusters de cómputo.

La estrategia debe buscar ventajas donde un proyecto pequeño todavía puede innovar:

- mejores algoritmos;
- arquitecturas más eficientes;
- sparsity y routing;
- memoria mejor diseñada;
- aprendizaje continuo;
- mejores datos;
- entrenamiento más eficiente;
- sistemas de agentes;
- automatización de investigación;
- optimización profunda del runtime;
- experimentación mucho más rápida.

## El papel de Rust

Rust es la base inicial porque permite controlar de cerca memoria, paralelismo, rendimiento y portabilidad manteniendo garantías fuertes de seguridad de memoria.

Auralis podrá incorporar backends especializados —GPU, aceleradores o kernels de bajo nivel— sin abandonar la idea de un núcleo comprensible y controlable.

## Horizonte

No existe una fecha honesta para alcanzar capacidades de frontera. El proyecto se estructura como una secuencia de sistemas completos cada vez mejores.

Cada generación debe dejar a la siguiente:

- código más sólido;
- mejores evaluaciones;
- más capacidad;
- más eficiencia;
- mejor infraestructura de experimentación;
- menos incertidumbre sobre qué funciona y qué no.

## Norte final

Construir un sistema que no solo responda preguntas, sino que sea capaz de **aprender, razonar, recordar, crear, investigar, utilizar herramientas y ayudar a diseñar una versión mejor de sí mismo**, siempre dentro de un proceso donde las mejoras puedan verificarse externamente.

Ese es Auralis.
