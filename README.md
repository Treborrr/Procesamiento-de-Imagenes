# 1ACC0235 — Procesamiento de Imágenes — Trabajo Parcial/Final 2026-01

## Objetivo del trabajo

Desarrollar un sistema de clasificación de acciones humanas en video aplicado a la
evaluación de productividad de empleados en operaciones de almacén (picking/packing),
comparando una técnica clásica de Procesamiento de Imágenes (HOG + SVM) contra un modelo
de Deep Learning (ResNet50 con transfer learning) para el mismo problema.

El sistema responde 3 preguntas de negocio a partir de clips de video cortos:

- **P1 — Estado del empleado:** Activo (Pick Up / Running) o Inactivo/Pausado (Standing).
- **P2 — Nivel de productividad:** Alto / Medio / Bajo, según la acción detectada.
- **P3 — Posible cuello de botella:** proporción de frames "Standing" vs. "Pick Up" en el clip.

## Alumnos participantes

| Nombre | Código |
|---|---|
| Alvarado Valle, Robert Leonardo | u202111912 |
| Bussalleu Salcedo, Fabrizio | u202315655 |
| Chavez Merino, Cielo Luwidka | u2019e443 |

## Descripción del dataset

**HMDB51** (Kuehne et al., 2011, Serre Lab - Brown University), descargado vía Kaggle
(mirror `easonlll/hmdb51`). Se usan 3 de sus 51 clases originales, elegidas por
corresponder exactamente a los 3 estados del caso de uso:

- `pick` (106 videos) → recogida de objetos → Activo / productividad Alta
- `run` (232 videos) → desplazamiento rápido → Activo / productividad Media
- `stand` (154 videos) → persona estática → Inactivo / posible cuello de botella

Total: 492 videos → 7,383 frames extraídos (224x224 RGB, hasta 30 frames representativos
por clip) → split 80/20 por video (sin fuga de datos entre train y test).

> Nota: el Hito 1 había propuesto UCF101, pero esa colección no contiene ninguna clase de
> tipo "persona inactiva" (ver `TF/DECISIONES_TECNICAS.md`, sección 1, para el detalle de
> por qué se corrigió a HMDB51).

## Modelos y resultados

Evaluación a nivel de video (mayoría de votos entre los frames de cada clip — la forma
representativa del uso real, ver `TF/DECISIONES_TECNICAS.md` sección 12.1):

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| HOG + SVM (clásico) | 59.6% | 0.722 | 0.520 | 0.542 |
| **ResNet50 (profundo, fine-tuning layer4)** | **80.8%** | **0.848** | **0.777** | **0.797** |

Tablas completas (por frame y por video) y matrices de confusión en `TF/results/`.

## Conclusiones

ResNet50 con transfer learning (fine-tuning del último bloque residual, `layer4`) superó
consistentemente al modelo clásico HOG + SVM (80.8% vs. 59.6% de accuracy por video, y
mejor balance entre clases). Se probaron 3 mejoras sobre los modelos base: evaluación a
nivel de video, balanceo de clases y fine-tuning real de ResNet50 (no solo su capa final).
El balanceo de clases resultó ser una mejora real para ResNet50, pero **no tuvo ningún
efecto medible en HOG+SVM** (0 predicciones cambiadas al comparar con/sin `class_weight`,
verificado empíricamente) — evidencia de que el sesgo de HOG+SVM hacia la clase "run" es
una limitación de separabilidad del descriptor de gradientes para posturas estáticas
(pick/stand), no un problema de desbalance de datos. Esto confirma que, para distinguir
posturas y acciones humanas completas en escenas variadas (distintos fondos, ángulos de
cámara e iluminación como los de HMDB51), las características aprendidas end-to-end de una
red pre-entrenada en ImageNet generalizan mejor que un descriptor fijo. El modelo clásico
sigue siendo válido como alternativa ligera (no requiere GPU, es interpretable), pero no
alcanza la precisión necesaria para un sistema de monitoreo de productividad en producción.
Como trabajo futuro queda: (1) incorporar detección de personas para escenas con múltiples
empleados simultáneos, y (2) extender el análisis de cuello de botella a ventanas de
monitoreo continuo en vez de clips aislados (ver limitaciones detalladas en
`TF/DECISIONES_TECNICAS.md`).

## Estructura del repositorio

```
TF/
  code/       # pipeline completo (descarga, extraccion, entrenamiento, GUI)
  data/       # dataset filtrado y frames procesados (no versionado en git)
  models/     # modelos entrenados (no versionado en git)
  results/    # metricas, matrices de confusion, tabla comparativa
  samples/    # clips de ejemplo para probar la GUI
  DECISIONES_TECNICAS.md   # bitacora de decisiones (antes/ahora/por que)
```

Ver `TF/code/README.md` para instrucciones de ejecución del pipeline.

## Licencia

Ver [`LICENSE`](LICENSE) — uso académico/educativo, desarrollado para el curso 1ACC0235
(Procesamiento de Imágenes) de la Universidad Peruana de Ciencias Aplicadas (UPC).
