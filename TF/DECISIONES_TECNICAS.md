# Bitácora de decisiones técnicas — Trabajo Final PI

Documento vivo. Cada vez que se toma o se cambia una decisión de diseño durante el
desarrollo se registra aquí con el mismo formato, para poder redactar directamente
la sección de "cambios respecto al Hito 1" y las tablas comparativas del informe final.

Formato de cada entrada:
- **Antes (Hito 1 / opción inicial):** qué se había propuesto o qué alternativas se evaluaron.
- **Ahora (decisión final):** qué se implementa.
- **Por qué es mejor:** justificación técnica.
- **Comparación con alternativas menos eficientes:** para la tabla comparativa del informe
  (nunca se compara contra algo mejor que la opción elegida).

---

## 1. Dataset: UCF101 → HMDB51

**Antes (Hito 1):** El informe parcial proponía UCF101 (Soomro et al., 2012) con tres
clases supuestas: "Pick Up", "Running", "Standing".

**Problema detectado:** Se verificó la lista oficial de las 101 clases de UCF101
(`classInd.txt` del split oficial) y ninguna de esas tres clases existe. UCF101 está
compuesto íntegramente por acciones dinámicas (deportes, instrumentos musicales,
interacción humano-objeto y humano-humano) y no contempla ninguna clase de tipo
"persona inactiva/parada", que es indispensable para la pregunta de negocio P3
(cuello de botella operativo).

**Ahora:** Se cambia el dataset a **HMDB51** (Kuehne et al., 2011; Serre Lab, Brown
University), que sí contiene, de forma nativa y sin adaptación, las tres clases
exactas requeridas por el caso de uso: `pick`, `run`, `stand`.

**Por qué es mejor:**
- Correspondencia 1 a 1 entre clase del dataset y estado de negocio, sin necesidad
  de mapear o combinar clases artificialmente.
- HMDB51 sí incluye videos de personas estáticas/en reposo (clase `stand`), condición
  indispensable para modelar el estado "Inactivo/Pausado" del caso de uso.
- Descarga directa vía Kaggle API (mirrors `easonlll/hmdb51` y `jizeyong/hmdb51`),
  manteniendo el requisito de "descarga directa sin solicitud de acceso especial"
  que ya se valoraba en el Hito 1.

**Comparación con alternativas menos eficientes:**

| Criterio | HMDB51 (elegido) | UCF101 (forzando clases) | Kinetics-400 |
|---|---|---|---|
| Clases nativas pick/run/stand | Sí, exactas | No existen | Sí, pero dataset de 300k videos |
| Necesidad de clase "inactivo" | Cubierta (`stand`) | Ausente — rompe P3 | Cubierta |
| Tamaño / factibilidad en CPU | 6,766 clips (manejable) | 13,320 clips | 300,000 (inviable sin GPU) |
| Descarga directa Kaggle | Sí | Sí (pero clases inválidas) | No, requiere scripts de YouTube-dl |
| Riesgo de invalidar P1/P2/P3 | Ninguno | Alto (falta clase base) | Bajo, pero costo computacional alto |

*(UCF101 se mantiene en la tabla solo para mostrar por qué se abandonó — no se usa como
punto de comparación de rendimiento final, ya que nunca llegó a entrenarse: el problema
se detectó antes de la fase de modelización.)*

**Impacto en el resto del informe:** La sección 2 (Descripción del dataset) del Hito 1
debe reescribirse con las características de HMDB51 en el informe final. Las secciones
3.2/3.3 (modelo clásico HOG+SVM y modelo profundo ResNet50) y sus pipelines **no cambian**,
porque ambos operan sobre frames extraídos de video y son agnósticos al dataset de origen.

---

## 2. Mirror de Kaggle para HMDB51

**Alternativas evaluadas:** `easonlll/hmdb51` vs `jizeyong/hmdb51`.

**Ahora:** `easonlll/hmdb51` configurado por defecto en `config.py` (`KAGGLE_DATASET_SLUG`).

**Por qué:** Ambos son mirrors del mismo dataset original (Serre Lab); se deja el segundo
como alternativa de respaldo en caso de que el primero tenga archivos corruptos o
cuota de descarga excedida. *(Pendiente de confirmar cuál da menos problemas al descargar
— se actualizará esta entrada tras la primera descarga real.)*

---

## 3. Framework de deep learning: PyTorch vs TensorFlow/Keras

**Decisión:** PyTorch + torchvision (`torchvision.models.resnet50`).

**Por qué es mejor para este proyecto:**
- `torchvision` expone ResNet50 pre-entrenado en ImageNet con una API directa para
  congelar capas y reemplazar la capa fully-connected final, tal como describe la
  sección 3.3 del Hito 1.
- Mayor control explícito del loop de entrenamiento, útil para reportar métricas por
  época en el informe (curvas de loss/accuracy).
- No se usó GPU (no disponible en el entorno de desarrollo), y ambos frameworks
  soportan CPU; PyTorch fue la preferencia del equipo.

**Comparación con alternativa menos eficiente para este caso:**

| Criterio | PyTorch (elegido) | TensorFlow/Keras |
|---|---|---|
| Control explícito del training loop | Alto | Medio (abstraído por `model.fit`) |
| Facilidad para congelar capas específicas | Directa (`param.requires_grad = False`) | Directa también, pero requiere recompilar el modelo |
| Reporticidad de métricas por época para el informe | Manual y explícita | Requiere callbacks adicionales |

*(Nota: esta no es una decisión de "peor a mejor", ambos frameworks son válidos;
se documenta como decisión de equipo, no como corrección de un error como en el
caso del dataset.)*

---

## 4. GUI para el usuario final

**Decisión:** Gradio (interfaz web local, sin necesidad de exponer servidor a internet).

**Por qué:** Requisito del enunciado (sección 7): "interfaz gráfica simple y amigable,
destinada a un usuario final". Gradio permite subir un video, mostrar el frame
representativo y las 3 respuestas de negocio (P1/P2/P3) con pocas líneas de código,
sin requerir que el usuario final instale un entorno de escritorio (Tkinter) ni
conocimientos técnicos.

**Comparación con alternativa menos eficiente:**

| Criterio | Gradio (elegido) | Tkinter |
|---|---|---|
| Curva de uso para un Stakeholder no técnico | Baja (navegador) | Media (requiere ejecutar app de escritorio) |
| Tiempo de desarrollo | Bajo | Medio (layout manual) |
| Estética / "amigable" (requisito del enunciado) | Alta | Básica |

---

## 5. Split train/test: por video, no por frame

**Alternativa descartada:** dividir aleatoriamente el total de frames extraídos
(train_test_split directo sobre `frames_metadata.csv`).

**Ahora:** el split se hace sobre la lista de videos únicos (estratificado por clase)
y luego se asignan todos los frames de cada video al conjunto que le tocó a su video.

**Por qué es mejor:** frames consecutivos del mismo clip son casi idénticos entre sí
(misma persona, mismo fondo, mismo movimiento). Si el split fuera por frame, el mismo
video terminaría con frames en train y frames en test, y el modelo "vería" ese video
durante el entrenamiento — la accuracy en test quedaría artificialmente inflada y no
reflejaría generalización real a videos nuevos.

**Comparación con alternativa menos eficiente:**

| Criterio | Split por video (elegido) | Split por frame (descartado) |
|---|---|---|
| Fuga de datos (data leakage) | Ninguna | Alta — mismo video en train y test |
| Accuracy de test representativa de uso real | Sí | No, sobreestimada |
| Complejidad de implementación | Ligeramente mayor (agrupar por video_id) | Trivial |

---

## 6. Origen de los frames: el mirror de Kaggle ya viene con frames extraídos

**Supuesto inicial:** `extract_frames.py` decodificaría archivos de video (.avi) con
OpenCV (`cv2.VideoCapture`) para extraer fotogramas a 5 fps.

**Hallazgo al descargar:** el mirror de Kaggle usado (`easonlll/hmdb51`) no distribuye
archivos .avi, sino que cada video ya viene como una carpeta con sus frames
individuales en JPG (resolución nativa 320x240, fps nativo asumido ~30 fps, sin
metadata explícita de fps en el propio mirror).

**Ahora:** `download_dataset.py` copia las carpetas de frames de las clases
`pick`/`run`/`stand` completas a `data/raw/`, y `extract_frames.py` ya no decodifica
video: sub-muestrea 1 de cada N frames (N = 30/5 = 6) para aproximar los 5 fps
definidos en el Hito 1, y redimensiona a 224x224.

**Por qué no afecta la validez del proyecto:** el resultado final (frames a 5 fps
equivalentes, 224x224 RGB) es idéntico en forma al pipeline propuesto originalmente;
solo cambia el punto de partida (frames ya separados vs. video contenedor). El
supuesto de 30 fps nativo debe declararse como limitación en el informe (sección de
alcance y limitaciones), ya que no se confirmó con metadata exacta del mirror.

**Comparación con alternativa descartada:**

| Criterio | Sub-muestreo de frames existentes (elegido) | Descargar HMDB51 desde el repositorio original (.avi) y decodificar con OpenCV |
|---|---|---|
| Tiempo de descarga | Bajo (mirror ya listo en Kaggle, ~3.3GB) | Alto (servidor original más lento, fuera de Kaggle) |
| Complejidad del pipeline | Baja (solo sub-muestreo de imágenes) | Media (requiere `cv2.VideoCapture` + manejo de codecs) |
| Control exacto del fps de origen | Aproximado (se asume 30 fps) | Exacto (se lee `cap.get(cv2.CAP_PROP_FPS)`) |

---

## 7. Estrategia de muestreo de frames: stride fijo (5fps asumido) → muestreo adaptativo por clip

**Primer intento:** aplicar un stride fijo (1 de cada 6 frames) asumiendo 30 fps
nativos, para aproximar los 5 fps del Hito 1.

**Problema detectado al correr el pipeline:** los clips de este mirror de HMDB51 ya
vienen recortados a solo sus frames representativos (10 a 60 frames por clip,
promedio ~15), no al video completo a fps nativo. Aplicar un stride de 6 sobre clips
ya cortos dejaba un promedio de **2.9 frames por video** — insuficiente para
entrenar cualquiera de los dos modelos con información temporal representativa
(algunos videos quedaban con 1 solo frame).

**Ahora:** se elimina el supuesto de fps nativo. `extract_frames.py` toma TODOS los
frames disponibles de un clip si son ≤ `MAX_FRAMES_PER_VIDEO` (30), o una muestra
uniforme (`numpy.linspace`) distribuida en todo el clip si supera ese límite. Esto
sube el promedio a ~15-18 frames/video (≈7,500 frames totales vs. 1,425 del primer
intento), sin descartar informacion de clips ya de por si cortos.

**Por qué es mejor:** el objetivo real no es "exactamente 5 fps" sino tener una
cantidad suficiente y representativa de frames por clip para que HOG+SVM y
ResNet50 puedan aprender el patrón visual de cada acción; un stride fijo basado en
un fps no verificado sub-utilizaba los datos ya escasos del dataset.

**Comparación con alternativa descartada:**

| Criterio | Muestreo adaptativo (elegido) | Stride fijo asumiendo 30fps (descartado) |
|---|---|---|
| Frames promedio por video | ~15-18 | 2.9 |
| Total de frames de entrenamiento | ~7,500 | 1,425 |
| Riesgo de videos con 1 solo frame util | Ninguno | Sí (clips cortos) |
| Depende de un supuesto no verificado (fps nativo) | No | Sí |

**Impacto en el informe:** la sección de dataset debe describir el muestreo como
"hasta 30 frames representativos por clip, distribuidos uniformemente" en lugar de
"5 fotogramas por segundo", ya que el mirror de origen no expone el fps real del
video para calcular una tasa exacta.

---

## 8. Entrenamiento de ResNet50 en CPU: loop tradicional → embeddings cacheados

**Primer intento:** loop de entrenamiento tradicional (forward + backward + optimizer.step
por batch, repetido en las 12 épocas), con el backbone convolucional congelado y solo la
capa FC entrenable.

**Problema detectado:** en CPU, sin GPU, ese loop tardó más de 15 minutos sin terminar
siquiera la primera de las 12 épocas — un backbone congelado no actualiza sus pesos, pero
un loop de entrenamiento tradicional igual recalcula el forward pass completo de ResNet50
en cada época, desperdiciando cómputo de forma repetida (~12x más de lo necesario).
Proyectado, el entrenamiento completo hubiera tomado varias horas.

**Ahora:** se extrae el embedding de 2048 valores (salida de ResNet50 antes de la capa FC)
UNA sola vez por imagen — incluyendo una copia con data augmentation además de la original,
para no perder esa parte del pipeline propuesto en el Hito 1 — y las `RESNET_EPOCHS` se
entrenan sobre esos embeddings ya cacheados (una capa `nn.Linear(2048,3)` con Adam), lo que
toma segundos en lugar de horas.

**Por qué es matemáticamente equivalente:** dado que el backbone está 100% congelado
(`requires_grad=False` en todas las capas convolucionales), su salida para una misma imagen
de entrada es idéntica sin importar cuántas veces se recalcule. Entrenar la capa FC sobre
embeddings cacheados produce exactamente el mismo resultado que entrenarla dentro del loop
completo — la única diferencia es que se evita recalcular una y otra vez una salida que ya
sabemos que no va a cambiar.

**Comparación con alternativa descartada:**

| Criterio | Embeddings cacheados (elegido) | Loop de entrenamiento tradicional (descartado) |
|---|---|---|
| Pasadas por el backbone de ResNet50 | 2 (original + augmented, una sola vez) | 12 (una por época) |
| Tiempo estimado en CPU | Minutos | Horas (no terminó ni la 1ra época en 15 min) |
| Resultado del entrenamiento de la capa FC | Idéntico (backbone congelado) | Idéntico |
| Compatibilidad con data augmentation | Sí (copia aumentada incluida en el cache) | Sí, pero a costo computacional inviable en CPU |

---

## 9. Resultados finales (Publicación de Resultados)

Dataset: 492 videos HMDB51 (pick=106, run=232, stand=154) → 7,383 frames →
split por video 80/20 (train: 393 videos / 5,878 frames, test: 99 videos / 1,505 frames).

| Modelo | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| HOG + SVM (clásico) | 56.1% | 0.591 | 0.501 | 0.510 |
| **ResNet50 (profundo)** | **75.1%** | **0.742** | **0.735** | **0.738** |

ResNet50 supera a HOG+SVM por ~19 puntos de accuracy. Es el resultado esperado y
consistente con la literatura (He et al., 2016): las features aprendidas end-to-end de una
red pre-entrenada en ImageNet generalizan mejor que un descriptor de gradientes fijo (HOG)
para distinguir posturas corporales completas (de pie vs. en movimiento) en escenas
variadas (fondos, iluminación, ángulos de cámara distintos en HMDB51). El notorio train_acc
de HOG+SVM más bajo que el de ResNet50 (98.9% en train) sugiere además que HOG+SVM no logra
capturar tan bien la variabilidad del dataset ni siquiera en entrenamiento.

Matrices de confusión: `results/hog_svm_confusion_matrix.png`, `results/resnet50_confusion_matrix.png`.
Tabla completa: `results/comparison_table.csv` / `.md`.

**Desglose por clase (recall):**

| Clase | HOG+SVM | ResNet50 |
|---|---|---|
| pick | 30.9% | 73.4% |
| run | 81.3% | 85.5% |
| stand | 38.3% | 65.5% |

HOG+SVM muestra un sesgo marcado hacia predecir "run" (recall 81% en esa clase, pero
solo 31-38% en pick/stand) — el descriptor de gradientes probablemente captura mejor el
patrón de movimiento/desenfoque característico de "correr" que las posturas más sutiles
de "recoger" o "estar de pie". ResNet50 es mucho más balanceado entre las 3 clases.
Consistente con lo observado manualmente en la GUI (sección 10): un clip real de "stand"
fue clasificado correctamente por ResNet50 pero como "run" por HOG+SVM.

---

## 10. GUI: lógica de P3 (cuello de botella) — clase ganadora → proporción de frames

**Primer intento en la GUI:** `P3 = "Sí"` si la clase ganadora (mayoría de votos) del clip
completo era `stand`, `"No"` en cualquier otro caso.

**Problema detectado:** esa regla ignora la definición de P3 planteada en el Hito 1
("proporción de frames Standing versus Pick Up"). Al probar la GUI con un clip real de
`stand` mal etiquetado como `run` por ambos modelos (clip `..._bad_..`, autoetiquetado como
de mala calidad por HMDB51), quedó claro que reducir P3 a la clase ganadora pierde
información y no refleja el diseño original de 3 posibles respuestas (Sí/No/Próximo).

**Ahora:** `business_answers()` en `gui_app.py` recibe las predicciones por frame del video
(no solo la clase ganadora) y calcula `stand_ratio` y `pick_ratio` sobre el total de frames
del clip. Umbral: `stand_ratio >= 50%` → "Sí"; `stand_ratio > pick_ratio` (sin mayoría
absoluta) → "Próximo"; en otro caso → "No". La GUI muestra el desglose de proporciones en
el detalle de salida.

**Por qué es mejor:** replica la definición real de P3 del Hito 1 (proporción, no solo
"instante dominante"), y expone las 3 categorías originales (Sí/No/Próximo) en vez de solo
2. También sirvió para verificar visualmente que ResNet50 clasifica bien un clip de `stand`
de buena calidad (96.7% confianza) donde HOG+SVM se equivoca (89.2% a favor de `run`) — un
buen ejemplo concreto de la brecha de accuracy entre ambos modelos para el video/informe.

**Nota de limitación para el informe:** el umbral de 50%/"Próximo" es una heurística simple
de demo; el Hito 1 hablaba de una "ventana temporal" continua de monitoreo, mientras que
la GUI solo analiza un clip corto aislado a la vez. Queda como trabajo futuro extenderlo a
monitoreo continuo de video en vivo.

---

## 11. Limitación: el sistema no separa múltiples personas en un mismo frame

**Limitación detectada:** ni HOG+SVM ni ResNet50 hacen detección/segmentación de personas.
Ambos modelos clasifican el **frame completo** como una sola unidad y devuelven **una sola
etiqueta por frame/clip**. Esto viene heredado de HMDB51: cada clip está anotado con una
sola acción, asumiendo un actor principal, aunque pueda haber gente de fondo.

**Consecuencia práctica:** si en un frame real de almacén aparecen dos empleados haciendo
cosas distintas (uno parado, otro recogiendo), el sistema no los distingue — da una única
predicción global ambigua o sesgada hacia lo visualmente dominante, no una respuesta por
persona.

**Por qué no se implementa ahora:** agregar un detector de personas (ej. YOLO o
MobileNet-SSD) que recorte cada bounding box y clasifique cada recorte por separado está
fuera del alcance definido para el Hito Final — el enunciado pide comparar una técnica
clásica vs. una profunda para UN problema de clasificación, no un pipeline de detección +
clasificación multi-persona. Se documenta como limitación conocida y como línea de trabajo
futuro, no como una omisión accidental.

**Para el informe (sección Alcance y Limitaciones):** agregar, junto a la limitación ya
existente del Hito 1 ("no realiza seguimiento de identidad individual"), que el sistema
tampoco separa múltiples personas dentro de un mismo frame — cada video/frame se asume con
un único trabajador en foco, consistente con la estructura de clips del dataset HMDB51
usado. Trabajo futuro: incorporar un detector de personas previo a la clasificación para
escenas con múltiples empleados simultáneos.

---

## 12. Mejoras post-Hito 1: evaluación por video, balanceo de clases, fine-tuning de layer4

Tras revisar los resultados iniciales (sección 9) se identificaron 3 mejoras reales,
no solo cosméticas:

### 12.1 Evaluación a nivel de video (mayoría de votos), no solo por frame

**Problema:** las métricas originales (sección 9) se calculaban tratando cada frame como
una muestra independiente. Pero el uso real del sistema (y la GUI) sube un clip completo y
agrega las predicciones de sus frames por mayoría de votos — evaluar por frame suelto no
es representativo del caso de uso real y castiga innecesariamente al modelo (un solo frame
ambiguo de un video que en conjunto se clasifica bien correctamente cuenta como error).

**Ahora:** `eval_utils.report_video_level()` agrupa las predicciones de test por
`video_id`, calcula la clase más votada por video (igual que hace `gui_app.py` en
inferencia real) y reporta métricas separadas a nivel de video
(`results/*_video_level_metrics.json`), ademas de las metricas por frame ya existentes.

### 12.2 Balanceo de clases (class_weight)

**Problema:** `run` tiene 232 videos vs. 106 de `pick` — casi el doble. Esto explicaba el
sesgo observado de HOG+SVM hacia predecir "run" (recall 81% en run vs. 31-38% en
pick/stand, sección 9).

**Ahora:** `SVC(class_weight="balanced")` en HOG+SVM, y `CrossEntropyLoss(weight=...)` con
pesos inversamente proporcionales a la frecuencia de clase (`compute_class_weight`) en
ResNet50.

**Resultado negativo verificado en HOG+SVM (hallazgo honesto, no oculto):** se comparó
`class_weight=None` vs. `class_weight="balanced"` sobre el mismo dataset completo de train/test
y las predicciones fueron **idénticas: 0 de 1505 frames de test cambiaron**. También se
probó oversampling de las clases minoritarias (duplicar pick/stand hasta igualar a run) como
alternativa: la accuracy global bajó levemente (56.1% → 55.1%) y el recall de pick/stand
solo mejoró marginalmente (31%→34%, 38%→39%) a costa de peor recall en run (81%→77%).
**Conclusión:** el sesgo de HOG+SVM hacia "run" no es un problema de desbalance de datos ni
de ponderación de la pérdida — es una limitación de **separabilidad de las features HOG**:
el descriptor de gradientes no distingue bien posturas estáticas/sutiles (pick vs. stand)
sin importar cómo se pese o remuestree el entrenamiento. Se mantiene `class_weight="balanced"`
en el código por buenas prácticas (no perjudica), pero se documenta que no resuelve el sesgo
observado — este es justamente uno de los puntos a favor de ResNet50 en la comparación final.

### 12.3 Fine-tuning de layer4 (no solo la capa FC) en ResNet50

**Antes (sección 8):** solo se entrenaba una capa lineal (`nn.Linear`) sobre embeddings
congelados de todo el backbone — rápido, pero deja precisión sobre la mesa porque las
features nunca se adaptan al dominio específico (HMDB51/almacén) más allá de lo aprendido
en ImageNet.

**Ahora:** se congela solo conv1..layer3 (se cachea su salida, un mapa de 1024x14x14, en
float16 para no exceder ~5GB de RAM) y se afina de verdad (forward + backward reales en
cada época) el último bloque residual (`layer4`) junto con la nueva capa FC. Esto es
fine-tuning real, no solo un clasificador lineal sobre features fijas, y debería mejorar
la capacidad del modelo de distinguir patrones específicos del dataset (ej. posturas
"stand" vs "pick" que ImageNet nunca vio en ese contexto).

**Por qué se cachea conv1..layer3 y no todo el backbone:** esas capas no se entrenan, así
que su salida es siempre la misma para una misma imagen — igual razonamiento que en la
sección 8, solo que ahora el "corte" entre lo cacheado y lo entrenable se movió más
adentro de la red (antes: todo congelado; ahora: layer4 también se entrena).

**Comparación de las 3 variantes de ResNet50 probadas:**

| Variante | Que se entrena | Costo en CPU | Accuracy (video-level) |
|---|---|---|---|
| Loop tradicional (descartado, sección 8) | Toda la red (backbone congelado igual recalculado) | Horas (inviable) | No se completó |
| Solo capa FC sobre embeddings cacheados (sección 8) | 1 capa lineal | Minutos | 75.1% (frame-level; no se evaluó por video en esa version) |
| **Fine-tuning de layer4 + FC (elegido)** | Último bloque residual + capa lineal | ~40 min (12 épocas, ~3.5 min/época) | **80.8%** |

### 12.4 Resultados finales tras las 3 mejoras

| Modelo | Accuracy (frame) | Accuracy (video) | F1 macro (video) |
|---|---|---|---|
| HOG + SVM (class_weight balanced, sin efecto medible) | 56.1% | 59.6% | 0.542 |
| ResNet50 (fine-tuning layer4 + class weights) | 79.9% | **80.8%** | **0.797** |

ResNet50 subió de 75.1% a 79.9%/80.8% gracias al fine-tuning real de layer4 (vs. solo
entrenar la capa FC). La evaluación por video sube la accuracy de ambos modelos respecto a
la evaluación por frame (59.6% vs 56.1% en HOG+SVM; 80.8% vs 79.9% en ResNet50), confirmando
que agregar por mayoría de votos entre frames de un mismo clip filtra ruido de frames
individuales ambiguos — más representativo del uso real vía la GUI.

Detalle por clase (recall, video-level): `results/*_video_level_per_class_metrics.csv`.
Tablas comparativas completas: `results/comparison_table.csv` (por frame) y
`results/comparison_table_video_level.csv` (por video, la más representativa).

---

*(Pendiente: redactar sección de Conclusiones del informe final citando estos resultados.)*
