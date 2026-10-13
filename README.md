---
title: EcoScan
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
---

# EcoScan · De la imagen a una decisión informada

**Integrantes:** Alexis, Gabriela y Agustín.

Proyecto de IA Generativa para la Gestión Industrial: clasificar una fotografía de un residuo entre seis materiales y generar una orientación en español mediante Gemini. Ayuda a explorar separación de residuos; no certifica reciclabilidad ni sustituye al gestor local.

## Estado de la entrega

El código calcula y exporta resultados reales. Si faltan pesos o `modelo/metrics.json`, la app lo indica y no inventa predicciones ni métricas. La publicación y la API deben probarse con las cuentas y secretos del grupo. Agregar aquí las URLs verificadas del repositorio público y del Space cuando existan.

## Datos y fuentes

TrashNet fue creado por **Gary Thung y Mindy Yang**, para un proyecto de Stanford CS229. Fotos de objetos sobre fondo blanco, con celulares y distintas condiciones de iluminación. Licencia MIT, conservar atribución y licencia de la fuente.

- Fuente entregada: https://huggingface.co/datasets/garythung/trashnet
- Descripción original: https://github.com/garythung/trashnet
- Licencia: https://github.com/garythung/trashnet/blob/master/LICENSE

| Clase original | Nombre en EcoScan | Fotos descritas en la fuente original |
|---|---|---:|
| cardboard | Cartón | 403 |
| glass | Vidrio | 501 |
| metal | Metal | 410 |
| paper | Papel | 594 |
| plastic | Plástico | 482 |
| trash | Otros residuos | 137 |
| **Total** | | **2.527** |

El visor de Hugging Face muestra 5.054 registros, mientras la fuente original describe 2.527 fotos. EcoScan descarga específicamente **dataset-resized.zip** del repositorio de Hugging Face (unos 43 MB), para usar una sola versión de la colección. Los conteos efectivamente procesados se guardan en `modelo/conteos.csv`; no se asume que todas las filas del visor son observaciones independientes.

Separación **70% entrenamiento, 15% validación, 15% prueba**, estratificada por clase, semilla 42 y redondeos de scikit-learn. Antes de dividir se eliminan duplicados exactos mediante hash de píxeles RGB y tamaño. Los hashes de los tres conjuntos deben ser disjuntos. No hay identificadores de objetos: esto no descarta fotos casi iguales del mismo objeto. Para una evaluación más exigente hace falta una colección externa o división por objeto.

Desbalance: `trash` es la clase minoritaria; `paper` tiene más de cuatro veces sus ejemplos. Se usa entropía cruzada ponderada calculada solo en entrenamiento y accuracy balanceada como criterio de selección. **trash no equivale a orgánicos.**

## Modelos y evaluación

Se comparan **EfficientNet-B0** y **ResNet18**, ambos preentrenados en ImageNet con `timm`. EfficientNet sigue la pauta docente y es un candidato compacto; ResNet18 ofrece un contraste de arquitectura. Se adapta la salida a seis clases y se ajustan todos los pesos con AdamW, lr=1e-4, weight_decay=1e-4, aumento de datos solo en entrenamiento y normalización propia de cada modelo.

Máximo 12 épocas y parada temprana tras 4 sin mejora. El mejor checkpoint de cada modelo se elige por validación. La arquitectura ganadora se decide **antes de evaluar prueba**; ambas se evalúan sobre las mismas fotos de prueba para reportar accuracy, accuracy balanceada, F1 macro, precisión/recall por clase y matriz de confusión. El modelo ganador se exporta a `modelo/modelo.pt`, con clases y configuración de preprocesamiento para que la app use exactamente el mismo orden y normalización. No requiere descargar pesos ImageNet al arrancar.

Error relevante: confundir otros residuos con un material potencialmente reciclable puede contaminar una carga. El reporte mide clasificación de material, **no aceptación real para reciclaje**. Analizar la fila `trash` y las confusiones hacia materiales; también mirar plástico/vidrio. La confianza softmax no está calibrada y no detecta de forma fiable objetos fuera de dominio. El umbral 70% es una política educativa, no un indicador validado de seguridad.

## Ejecutar el notebook (recomendado: Colab con GPU T4)

1. Abrir `notebooks/entrenamiento.ipynb` en https://colab.research.google.com/.
2. Activar GPU en Entorno de ejecución → Cambiar tipo de entorno.
3. Ejecutar todas las celdas. El notebook incorpora el código del proyecto, descarga los datos, entrena y muestra métricas, matrices y curvas.
4. Descargar `ecoscan_entrega.zip` y el notebook ejecutado (`Archivo → Descargar → .ipynb`). Sustituir el notebook del repositorio por la versión ejecutada.
5. Revisar las conclusiones según los números obtenidos. No afirmar un rendimiento que no aparece en la evaluación.

## Ejecutar localmente

Python 3.11 recomendado. En PowerShell, desde la carpeta del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.train --epochs 12 --prepare
$env:GEMINI_API_KEY = "CONFIGURAR_LOCALMENTE"
$env:GEMINI_MODEL = "gemini-2.5-flash"
.\.venv\Scripts\python.exe app.py
```

La CPU funciona, pero entrenar puede tardar considerablemente más que en Colab. Para abrir la app después de Colab basta copiar `modelo/` y `ejemplos/`; no hace falta entrenar otra vez ni subir `datos/`.

## Gemini y privacidad

Se usa la API oficial `google-genai`; recibe clase, confianza y contexto limitado a 500 caracteres. **No recibe la imagen.** El prompt limita el tema, pide revisión humana con confianza baja y evita inventar reglas municipales, colores de contenedores o aceptación de plásticos. Si falta clave o la API falla, se muestra el estado y no se presenta texto simulado como IA. La disponibilidad del modelo y la cuota dependen de la cuenta; `GEMINI_MODEL` permite cambiar a uno habilitado.

La foto se procesa en el servidor. La app no la guarda explícitamente en una colección, pero la plataforma puede manejar archivos temporales y registros. No subir fotos con personas ni datos sensibles. La API tiene timeout de 30 segundos y no expone errores del proveedor o claves al visitante.

## Publicar en GitHub y Hugging Face Spaces

1. Crear un repositorio **público** llamado `ecoscan`; invitar a los integrantes como colaboradores.
2. Subir el contenido del proyecto: `app.py`, `src/`, `requirements.txt`, README, notebook ejecutado, `modelo/modelo.pt`, `modelo/metrics.json`, matrices, conteos, clases y `ejemplos/`. No subir `.venv/`, `datos/`, secretos ni los archivos privados de clases.
3. Crear un Space en https://huggingface.co/new-space, SDK **Gradio**, hardware CPU Basic, visibilidad pública. Subir los mismos archivos de ejecución. Este README incluye el encabezado del Space.
4. En Settings → Variables and secrets, crear **Secret** `GEMINI_API_KEY`. Opcional: variable `GEMINI_MODEL`. Nunca escribir la clave en README o Python.
5. Agregar el enlace GitHub al README del Space y ambos enlaces a la ficha de entrega. Abrir la URL del Space desde el celular; probar foto propia, baja confianza y botón Gemini.
6. Si se cambia una foto después de clasificar, la app invalida el resultado anterior: hay que volver a clasificar antes de recomendar.

No se puede garantizar la disponibilidad de servicios gratuitos. Conservar capturas de la demo y el modelo local para la exposición.

## Estructura

```text
app.py                    interfaz Gradio
src/data.py               descarga, duplicados y particiones
src/train.py              entrenamiento, comparación y exportación
src/inference.py          carga, preprocesamiento e inferencia
src/recommend.py          prompt responsable y Gemini
notebooks/entrenamiento.ipynb  flujo reproducible en Colab
modelo/                   pesos y evidencia de evaluación
ejemplos/                 cinco fotos de entrenamiento, con atribución TrashNet
docs/                     ficha y preparación de la exposición
tests/                    verificaciones de comportamiento
```

## Uso de IA y referencia docente

Se utilizó **ChatGPT/Codex** para estructurar y programar el proyecto, adaptar el flujo al dataset TrashNet, revisar casos de error y preparar documentación. Gemini genera recomendaciones en tiempo de ejecución. El grupo debe ejecutar, revisar, interpretar y explicar el trabajo; el uso de IA no reemplaza el dominio del código.

Referencias de clase: `Pauta_Clase_2_Alzheimer_3_Modelos.ipynb` (transfer learning con EfficientNet/timm, comparación y accuracy balanceada) y `Guia_1_Clasificacion_Binaria_CNN_1-2.ipynb` (aumento de datos, early stopping, matrices y errores costosos). Se adapta a residuos multiclase y se reserva prueba independiente de validación. Los materiales del profesor no se redistribuyen en el repositorio.

Documentación técnica: https://huggingface.co/docs/timm/quickstart · https://www.gradio.app/guides/quickstart · https://ai.google.dev/gemini-api/docs/text-generation
