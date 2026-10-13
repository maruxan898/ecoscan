# Exposición de 10 minutos

**Alexis (0:00-3:00):** problema, usuarios y datos. Explicar por qué conocer el material ayuda pero no garantiza reciclabilidad. Mostrar el desbalance y la separación 70/15/15. Aclarar la diferencia entre 5.054 filas del visor y la versión reducida usada.

**Gabriela (3:00-5:00):** comparación EfficientNet-B0 / ResNet18. Mostrar accuracy balanceada, F1 macro y matrices reales. Explicar que se selecciona por validación y se reporta prueba después, evitando usar prueba como entrenamiento. Interpretar un error costoso a partir del CSV de predicciones.

**Agustín (5:00-8:00):** demo: subir un objeto, leer seis probabilidades, explicar la confianza, generar la recomendación, cambiar imagen y mostrar que hay que reclasificar. Mostrar las métricas de prueba de la pestaña de desempeño.

**Todos (8:00-10:00):** explicar cada módulo, limitaciones, privacidad y mejoras. No memorizar números sin entenderlos. Ensayar que cualquiera pueda responder sobre todas las partes.

## Preguntas que deben poder responder

1. ¿Qué cambia del modelo ImageNet? La capa de salida pasa a seis clases y se entrena la cabeza de seis clases sobre características ImageNet; el encoder queda congelado en la modalidad CPU. La modalidad fine_tune ajusta todos los pesos.
2. ¿Por qué softmax? Convierte seis logits en puntuaciones positivas que suman uno; no demuestra certeza ni calibración.
3. ¿Por qué accuracy balanceada? Promedia recall por clase, evitando que papel o vidrio oculten fallos en trash.
4. ¿Por qué tres particiones? Train ajusta pesos, val selecciona el checkpoint, test estima desempeño después de cerrar decisiones.
5. ¿Por qué pérdida ponderada? Penaliza más los errores de clases con menos ejemplos, usando solo conteos de entrenamiento.
6. ¿Qué es aumento de datos? Cambios en imágenes de train para reducir memorización; val y test tienen transformación determinista.
7. ¿Qué ve Gemini? Texto con clase, confianza y contexto; no la foto ni todas sus características.
8. ¿Dónde queda la clave? Variable de entorno o Secret de Hugging Face; nunca en GitHub.
9. ¿Qué hace el umbral de 70%? Activa revisión humana. Es una elección educativa que debe validarse si el sistema se usa realmente.
10. ¿Qué error importa? Enviar residuos no aceptados a una carga reciclable; no se puede medir la aceptación real solo con etiquetas de material.
11. ¿Qué harían para mejorar? Más datos de escenarios reales, revisión de objetos similares, validación externa y calibración en validación.
12. ¿Qué pasa sin modelo o API? La app explica qué falta; no fabrica una clasificación ni una respuesta de IA.

## Mapa del código

`data.py`: archivo reducido → fotos RGB → hashes → particiones y conteos.

`train_cached.py`: encoder congelado → características en caché → cabeza entrenable → validación y prueba.

`train.py` (alternativa GPU): Dataset/DataLoader → timm → AdamW/pérdida → validación → checkpoint → prueba y exportación.

`inference.py`: checkpoint → misma transformación → softmax → nombres en español.

`app.py`: componentes Gradio y eventos → funciones → estado de la predicción por sesión.

`recommend.py`: resultado → prompt con límites → Gemini → texto o error controlado.
