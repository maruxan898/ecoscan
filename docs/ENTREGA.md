# Ficha de entrega · EcoScan

Integrantes: Alexis, Gabriela y Agustín.

Dataset: https://huggingface.co/datasets/garythung/trashnet

Origen: Gary Thung y Mindy Yang, Stanford CS229. Licencia MIT. Archivo utilizado: dataset-resized.zip. Seis clases; fuente original: cartón 403, vidrio 501, metal 410, papel 594, plástico 482 y otros residuos 137. Conteos procesados y por partición: `modelo/conteos.csv`.

División: 70/15/15 estratificada con semilla 42 tras excluir tres parejas contradictorias: **2.521 fotos, 1.764 train / 378 val / 379 test**. Sesgos: fondo blanco, fuente única, pocos otros residuos y ausencia de IDs de objetos. Ninguna prueba externa acredita rendimiento en una planta real.

Repositorio público: https://github.com/maruxan898/ecoscan.

App pública: https://ecoscan-alexis-gabriela-agustin.streamlit.app/ (clasificación verificada; Gemini en configuración).

Notebook ejecutado: `notebooks/entrenamiento.ipynb`, con comparación, reportes, seis gráficas y análisis de errores. Presentación: `docs/presentacion_ecoscan.pdf`, con métricas y enlaces reales.

## Lista antes de Canvas

- [x] Notebook ejecutado con conteos, comparación, métricas y matrices visibles.
- [x] Modelo elegido mediante validación; no volver a ajustar tras observar prueba.
- [x] Analizar errores del modelo y redactar la conclusión usando cifras reales.
- [ ] GitHub público con los tres integrantes y declaración de IA.
- [ ] Space público responde desde un dispositivo externo.
- [ ] Botón Gemini probado con un secreto válido.
- [ ] Presentación actualizada y ensayo de 10 minutos.
- [ ] Capturas o video breve de respaldo sin claves visibles.
