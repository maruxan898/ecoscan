Esta carpeta recibe los artefactos del entrenamiento real. No contiene pesos ficticios.

`python -m src.train --prepare` genera modelo.pt, clases.json, metrics.json, conteos.csv, las matrices de confusión y predicciones_test.csv.

Para despliegue solo hace falta el checkpoint ganador modelo.pt, metrics.json, conteos.csv, clases.json y las matrices. Los checkpoints de los otros candidatos se pueden conservar como evidencia local; cada uno debe respetar el límite de 100 MB de GitHub.
