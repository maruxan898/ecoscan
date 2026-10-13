from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
SPANISH = dict(zip(CLASSES, ['Cartón', 'Vidrio', 'Metal', 'Papel', 'Plástico', 'Otros residuos']))
SEED = 42
LOW_CONFIDENCE = 0.70  # Umbral educativo; no demuestra detección de imágenes fuera del dominio.
