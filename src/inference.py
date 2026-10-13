from functools import lru_cache
import json
import timm
import torch
from PIL import ImageOps

from .config import CLASSES, ROOT, SPANISH, LOW_CONFIDENCE


@lru_cache(maxsize=1)
def load_model():
    path = ROOT / 'modelo/modelo.pt'
    if not path.exists():
        raise FileNotFoundError('Falta el modelo entrenado. Ejecuta el notebook de Colab y copia la carpeta modelo.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['classes'] != CLASSES:
        raise ValueError('El orden de clases del modelo no coincide con la app')
    model = timm.create_model(checkpoint['architecture'], pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    transform = timm.data.create_transform(**checkpoint['data_config'], is_training=False)
    return model, transform


@torch.inference_mode()
def predict(image):
    if image is None:
        raise ValueError('Sube primero una foto de un residuo')
    if image.width * image.height > 25_000_000:
        raise ValueError('La imagen es demasiado grande. Usa una foto de hasta 25 megapíxeles')
    model, transform = load_model()
    image = ImageOps.exif_transpose(image).convert('RGB')
    probabilities = model(transform(image).unsqueeze(0)).softmax(1)[0].tolist()
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    result = {'class': CLASSES[index], 'confidence': probabilities[index],
              'probabilities': dict(zip(CLASSES, probabilities))}
    warning = 'Confianza baja: toma otra foto y solicita revisión humana.' if result['confidence'] < LOW_CONFIDENCE else 'Verifica el material antes de decidir su destino.'
    summary = f"### {SPANISH[result['class']]} · {result['confidence']:.1%}\n{warning}\n\nLa confianza del modelo no garantiza que el objeto sea reciclable."
    return {SPANISH[k]: v for k, v in result['probabilities'].items()}, summary, result


def performance_text():
    path = ROOT / 'modelo/metrics.json'
    if not path.exists():
        return 'Evaluación pendiente: todavía no hay métricas verificadas de prueba.'
    data = json.loads(path.read_text(encoding='utf-8'))
    selected = next(c for c in data['candidates'] if c['architecture'] == data['selected'])
    m = selected['test']
    return (f"**Modelo:** {data['selected']} | **Prueba:** {data['test_size']} imágenes\n\n"
            f"Accuracy: **{m['accuracy']:.1%}** · Accuracy balanceada: **{m['balanced_accuracy']:.1%}** · F1 macro: **{m['macro_f1']:.3f}**\n\n"
            'Estas métricas describen el conjunto de prueba; la confianza corresponde a una foto particular.')
