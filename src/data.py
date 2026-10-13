"""Descarga solo la versión reducida de la fuente que entregó el usuario."""
import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import CLASSES, ROOT, SEED


def prepare_data(root=ROOT):
    root = Path(root)
    destination = root / 'datos'
    destination.mkdir(parents=True, exist_ok=True)
    archive = hf_hub_download('garythung/trashnet', 'dataset-resized.zip', repo_type='dataset')
    # Verifica rutas antes de extraer un archivo externo.
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError('Ruta insegura en el archivo comprimido')
        z.extractall(destination)
    rows, seen, duplicates, conflicting = [], {}, 0, set()
    for path in sorted(destination.rglob('*')):
        if '__MACOSX' in path.parts or path.name.startswith('.'):
            continue  # Metadatos de macOS; no son imágenes aunque terminen en .jpg.
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'} or path.parent.name not in CLASSES:
            continue
        with Image.open(path) as im:
            im = im.convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        label = path.parent.name
        if digest in seen:
            if seen[digest] != label:
                conflicting.add(digest)
            duplicates += 1
            continue
        seen[digest] = label
        rows.append({'path': str(path.relative_to(root)), 'label': label, 'sha256': digest})
    frame = pd.DataFrame(rows)
    # No decide arbitrariamente una etiqueta cuando la fuente se contradice.
    excluded = frame[frame.sha256.isin(conflicting)]
    excluded.to_csv(root / 'modelo/conflictos_etiquetas.csv', index=False)
    frame = frame[~frame.sha256.isin(conflicting)].copy()
    audit = {'exact_duplicates': duplicates, 'conflicting_hashes_excluded': len(conflicting), 'unique_images_used': len(frame)}
    (root / 'modelo/auditoria_datos.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    if set(frame.label) != set(CLASSES):
        raise ValueError('El archivo no contiene las seis clases esperadas')
    train, remainder = train_test_split(frame, test_size=0.30, stratify=frame.label, random_state=SEED)
    val, test = train_test_split(remainder, test_size=0.50, stratify=remainder.label, random_state=SEED)
    parts = []
    for name, subset in [('train', train), ('val', val), ('test', test)]:
        parts.append(subset.assign(split=name))
    manifest = pd.concat(parts).sort_values('path').reset_index(drop=True)
    manifest.to_csv(destination / 'manifest.csv', index=False)
    counts = pd.crosstab(manifest.label, manifest.split).reindex(CLASSES)
    counts.to_csv(root / 'modelo' / 'conteos.csv')
    for label in CLASSES[:5]:
        # Ejemplos de entrenamiento: no se usan para demostrar desempeño en prueba.
        path = root / manifest[(manifest.label == label) & (manifest.split == 'train')].iloc[0].path
        with Image.open(path) as im:
            im.convert('RGB').save(root / 'ejemplos' / f'{label}.jpg')
    print(counts)
    print('Imágenes únicas:', len(manifest), '| duplicados exactos eliminados:', duplicates)
    print('Hashes con etiquetas contradictorias excluidos:', len(conflicting))
    return manifest
