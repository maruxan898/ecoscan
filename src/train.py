"""Comparación reproducible; selección por validación, prueba solo al terminar."""
import argparse
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from torch.utils.data import Dataset, DataLoader

from .config import CLASSES, ROOT, SEED
from .data import prepare_data


class WasteDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(ROOT / row.path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, CLASSES.index(row.label)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    for images, targets in loader:
        probabilities.extend(model(images.to(device)).softmax(1).cpu().numpy())
        labels.extend(targets.numpy())
    return np.asarray(labels), np.asarray(probabilities)


def metrics(y, probs):
    pred = probs.argmax(1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'report': classification_report(y, pred, labels=list(range(6)), target_names=CLASSES, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=list(range(6))).tolist()}


def run(epochs=12, batch_size=32, prepare=False):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if prepare or not (ROOT / 'datos/manifest.csv').exists():
        frame = prepare_data()
    else:
        frame = pd.read_csv(ROOT / 'datos/manifest.csv')
    hashes = [set(frame[frame.split == s].sha256) for s in ['train', 'val', 'test']]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    candidates, histories = [], {}
    for architecture in ['efficientnet_b0', 'resnet18']:
        # Reinicia la semilla para que cada ejecución tenga inicialización reproducible.
        torch.manual_seed(SEED)
        model = timm.create_model(architecture, pretrained=True, num_classes=6).to(device)
        config = timm.data.resolve_model_data_config(model)
        train_transform = timm.data.create_transform(**config, is_training=True, hflip=0.5, color_jitter=0.2)
        eval_transform = timm.data.create_transform(**config, is_training=False)
        loaders = {s: DataLoader(WasteDataset(frame[frame.split == s], train_transform if s == 'train' else eval_transform),
                    batch_size=batch_size, shuffle=s == 'train', num_workers=0, pin_memory=device == 'cuda')
                   for s in ['train', 'val', 'test']}
        counts = frame[frame.split == 'train'].label.value_counts().reindex(CLASSES).to_numpy()
        weights = torch.tensor(counts.sum() / (6 * counts), dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best, best_state, stale, history = -1, None, 0, []
        start = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            loss_sum = 0.0
            for images, targets in loaders['train']:
                images, targets = images.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(images), targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(targets)
            scheduler.step()
            y, probs = evaluate(model, loaders['val'], device)
            score = balanced_accuracy_score(y, probs.argmax(1))
            history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(loaders['train'].dataset), 'val_balanced_accuracy': float(score)})
            print(architecture, history[-1], flush=True)
            (ROOT / 'modelo/progreso.json').write_text(json.dumps({'architecture': architecture, **history[-1]}), encoding='utf-8')
            if score > best:
                best, stale = float(score), 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({'architecture': architecture, 'classes': CLASSES, 'state_dict': best_state, 'data_config': config}, ROOT / 'modelo' / f'{architecture}.pt')
            else:
                stale += 1
            if stale >= 4:
                break
        checkpoint = {'architecture': architecture, 'classes': CLASSES, 'state_dict': best_state, 'data_config': config}
        torch.save(checkpoint, ROOT / 'modelo' / f'{architecture}.pt')
        candidates.append({'architecture': architecture, 'val_balanced_accuracy': best, 'training_seconds': time.perf_counter() - start})
        histories[architecture] = history
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
    # Decisión cerrada antes de mirar el conjunto de prueba.
    selected = max(candidates, key=lambda c: c['val_balanced_accuracy'])['architecture']
    for candidate in candidates:
        checkpoint = torch.load(ROOT / 'modelo' / f"{candidate['architecture']}.pt", map_location='cpu', weights_only=True)
        model = timm.create_model(checkpoint['architecture'], pretrained=False, num_classes=6).to(device)
        model.load_state_dict(checkpoint['state_dict'])
        transform = timm.data.create_transform(**checkpoint['data_config'], is_training=False)
        loader = DataLoader(WasteDataset(frame[frame.split == 'test'], transform), batch_size=batch_size)
        y, probs = evaluate(model, loader, device)
        candidate['test'] = metrics(y, probs)
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(np.asarray(candidate['test']['confusion_matrix']), display_labels=CLASSES).plot(ax=ax, cmap='Greens', xticks_rotation=45)
        fig.tight_layout()
        fig.savefig(ROOT / 'modelo' / f"matriz_{candidate['architecture']}.png", dpi=150)
        plt.close(fig)
        if candidate['architecture'] == selected:
            torch.save(checkpoint, ROOT / 'modelo/modelo.pt')
            pd.DataFrame(probs, columns=CLASSES).assign(real=[CLASSES[i] for i in y], path=frame[frame.split == 'test'].path.to_list()).to_csv(ROOT / 'modelo/predicciones_test.csv', index=False)
    output = {'selected': selected, 'selection_rule': 'Máxima balanced accuracy en validación', 'seed': SEED,
              'classes': CLASSES, 'test_size': int((frame.split == 'test').sum()), 'candidates': candidates, 'history': histories,
              'torch_version': str(torch.__version__), 'timm_version': str(timm.__version__)}
    (ROOT / 'modelo/metrics.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    (ROOT / 'modelo/clases.json').write_text(json.dumps(CLASSES), encoding='utf-8')
    print(json.dumps({'selected': selected, 'results': candidates}, indent=2), flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs y batch-size deben ser positivos')
    run(args.epochs, args.batch_size, args.prepare)
