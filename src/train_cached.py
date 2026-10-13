"""Transfer learning eficiente en CPU: encoder congelado y características en caché."""
import json
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score, ConfusionMatrixDisplay
from .config import ROOT, CLASSES, SEED
from .train import WasteDataset, evaluate, metrics


@torch.inference_mode()
def features(model, loader, stage):
    model.eval()
    xs, ys = [], []
    for batch, (images, labels) in enumerate(loader):
        tensor = model.forward_head(model.forward_features(images), pre_logits=True)
        xs.append(tensor.cpu().clone())
        ys.append(labels.clone())
        if batch % 10 == 0:
            (ROOT/'modelo/progreso.json').write_text(json.dumps({'stage':stage, 'batch':batch+1, 'batches':len(loader)}), encoding='utf-8')
    return torch.cat(xs), torch.cat(ys)


def run_cached(epochs=80, batch_size=32):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    frame = pd.read_csv(ROOT/'datos/manifest.csv')
    candidates, histories = [], {}
    for name in ['efficientnet_b0','resnet18']:
        torch.manual_seed(SEED)
        started = time.perf_counter()
        model = timm.create_model(name, pretrained=True, num_classes=6).eval()
        for p in model.parameters(): p.requires_grad_(False)
        head = model.get_classifier()
        for p in head.parameters(): p.requires_grad_(True)
        config = timm.data.resolve_model_data_config(model)
        prep = timm.data.create_transform(**config,is_training=False)
        aug = timm.data.create_transform(**config,is_training=True,hflip=.5,color_jitter=.2)
        loaders = {s:DataLoader(WasteDataset(frame[frame.split==s],prep),batch_size=batch_size) for s in ['train','val','test']}
        # Dos vistas por imagen de train: original y una aumentada; val es determinista.
        x_base,y_base = features(model,loaders['train'],name+' train original')
        augmented = DataLoader(WasteDataset(frame[frame.split=='train'],aug),batch_size=batch_size)
        x_aug,y_aug = features(model,augmented,name+' train aumentado')
        x_train,y_train = torch.cat([x_base,x_aug]),torch.cat([y_base,y_aug])
        x_val,y_val = features(model,loaders['val'],name+' validación')
        # Inference mode crea tensores sin autograd; clone fuera del contexto los normaliza.
        x_train,y_train,x_val,y_val = [x.clone() for x in [x_train,y_train,x_val,y_val]]
        counts = frame[frame.split=='train'].label.value_counts().reindex(CLASSES).to_numpy()
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(counts.sum()/(6*counts),dtype=torch.float32))
        optimizer = torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
        cached_loader = DataLoader(TensorDataset(x_train,y_train),batch_size=256,shuffle=True)
        best, stale, history, best_head = -1,0,[],None
        for epoch in range(epochs):
            head.train(); total = 0
            for x,y in cached_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(head(x),y); loss.backward(); optimizer.step()
                total += loss.item()*len(y)
            head.eval()
            with torch.no_grad(): probs = head(x_val).softmax(1).numpy()
            score = float(balanced_accuracy_score(y_val.numpy(),probs.argmax(1)))
            history.append({'epoch':epoch+1,'train_loss':total/len(y_train),'val_balanced_accuracy':score})
            (ROOT/'modelo/progreso.json').write_text(json.dumps({'architecture':name, 'mode':'cached_linear_probe',**history[-1]}),encoding='utf-8')
            if score>best:
                best, stale = score,0
                best_head = {k:v.detach().clone() for k,v in head.state_dict().items()}
            else: stale += 1
            if stale>=15: break
        head.load_state_dict(best_head)
        checkpoint = {'architecture':name,'classes':CLASSES,'data_config':config,'state_dict':model.state_dict()}
        torch.save(checkpoint,ROOT/'modelo'/f'{name}.pt')
        candidates.append({'architecture':name,'val_balanced_accuracy':best,'training_seconds':time.perf_counter()-started})
        histories[name] = history
        print(name,'validación',best,'épocas',len(history),flush=True)
        del model,head,x_train,x_val,x_base,x_aug
    selected = max(candidates,key=lambda c:c['val_balanced_accuracy'])['architecture']
    # Prueba solo después de cerrar la elección por validación.
    for c in candidates:
        checkpoint = torch.load(ROOT/'modelo'/f"{c['architecture']}.pt",map_location='cpu',weights_only=True)
        model = timm.create_model(c['architecture'],pretrained=False,num_classes=6)
        model.load_state_dict(checkpoint['state_dict'])
        prep = timm.data.create_transform(**checkpoint['data_config'],is_training=False)
        loader = DataLoader(WasteDataset(frame[frame.split=='test'],prep),batch_size=batch_size)
        y,probs = evaluate(model,loader,'cpu'); c['test'] = metrics(y,probs)
        fig,ax=plt.subplots(figsize=(8,6))
        ConfusionMatrixDisplay(np.asarray(c['test']['confusion_matrix']),display_labels=CLASSES).plot(ax=ax,cmap='Greens',xticks_rotation=45)
        fig.tight_layout(); fig.savefig(ROOT/'modelo'/f"matriz_{c['architecture']}.png",dpi=150); plt.close(fig)
        if c['architecture']==selected:
            torch.save(checkpoint,ROOT/'modelo/modelo.pt')
            pd.DataFrame(probs,columns=CLASSES).assign(real=[CLASSES[i] for i in y],path=frame[frame.split=='test'].path.to_list()).to_csv(ROOT/'modelo/predicciones_test.csv',index=False)
    output = {'selected':selected,'selection_rule':'Máxima balanced accuracy en validación','mode':'cached_linear_probe',
              'protocol':'Encoder ImageNet congelado; dos vistas train; AdamW lr=1e-3; hasta 80 épocas de la cabeza; paciencia 15',
              'classes':CLASSES,'seed':SEED,'test_size':int((frame.split=='test').sum()),'candidates':candidates,'history':histories,
              'torch_version':str(torch.__version__),'timm_version':str(timm.__version__)}
    (ROOT/'modelo/metrics.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'modelo/clases.json').write_text(json.dumps(CLASSES),encoding='utf-8')
    return output
