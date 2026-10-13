"""Regenera el notebook autocontenido después de modificar el código fuente."""
import json
from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# ♻️ EcoScan · Reconocimiento de residuos e IA generativa
**Alexis, Gabriela y Agustín** · IA Generativa para la Gestión Industrial

Flujo: foto → modelo preentrenado adaptado → probabilidades → Gemini → orientación.

Este notebook es **autocontenido**: escribe el código necesario en `/content/ecoscan`, sin subir archivos adicionales. En Colab activar **GPU T4** y ejecutar todas las celdas. Descargar también esta versión **con sus salidas** al terminar.

Sigue la pauta docente de EfficientNet/timm, ampliada con una prueba independiente, pérdida ponderada y exportación coherente con la app. Los resultados se calculan; no hay métricas prefabricadas.''')
md('## 1. Instalar las librerías\nPyTorch ya viene en Colab; pip resuelve las dependencias compatibles. Para repetir exactamente una ejecución conservar también el archivo de versiones exportado.')
code('%pip install -q "gradio>=5.20,<6" "timm>=1.0.15,<2" "huggingface-hub>=0.28,<1" "google-genai>=1,<2" "scikit-learn>=1.5,<2" pandas matplotlib Pillow')
md('## 2. Código del proyecto\nCada módulo tiene una responsabilidad. Esta celda escribe los mismos archivos que usa la aplicación. No contiene secretos. Si modificas el proyecto, ejecutar `build_notebook.py` para actualizar esta copia.')
paths = [root / 'app.py', root / 'streamlit_app.py', root / '.streamlit/config.toml', root / 'requirements.txt', root / 'README.md', root / '.gitignore'] + sorted((root/'src').glob('*.py')) + sorted((root/'docs').glob('*.md'))
payload = {str(p.relative_to(root)).replace('\\','/'):p.read_text(encoding='utf-8') for p in paths}
code('from pathlib import Path\nimport os, sys, json\nWORK = Path("/content/ecoscan") if Path("/content").exists() else Path.cwd()\nWORK.mkdir(parents=True, exist_ok=True)\nFILES = '+repr(payload)+'\nfor name, contents in FILES.items():\n    path = WORK / name\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(contents, encoding="utf-8")\nfor folder in ["modelo", "ejemplos", "datos"]:\n    (WORK / folder).mkdir(exist_ok=True)\nos.chdir(WORK)\nsys.path.insert(0, str(WORK))\nprint("Proyecto:", WORK)')
md('''## 3. Datos: origen, licencia, clases y limitaciones
Gary Thung y Mindy Yang crearon TrashNet para Stanford CS229. **MIT**. Fuente del trabajo: https://huggingface.co/datasets/garythung/trashnet ; descripción original: https://github.com/garythung/trashnet .

Original: 2.527 fotos (cartón 403, vidrio 501, metal 410, papel 594, plástico 482, otros 137). El visor muestra 5.054 registros; descargamos **solo dataset-resized.zip**, no ambas versiones. Se eliminan duplicados exactos antes de dividir. Las variantes del mismo objeto pueden seguir siendo similares: faltan IDs de objeto.

Fotos sobre fondo blanco; no representan una planta de reciclaje. `trash` es minoritaria y significa **otros residuos**, no orgánicos. Guardamos los conteos reales y estratificamos 70/15/15 con semilla 42.''')
code('from src.data import prepare_data\nfrom src.config import CLASSES, SPANISH, SEED\nimport pandas as pd\nimport matplotlib.pyplot as plt\nmanifest = prepare_data()\ncounts = pd.crosstab(manifest.label, manifest.split).reindex(CLASSES)\ndisplay(counts)\ncounts.plot.bar(stacked=True, figsize=(9,4), color=["#176b52", "#62b99a", "#c5e4d7"])\nplt.ylabel("Imágenes")\nplt.title("Distribución real por clase y partición")\nplt.tight_layout()\nplt.show()\ngroups = {s:set(manifest[manifest.split==s].sha256) for s in ["train","val","test"]}\nassert not (groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"])\nprint("Sin duplicados exactos compartidos entre particiones")')
code('from PIL import Image\nfig, axes = plt.subplots(2,3,figsize=(10,6))\nfor ax, label in zip(axes.flat, CLASSES):\n    row = manifest[(manifest.label==label)&(manifest.split=="train")].iloc[0]\n    with Image.open(WORK / row.path) as im:\n        ax.imshow(im.convert("RGB"))\n    ax.set_title(SPANISH[label]); ax.axis("off")\nplt.tight_layout(); plt.show()')
md('''## 4. Entrenar y comparar dos enfoques
**EfficientNet-B0**: modelo de la clase, buena relación tamaño/rendimiento. **ResNet18**: otra CNN preentrenada que sirve como comparación.

Ambos usan los mismos conjuntos, AdamW (lr=1e-4), 12 épocas como máximo, early stopping (paciencia 4) y entropía cruzada ponderada por los conteos de **train**. Aumentos solo en train. La normalización se obtiene de la configuración del modelo.

Seleccionamos checkpoint y arquitectura por **accuracy balanceada de val**. Cerrada esa elección, reportamos prueba para ambos. No elegir por test ni continuar afinando después de observarlo. GPU recomendada; CPU tarda más.''')
code('import torch\nprint("Dispositivo:", "GPU" if torch.cuda.is_available() else "CPU")\nMODE = "cached_linear_probe"  # Comparación eficiente y reproducible; cambiar a fine_tune para ajuste completo con GPU\nif MODE == "cached_linear_probe":\n    from src.train_cached import run_cached\n    results = run_cached(epochs=80, batch_size=32)\nelse:\n    from src.train import run\n    results = run(epochs=12, batch_size=32)\nprint("Modalidad ejecutada:", MODE)')
md('## 5. Resultados de prueba\nLa accuracy balanceada promedia el recall de las seis clases. F1 macro da a cada clase el mismo peso. No confundir estas métricas agregadas con la confianza de una foto.')
code('comparison = pd.DataFrame([{ "modelo":c["architecture"], "balanced_accuracy_val":c["val_balanced_accuracy"], "accuracy_test":c["test"]["accuracy"], "balanced_accuracy_test":c["test"]["balanced_accuracy"], "f1_macro_test":c["test"]["macro_f1"], "minutos_entrenamiento":c["training_seconds"]/60} for c in results["candidates"]])\ndisplay(comparison.round(4))\nprint("Elegido por validación:", results["selected"])\nfor c in results["candidates"]:\n    print("\\nReporte:", c["architecture"])\n    display(pd.DataFrame(c["test"]["report"]).T)\n    plt.figure(figsize=(9,6))\n    plt.imshow(plt.imread(WORK / "modelo" / f"matriz_{c[\'architecture\']}.png"))\n    plt.axis("off"); plt.show()')
md('## 6. Curvas e interpretación de errores\nLa pérdida debe disminuir; si la validación deja de mejorar, puede haber sobreajuste. Error relevante: otros residuos enviados erróneamente a categorías potencialmente reciclables. El dataset no etiqueta peligros ni aceptación del municipio.')
code('fig, axes = plt.subplots(1,2,figsize=(12,4))\nfor name, history in results["history"].items():\n    h = pd.DataFrame(history)\n    axes[0].plot(h.epoch,h.train_loss,label=name)\n    axes[1].plot(h.epoch,h.val_balanced_accuracy,label=name)\naxes[0].set_title("Pérdida de entrenamiento"); axes[1].set_title("Accuracy balanceada de validación")\nfor ax in axes:\n    ax.set_xlabel("Época"); ax.legend(); ax.grid(alpha=.2)\nplt.tight_layout(); plt.show()\nselected = next(c for c in results["candidates"] if c["architecture"]==results["selected"])\ncm = selected["test"]["confusion_matrix"]\nidx = CLASSES.index("trash")\nprint("Otros residuos confundidos con materiales:",sum(cm[idx])-cm[idx][idx],"de",sum(cm[idx]))\npreds = pd.read_csv(WORK / "modelo/predicciones_test.csv")\npreds["predicha"] = preds[CLASSES].idxmax(axis=1)\nerrors = preds[preds.real != preds.predicha]\ndisplay(errors[["path","real","predicha"]].head(10))\nif len(errors):\n    fig, axes = plt.subplots(1,min(4,len(errors)),figsize=(12,4),squeeze=False)\n    for ax, (_, row) in zip(axes.flat,errors.head(4).iterrows()):\n        with Image.open(WORK / row.path) as im: ax.imshow(im.convert("RGB"))\n        ax.set_title(f"Real: {row.real}\\nPredicción: {row.predicha}"); ax.axis("off")\n    plt.tight_layout(); plt.show()')
md('''## 7. Conclusiones para la exposición
Completar después de ejecutar: ¿qué modelo ganó por validación? ¿Coincide la comparación en prueba? ¿Qué clase tiene menor recall? ¿Qué error observable podría contaminar un flujo de reciclaje? ¿Qué limitaciones impiden extrapolar estas cifras a fotos de una planta real?

No se demuestra que una confianza de 70% implique 70% de aciertos. Antes de uso real se necesita validación externa, calibración y conocer las reglas del gestor local.''')
md('## 8. Exportación y prueba de inferencia\nEl checkpoint lleva pesos, clases y transformación. Se verifica que la app pueda cargarlo y producir seis probabilidades que sumen uno.')
code('from src.inference import predict\nrow = manifest[manifest.split=="test"].iloc[0]\nwith Image.open(WORK / row.path) as im:\n    probabilities, summary, state = predict(im)\nprint(summary)\nprint(probabilities)\nassert len(probabilities)==6 and abs(sum(probabilities.values())-1)<1e-5\nprint("Exportación e inferencia verificadas")')
md('## 9. Probar la app y Gemini (opcional en Colab)\nPara Gemini crear el secreto `GEMINI_API_KEY` en el panel de secretos de Colab y permitir acceso al notebook. No escribirlo en una celda. El despliegue permanente se hace en un Space; `share=True` es solo una demo temporal.')
code('START_DEMO = False  # Cambiar a True para la demo temporal\nif START_DEMO:\n    from google.colab import userdata\n    try:\n        os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")\n    except Exception:\n        print("Sin secreto Gemini; clasificación disponible, recomendación pendiente")\n    from app import demo\n    demo.queue(max_size=16).launch(share=True)')
md('## 10. Descargar la entrega\nEl ZIP excluye datos completos, entornos y secretos. Conservar los checkpoints candidatos localmente; al desplegar basta el ganador. Descargar también este notebook ejecutado desde el menú Archivo.')
code('import zipfile, subprocess\nversions = subprocess.check_output([sys.executable,"-m","pip","freeze"], text=True)\n(WORK / "modelo/versiones_entrenamiento.txt").write_text(versions, encoding="utf-8")\narchive = WORK.parent / "ecoscan_entrega.zip"\nwith zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:\n    for rel in ["app.py","requirements.txt","README.md",".gitignore"]:\n        z.write(WORK/rel,rel)\n    for folder in ["src","modelo","ejemplos","docs"]:\n        for p in (WORK/folder).rglob("*"):\n            if p.is_file() and "__pycache__" not in p.parts:\n                z.write(p,p.relative_to(WORK))\nprint("Entrega:",archive)\nif "google.colab" in sys.modules:\n    from google.colab import files\n    files.download(str(archive))')
nb['cells'] = cells
nb['metadata'] = {'kernelspec': {'display_name':'Python 3','language':'python','name':'python3'}, 'language_info': {'name':'python','version':'3.11'}, 'colab': {'name':'EcoScan_entrenamiento.ipynb'}, 'accelerator':'GPU'}
nbf.validate(nb)
nbf.write(nb,root/'notebooks/entrenamiento.ipynb')
print('Notebook generado y validado')
