"""Ejecuta el notebook con el Python del entorno actual y conserva salidas reales."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
if '--report-only' in sys.argv:
    os.environ['ECOSCAN_REPORT_ONLY'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / '.jupyter')
os.environ['MPLBACKEND'] = 'Agg'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
path = ROOT / 'notebooks/entrenamiento.ipynb'
nb = nbformat.read(path, as_version=4)
install = nb.cells[2]
assert '%pip' in install.source
original = install.source
install.source = 'print("Dependencias instaladas previamente en el entorno aislado; se conserva la celda de instalación para Colab.")'
training = next(c for c in nb.cells if c.cell_type == 'code' and 'results = run_cached(' in c.source)
training_source = training.source
previous_outputs = list(training.outputs)
if os.environ.get('ECOSCAN_REPORT_ONLY') == '1':
    training.source = 'results = json.loads((WORK / "modelo/metrics.json").read_text(encoding="utf-8"))\nMODE = results["mode"]\nprint("Informe de la ejecución ya realizada; se reutilizan sus pesos y métricas sin volver a entrenar.")'
manager = KernelManager(kernel_name='python3')
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(nb, timeout=None, km=manager, resources={'metadata': {'path': str(ROOT)}}, record_timing=True)
try:
    client.execute()
finally:
    install.source = original
    if os.environ.get('ECOSCAN_REPORT_ONLY') == '1':
        training.source = training_source
        training.outputs = previous_outputs + training.outputs
    nbformat.write(nb, path)
    print('Notebook y salidas guardados:', path, flush=True)
