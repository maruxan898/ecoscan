"""Presentación PDF con resultados dinámicos, sin métricas inventadas."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import simpleSplit

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/presentacion_ecoscan.pdf'
GREEN, BG, INK = '#1b7d61', '#f4f8f6', '#173d33'


def build():
    links_path = ROOT / 'docs/enlaces.json'
    links = json.loads(links_path.read_text(encoding='utf-8')) if links_path.exists() else {}
    final_links = ['Repositorio: github.com/maruxan898/ecoscan']
    final_links += ['App: '+links['app'].removeprefix('https://')] if links.get('app') else ['App: pendiente de publicación.']
    metrics = ROOT / 'modelo/metrics.json'
    if metrics.exists():
        data = json.loads(metrics.read_text(encoding='utf-8'))
        result = [f"{c['architecture']}: accuracy {c['test']['accuracy']:.1%} | balanceada {c['test']['balanced_accuracy']:.1%} | F1 macro {c['test']['macro_f1']:.3f}" for c in data['candidates']]
        result += [f"Elegido por validación: {data['selected']}", f"Conjunto de prueba: {data['test_size']} imágenes."]
    else:
        result = ['Evaluación pendiente de entrenamiento.', 'Comparar accuracy, accuracy balanceada y F1 macro en las mismas fotos de prueba.', 'Actualizar esta diapositiva con metrics.json; no presentar cifras supuestas.']
    slides = [
        ('EcoScan', 'De la imagen a una decisión informada', ['Alexis · Gabriela · Agustín', 'IA Generativa para la Gestión Industrial', 'Clasificación de residuos + recomendaciones con Gemini'], '0:00 - 0:30'),
        ('01 / Problema', 'Separar mejor empieza por reconocer el material', ['Una persona sube una fotografía de un solo residuo.', 'EcoScan propone una clase, muestra probabilidades y orienta la separación.', 'La aceptación para reciclaje depende del estado del residuo y del gestor local.'], '0:30 - 1:30'),
        ('02 / Datos', 'TrashNet: seis clases y una fuente pública', ['Gary Thung y Mindy Yang · Stanford CS229 · licencia MIT.', '2.527 fotos en la colección original: cartón 403, vidrio 501, metal 410, papel 594, plástico 482 y otros 137.', 'Se usa dataset-resized.zip de Hugging Face: una sola versión de cada foto.', 'Sesgos: fondo blanco, fuente única, pocos otros residuos y falta de IDs de objeto.'], '1:30 - 2:30'),
        ('03 / Diseño experimental', 'Comparación con prueba reservada', ['Duplicados exactos eliminados antes de dividir.', '70% train / 15% val / 15% test · estratificación · semilla 42.', 'Train ajusta pesos; val elige modelo; test reporta desempeño.', 'Misma partición y mismas métricas para ambos candidatos.'], '2:30 - 3:00'),
        ('04 / Modelos', 'Transfer learning con timm', ['EfficientNet-B0: candidato compacto, alineado con la pauta docente.', 'ResNet18: comparación con otra arquitectura preentrenada.', 'Encoder congelado + cabeza entrenable: características en caché para CPU.', 'AdamW · pérdida ponderada · dos vistas de train · parada temprana.'], '3:00 - 4:00'),
        ('05 / Resultados', 'Desempeño medido en prueba', result, '4:00 - 5:00'),
        ('06 / Demo', 'Imagen / modelo / recomendación', ['1. Subir una foto de un solo residuo y clasificar.', '2. Leer la clase, la confianza y las seis probabilidades.', '3. Generar orientación con Gemini y revisar las advertencias.', '4. Mostrar las métricas de prueba; cambiar la imagen y reclasificar.'], '5:00 - 8:00'),
        ('07 / Código e IA', 'Cinco módulos que se pueden explicar', ['data.py: descarga, duplicados y particiones.', 'train.py: ajuste, validación, comparación y exportación.', 'inference.py + app.py: mismo preprocesamiento, softmax e interfaz.', 'recommend.py: solo texto a Gemini, clave en Secrets y prompt responsable.'], '8:00 - 9:00'),
        ('08 / Límites y cierre', 'Una herramienta educativa que requiere revisión', ['Confianza no es accuracy ni una garantía de reciclabilidad.', 'Error costoso: enviar residuos no aceptados a un flujo reciclable.', 'Mejoras: fotos reales, IDs de objetos, evaluación externa y calibración.', 'IA usada: ChatGPT/Codex para código y documentación; Gemini en la app.'], '9:00 - 10:00'),
        ('Fuentes y respaldo', 'Referencias para la evaluación', ['Dataset: huggingface.co/datasets/garythung/trashnet', 'Origen y licencia: github.com/garythung/trashnet', 'Modelo: huggingface.co/docs/timm/quickstart', 'Gemini: ai.google.dev/gemini-api/docs/text-generation']+final_links, 'Apoyo / preguntas'),
    ]
    c = canvas.Canvas(str(OUTPUT), pagesize=(960,540))
    c.setTitle('EcoScan · Proyecto de clasificación de residuos')
    c.setAuthor('Alexis, Gabriela y Agustín')
    for i, (eyebrow, title, lines, timing) in enumerate(slides):
        c.setFillColor(HexColor(BG)); c.rect(0,0,960,540,fill=1,stroke=0)
        c.setFillColor(HexColor(GREEN)); c.rect(0,0,16,540,fill=1,stroke=0)
        c.setFont('Helvetica-Bold',14); c.drawString(55,487,eyebrow.upper())
        c.setFillColor(HexColor(INK)); c.setFont('Helvetica-Bold',30)
        title_lines = simpleSplit(title,'Helvetica-Bold',30,850)
        y = 430
        for line in title_lines:
            c.drawString(55,y,line); y -= 38
        y -= 24
        for paragraph in lines:
            wrap = simpleSplit(paragraph,'Helvetica',20,805)
            c.setFillColor(HexColor(GREEN)); c.circle(61,y+6,3,fill=1,stroke=0)
            c.setFillColor(HexColor(INK)); c.setFont('Helvetica',20)
            for line in wrap:
                c.drawString(79,y,line); y -= 27
            y -= 16
        if y < 55:
            raise ValueError(f'Contenido excede la diapositiva {i+1}')
        c.setFillColor(HexColor(GREEN)); c.setFont('Helvetica',11)
        c.drawString(55,25,'EcoScan · Uso educativo · '+timing)
        c.drawRightString(915,25,f'{i+1:02d} / {len(slides):02d}')
        c.showPage()
    c.save()
    print(OUTPUT)


if __name__ == '__main__':
    build()
