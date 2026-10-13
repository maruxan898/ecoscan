"""EcoScan: clasificación de residuos con un modelo entrenado y recomendaciones Gemini."""
import gradio as gr
from src.config import ROOT
from src.inference import predict, performance_text
from src.recommend import recommendation


def classify(image):
    try:
        return (*predict(image), '')
    except (ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from None


def advise(result, context):
    try:
        return recommendation(result, context)
    except ValueError as exc:
        raise gr.Error(str(exc)) from None


CSS = '''.gradio-container {max-width: 1120px !important;}
#hero {background: linear-gradient(120deg,#103f36,#216652); border-radius:20px; padding:26px;}
#hero h1, #hero p {color:white !important;}'''

with gr.Blocks(theme=gr.themes.Soft(primary_hue='emerald', secondary_hue='teal'), css=CSS, title='EcoScan · Reciclaje informado') as demo:
    gr.Markdown('# ♻️ EcoScan\n### Una foto. Un material probable. Una decisión informada.\nClasifica un residuo y recibe orientación para separarlo con cuidado.', elem_id='hero')
    gr.Markdown('**Prototipo educativo.** Fotografía un solo objeto, con buena luz. No detecta peligros ni certifica reciclabilidad. Incluso una confianza alta puede ser incorrecta con objetos ajenos al dataset.')
    state = gr.State(None)
    with gr.Tab('Analizar un residuo'):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', sources=['upload', 'webcam'], label='Foto del residuo')
                button = gr.Button('Identificar material', variant='primary')
                samples = [[str(p)] for p in sorted((ROOT / 'ejemplos').glob('*.jpg'))]
                if samples:
                    gr.Examples(samples, inputs=image, label='Ejemplos del conjunto de entrenamiento')
            with gr.Column():
                summary = gr.Markdown('Sube una foto para comenzar.')
                probabilities = gr.Label(num_top_classes=6, label='Probabilidades de las seis clases')
                context = gr.Textbox(label='Contexto opcional', placeholder='Por ejemplo: envase con restos de comida', max_lines=3)
                recommend = gr.Button('Generar recomendación con IA')
                advice = gr.Markdown()
        button.click(classify, inputs=image, outputs=[probabilities, summary, state, advice])
        # Evita recomendar usando el resultado de una fotografía anterior.
        image.change(lambda: (None, {}, 'Foto actualizada: vuelve a clasificar.', ''), outputs=[state, probabilities, summary, advice])
        recommend.click(advise, inputs=[state, context], outputs=advice, concurrency_limit=1)
    with gr.Tab('Desempeño y límites'):
        gr.Markdown(performance_text())
        gr.Markdown('### Qué reconoce\nCartón, vidrio, metal, papel, plástico y otros residuos.\n\n### Alcance\nTrashNet contiene objetos fotografiados principalmente sobre un fondo blanco. Fotos con varios residuos, bolsas cerradas, suciedad o fondos complejos pueden reducir el desempeño. No hay identificadores de objeto para descartar todos los parecidos entre particiones.\n\n### Error costoso\nClasificar un residuo no aceptado como reciclable puede contaminar una carga. El material visual no basta para conocer composición, limpieza o aceptación del gestor.\n\n### Privacidad\nNo subas fotos con personas o información sensible. La imagen se procesa en el servidor; Gemini recibe solo la clase, confianza y el contexto que escribas.')
        for p in sorted((ROOT / 'modelo').glob('matriz_*.png')):
            gr.Image(str(p), label=p.stem.replace('_', ' '), interactive=False)

if __name__ == '__main__':
    demo.queue(max_size=16).launch()
