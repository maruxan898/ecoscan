"""Interfaz Streamlit: despliegue gratuito en Community Cloud."""
import hashlib
import os
from io import BytesIO
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
from src.config import ROOT
from src.inference import predict, performance_text
from src.recommend import recommendation

st.set_page_config(page_title='EcoScan · Reciclaje informado', page_icon='♻️', layout='wide')
try:
    for name in ['GEMINI_API_KEY', 'GEMINI_MODEL']:
        if name in st.secrets:
            os.environ[name] = str(st.secrets[name])
except FileNotFoundError:
    pass

st.markdown('''<style>
.block-container {max-width:1150px; padding-top:2rem;}
.hero {background:linear-gradient(120deg,#103f36,#216652);padding:28px;border-radius:18px;color:white;margin-bottom:20px;}
.hero h1 {color:white; padding:0; font-size:2.6rem;}
.hero p {margin:10px 0 0; font-size:1.1rem;}
</style><div class="hero"><h1>♻️ EcoScan</h1><p>Una foto. Un material probable. Una decisión informada.</p></div>''', unsafe_allow_html=True)
st.caption('Prototipo educativo · Alexis, Gabriela y Agustín')
st.info('Fotografía un solo residuo con buena luz. Verifica el material y la aceptación del gestor local: la app no certifica reciclabilidad ni detecta peligros.')
analyze, performance, about = st.tabs(['🔎 Analizar un residuo', '📊 Desempeño', '🌱 Proyecto'])
with analyze:
    left, right = st.columns([1,1], gap='large')
    with left:
        upload = st.file_uploader('Sube la foto de un residuo', type=['jpg','jpeg','png','webp'])
        examples = sorted((ROOT/'ejemplos').glob('*.jpg'))
        example = st.selectbox('O usa un ejemplo de entrenamiento', ['Ninguno']+[p.stem for p in examples])
        raw = upload.getvalue() if upload else next((p.read_bytes() for p in examples if p.stem==example), None)
        current_id = hashlib.sha256(raw).hexdigest() if raw else None
        if st.session_state.get('image_id') != current_id:
            st.session_state.update(image_id=current_id, result=None, probabilities=None, summary=None, advice=None)
        image = None
        if raw:
            try:
                image = Image.open(BytesIO(raw))
                if image.width*image.height > 25_000_000:
                    raise ValueError('Imagen demasiado grande')
                st.image(image, caption='Imagen actual', width='stretch')
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                st.error('La imagen no es válida o excede 25 megapíxeles.')
                image = None
        if st.button('Identificar material', type='primary', disabled=image is None, width='stretch'):
            try:
                with st.spinner('Analizando el material…'):
                    probs, summary, result = predict(image)
                st.session_state.update(probabilities=probs, summary=summary, result=result, advice=None)
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
    with right:
        if st.session_state.get('result'):
            st.markdown(st.session_state.summary)
            data = pd.DataFrame({'Material':st.session_state.probabilities.keys(), 'Probabilidad':st.session_state.probabilities.values()}).set_index('Material')
            st.bar_chart(data, color='#1b7d61', horizontal=True)
            st.dataframe(data.style.format({'Probabilidad':'{:.1%}'}), width='stretch')
        else:
            st.markdown('### Reconoce el material')
            st.write('Sube una foto o elige un ejemplo y pulsa **Identificar material** para ver las seis probabilidades.')
        context = st.text_area('Contexto opcional', placeholder='Por ejemplo: envase con restos de comida', max_chars=500)
        if st.button('Generar recomendación con IA', disabled=not st.session_state.get('result'), width='stretch'):
            with st.spinner('Generando orientación con Gemini…'):
                st.session_state.advice = recommendation(st.session_state.result, context)
        if st.session_state.get('advice'):
            st.markdown(st.session_state.advice)
with performance:
    st.markdown(performance_text())
    st.warning('La confianza de una foto es distinta de la accuracy de prueba. Softmax no está calibrado y puede dar confianza alta en imágenes ajenas al dataset.')
    for p in sorted((ROOT/'modelo').glob('matriz_*.png')):
        st.image(str(p), caption=p.stem.replace('_',' '))
    if (ROOT/'modelo/conteos.csv').exists():
        st.subheader('Distribución de los datos utilizados')
        st.dataframe(pd.read_csv(ROOT/'modelo/conteos.csv').set_index('label'), width='stretch')
with about:
    st.markdown('''### Del reconocimiento a la orientación
TrashNet contiene cartón, vidrio, metal, papel, plástico y otros residuos.
La colección se fotografió principalmente sobre fondo blanco. Fotos con varios
objetos, suciedad o fondos complejos pueden reducir el desempeño.

**Modelo:** comparación EfficientNet-B0 y ResNet18 con la misma partición.
Selección mediante validación; prueba reservada para evaluar al final.

**Error costoso:** enviar residuos no aceptados a una carga puede contaminarla.
Material probable no equivale a reciclabilidad.

**Privacidad:** la foto se procesa en el servidor. Gemini recibe clase,
confianza y contexto escrito. No subas fotos de personas ni datos sensibles.

**Uso de IA:** ChatGPT/Codex apoyó programación y documentación; Gemini genera
recomendaciones. El grupo debe revisar y explicar todo el código.

[Dataset](https://huggingface.co/datasets/garythung/trashnet) ·
[Origen y licencia](https://github.com/garythung/trashnet) ·
[Repositorio del grupo](https://github.com/maruxan898/ecoscan)
''')
