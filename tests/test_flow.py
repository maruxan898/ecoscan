"""Comprobaciones del contrato entre modelo, interfaz y recomendación."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.config import CLASSES
from src.recommend import recommendation


class FlowTests(unittest.TestCase):
    def test_requires_current_prediction(self):
        with self.assertRaises(ValueError):
            recommendation(None)

    def test_missing_key_is_not_presented_as_ai(self):
        with patch.dict(os.environ, {}, clear=True):
            text = recommendation({'class':'plastic','confidence':0.9})
            self.assertIn('No se ha generado', text)

    def test_probabilities_from_exported_checkpoint(self):
        from src.config import ROOT
        if not (ROOT/'modelo/modelo.pt').exists():
            self.skipTest('Requiere entrenamiento real')
        from PIL import Image
        from src.inference import predict
        with Image.open(next((ROOT/'ejemplos').glob('*.jpg'))) as im:
            probabilities, _, state = predict(im)
        self.assertEqual(len(probabilities), 6)
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=5)
        self.assertIn(state['class'], CLASSES)

    def test_llm_sends_text_and_disables_storage(self):
        client = SimpleNamespace(interactions=SimpleNamespace(create=lambda **kw: None), close=lambda: None)
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(outputs=[SimpleNamespace(type='text',text='Solicita revisión humana.')])
        client.interactions.create = generate
        with patch.dict(os.environ, {'GEMINI_API_KEY':'test-placeholder'}), patch('google.genai.Client',return_value=client):
            answer = recommendation({'class':'glass','confidence':0.4}, 'x'*1000)
        self.assertEqual(answer,'Solicita revisión humana.')
        self.assertFalse(captured['store'])
        self.assertIsInstance(captured['input'],str)
        self.assertIn('True',captured['input'])
        self.assertNotIn('x'*501,captured['input'])


if __name__ == '__main__':
    unittest.main()
