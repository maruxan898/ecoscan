# Resultados reales · EcoScan

Modalidad ejecutada: **extracción de características con encoder ImageNet congelado**. Se entrenó la cabeza de cada modelo usando dos vistas de train y se eligió el checkpoint por accuracy balanceada de validación. No se ajustó el modelo después de observar prueba.

| Modelo | Balanceada validación | Accuracy prueba | Balanceada prueba | F1 macro prueba |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 83,83% | 85,49% | 82,47% | 0,827 |
| ResNet18 | 79,54% | 84,17% | 81,10% | 0,809 |

**Selección:** EfficientNet-B0, por el mejor resultado de validación. También supera a ResNet18 en las tres métricas reportadas de prueba, con un checkpoint de aproximadamente 16,35 MB. La diferencia de accuracy es pequeña (1,32 puntos porcentuales), por lo que no se debe afirmar superioridad estadística sin un análisis adicional. Las métricas de la app provienen de `modelo/metrics.json`, no de valores escritos en la interfaz.

**Datos efectivos:** 2.521 imágenes luego de excluir tres parejas de fotos idénticas con etiquetas contradictorias. Train 1.764, validación 378, prueba 379. La prueba se comparte entre ambos modelos.

## Error costoso y límites

El modelo elegido reconoce **12 de 20 otros residuos**: recall de `trash` **60%**. Los ocho errores fueron tres clasificados como vidrio, tres como metal y dos como papel. En un uso real, interpretar esas clases como una autorización para reciclar podría contaminar una carga. Sin embargo, TrashNet etiqueta material, no la aceptación real del gestor: este análisis es una aproximación al riesgo, no una medición de contaminación industrial.

El fondo blanco y la escasez de otros residuos limitan la generalización. Las 379 fotos de prueba proceden de la misma fuente, y faltan IDs de objetos para separar todos los casos visualmente parecidos. Se requiere una evaluación externa con fotos del lugar de uso.

La confianza softmax de una imagen no es la accuracy agregada ni una probabilidad calibrada de acierto. El umbral de 70% activa revisión humana como decisión educativa. Incluso por encima del umbral, la app recomienda verificar material y reglas locales.

## Qué mejoraría el proyecto

Recolectar fotos de residuos de una planta o campus, con IDs de objeto y varios fondos; ampliar la clase minoritaria; validar fuera de TrashNet; y estudiar calibración en validación. El código incluye ajuste completo en GPU como alternativa, pero **no se atribuyen a esa modalidad las métricas de esta ejecución**.
