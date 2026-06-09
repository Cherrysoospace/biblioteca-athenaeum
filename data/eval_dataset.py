"""
data/eval_dataset.py — Dataset de evaluación RAGAS.

20 pares (pregunta, ground_truth) basados en el contenido REAL
de la Biblioteca Athenaeum (seed: biblioteca_athenaeum_seed.sql).
"""

EVAL_DATASET = [
    {
        "pregunta": "¿Qué libro de Gabriel García Márquez está disponible en la biblioteca?",
        "ground_truth": "'Cien años de soledad' de Gabriel García Márquez, novela fundacional del realismo mágico que narra la saga de la familia Buendía en Macondo.",
    },
    {
        "pregunta": "¿Qué obra de Octavio Paz hay en el catálogo?",
        "ground_truth": "'El laberinto de la soledad', ensayo de Octavio Paz que explora la identidad y la psicología del mexicano a través de la historia y los mitos nacionales.",
    },
    {
        "pregunta": "¿Qué recursos hay sobre inteligencia artificial en medicina?",
        "ground_truth": "'Inteligencia artificial en medicina', artículo que revisa el uso de modelos de aprendizaje profundo para diagnóstico clínico en radiología e histopatología.",
    },
    {
        "pregunta": "¿Qué mapas históricos están disponibles?",
        "ground_truth": "'Atlas histórico de la Revolución Francesa', colección de 48 mapas que documenta los movimientos militares y cambios administrativos de 1789 a 1799.",
    },
    {
        "pregunta": "¿Hay revistas de filosofía en la biblioteca?",
        "ground_truth": "'Revista de Filosofía Latinoamericana', publicación semestral sobre filosofía existencialista, fenomenología y pensamiento decolonial en América Latina.",
    },
    {
        "pregunta": "¿Qué recursos hay sobre cambio climático?",
        "ground_truth": "'Cambio climático: evidencias y proyecciones', síntesis del sexto informe del IPCC adaptada para divulgación científica con énfasis en América del Sur.",
    },
    {
        "pregunta": "¿Qué libros de arquitectura moderna están disponibles?",
        "ground_truth": "'Historia de la arquitectura moderna', recorrido por los movimientos arquitectónicos del siglo XX desde el funcionalismo bauhaus hasta el high-tech.",
    },
    {
        "pregunta": "¿Hay fotografías históricas del Bogotazo?",
        "ground_truth": "'Fotografías del Bogotazo 1948', archivo de 220 imágenes que documentan los eventos del 9 de abril de 1948 tras el asesinato de Jorge Eliécer Gaitán.",
    },
    {
        "pregunta": "¿Qué libros de neurociencia hay?",
        "ground_truth": "'Neurociencia cognitiva: fundamentos', introducción a los mecanismos neuronales de la percepción, la memoria, el lenguaje y la toma de decisiones.",
    },
    {
        "pregunta": "¿Qué obra de Hannah Arendt está disponible?",
        "ground_truth": "'La condición humana' de Hannah Arendt, análisis filosófico sobre la labor, el trabajo y la acción como actividades fundamentales de la vida activa.",
    },
    {
        "pregunta": "¿Hay revistas de ecología en la biblioteca?",
        "ground_truth": "'Revista Iberoamericana de Ecología', revista científica de acceso abierto especializada en ecología de ecosistemas tropicales y biodiversidad.",
    },
    {
        "pregunta": "¿Qué obra de Charles Darwin está disponible?",
        "ground_truth": "'El origen de las especies' de Charles Darwin, traducción anotada que expone la teoría de la evolución por selección natural.",
    },
    {
        "pregunta": "¿Quién escribió 'El laberinto de la soledad'?",
        "ground_truth": "'El laberinto de la soledad' fue escrito por Octavio Paz, poeta y ensayista mexicano, Premio Nobel de Literatura 1990.",
    },
    {
        "pregunta": "¿Hay recursos sobre el IPCC en la biblioteca?",
        "ground_truth": "Sí, 'Cambio climático: evidencias y proyecciones' es una síntesis del sexto informe del IPCC, adaptada para divulgación científica.",
    },
    {
        "pregunta": "¿Qué autores colombianos están en el catálogo?",
        "ground_truth": "Gabriel García Márquez (autor de 'Cien años de soledad'), Jorge Eliécer Gaitán (figura histórica documentada en 'Fotografías del Bogotazo 1948') y María Cristina Suárez (editora del atlas histórico y del archivo fotográfico).",
    },
    {
        "pregunta": "¿Qué idiomas tienen los recursos de la biblioteca?",
        "ground_truth": "La mayoría de los recursos están en español, y hay uno en francés ('Atlas histórico de la Revolución Francesa').",
    },
    {
        "pregunta": "¿Qué libros tienen calificación 5 según las reseñas?",
        "ground_truth": "'El laberinto de la soledad', 'Cien años de soledad', 'Cambio climático: evidencias y proyecciones', 'Neurociencia cognitiva: fundamentos', 'La condición humana' y 'Fotografías del Bogotazo 1948' tienen calificación 5.",
    },
    {
        "pregunta": "¿Qué recursos tratan sobre la Revolución Francesa?",
        "ground_truth": "'Atlas histórico de la Revolución Francesa', colección cartográfica de 48 mapas que documenta los movimientos militares y cambios territoriales de 1789 a 1799.",
    },
    {
        "pregunta": "¿Hay artículos científicos sobre diagnóstico clínico?",
        "ground_truth": "'Inteligencia artificial en medicina', artículo que revisa el uso de aprendizaje profundo para diagnóstico en radiología e histopatología.",
    },
    {
        "pregunta": "¿Qué recursos de filosofía están en español?",
        "ground_truth": "'La condición humana' de Hannah Arendt, 'Revista de Filosofía Latinoamericana' y 'El laberinto de la soledad' de Octavio Paz, todos en español.",
    },
]
