Bases de datos relacionales 
Proyecto Final 
Sistema Híbrido Relacional-Vectorial para Recuperación Aumentada (RAG) 
 
1. Información General del Proyecto 
 
Los estudiantes desarrollarán un sistema de Recuperación y Generación Aumentada (RAG) 
que combine un motor de base de datos relacional con extensiones vectoriales. El sistema 
permitirá almacenar datos estructurados y no estructurados (texto e imágenes), procesarlos 
mediante embeddings y realizar consultas inteligentes que combinen SQL tradicional con 
búsqueda vectorial semántica. 
 
El proyecto finalizará con la implementación de un pipeline RAG que integre un LLM 
accesible mediante API gratuita (ej. Groq API, HuggingFace Inference API o OpenAI free 
tier), y que incorpore un módulo de evaluación automática de calidad usando métricas 
estándar de la industria. 
 
2. Objetivos de Aprendizaje 
 
• 
Diseñar e implementar una arquitectura híbrida relacional-vectorial. 
• 
Construir un modelo entidad-relación y un modelo lógico en 3FN que incorpore 
atributos vectorizables. 
• 
Procesar y vectorizar al menos texto e imágenes. 
• 
Ejecutar consultas relacionales, vectoriales e híbridas. 
• 
Integrar un pipeline RAG con un LLM gratuito. 
• 
Implementar un sistema de evaluación automática del pipeline RAG usando métricas 
RAGAS. 
• 
Comparar experimentalmente distintas estrategias de chunking y seleccionar la más 
adecuada para el dominio de datos. 
 
Ejemplo de referencia 
 
3. Diseño Conceptual 
 
Entidades principales 
• 
Usuario (id_usuario, nombre, email, rol). 
• 
Documento (id_doc, título, idioma, fecha, fuente, contenido_texto). 
• 
Imagen (id_img, id_doc FK, ruta_archivo, descripción, etiquetas). 

• 
Consulta (id_consulta, texto_pregunta, fecha, usuario). 
• 
Evaluacion (id_eval, id_consulta FK, faithfulness, answer_relevancy, 
context_recall, modelo_eval, fecha). 
 
Relaciones 
• 
Un Usuario puede realizar muchas consultas. 
• 
Un Documento puede tener múltiples Imágenes asociadas. 
• 
Una Consulta puede recuperar tanto Documentos como Imágenes relevantes. 
• 
Una Consulta tiene asociada exactamente una evaluacion de calidad. 
 
Atributos vectorizables (para embeddings) 
• 
contenido_texto (campo de documento). 
• 
título (opcional para búsquedas rápidas). 
• 
texto_pregunta (de Consulta, usado para matching semántico). 
• 
ruta_archivo o representación binaria de Imagen (vectorizada con CLIP). 
• 
descripción / etiquetas (metadatos de imagen que enriquecen el embedding). 
 
4. Diseño Lógico 
 
Tablas Relacionales 
• 
Usuarios (id_usuario PK, nombre, email, rol). 
• 
Documentos (id_doc PK, título, idioma, fecha, fuente, contenido_texto). 
• 
Imagenes (id_img PK, id_doc FK, ruta_archivo, descripción, etiquetas). 
• 
Consultas (id_consulta PK, texto_pregunta, fecha, id_usuario FK). 
• 
Resultados (id_resultado PK, id_consulta FK, id_doc FK NULL, id_img FK 
NULL, score_similitud). 
• 
Evaluaciones (id_eval PK, id_consulta FK, faithfulness FLOAT, 
answer_relevancy FLOAT, context_recall FLOAT, modelo_eval VARCHAR, 
fecha TIMESTAMP). 
 
Tablas con soporte vectorial 
• 
Embeddings_Texto (id_embedding PK, id_doc FK, chunk_id INT, 
estrategia_chunking VARCHAR, vector_embedding VECTOR(384), modelo). 
• 
Embeddings_Imagen (id_embedding PK, id_img FK, vector_embedding 
VECTOR(512), modelo). 
• 
QueryEmbeddings (id_qemb PK, id_consulta FK, vector_embedding 
VECTOR(384), modelo). 
 
Nota sobre Evaluaciones 
La tabla Evaluaciones registra las métricas RAGAS por consulta, permitiendo consultas 
analíticas sobre la calidad del sistema a lo largo del tiempo. 
El 
campo 
estrategia_chunking 
en 
Embeddings_Texto 
permite 
comparar 
distintas 
configuraciones en la misma base de datos. 
 

5. Estrategias de Chunking 
 
Una de las decisiones más críticas en un sistema RAG es cómo dividir los documentos en 
fragmentos (chunks) antes de vectorizarlos. Distintas estrategias producen resultados muy 
diferentes según el tipo de contenido y las consultas esperadas. 
 
5.1 Estrategias a implementar y comparar 
 
Estrategia 
Descripción 
Cuándo es mejor 
Fixed-size chunking 
Divide el texto en fragmentos de N 
tokens con solapamiento (overlap). 
Simple y predecible. 
Textos homogéneos, documentos sin 
estructura semántica clara. 
Sentence-aware 
chunking 
Respeta los límites de oraciones. No 
corta en medio de una frase. 
Textos narrativos, artículos de 
noticias, descripciones. 
Semantic chunking 
Agrupa oraciones por similitud 
semántica usando embeddings. 
Produce chunks temáticamente 
coherentes. 
Documentos técnicos, papers, textos 
con cambios de tema. 
 
5.2 Parámetros de configuración sugeridos 
• 
Fixed-size: chunk_size = 256 tokens, overlap = 32 tokens. 
• 
Sentence-aware: máximo 5 oraciones por chunk, overlap de 1 oración. 
• 
Semantic: umbral de similitud coseno entre 0.75 y 0.85. 
 
5.3 Experimento de comparación 
Cada grupo debe ejecutar el mismo conjunto de 10 consultas de prueba sobre los tres tipos 
de chunking y registrar las métricas RAGAS resultantes. 
 
6. Evaluación con RAGAS (Nota extra) 
 
RAGAS (Retrieval Augmented Generation Assessment) es un framework de evaluación 
automática diseñado específicamente para sistemas RAG. Permite medir la calidad tanto 
de la recuperación como de la generación sin necesidad de anotación humana para todas 
las métricas. 
 
6.1 Métricas principales 
 
Métrica 
Descripción 
Rango 

Faithfulness 
Mide si la respuesta generada es factualmente 
consistente con el contexto recuperado. Una 
respuesta fiel no inventa información. 
0.0 – 1.0 
Answer Relevancy 
Evalúa si la respuesta es pertinente a la 
pregunta original. Penaliza respuestas 
incompletas o fuera de tema. 
0.0 – 1.0 
Context Recall 
Compara el contexto recuperado con la 
respuesta esperada (ground truth). Requiere 
anotación manual de un subconjunto. 
0.0 – 1.0 
 
6.2 Métricas opcionales (para nota extra) 
• 
Context Precision: mide cuánto del contexto recuperado es realmente útil. 
• 
Answer Correctness: compara semánticamente la respuesta con el ground 
truth. 
 
6.3 Implementación sugerida 
• 
Instalar ragas con: pip install ragas. 
• 
Preparar un dataset de evaluación con al menos 20 pares (pregunta, 
ground_truth). 
• 
Ejecutar la evaluación sobre las respuestas generadas por el LLM. 
• 
Almacenar los scores en la tabla Evaluaciones de la base de datos. 
 
Ejemplo de código mínimo 
 
from ragas import evaluate 
from ragas.metrics import faithfulness, answer_relevancy, context_recall 
 
result = evaluate( 
    dataset=eval_dataset, 
    metrics=[faithfulness, answer_relevancy, context_recall] 
) 
 
7. Entregas del Proyecto 
Entrega 1: Diseño y Configuración 
• 
Universo del discurso y análisis de requerimientos. 
• 
Diseño conceptual (ERD). 
• 
Diseño lógico en 3FN. 
• 
Identificación de campos vectorizables (texto e imágenes). 
• 
Esquema SQL con tablas relacionales y vectoriales. 
• 
Entorno ejecutable: Supabase (Colab) o Docker (Postgres + pgvector). 

• 
Plan de experimento de chunking: qué estrategias y parámetros se probarán. 
 
 
Entrega 2: Procesamiento, RAG y Evaluación 
Ingesta de datos 
• 
Chunking de textos con las tres estrategias definidas. 
• 
Extracción de embeddings (ej. MiniLM) por cada estrategia. 
• 
Vectorización de imágenes con modelo multimodal (ej. CLIP). 
• 
Inserción en tablas Embeddings_Texto (con campo estrategia_chunking) y 
Embeddings_Imagen. 
 
Consultas de prueba 
• 
SQL clásicas (ej. filtrar imágenes por etiqueta). 
• 
Vectoriales (similitud texto-texto, imagen-imagen, texto→imagen o 
imagen→texto). 
• 
Híbridas (ej. "imágenes de turismo asociadas a documentos en inglés 
publicados en 2024"). 
 
Pipeline RAG 
• 
Recuperación de documentos e imágenes por embeddings. 
• 
Generación de respuesta contextualizada en un LLM gratuito. 
 
Informe final (máx. 12 páginas) 
• 
Descripción de la arquitectura implementada. 
• 
Resultados de las consultas híbridas. 
• 
Resultados del experimento de chunking con interpretación. 
 

