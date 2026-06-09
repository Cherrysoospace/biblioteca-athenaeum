"""
scripts/agregar_20_recursos.py

Añade 20 recursos bibliográficos adicionales (libros, artículos, revistas,
fotografías) en múltiples idiomas con todas sus relaciones:
  - Autores, géneros, tags
  - Imágenes
  - Colecciones, reseñas, préstamos
  - Embeddings de texto (all-MiniLM-L6-v2) e imagen (CLIP)

Ejecutar: python scripts/agregar_20_recursos.py
"""

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from sqlalchemy import text

from core.database import get_session
from pipeline.embeddings_minilm import get_embedding, get_embeddings_batch

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agregar_20_recursos")

# ═══════════════════════════════════════════════════════════
# DATOS: 20 recursos en múltiples idiomas
# ═══════════════════════════════════════════════════════════

RECURSOS = [
    {"titulo": "The Great Gatsby", "fecha": date(1925, 4, 10), "idioma": "English", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "A novel set in the Jazz Age that explores themes of wealth, love, and the American Dream through the mysterious millionaire Jay Gatsby and his obsession with Daisy Buchanan."},
    {"titulo": "Les Misérables", "fecha": date(1862, 1, 1), "idioma": "Français", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "Roman monumental qui suit le parcours de Jean Valjean, ancien bagnard en quête de rédemption, dans la France du XIXe siècle, entrelaçant histoire, justice sociale et amour."},
    {"titulo": "Dom Casmurro", "fecha": date(1899, 1, 1), "idioma": "Português", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "Romance clássico da literatura brasileira que narra a história de Bentinho e Capitu, explorando ciúmes, dúvida e memória através de uma narrativa engenhosa e irônica de Machado de Assis."},
    {"titulo": "Die Verwandlung (La metamorfosis)", "fecha": date(1915, 1, 1), "idioma": "Deutsch", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "Novela existencialista que narra la transformación de Gregorio Samsa en un insecto gigante y el consecuente aislamiento de su familia, explorando la alienación y la culpa."},
    {"titulo": "1984", "fecha": date(1949, 6, 8), "idioma": "English", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Dystopian novel depicting a totalitarian regime where Big Brother watches everything, language is controlled through Newspeak, and independent thought is punished as thoughtcrime."},
    {"titulo": "Le Petit Prince", "fecha": date(1943, 4, 6), "idioma": "Français", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Conte philosophique et poétique qui raconte la rencontre entre un aviateur et un petit prince venu d'un autre astre, explorant les thèmes de l'amitié, de l'amour et du sens de la vie."},
    {"titulo": "O Alquimista", "fecha": date(1988, 1, 1), "idioma": "Português", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Narração da jornada de Santiago, um pastor andaluz, em busca de um tesouro nas pirâmides do Egito, descobrindo lições sobre a vida, o destino e a importância de seguir os próprios sonhos."},
    {"titulo": "The Art of War", "fecha": None, "idioma": "Chinese", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "Ancient Chinese military treatise attributed to Sun Tzu, covering strategy, tactics, and philosophy of warfare. Its principles have been applied to business, sports, and diplomacy worldwide."},
    {"titulo": "Sapiens: A Brief History of Humankind", "fecha": date(2011, 1, 1), "idioma": "English", "tipo": "libro", "licencia": "CC BY-NC",
     "descripcion": "A sweeping narrative of humanity's history from the Cognitive Revolution to the present, exploring how Homo sapiens came to dominate the planet through cooperation, language, and shared myths."},
    {"titulo": "Neuromancer", "fecha": date(1984, 7, 1), "idioma": "English", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Foundational cyberpunk novel that follows Case, a washed-up computer hacker, through a world of artificial intelligence, corporate espionage, and virtual reality cyberspace."},
    {"titulo": "La sombra del viento", "fecha": date(2001, 1, 1), "idioma": "Español", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Novela ambientada en la Barcelona de posguerra donde Daniel Sempere descubre un libro maldito que desencadena una historia de misterio, amor y venganza literaria."},
    {"titulo": "Fausto", "fecha": date(1808, 1, 1), "idioma": "Español", "tipo": "libro", "licencia": "Dominio público",
     "descripcion": "Drama filosófico de Goethe que narra el pacto del erudito Fausto con Mefistófeles, explorando la ambición del conocimiento, la condición humana y la redención."},
    {"titulo": "Deep Learning for Computer Vision: Architectures and Applications", "fecha": date(2023, 6, 15), "idioma": "English", "tipo": "articulo", "licencia": "CC BY",
     "descripcion": "Comprehensive survey of deep learning architectures for computer vision tasks including object detection, segmentation, and image generation using CNN and transformer-based models."},
    {"titulo": "Transformer Models in NLP: A Comprehensive Survey", "fecha": date(2024, 1, 10), "idioma": "English", "tipo": "articulo", "licencia": "CC BY",
     "descripcion": "Systematic review of transformer-based language models such as BERT, GPT-4, LLaMA, and Mistral, analyzing architectures, training methods, fine-tuning strategies, and NLP applications."},
    {"titulo": "Revista Internacional de Estudios Lingüísticos", "fecha": date(2023, 3, 1), "idioma": "Español", "tipo": "revista", "licencia": "CC BY-NC-SA",
     "descripcion": "Publicación académica semestral que recoge investigaciones sobre lingüística teórica y aplicada, sociolingüística, lenguas indígenas de América Latina y políticas lingüísticas."},
    {"titulo": "Journal of Climate Science and Policy", "fecha": date(2023, 9, 1), "idioma": "English", "tipo": "revista", "licencia": "CC BY",
     "descripcion": "Peer-reviewed journal publishing interdisciplinary research on climate science, mitigation strategies, carbon neutrality, policy frameworks, and sustainable development goals."},
    {"titulo": "The Structure of Scientific Revolutions", "fecha": date(1962, 1, 1), "idioma": "English", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Influential philosophical work introducing paradigm shifts in scientific progress, challenging cumulative views and redefining how scientific communities evolve."},
    {"titulo": "Fotografías del conflicto armado colombiano 1990-2000", "fecha": date(2010, 7, 20), "idioma": "Español", "tipo": "libro", "licencia": "CC BY-NC",
     "descripcion": "Archivo fotográfico documental con 150 imágenes del conflicto armado interno colombiano en los 90: desplazamiento forzado, masacres, y procesos de paz."},
    {"titulo": "International Journal of Quantum Computing Research", "fecha": date(2024, 1, 1), "idioma": "English", "tipo": "revista", "licencia": "CC BY",
     "descripcion": "Academic journal on quantum algorithms, error correction, hardware architectures, quantum machine learning, and quantum cryptography protocols."},
    {"titulo": "El amor en los tiempos del cólera", "fecha": date(1985, 1, 1), "idioma": "Español", "tipo": "libro", "licencia": "Todos los derechos reservados",
     "descripcion": "Novela que narra la historia de amor entre Florentino Ariza y Fermina Daza a lo largo de medio siglo, explorando la persistencia del amor frente al tiempo, la muerte y las convenciones."},
]

AUTORES_NUEVOS = [
    ("F. Scott Fitzgerald",          "persona", "Escritor estadounidense, figura clave de la Generación Perdida. Su obra explora la riqueza, el glamour y la desilusión del sueño americano."),
    ("Victor Hugo",                  "persona", "Écrivain, poète et homme politique français, figure majeure du romantisme. Auteur de romans monumentaux dénonçant l'injustice sociale."),
    ("Machado de Assis",             "persona", "Escritor brasileiro, fundador da Academia Brasileira de Letras. Conhecido por seu estilo irônico e análise profunda da psicologia humana."),
    ("Franz Kafka",                  "persona", "Escritor checo en lengua alemana, cuya obra existencialista explora la alienación, la burocracia y la angustia del individuo moderno."),
    ("George Orwell",                "persona", "British writer known for sharp social criticism and dystopian visions of totalitarianism, language manipulation, and political power abuse."),
    ("Antoine de Saint-Exupéry",     "persona", "Aviateur et écrivain français, dont l'œuvre célèbre l'aviation, l'amitié et la condition humaine. Auteur du Petit Prince."),
    ("Paulo Coelho",                 "persona", "Escritor brasileiro, um dos autores mais lidos do mundo. Sua obra combina espiritualidade e jornadas de autodescoberta."),
    ("Sun Tzu",                      "persona", "Ancient Chinese military strategist and philosopher. The Art of War has influenced military, business, and political strategy for millennia."),
    ("Yuval Noah Harari",            "persona", "Israeli historian exploring the intersection of history, biology, and technology in understanding human civilization."),
    ("William Gibson",               "persona", "American-Canadian writer who pioneered the cyberpunk genre, exploring AI, virtual reality, and the fusion of human and machine."),
    ("Carlos Ruiz Zafón",            "persona", "Escritor español, autor de la saga El Cementerio de los Libros Olvidados, combinando misterio y crítica social en la Barcelona del siglo XX."),
    ("Johann Wolfgang von Goethe",   "persona", "Escritor, poeta y pensador alemán, figura cumbre de la literatura universal. Fausto es su creación más emblemática."),
    ("Thomas Kuhn",                  "persona", "American philosopher of science who revolutionized understanding of scientific progress with paradigm shifts."),
    ("Centro de Investigación en IA", "organizacion", "Centro interdisciplinario dedicado a la investigación en inteligencia artificial, aprendizaje automático y procesamiento del lenguaje natural."),
]

# Para cada recurso (índice 0-19), lista de (tipo, id):
#   tipo='new' → índice en AUTORES_NUEVOS
#   tipo='old' → id existente en DB
AUTORES_POR_RECURSO = [
    [("new", 0)],                         # 0: Great Gatsby → Fitzgerald
    [("new", 1)],                         # 1: Les Misérables → Hugo
    [("new", 2)],                         # 2: Dom Casmurro → Machado de Assis
    [("new", 3)],                         # 3: Metamorfosis → Kafka
    [("new", 4)],                         # 4: 1984 → Orwell
    [("new", 5)],                         # 5: Petit Prince → Saint-Exupéry
    [("new", 6)],                         # 6: O Alquimista → Coelho
    [("new", 7)],                         # 7: Art of War → Sun Tzu
    [("new", 8)],                         # 8: Sapiens → Harari
    [("new", 9)],                         # 9: Neuromancer → Gibson
    [("new", 10)],                        # 10: Sombra del viento → Ruiz Zafón
    [("new", 11)],                        # 11: Fausto → Goethe
    [("new", 13)],                        # 12: Deep Learning → Centro IA
    [("new", 13)],                        # 13: Transformers → Centro IA
    [("old", 9)],                         # 14: Revista Lingüística → Academia Colombiana (id=9)
    [("old", 5)],                         # 15: Journal Climate → IPCC (id=5)
    [("new", 12)],                        # 16: Kuhn → Thomas Kuhn
    [("old", 6)],                         # 17: Fotografías conflicto → María Cristina Suárez (id=6)
    [("new", 13)],                        # 18: Quantum Computing → Centro IA
    [("old", 2)],                         # 19: El amor en tiempos del cólera → Gabriel García Márquez (id=2)
]

GENEROS_NUEVOS = [
    ("Literatura Anglófona", "Obras literarias en lengua inglesa de autores británicos, estadounidenses y otras regiones angloparlantes."),
    ("Literatura Francesa",  "Œuvres littéraires en français couvrant poésie, roman, théâtre et essai du Moyen Âge à l'époque contemporaine."),
    ("Literatura Lusófona",  "Obras literárias em português de Portugal, Brasil e países africanos lusófonos."),
    ("Literatura Germánica", "Obras literarias en alemán desde la épica medieval hasta la narrativa contemporánea y la filosofía."),
    ("Ciencia Política",     "Estudio de sistemas políticos, teoría del estado, relaciones internacionales y políticas públicas."),
    ("Computación",          "Publicaciones sobre ciencias de la computación, inteligencia artificial y avances tecnológicos."),
    ("Lingüística",          "Estudio científico del lenguaje humano, incluyendo fonética, sintaxis, semántica y sociolingüística."),
    ("Física Cuántica",      "Estudio de fenómenos a escalas atómicas, incluyendo computación y criptografía cuántica."),
]

# (tipo, id) — tipo='new' para índice en GENEROS_NUEVOS, tipo='old' para ID existente en DB
GENEROS_POR_RECURSO = [
    [("new", 0), ("old", 1)],   # 0: Gatsby → Anglófona + Filosofía
    [("new", 1)],                # 1: Les Misérables → Francesa
    [("new", 2)],                # 2: Dom Casmurro → Lusófona
    [("new", 3), ("old", 1)],   # 3: Metamorfosis → Germánica + Filosofía
    [("new", 0), ("new", 4)],   # 4: 1984 → Anglófona + Ciencia Política
    [("new", 1)],                # 5: Petit Prince → Francesa
    [("new", 2), ("old", 1)],   # 6: O Alquimista → Lusófona + Filosofía
    [("new", 0), ("old", 1)],   # 7: Art of War → Anglófona + Filosofía
    [("new", 0), ("new", 4)],   # 8: Sapiens → Anglófona + Ciencia Política
    [("new", 0), ("new", 5)],   # 9: Neuromancer → Anglófona + Computación
    [("old", 2)],                # 10: Sombra del viento → Literatura
    [("new", 3), ("old", 1)],   # 11: Fausto → Germánica + Filosofía
    [("new", 5), ("old", 4)],   # 12: Deep Learning → Computación + Ciencia
    [("new", 5), ("old", 4)],   # 13: Transformers → Computación + Ciencia
    [("new", 6), ("old", 10)],  # 14: Revista Lingüística → Lingüística + Cs. Sociales
    [("old", 4), ("old", 5)],   # 15: Journal Climate → Ciencia + Medio Ambiente
    [("old", 1), ("old", 4)],   # 16: Kuhn → Filosofía + Ciencia
    [("old", 3)],                # 17: Fotografías → Historia
    [("new", 5), ("new", 7)],   # 18: Quantum → Computación + Física Cuántica
    [("old", 2), ("old", 12)],  # 19: El amor → Literatura + Realismo Mágico
]

TAGS_NUEVOS = [
    "ciencia-ficción", "procesamiento-lenguaje-natural", "literatura-francesa",
    "literatura-portuguesa", "literatura-inglesa", "literatura-alemana",
    "lingüística", "computación-cuántica", "fotografía-documental",
    "deep-learning", "grandes-modelos-lenguaje", "conflicto-armado-colombia",
]

TAGS_POR_RECURSO = [
    [("new", 4)],              # 0: Gatsby → literatura-inglesa
    [("new", 2)],              # 1: Misérables → literatura-francesa
    [("new", 3)],              # 2: Dom Casmurro → literatura-portuguesa
    [("new", 5)],              # 3: Metamorfosis → literatura-alemana
    [("new", 4), ("new", 0)], # 4: 1984 → literatura-inglesa + ciencia-ficción
    [("new", 2)],              # 5: Petit Prince → literatura-francesa
    [("new", 3)],              # 6: Alquimista → literatura-portuguesa
    [("new", 4)],              # 7: Art of War → literatura-inglesa
    [("new", 4)],              # 8: Sapiens → literatura-inglesa
    [("new", 4), ("new", 0)], # 9: Neuromancer → literatura-inglesa + ciencia-ficción
    [],                        # 10: Sombra del viento → sin tags nuevos
    [],                        # 11: Fausto → sin tags nuevos
    [("new", 9)],              # 12: Deep Learning → deep-learning
    [("new", 1), ("new", 10)],# 13: Transformers → NLP + LLM
    [("new", 6)],              # 14: Revista Lingüística → lingüística
    [],                        # 15: Journal Climate → sin tags nuevos
    [],                        # 16: Kuhn → sin tags nuevos
    [("new", 8), ("new", 11)],# 17: Fotografías → fotografía-documental + conflicto
    [("new", 7)],              # 18: Quantum → computación-cuántica
    [],                        # 19: El amor → sin tags nuevos
]

IMAGENES = [
    ("portada",    "Cover of The Great Gatsby with luminous eyes overlooking a cityscape, Jazz Age opulence."),
    ("portada",    "Couverture des Misérables montrant Jean Valjean dans les rues de Paris au XIXe siècle."),
    ("portada",    "Capa de Dom Casmurro com retrato de Machado de Assis e ilustração de Bentinho e Capitu."),
    ("portada",    "Portada de La metamorfosis con la figura icónica de un insecto sobre una cama, estilo expresionista."),
    ("portada",    "Cover of 1984 showing a giant eye symbolizing the surveillance state of Big Brother."),
    ("portada",    "Couverture du Petit Prince montrant le garçon blond sur son astéroïde avec sa rose et le serpent."),
    ("portada",    "Capa de O Alquimista com um andarilho no deserto sob céu estrelado em tons dourados."),
    ("portada",    "Cover of The Art of War with ancient Chinese calligraphy on a bamboo background."),
    ("portada",    "Cover of Sapiens with abstract depiction of human evolution from hunters to civilization."),
    ("portada",    "Cover of Neuromancer with neon-lit futuristic cityscape and hacker silhouette."),
    ("portada",    "Portada de La sombra del viento con el Cementerio de los Libros Olvidados en tonos oscuros."),
    ("portada",    "Portada de Fausto con ilustración de Mefistófeles y Fausto en su estudio medieval."),
    ("ilustracion","Diagram of a CNN architecture for image classification with multiple filter layers."),
    ("ilustracion","Diagram of the Transformer architecture showing multi-head attention and encoder-decoder structure."),
    ("portada",    "Portada de la Revista de Estudios Lingüísticos con mapa de familias lingüísticas de América Latina."),
    ("portada",    "Cover of Journal of Climate Science and Policy featuring satellite imagery of melting ice caps."),
    ("portada",    "Cover of The Structure of Scientific Revolutions with abstract representation of shifting paradigms."),
    ("fotografia", "Fotografía de desplazados por el conflicto armado colombiano caminando por carretera rural en 1995."),
    ("portada",    "Cover of International Journal of Quantum Computing Research with Bloch sphere and quantum circuit."),
    ("portada",    "Portada de El amor en los tiempos del cólera con barco fluvial en el río Magdalena al atardecer."),
]

TEXT_CHUNKS = [
    ["The Great Gatsby explores the American Dream through Jay Gatsby, a mysterious millionaire who throws lavish parties in West Egg hoping to reunite with his lost love Daisy Buchanan.",
     "Fitzgerald's novel captures the Jazz Age in all its glamour and despair, revealing the corruption beneath wealth and the impossibility of repeating the past."],
    ["Les Misérables est une fresque historique qui suit Jean Valjean, ancien forçat cherchant la rédemption dans une société française marquée par les inégalités.",
     "Victor Hugo entrelace les destins de Fantine, Cosette, Marius et Javert dans un récit puissant sur la justice, la compassion et la révolution."],
    ["Dom Casmurro é um romance introspectivo onde Bentinho revisita sua juventude e o relacionamento com Capitu, questionando-se sobre a possível traição que marcou sua vida.",
     "Machado de Assis constrói uma narrativa ambígua e irônica que desafia o leitor a interpretar os sinais de ciúme e dúvida na memória do protagonista."],
    ["La metamorfosis narra la transformación de Gregorio Samsa en un insecto monstruoso y el progresivo rechazo de su familia, metáfora de la alienación humana.",
     "Kafka explora la culpa, la responsabilidad familiar y la deshumanización en una sociedad burocrática que reduce al individuo a su utilidad económica."],
    ["1984 by George Orwell presents a terrifying vision of a totalitarian future where the Party controls every aspect of life through surveillance and language manipulation.",
     "Big Brother, doublethink, and Newspeak have become enduring symbols of political oppression and the erosion of truth in modern society."],
    ["Le Petit Prince est un conte philosophique sur la rencontre entre un aviateur perdu dans le désert et un petit garçon venu d'une autre planète.",
     "À travers ses voyages entre astéroïdes, le petit prince découvre l'amitié, l'amour et la responsabilité, rappelant que l'essentiel est invisible pour les yeux."],
    ["O Alquimista acompanha Santiago, um pastor andaluz que viaja ao Egito em busca de um tesouro, descobrindo que a verdadeira riqueza está na jornada e no autoconhecimento.",
     "Paulo Coelho tece uma fábula espiritual sobre ouvir o coração, reconhecer oportunidades e perseguir a Lenda Pessoal."],
    ["The Art of War by Sun Tzu emphasizes knowing oneself and one's enemy, adaptability, and winning without fighting.",
     "Its principles of strategic thinking have been applied to business, sports, and negotiation for over two millennia."],
    ["Sapiens traces human history from the Cognitive Revolution, examining how language, cooperation, and shared beliefs enabled Homo sapiens to dominate the planet.",
     "Harari argues that shared fictions like money, nations, and laws are what distinguish humans and enable large-scale cooperation."],
    ["Neuromancer follows Case, a washed-up hacker in a world where cyberspace is tangible, AIs seek freedom, and corporations hold ultimate power.",
     "Gibson's novel defined cyberpunk, envisioning neural interfaces and AI consciousness long before the internet age."],
    ["La sombra del viento transcurre en la Barcelona de 1945, donde Daniel Sempere descubre un libro maldito y se obsesiona con su misterioso autor.",
     "Ruiz Zafón construye un misterio literario que entrelaza el amor por los libros, las heridas de la guerra civil y una venganza generacional."],
    ["Fausto es un drama filosófico donde el doctor Fausto pacta con Mefistófeles su alma a cambio de conocimiento absoluto y placeres terrenales.",
     "Goethe explora la tensión entre ambición humana y moral divina, la búsqueda del conocimiento como camino de perdición y redención."],
    ["This survey reviews deep learning for computer vision: CNNs, Vision Transformers, diffusion models for classification, detection, and generation.",
     "Vision-language models like CLIP and DALL-E bridge visual understanding and natural language, enabling zero-shot learning and multimodal reasoning."],
    ["This survey examines transformer LMs from BERT and GPT to LLaMA and Mistral, analyzing scaling laws, training data, and fine-tuning strategies.",
     "Transformer architectures revolutionized NLP by enabling parallel sequence processing and capturing long-range dependencies via self-attention."],
    ["La Revista Internacional de Estudios Lingüísticos publica investigaciones sobre lenguas indígenas latinoamericanas, sociolingüística y análisis del discurso.",
     "Este volumen incluye artículos sobre revitalización de lenguas originarias y políticas de educación intercultural bilingüe en la región andina."],
    ["The Journal of Climate Science and Policy publishes research on climate impacts, mitigation, carbon capture, renewable energy, and policy frameworks.",
     "This issue features nature-based solutions for adaptation, the economics of the green transition, and indigenous knowledge in environmental stewardship."],
    ["Kuhn challenges the cumulative view of science, arguing that progress occurs through paradigm shifts that transform how scientists understand the world.",
     "Paradigm shifts have influenced philosophy, sociology, and organizational theory as a foundational idea in understanding intellectual change."],
    ["Este archivo fotográfico documenta el conflicto armado colombiano en los 90: desplazamiento forzado, masacres rurales y negociaciones de paz.",
     "Las imágenes capturan la resiliencia de las comunidades afectadas y la memoria colectiva como herramienta para la reconciliación nacional."],
    ["This journal publishes research on quantum algorithms, error correction, superconducting qubits, topological computing, and quantum ML.",
     "Advances in quantum supremacy and fault-tolerant computers promise to revolutionize cryptography, drug discovery, and optimization."],
    ["El amor en los tiempos del cólera narra la historia de Florentino Ariza y Fermina Daza, cuyo amor persiste durante más de cincuenta años.",
     "García Márquez explora las múltiples formas del amor frente al tiempo, la muerte y las convenciones sociales en una meditación sobre la pasión."],
]


# ═══════════════════════════════════════════════════════════
# FUNCIONES DE INSERCIÓN
# ═══════════════════════════════════════════════════════════

def insertar_autores(session) -> list[int]:
    ids = []
    for nombre, tipo, bio in AUTORES_NUEVOS:
        rp = session.execute(
            text("INSERT INTO autores (nombre, tipo, biografia) VALUES (:n, :t, :b) RETURNING id"),
            {"n": nombre, "t": tipo, "b": bio},
        )
        ids.append(rp.scalar())
    session.flush()
    logger.info("Insertados %d autores nuevos (IDs %d-%d)", len(ids), ids[0], ids[-1])
    return ids


def insertar_generos(session) -> list[int]:
    ids = []
    for nombre, desc in GENEROS_NUEVOS:
        rp = session.execute(
            text("INSERT INTO generos (nombre, descripcion) VALUES (:n, :d) ON CONFLICT (nombre) DO NOTHING RETURNING id"),
            {"n": nombre, "d": desc},
        )
        row = rp.fetchone()
        if row:
            ids.append(row[0])
    session.flush()
    if ids:
        logger.info("Insertados %d géneros nuevos (IDs %d-%d)", len(ids), ids[0], ids[-1])
    return ids


def insertar_tags(session) -> list[int]:
    ids = []
    for nombre in TAGS_NUEVOS:
        rp = session.execute(
            text("INSERT INTO tags (nombre) VALUES (:n) ON CONFLICT (nombre) DO NOTHING RETURNING id"),
            {"n": nombre},
        )
        row = rp.fetchone()
        if row:
            ids.append(row[0])
    session.flush()
    if ids:
        logger.info("Insertados %d tags nuevos (IDs %d-%d)", len(ids), ids[0], ids[-1])
    return ids


def insertar_recursos(session) -> list[int]:
    ids = []
    for r in RECURSOS:
        rp = session.execute(
            text("""INSERT INTO recursos (titulo, fecha_publicacion, idioma, tipo, licencia, descripcion)
                    VALUES (:t, :f, :i, :tp, :l, :d) RETURNING id"""),
            {"t": r["titulo"], "f": r["fecha"], "i": r["idioma"], "tp": r["tipo"],
             "l": r["licencia"], "d": r["descripcion"]},
        )
        ids.append(rp.scalar())
    session.flush()
    logger.info("Insertados %d recursos (IDs %d-%d)", len(ids), ids[0], ids[-1])
    return ids


def resolver_id(tipo_id, idx_o_id, mapa_ids_nuevos):
    """Dado ('new', idx) o ('old', db_id), retorna el ID numérico."""
    if tipo_id == "new":
        return mapa_ids_nuevos[idx_o_id]
    return idx_o_id  # 'old' → es directamente el ID en DB


def insertar_relaciones(session, tabla, col_a, col_b, recurso_ids, mapping, nuevos_ids):
    count = 0
    for i_recurso, entries in enumerate(mapping):
        rid = recurso_ids[i_recurso]
        for tipo_id, idx_o_id in entries:
            otro_id = resolver_id(tipo_id, idx_o_id, nuevos_ids)
            session.execute(
                text(f"INSERT INTO {tabla} ({col_a}, {col_b}) VALUES (:v1, :v2) ON CONFLICT DO NOTHING"),
                {"v1": rid, "v2": otro_id},
            )
            count += 1
    session.flush()
    logger.info("Insertadas %d relaciones en %s", count, tabla)


def insertar_imagenes(session, recurso_ids: list[int]) -> list[int]:
    ids = []
    for i, (tipo, desc) in enumerate(IMAGENES):
        rid = recurso_ids[i]
        ruta = f"/media/adicionales/recurso_{rid}_{tipo}.jpg"
        rp = session.execute(
            text("""INSERT INTO imagenes_recurso (recurso_id, ruta_archivo, tipo_imagen, descripcion)
                    VALUES (:r, :ruta, :t, :d) RETURNING id"""),
            {"r": rid, "ruta": ruta, "t": tipo, "d": desc},
        )
        ids.append(rp.scalar())
    session.flush()
    logger.info("Insertadas %d imágenes (IDs %d-%d)", len(ids), ids[0], ids[-1])
    return ids


def insertar_coleccion_recursos(session, recurso_ids: list[int]):
    asignaciones = [
        (1,  [2, 6, 11]),     # Filosofía → Dom Casmurro, O Alquimista, Fausto
        (2,  [12, 13, 18]),   # Ciencia → Deep Learning, Transformers, Quantum
        (4,  [0, 3, 4, 7]),   # Clásicos → Gatsby, Metamorfosis, 1984, Art of War
        (5,  [15]),            # Clima → Journal Climate
        (8,  [5, 8, 10]),     # Favoritos → Petit Prince, Sapiens, Sombra viento
        (10, [1, 11]),         # Doctorado → Les Misérables, Fausto
        (11, [9, 14, 16]),    # Repositorio → Neuromancer, Revista Lingüística, Kuhn
    ]
    count = 0
    for col_id, indices in asignaciones:
        for idx in indices:
            session.execute(
                text("INSERT INTO coleccion_recursos (coleccion_id, recurso_id) VALUES (:c, :r) ON CONFLICT DO NOTHING"),
                {"c": col_id, "r": recurso_ids[idx]},
            )
            count += 1
    session.flush()
    logger.info("Insertadas %d relaciones colección-recurso", count)


def insertar_resenas(session, recurso_ids: list[int]):
    resenas = [
        (1,  0, 5, "A masterful exploration of the American Dream. Fitzgerald's prose is luminous and the tragedy of Gatsby remains profoundly moving."),
        (2,  1, 5, "Une œuvre monumentale mêlant histoire et destins individuels avec une puissance narrative inégalée."),
        (4,  2, 4, "Machado de Assis constrói uma narrativa ambígua que desafia o leitor. A dúvida sobre Capitu é o verdadeiro tema."),
        (5,  4, 5, "Orwell's vision remains chillingly relevant in the age of surveillance capitalism. Essential for understanding power and truth."),
        (6,  7, 4, "Timeless wisdom on strategy. Sun Tzu's insights on knowing oneself and one's adversary apply to life as to warfare."),
        (7,  8, 5, "Harari synthesizes complex ideas into a compelling narrative. Shared fictions as the basis for cooperation is brilliant."),
        (9,  10, 4, "Una novela que es un homenaje a los libros. Ruiz Zafón construye un misterio fascinante en una Barcelona mágica."),
        (10, 16, 5, "Kuhn's paradigm shifts fundamentally changed how we understand scientific progress. A must-read for philosophy of science."),
        (11, 17, 5, "Archivo fotográfico desgarrador del conflicto colombiano. Testimonio necesario para la memoria histórica."),
        (12, 19, 5, "García Márquez demuestra que el amor puede ser tan persistente como el cólera. Meditación bellísima sobre la pasión y el tiempo."),
    ]
    count = 0
    for uid, idx, cal, txt in resenas:
        session.execute(
            text("INSERT INTO reseñas (usuario_id, recurso_id, calificacion, texto) VALUES (:u, :r, :c, :t)"),
            {"u": uid, "r": recurso_ids[idx], "c": cal, "t": txt},
        )
        count += 1
    session.flush()
    logger.info("Insertadas %d reseñas", count)


def insertar_prestamos(session, recurso_ids: list[int]):
    data = [
        (1,  0,  date(2026, 5, 1),  True,  date(2026, 5, 28)),
        (2,  1,  date(2026, 5, 10), True,  date(2026, 6, 5)),
        (4,  4,  date(2026, 5, 15), True,  date(2026, 6, 10)),
        (5,  8,  date(2026, 5, 20), False, None),
        (6,  10, date(2026, 6, 1),  False, None),
        (8,  16, date(2026, 6, 1),  False, None),
        (10, 19, date(2026, 6, 5),  False, None),
    ]
    prestamo_ids = []
    for uid, idx, inicio, dev, devol in data:
        fin = inicio + timedelta(days=30)
        rp = session.execute(
            text("""INSERT INTO prestamos (usuario_id, recurso_id, fecha_inicio, fecha_fin_esperada, devuelto, fecha_devolucion)
                    VALUES (:u, :r, :i, :f, :dv, :dl) RETURNING id"""),
            {"u": uid, "r": recurso_ids[idx], "i": inicio, "f": fin, "dv": dev, "dl": devol},
        )
        prestamo_ids.append(rp.scalar())
    session.flush()

    historial = [
        (prestamo_ids[0], date(2026, 5, 5),  80,  False),
        (prestamo_ids[0], date(2026, 5, 20), 180, True),
        (prestamo_ids[1], date(2026, 5, 15), 120, False),
        (prestamo_ids[1], date(2026, 6, 3),  350, True),
        (prestamo_ids[2], date(2026, 5, 20), 100, False),
        (prestamo_ids[2], date(2026, 6, 8),  280, True),
        (prestamo_ids[3], date(2026, 5, 25), 50,  False),
        (prestamo_ids[4], date(2026, 6, 5),  30,  False),
        (prestamo_ids[5], date(2026, 6, 7),  45,  False),
        (prestamo_ids[6], date(2026, 6, 8),  60,  False),
    ]
    for pid, fecha, pag, comp in historial:
        session.execute(
            text("""INSERT INTO historial_lectura (prestamo_id, fecha_acceso, pagina_hasta, completado)
                    VALUES (:p, :f, :pg, :c)"""),
            {"p": pid, "f": fecha, "pg": pag, "c": comp},
        )
    session.flush()
    logger.info("Insertados %d préstamos con historial de lectura", len(data))


def insertar_embeddings_texto(session, recurso_ids: list[int]):
    logger.info("Generando embeddings de texto con all-MiniLM-L6-v2...")
    count = 0
    for i, chunks in enumerate(TEXT_CHUNKS):
        rid = recurso_ids[i]
        if chunks:
            vectores = get_embeddings_batch(chunks)
            for cid, (txt, vec) in enumerate(zip(chunks, vectores), start=1):
                session.execute(
                    text("""INSERT INTO embeddings_texto (recurso_id, chunk_id, chunk_texto, estrategia_chunking, vector_texto_384)
                            VALUES (:r, :cid, :txt, 'sentence_aware', CAST(:vec AS vector))"""),
                    {"r": rid, "cid": cid, "txt": txt, "vec": str(vec)},
                )
                count += 1
        vec_tit = get_embedding(RECURSOS[i]["titulo"])
        session.execute(
            text("""INSERT INTO embeddings_texto (recurso_id, chunk_id, chunk_texto, estrategia_chunking, vector_texto_384)
                    VALUES (:r, 0, :txt, 'title', CAST(:vec AS vector))"""),
            {"r": rid, "txt": RECURSOS[i]["titulo"], "vec": str(vec_tit)},
        )
        count += 1
    session.flush()
    logger.info("Insertados %d embeddings de texto con vectores reales", count)


def insertar_embeddings_imagen(session, imagen_ids: list[int]):
    logger.info("Insertando embeddings de imagen como placeholders (ceros)...")
    logger.info("Para generar vectores reales, ejecuta: python scripts/generar_embeddings_seed.py")
    count = 0
    for img_id in imagen_ids:
        session.execute(
            text("INSERT INTO embeddings_imagen (imagen_id, vector_embedding) VALUES (:id, array_fill(0.0, ARRAY[512])::vector)"),
            {"id": img_id},
        )
        count += 1
    session.flush()
    logger.info("Insertados %d embeddings de imagen (placeholders)", count)


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def limpiar_run_anterior(session):
    logger.warning("Limpiando datos de corridas anteriores...")

    titulos = [r["titulo"] for r in RECURSOS]
    nombres_autores = [a[0] for a in AUTORES_NUEVOS]

    session.execute(text("DELETE FROM embeddings_imagen WHERE imagen_id IN (SELECT id FROM imagenes_recurso WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits)))"), {"tits": titulos})
    session.execute(text("DELETE FROM embeddings_texto WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM historial_lectura WHERE prestamo_id IN (SELECT id FROM prestamos WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits)))"), {"tits": titulos})
    session.execute(text("DELETE FROM prestamos WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM reseñas WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM coleccion_recursos WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM imagenes_recurso WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM recurso_tags WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM recurso_generos WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM recurso_autores WHERE recurso_id IN (SELECT id FROM recursos WHERE titulo = ANY(:tits))"), {"tits": titulos})
    session.execute(text("DELETE FROM recursos WHERE titulo = ANY(:tits)"), {"tits": titulos})
    session.execute(text("DELETE FROM autores WHERE nombre = ANY(:nombres)"), {"nombres": nombres_autores})

    for seq in ["recursos_id_seq", "autores_id_seq", "imagenes_recurso_id_seq", "prestamos_id_seq", "reseñas_id_seq"]:
        session.execute(text(f"SELECT setval('{seq}', COALESCE((SELECT MAX(id) FROM {seq.split('_id_seq')[0]}), 0))"))

    session.flush()
    logger.warning("Limpieza completada. Secuencias reiniciadas.")


def main():
    logger.info("=== Iniciando inserción de 20 recursos adicionales ===")

    with get_session() as session:
        limpiar_run_anterior(session)
        autor_ids = insertar_autores(session)
        gen_ids = insertar_generos(session)
        tag_ids = insertar_tags(session)
        recurso_ids = insertar_recursos(session)

        insertar_relaciones(session, "recurso_autores", "recurso_id", "autor_id",
                            recurso_ids, AUTORES_POR_RECURSO, autor_ids)
        insertar_relaciones(session, "recurso_generos", "recurso_id", "genero_id",
                            recurso_ids, GENEROS_POR_RECURSO, gen_ids)
        insertar_relaciones(session, "recurso_tags", "recurso_id", "tag_id",
                            recurso_ids, TAGS_POR_RECURSO, tag_ids)

        imagen_ids = insertar_imagenes(session, recurso_ids)
        insertar_coleccion_recursos(session, recurso_ids)
        insertar_resenas(session, recurso_ids)
        insertar_prestamos(session, recurso_ids)

        insertar_embeddings_texto(session, recurso_ids)
        insertar_embeddings_imagen(session, imagen_ids)

    logger.info("=== Inserción completada exitosamente ===")
    logger.info("Se añadieron %d recursos con todas sus relaciones y embeddings reales.", len(RECURSOS))


if __name__ == "__main__":
    main()
