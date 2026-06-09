-- ============================================================
-- BIBLIOTECA ATHENAEUM — Script completo de creación y datos
-- Compatible con Neon (PostgreSQL + pgvector)
-- Entrega 1 Corregida — Bases de Datos Relacionales 2026
-- ============================================================

-- Extensión vectorial (Neon ya la tiene disponible)
CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- SECCIÓN 1: TABLAS BASE (relacionales)
-- ============================================================

CREATE TABLE Usuarios (
    id              SERIAL PRIMARY KEY,
    nombre          VARCHAR(200) NOT NULL,
    email           VARCHAR(300) NOT NULL UNIQUE,
    contrasena_hash VARCHAR(512) NOT NULL,
    rol             VARCHAR(50)  NOT NULL DEFAULT 'lector'
                        CHECK (rol IN ('lector', 'bibliotecario', 'admin')),
    fecha_registro  TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE Recursos (
    id                 SERIAL PRIMARY KEY,
    titulo             VARCHAR(500) NOT NULL,
    fecha_publicacion  DATE,
    idioma             VARCHAR(50),
    tipo               VARCHAR(100) NOT NULL
                           CHECK (tipo IN ('libro', 'articulo', 'revista', 'video', 'mapa', 'fotografia', 'otro')),
    licencia           VARCHAR(100),
    descripcion        TEXT
);

CREATE TABLE Autores (
    id        SERIAL PRIMARY KEY,
    nombre    VARCHAR(300) NOT NULL,
    tipo      VARCHAR(50)  NOT NULL CHECK (tipo IN ('persona', 'organizacion')),
    biografia TEXT
);

CREATE TABLE Recurso_Autores (
    recurso_id INT NOT NULL REFERENCES Recursos(id) ON DELETE CASCADE,
    autor_id   INT NOT NULL REFERENCES Autores(id)  ON DELETE CASCADE,
    rol_autor  VARCHAR(100),
    PRIMARY KEY (recurso_id, autor_id)
);

CREATE TABLE Generos (
    id          SERIAL PRIMARY KEY,
    nombre      VARCHAR(150) NOT NULL UNIQUE,
    descripcion TEXT
);

CREATE TABLE Recurso_Generos (
    recurso_id INT NOT NULL REFERENCES Recursos(id) ON DELETE CASCADE,
    genero_id  INT NOT NULL REFERENCES Generos(id)  ON DELETE CASCADE,
    PRIMARY KEY (recurso_id, genero_id)
);

CREATE TABLE Tags (
    id     SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE Recurso_Tags (
    recurso_id INT NOT NULL REFERENCES Recursos(id) ON DELETE CASCADE,
    tag_id     INT NOT NULL REFERENCES Tags(id)     ON DELETE CASCADE,
    PRIMARY KEY (recurso_id, tag_id)
);

CREATE TABLE Imagenes_Recurso (
    id           SERIAL PRIMARY KEY,
    recurso_id   INT          NOT NULL REFERENCES Recursos(id) ON DELETE CASCADE,
    ruta_archivo VARCHAR(1000) NOT NULL,
    tipo_imagen  VARCHAR(100)  CHECK (tipo_imagen IN ('portada', 'ilustracion', 'mapa', 'fotografia', 'otro')),
    descripcion  TEXT
);

CREATE TABLE Colecciones (
    id             SERIAL PRIMARY KEY,
    usuario_id     INT          NOT NULL REFERENCES Usuarios(id) ON DELETE CASCADE,
    nombre         VARCHAR(300) NOT NULL,
    descripcion    TEXT,
    es_publica     BOOLEAN      NOT NULL DEFAULT FALSE,
    fecha_creacion TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE TABLE Coleccion_Recursos (
    coleccion_id   INT       NOT NULL REFERENCES Colecciones(id) ON DELETE CASCADE,
    recurso_id     INT       NOT NULL REFERENCES Recursos(id)    ON DELETE CASCADE,
    fecha_agregado TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (coleccion_id, recurso_id)
);

CREATE TABLE Reseñas (
    id          SERIAL PRIMARY KEY,
    usuario_id  INT      NOT NULL REFERENCES Usuarios(id)  ON DELETE CASCADE,
    recurso_id  INT      NOT NULL REFERENCES Recursos(id)  ON DELETE CASCADE,
    calificacion SMALLINT CHECK (calificacion BETWEEN 1 AND 5),
    texto       TEXT,
    fecha       TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE Prestamos (
    id                  SERIAL PRIMARY KEY,
    usuario_id          INT       NOT NULL REFERENCES Usuarios(id)  ON DELETE CASCADE,
    recurso_id          INT       NOT NULL REFERENCES Recursos(id)  ON DELETE CASCADE,
    fecha_inicio        TIMESTAMP NOT NULL DEFAULT NOW(),
    fecha_fin_esperada  TIMESTAMP NOT NULL,
    devuelto            BOOLEAN   NOT NULL DEFAULT FALSE,
    fecha_devolucion    TIMESTAMP
);

-- Índice parcial: solo un préstamo activo por usuario/recurso
CREATE UNIQUE INDEX idx_prestamo_activo_unico
    ON Prestamos (usuario_id, recurso_id)
    WHERE devuelto = FALSE;

CREATE TABLE Historial_Lectura (
    id           SERIAL PRIMARY KEY,
    prestamo_id  INT       NOT NULL REFERENCES Prestamos(id) ON DELETE CASCADE,
    fecha_acceso TIMESTAMP NOT NULL DEFAULT NOW(),
    pagina_hasta INT       CHECK (pagina_hasta >= 0),
    completado   BOOLEAN   NOT NULL DEFAULT FALSE
);

CREATE TABLE Consultas (
    id              SERIAL PRIMARY KEY,
    usuario_id      INT       NOT NULL REFERENCES Usuarios(id) ON DELETE CASCADE,
    texto_pregunta  TEXT      NOT NULL,
    fecha           TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE Resultados_Consulta (
    id                   SERIAL PRIMARY KEY,
    consulta_id          INT            NOT NULL REFERENCES Consultas(id) ON DELETE CASCADE,
    embedding_texto_id   INT            REFERENCES Embeddings_Texto(id)   ON DELETE CASCADE,
    embedding_imagen_id  INT            REFERENCES Embeddings_Imagen(id)  ON DELETE CASCADE,
    score_similitud      NUMERIC(6, 4)  NOT NULL CHECK (score_similitud BETWEEN 0 AND 1),
    posicion             SMALLINT       NOT NULL CHECK (posicion > 0),
    CONSTRAINT chk_exactamente_un_embedding CHECK (
        (embedding_texto_id  IS NOT NULL AND embedding_imagen_id IS NULL) OR
        (embedding_texto_id  IS NULL     AND embedding_imagen_id IS NOT NULL)
    )
);

CREATE TABLE Evaluaciones (
    id                SERIAL PRIMARY KEY,
    consulta_id       INT           NOT NULL UNIQUE REFERENCES Consultas(id) ON DELETE CASCADE,
    faithfulness      NUMERIC(5, 4) CHECK (faithfulness      BETWEEN 0 AND 1),
    answer_relevancy  NUMERIC(5, 4) CHECK (answer_relevancy  BETWEEN 0 AND 1),
    context_recall    NUMERIC(5, 4) CHECK (context_recall    BETWEEN 0 AND 1),
    fecha             TIMESTAMP     NOT NULL DEFAULT NOW()
);


-- ============================================================
-- SECCIÓN 2: TABLAS VECTORIALES (pgvector)
-- ============================================================

CREATE TABLE Embeddings_Texto (
    id                  SERIAL PRIMARY KEY,
    recurso_id          INT          NOT NULL REFERENCES Recursos(id) ON DELETE CASCADE,
    chunk_id            INT          NOT NULL,
    chunk_texto         TEXT         NOT NULL,
    estrategia_chunking VARCHAR(50)  NOT NULL
                            CHECK (estrategia_chunking IN ('fixed_size', 'sentence_aware', 'semantic', 'title')),
    vector_embedding    vector(384)  NOT NULL,
    modelo              VARCHAR(200) NOT NULL DEFAULT 'all-MiniLM-L6-v2'
);

CREATE INDEX idx_emb_texto_vector
    ON Embeddings_Texto
    USING hnsw (vector_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX idx_emb_texto_estrategia ON Embeddings_Texto (estrategia_chunking);
CREATE INDEX idx_emb_texto_recurso    ON Embeddings_Texto (recurso_id);

CREATE TABLE Embeddings_Imagen (
    id               SERIAL PRIMARY KEY,
    imagen_id        INT          NOT NULL UNIQUE REFERENCES Imagenes_Recurso(id) ON DELETE CASCADE,
    vector_embedding vector(512)  NOT NULL,
    modelo           VARCHAR(200) NOT NULL DEFAULT 'clip-vit-base-patch32'
);

CREATE INDEX idx_emb_imagen_vector
    ON Embeddings_Imagen
    USING hnsw (vector_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE TABLE Embeddings_Consulta (
    id               SERIAL PRIMARY KEY,
    consulta_id      INT          NOT NULL UNIQUE REFERENCES Consultas(id) ON DELETE CASCADE,
    vector_embedding vector(384)  NOT NULL,
    modelo           VARCHAR(200) NOT NULL DEFAULT 'all-MiniLM-L6-v2'
);

CREATE INDEX idx_emb_consulta_vector
    ON Embeddings_Consulta
    USING hnsw (vector_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);


-- ============================================================
-- SECCIÓN 3: DATOS DE PRUEBA (mínimo 10 filas por tabla)
-- ============================================================

-- ------------------------------------------------------------
-- Usuarios (12 filas)
-- ------------------------------------------------------------
INSERT INTO Usuarios (nombre, email, contrasena_hash, rol) VALUES
('Laura Martínez',      'laura.martinez@athenaeum.co',    '$2b$12$abc1hashedpassword001', 'lector'),
('Carlos Rodríguez',    'carlos.rodriguez@athenaeum.co',  '$2b$12$abc2hashedpassword002', 'lector'),
('Ana Gómez',           'ana.gomez@athenaeum.co',         '$2b$12$abc3hashedpassword003', 'bibliotecario'),
('Pedro Sánchez',       'pedro.sanchez@athenaeum.co',     '$2b$12$abc4hashedpassword004', 'lector'),
('María Fernández',     'maria.fernandez@athenaeum.co',   '$2b$12$abc5hashedpassword005', 'lector'),
('Jorge Ramírez',       'jorge.ramirez@athenaeum.co',     '$2b$12$abc6hashedpassword006', 'lector'),
('Sofía Torres',        'sofia.torres@athenaeum.co',      '$2b$12$abc7hashedpassword007', 'bibliotecario'),
('Andrés Vargas',       'andres.vargas@athenaeum.co',     '$2b$12$abc8hashedpassword008', 'lector'),
('Valentina Ríos',      'valentina.rios@athenaeum.co',    '$2b$12$abc9hashedpassword009', 'lector'),
('Sebastián Cruz',      'sebastian.cruz@athenaeum.co',    '$2b$12$abc10hashedpassword010','lector'),
('Isabela Herrera',     'isabela.herrera@athenaeum.co',   '$2b$12$abc11hashedpassword011','admin'),
('Tomás Molina',        'tomas.molina@athenaeum.co',      '$2b$12$abc12hashedpassword012','lector');

-- ------------------------------------------------------------
-- Recursos (12 filas)
-- ------------------------------------------------------------
INSERT INTO Recursos (titulo, fecha_publicacion, idioma, tipo, licencia, descripcion) VALUES
('El laberinto de la soledad',           '1950-01-01', 'español', 'libro',     'CC BY-SA', 'Ensayo de Octavio Paz que explora la identidad y la psicología del mexicano a través de la historia, la soledad y los mitos nacionales.'),
('Cien años de soledad',                 '1967-05-30', 'español', 'libro',     'Todos los derechos reservados', 'Novela fundacional del realismo mágico latinoamericano que narra la saga de la familia Buendía en el pueblo ficticio de Macondo.'),
('Inteligencia artificial en medicina',  '2022-03-15', 'español', 'articulo',  'CC BY', 'Revisión sistemática del uso de modelos de aprendizaje profundo para el diagnóstico clínico en radiología e histopatología entre 2018 y 2023.'),
('Atlas histórico de la Revolución Francesa', '1989-07-14', 'francés', 'mapa', 'Dominio público', 'Colección cartográfica de 48 mapas que documenta los movimientos militares, políticas territoriales y cambios administrativos de 1789 a 1799.'),
('Revista de Filosofía Latinoamericana', '2021-06-01', 'español', 'revista',   'CC BY-NC', 'Publicación académica semestral que recoge investigaciones sobre filosofía existencialista, fenomenología y pensamiento decolonial en América Latina.'),
('Cambio climático: evidencias y proyecciones', '2023-11-20', 'español', 'articulo', 'CC BY', 'Síntesis del sexto informe del IPCC adaptada para divulgación científica, con énfasis en proyecciones regionales para América del Sur.'),
('Historia de la arquitectura moderna',  '2005-09-10', 'español', 'libro',     'Todos los derechos reservados', 'Recorrido exhaustivo por los movimientos arquitectónicos del siglo XX, desde el funcionalismo bauhaus hasta el posmodernismo y el high-tech.'),
('Fotografías del Bogotazo 1948',        '2008-04-09', 'español', 'fotografia','Dominio público', 'Archivo fotográfico de 220 imágenes que documentan los eventos del 9 de abril de 1948 en Bogotá tras el asesinato de Jorge Eliécer Gaitán.'),
('Neurociencia cognitiva: fundamentos',  '2019-02-28', 'español', 'libro',     'CC BY-NC-SA', 'Introducción a los mecanismos neuronales de la percepción, la memoria, el lenguaje y la toma de decisiones, con revisión de casos clínicos.'),
('La condición humana',                  '1958-01-01', 'español', 'libro',     'Todos los derechos reservados', 'Análisis filosófico de Hannah Arendt sobre las tres actividades fundamentales de la vida activa: labor, trabajo y acción, en el contexto de la modernidad.'),
('Revista Iberoamericana de Ecología',   '2020-03-01', 'español', 'revista',   'CC BY', 'Revista científica de acceso abierto especializada en ecología de ecosistemas tropicales, biodiversidad y servicios ecosistémicos en Iberoamérica.'),
('El origen de las especies',            '1859-11-24', 'español', 'libro',     'Dominio público', 'Traducción anotada de la obra de Charles Darwin que expone la teoría de la evolución por selección natural, con notas comparativas de ediciones posteriores.');

-- ------------------------------------------------------------
-- Autores (12 filas)
-- ------------------------------------------------------------
INSERT INTO Autores (nombre, tipo, biografia) VALUES
('Octavio Paz',               'persona',       'Poeta y ensayista mexicano, Premio Nobel de Literatura 1990. Su obra explora la identidad latinoamericana, la modernidad y la tradición desde una perspectiva filosófica y poética.'),
('Gabriel García Márquez',    'persona',       'Novelista colombiano, Premio Nobel de Literatura 1982. Máximo exponente del realismo mágico y una de las figuras más influyentes de la literatura latinoamericana del siglo XX.'),
('Hannah Arendt',             'persona',       'Filósofa política alemana, teórica del totalitarismo y la condición humana. Su pensamiento aborda el mal radical, la banalidad del mal y los fundamentos de la democracia republicana.'),
('Charles Darwin',            'persona',       'Naturalista británico cuya teoría de la evolución por selección natural transformó radicalmente la biología, la filosofía y la teología del siglo XIX.'),
('Instituto IPCC',            'organizacion',  'Panel Intergubernamental de Expertos sobre el Cambio Climático, organismo científico de la ONU que evalúa la información científica relacionada con el cambio climático.'),
('María Cristina Suárez',     'persona',       'Historiadora colombiana especializada en historia urbana del siglo XX y violencia política en Colombia. Docente de la Universidad Nacional de Colombia.'),
('Jürgen Habermas',           'persona',       'Filósofo y sociólogo alemán, representante de la segunda generación de la Escuela de Frankfurt. Teórico de la acción comunicativa y el espacio público democrático.'),
('Elena Villanueva',          'persona',       'Neurocientífica española, investigadora del Instituto Cajal. Sus trabajos sobre memoria episódica y plasticidad sináptica han sido publicados en Nature Neuroscience y Cell.'),
('Academia Colombiana de Ciencias', 'organizacion', 'Corporación científica colombiana fundada en 1933, dedicada al fomento de la investigación en ciencias exactas, físicas, naturales y sociales.'),
('Walter Gropius',            'persona',       'Arquitecto alemán fundador de la Bauhaus en 1919. Pionero del modernismo arquitectónico y defensor de la integración del arte, artesanía y tecnología en el diseño.'),
('Rigoberta Menchú',          'persona',       'Activista indígena guatemalteca, Premio Nobel de la Paz 1992. Su obra testimonial documenta la represión de las comunidades indígenas mayas durante el conflicto armado interno de Guatemala.'),
('Jorge Eliécer Gaitán',      'persona',       'Político y abogado colombiano, líder del movimiento liberal popular. Su asesinato el 9 de abril de 1948 desencadenó el Bogotazo y marcó el inicio de La Violencia en Colombia.');

-- ------------------------------------------------------------
-- Recurso_Autores (12 filas)
-- ------------------------------------------------------------
INSERT INTO Recurso_Autores (recurso_id, autor_id, rol_autor) VALUES
(1,  1,  'autor'),
(2,  2,  'autor'),
(3,  5,  'autor'),
(4,  6,  'editor'),
(5,  7,  'autor'),
(6,  5,  'autor'),
(7,  10, 'autor'),
(8,  6,  'editor'),
(9,  8,  'autor'),
(10, 3,  'autor'),
(11, 9,  'editor'),
(12, 4,  'autor');

-- ------------------------------------------------------------
-- Géneros (12 filas)
-- ------------------------------------------------------------
INSERT INTO Generos (nombre, descripcion) VALUES
('Filosofía',           'Textos de pensamiento filosófico, metafísica, epistemología y ética.'),
('Literatura',          'Obras de ficción, poesía, ensayo literario y narrativa.'),
('Historia',            'Documentos, análisis y relatos sobre eventos históricos.'),
('Ciencia',             'Publicaciones de investigación científica y divulgación.'),
('Medio Ambiente',      'Estudios sobre ecología, cambio climático y biodiversidad.'),
('Medicina',            'Investigaciones clínicas, artículos biomédicos y salud pública.'),
('Arquitectura',        'Teoría, historia y práctica del diseño arquitectónico.'),
('Cartografía',         'Mapas, atlas y estudios geoespaciales.'),
('Fotografía',          'Archivos fotográficos y estudios sobre fotografía documental.'),
('Ciencias Sociales',   'Sociología, antropología, ciencias políticas y estudios culturales.'),
('Neurociencia',        'Estudios del sistema nervioso, comportamiento y cognición.'),
('Realismo Mágico',     'Corriente literaria latinoamericana que integra lo fantástico en lo cotidiano.');

-- ------------------------------------------------------------
-- Recurso_Generos (14 filas)
-- ------------------------------------------------------------
INSERT INTO Recurso_Generos (recurso_id, genero_id) VALUES
(1,  1),  -- El laberinto → Filosofía
(1,  10), -- El laberinto → Ciencias Sociales
(2,  2),  -- Cien años → Literatura
(2,  12), -- Cien años → Realismo Mágico
(3,  4),  -- IA medicina → Ciencia
(3,  6),  -- IA medicina → Medicina
(4,  3),  -- Atlas RF → Historia
(4,  8),  -- Atlas RF → Cartografía
(5,  1),  -- Revista Filosofía → Filosofía
(6,  4),  -- Cambio climático → Ciencia
(6,  5),  -- Cambio climático → Medio Ambiente
(7,  7),  -- Arquitectura moderna → Arquitectura
(9,  11), -- Neurociencia → Neurociencia
(10, 1);  -- La condición humana → Filosofía

-- ------------------------------------------------------------
-- Tags (12 filas)
-- ------------------------------------------------------------
INSERT INTO Tags (nombre) VALUES
('identidad-latinoamericana'),
('realismo-mágico'),
('cambio-climático'),
('inteligencia-artificial'),
('historia-colombia'),
('filosofía-existencialista'),
('open-access'),
('revolución-francesa'),
('neurología'),
('arquitectura-moderna'),
('darwin'),
('derechos-humanos');

-- ------------------------------------------------------------
-- Recurso_Tags (14 filas)
-- ------------------------------------------------------------
INSERT INTO Recurso_Tags (recurso_id, tag_id) VALUES
(1,  1),
(2,  2),
(2,  1),
(3,  4),
(4,  8),
(5,  6),
(5,  7),
(6,  3),
(6,  7),
(7,  10),
(8,  5),
(9,  9),
(10, 6),
(12, 11);

-- ------------------------------------------------------------
-- Imagenes_Recurso (12 filas)
-- ------------------------------------------------------------
INSERT INTO Imagenes_Recurso (recurso_id, ruta_archivo, tipo_imagen, descripcion) VALUES
(1,  '/media/portadas/laberinto_soledad.jpg',         'portada',     'Portada de la edición del Fondo de Cultura Económica de 1959, diseño minimalista con máscara prehispánica en tonos ocre.'),
(2,  '/media/portadas/cien_años_soledad.jpg',         'portada',     'Portada icónica de la primera edición de Editorial Sudamericana, Buenos Aires 1967, con ilustración de Vicente Rojo.'),
(3,  '/media/ilustraciones/ia_medicina_fig1.jpg',     'ilustracion', 'Diagrama de arquitectura de red neuronal convolucional aplicada a clasificación de imágenes histopatológicas.'),
(4,  '/media/mapas/revolucion_francesa_1789.jpg',     'mapa',        'Mapa de Francia en 1789 mostrando la división en provincias antes de la reorganización departamental de 1790.'),
(4,  '/media/mapas/revolucion_francesa_1793.jpg',     'mapa',        'Mapa de los movimientos del ejército durante el período del Terror, 1793-1794, con rutas de campaña y posiciones de frentes.'),
(7,  '/media/portadas/arquitectura_moderna.jpg',      'portada',     'Portada con fotografía de la Bauhaus de Dessau, edificio diseñado por Walter Gropius en 1926.'),
(8,  '/media/fotografias/bogotazo_001.jpg',           'fotografia',  'Fotografía de la carrera séptima de Bogotá el 9 de abril de 1948, con evidencia de incendios y disturbios en el centro histórico.'),
(8,  '/media/fotografias/bogotazo_002.jpg',           'fotografia',  'Retrato de Jorge Eliécer Gaitán tomado en 1947 durante un mitin en la Plaza de Bolívar, Bogotá.'),
(9,  '/media/ilustraciones/neurociencia_fig1.jpg',    'ilustracion', 'Diagrama del hipocampo y sus conexiones con la corteza entorrinal, relevante para la formación de memorias episódicas.'),
(10, '/media/portadas/condicion_humana.jpg',          'portada',     'Portada de la edición española de Paidós con imagen de escultura clásica representando la acción política.'),
(11, '/media/portadas/revista_ecologia.jpg',          'portada',     'Portada del volumen 2020 de la Revista Iberoamericana de Ecología con fotografía aérea de selva amazónica.'),
(12, '/media/portadas/origen_especies.jpg',           'portada',     'Portada de la traducción anotada con reproducción del árbol evolutivo dibujado por Darwin en su cuaderno B (1837).');

-- ------------------------------------------------------------
-- Colecciones (11 filas)
-- ------------------------------------------------------------
INSERT INTO Colecciones (usuario_id, nombre, descripcion, es_publica) VALUES
(1,  'Filosofía latinoamericana',     'Lecturas sobre identidad, existencialismo y pensamiento decolonial en América Latina.', TRUE),
(2,  'Ciencia y tecnología',          'Artículos y libros sobre IA, neurociencia y avances científicos recientes.', TRUE),
(3,  'Colección patrimonial Colombia','Recursos documentales sobre historia y cultura colombiana del siglo XX.', TRUE),
(4,  'Clásicos de la literatura',     'Obras fundamentales de la narrativa en lengua española.', FALSE),
(5,  'Cambio climático',              'Bibliografía esencial para entender la crisis ambiental actual.', TRUE),
(6,  'Historia cartográfica',         'Atlas y mapas históricos de Europa y América Latina.', FALSE),
(7,  'Arquitectura del siglo XX',     'Referentes del modernismo y posmodernismo arquitectónico.', TRUE),
(8,  'Mis favoritos',                 'Selección personal de recursos destacados de diversas disciplinas.', FALSE),
(9,  'Neurociencia básica',           'Introducción a las bases neurobiológicas del comportamiento.', TRUE),
(10, 'Lecturas de doctorado',         'Textos de referencia para investigación en filosofía política.', FALSE),
(11, 'Repositorio institucional',     'Colección oficial de recursos digitalizados de la Biblioteca Athenaeum.', TRUE);

-- ------------------------------------------------------------
-- Coleccion_Recursos (14 filas)
-- ------------------------------------------------------------
INSERT INTO Coleccion_Recursos (coleccion_id, recurso_id) VALUES
(1,  1),
(1,  5),
(1,  10),
(2,  3),
(2,  9),
(3,  8),
(3,  4),
(4,  2),
(4,  12),
(5,  6),
(5,  11),
(6,  4),
(7,  7),
(11, 1);

-- ------------------------------------------------------------
-- Reseñas (12 filas)
-- ------------------------------------------------------------
INSERT INTO Reseñas (usuario_id, recurso_id, calificacion, texto) VALUES
(1,  1,  5, 'Una obra imprescindible para entender la psicología colectiva de México. El análisis sobre la máscara y la soledad resulta vigente a más de setenta años de su publicación.'),
(2,  2,  5, 'Lectura abrumadora por su riqueza narrativa. García Márquez construye una mitología propia que trasciende lo latinoamericano para tocar lo universalmente humano.'),
(4,  3,  4, 'Revisión rigurosa del estado del arte. Especialmente valioso el apartado de redes convolucionales para diagnóstico radiológico. Le falta más énfasis en sesgos algorítmicos.'),
(5,  6,  5, 'Excelente síntesis del informe IPCC. Las proyecciones para América del Sur son alarmantes y bien documentadas. Recomendable para cualquier persona interesada en política ambiental.'),
(6,  7,  4, 'Tratamiento exhaustivo de la arquitectura moderna. El capítulo sobre Bauhaus es el más logrado. La sección posmoderna podría profundizar más en el contexto latinoamericano.'),
(7,  9,  5, 'Manual claro y bien estructurado para introducirse a la neurociencia cognitiva. Los casos clínicos integrados facilitan enormemente la comprensión de conceptos abstractos.'),
(8,  10, 5, 'La distinción entre labor, trabajo y acción sigue siendo uno de los marcos conceptuales más potentes para pensar la política contemporánea. Lectura obligatoria.'),
(9,  12, 4, 'La traducción anotada añade un valor pedagógico enorme. Las notas de pie de página que contrastan ediciones permiten seguir la evolución del pensamiento darwiniano.'),
(10, 2,  4, 'Segunda lectura aún más rica que la primera. La estructura cíclica del tiempo en Macondo adquiere nuevas dimensiones con cada relectura.'),
(1,  6,  5, 'Los mapas interactivos complementan perfectamente el análisis textual. Fundamental para cursos de ciencias ambientales y políticas públicas climáticas.'),
(3,  8,  5, 'Archivo fotográfico de un valor histórico incalculable. Las imágenes del Bogotazo son un testimonio directo de uno de los momentos más traumáticos de la historia colombiana.'),
(12, 5,  3, 'La revista tiene artículos de calidad desigual. Los ensayos sobre fenomenología son sobresalientes, pero los de pensamiento decolonial resultan repetitivos en sus argumentos.');

-- ------------------------------------------------------------
-- Prestamos (12 filas)
-- Nota: devuelto=TRUE en la mayoría para poder crear el índice
-- parcial sin conflictos en los registros activos
-- ------------------------------------------------------------
INSERT INTO Prestamos (usuario_id, recurso_id, fecha_inicio, fecha_fin_esperada, devuelto, fecha_devolucion) VALUES
(1,  1,  '2026-01-10 09:00', '2026-02-10 09:00', TRUE,  '2026-02-08 14:30'),
(2,  2,  '2026-01-15 10:00', '2026-02-15 10:00', TRUE,  '2026-02-12 11:00'),
(4,  3,  '2026-02-01 08:00', '2026-03-01 08:00', TRUE,  '2026-02-28 16:00'),
(5,  6,  '2026-02-10 11:00', '2026-03-10 11:00', TRUE,  '2026-03-05 10:00'),
(6,  7,  '2026-02-20 14:00', '2026-03-20 14:00', TRUE,  '2026-03-18 09:00'),
(7,  9,  '2026-03-01 09:00', '2026-04-01 09:00', TRUE,  '2026-03-30 15:00'),
(8,  10, '2026-03-05 10:00', '2026-04-05 10:00', TRUE,  '2026-04-03 12:00'),
(9,  12, '2026-03-15 13:00', '2026-04-15 13:00', TRUE,  '2026-04-14 11:00'),
(10, 2,  '2026-04-01 09:00', '2026-05-01 09:00', TRUE,  '2026-04-29 10:00'),
(1,  5,  '2026-04-10 10:00', '2026-05-10 10:00', FALSE, NULL),
(2,  6,  '2026-05-01 09:00', '2026-06-01 09:00', FALSE, NULL),
(12, 9,  '2026-05-15 11:00', '2026-06-15 11:00', FALSE, NULL);

-- ------------------------------------------------------------
-- Historial_Lectura (12 filas)
-- ------------------------------------------------------------
INSERT INTO Historial_Lectura (prestamo_id, fecha_acceso, pagina_hasta, completado) VALUES
(1,  '2026-01-10 20:00', 45,  FALSE),
(1,  '2026-01-15 21:00', 112, FALSE),
(1,  '2026-02-07 19:00', 198, TRUE),
(2,  '2026-01-17 22:00', 67,  FALSE),
(2,  '2026-02-10 20:00', 180, FALSE),
(3,  '2026-02-05 18:00', 22,  FALSE),
(3,  '2026-02-25 19:00', 58,  FALSE),
(4,  '2026-02-12 21:00', 90,  FALSE),
(5,  '2026-02-22 20:00', 140, FALSE),
(5,  '2026-03-15 19:00', 310, TRUE),
(7,  '2026-03-07 21:00', 75,  FALSE),
(8,  '2026-03-17 20:00', 200, TRUE);

-- ------------------------------------------------------------
-- Consultas (12 filas)
-- ------------------------------------------------------------
INSERT INTO Consultas (usuario_id, texto_pregunta) VALUES
(1,  '¿Qué libros abordan la memoria histórica en América Latina?'),
(2,  '¿Qué artículos científicos sobre inteligencia artificial en medicina se publicaron entre 2018 y 2024?'),
(4,  '¿Qué recursos tienen reseñas que mencionen narrativa compleja o estructura no lineal?'),
(5,  '¿Qué documentos abordan temas similares a la Revolución Industrial y las estructuras sociales urbanas?'),
(6,  '¿Qué libros en español con calificación mayor a 4 tratan sobre filosofía existencialista?'),
(7,  '¿Qué documentos contienen imágenes visualmente similares a esta fotografía de un mapa medieval?'),
(8,  '¿Qué recursos están relacionados con el contenido visual de esta portada de libro?'),
(9,  '¿Existen mapas históricos sobre la Revolución Francesa anteriores a 1950?'),
(10, '¿Quién es el autor de El laberinto de la soledad y qué otros recursos similares hay?'),
(1,  '¿Qué revistas científicas han publicado investigaciones sobre cambio climático?'),
(3,  '¿Cuáles son los recursos digitalizados sobre el Bogotazo disponibles en la biblioteca?'),
(12, '¿Qué libros de filosofía política están disponibles en idioma español?');

-- ------------------------------------------------------------
-- Evaluaciones (10 filas)
-- ------------------------------------------------------------
INSERT INTO Evaluaciones (consulta_id, faithfulness, answer_relevancy, context_recall) VALUES
(1,  0.8900, 0.9100, 0.8500),
(2,  0.9200, 0.8800, 0.9000),
(3,  0.7500, 0.8200, 0.7800),
(4,  0.8100, 0.8700, 0.8300),
(5,  0.8800, 0.9300, 0.8700),
(6,  0.7200, 0.7800, 0.6900),
(7,  0.7800, 0.8100, 0.7500),
(8,  0.8500, 0.8900, 0.8200),
(9,  0.9300, 0.9500, 0.9100),
(10, 0.8600, 0.9000, 0.8800);

-- ------------------------------------------------------------
-- Embeddings_Texto (12 filas — vectores placeholder de 384 dims)
-- En producción estos vectores se generan con all-MiniLM-L6-v2
-- Para el seed usamos vectores de ceros normalizados como placeholder
-- ------------------------------------------------------------
INSERT INTO Embeddings_Texto (recurso_id, chunk_id, chunk_texto, estrategia_chunking, vector_embedding) VALUES
(1, 1, 'El laberinto de la soledad explora la psicología del mexicano a través de la soledad, los mitos nacionales y la identidad histórica.',                              'sentence_aware', array_fill(0.0, ARRAY[384])::vector),
(1, 2, 'Octavio Paz analiza cómo la máscara social del mexicano encubre una soledad profunda heredada de la conquista y el mestizaje cultural.',                            'sentence_aware', array_fill(0.0, ARRAY[384])::vector),
(2, 1, 'Macondo nació como un pueblo de veinte casas de barro y cañabrava construidas a la orilla de un río de aguas diáfanas que se precipitaban por un lecho de piedras pulidas.',  'fixed_size',     array_fill(0.0, ARRAY[384])::vector),
(2, 2, 'La saga de los Buendía refleja el ciclo de la historia latinoamericana: guerras civiles, dictaduras, prosperidad efímera y olvido colectivo.',                       'fixed_size',     array_fill(0.0, ARRAY[384])::vector),
(3, 1, 'Las redes neuronales convolucionales han demostrado precisión diagnóstica comparable o superior a la de radiólogos expertos en la detección de tumores pulmonares.', 'sentence_aware', array_fill(0.0, ARRAY[384])::vector),
(3, 2, 'El aprendizaje profundo aplicado a histopatología permite clasificar automáticamente tejidos cancerosos con sensibilidad del 94% y especificidad del 97%.',           'sentence_aware', array_fill(0.0, ARRAY[384])::vector),
(6, 1, 'El sexto informe del IPCC confirma que la temperatura media global ha aumentado 1.1 grados Celsius desde la era preindustrial, con proyecciones de 2.7 grados para 2100.', 'semantic',   array_fill(0.0, ARRAY[384])::vector),
(9, 1, 'El hipocampo es fundamental para la consolidación de memorias episódicas, transformando recuerdos de corto plazo en representaciones estables de largo plazo.',      'sentence_aware', array_fill(0.0, ARRAY[384])::vector),
(10,1, 'Hannah Arendt distingue tres actividades: labor como ciclo biológico, trabajo como producción duradera, y acción como inicio espontáneo en la esfera pública.',      'fixed_size',     array_fill(0.0, ARRAY[384])::vector),
(12,1, 'Darwin propone que las especies evolucionan mediante variación hereditaria y selección natural, sobreviviendo los individuos mejor adaptados al entorno.',           'fixed_size',     array_fill(0.0, ARRAY[384])::vector),
(5, 1, 'El existencialismo latinoamericano adapta las categorías sartreanas y heideggerianas a la realidad colonial y poscolonial del subcontinente.',                       'semantic',       array_fill(0.0, ARRAY[384])::vector),
(7, 1, 'La Bauhaus integró por primera vez arte, artesanía e ingeniería en un programa educativo unificado, revolucionando el diseño industrial y arquitectónico del siglo XX.', 'sentence_aware', array_fill(0.0, ARRAY[384])::vector);

-- ------------------------------------------------------------
-- Embeddings_Imagen (12 filas — vectores placeholder de 512 dims)
-- En producción generados con CLIP ViT-B/32
-- ------------------------------------------------------------
INSERT INTO Embeddings_Imagen (imagen_id, vector_embedding) VALUES
(1,  array_fill(0.0, ARRAY[512])::vector),
(2,  array_fill(0.0, ARRAY[512])::vector),
(3,  array_fill(0.0, ARRAY[512])::vector),
(4,  array_fill(0.0, ARRAY[512])::vector),
(5,  array_fill(0.0, ARRAY[512])::vector),
(6,  array_fill(0.0, ARRAY[512])::vector),
(7,  array_fill(0.0, ARRAY[512])::vector),
(8,  array_fill(0.0, ARRAY[512])::vector),
(9,  array_fill(0.0, ARRAY[512])::vector),
(10, array_fill(0.0, ARRAY[512])::vector),
(11, array_fill(0.0, ARRAY[512])::vector),
(12, array_fill(0.0, ARRAY[512])::vector);

-- ------------------------------------------------------------
-- Embeddings_Consulta (12 filas — vectores placeholder de 384 dims)
-- ------------------------------------------------------------
INSERT INTO Embeddings_Consulta (consulta_id, vector_embedding) VALUES
(1,  array_fill(0.0, ARRAY[384])::vector),
(2,  array_fill(0.0, ARRAY[384])::vector),
(3,  array_fill(0.0, ARRAY[384])::vector),
(4,  array_fill(0.0, ARRAY[384])::vector),
(5,  array_fill(0.0, ARRAY[384])::vector),
(6,  array_fill(0.0, ARRAY[384])::vector),
(7,  array_fill(0.0, ARRAY[384])::vector),
(8,  array_fill(0.0, ARRAY[384])::vector),
(9,  array_fill(0.0, ARRAY[384])::vector),
(10, array_fill(0.0, ARRAY[384])::vector),
(11, array_fill(0.0, ARRAY[384])::vector),
(12, array_fill(0.0, ARRAY[384])::vector);

-- ------------------------------------------------------------
-- Resultados_Consulta (12 filas)
-- Mezcla de resultados texto e imagen para probar el CHECK constraint
-- ------------------------------------------------------------
INSERT INTO Resultados_Consulta (consulta_id, embedding_texto_id, embedding_imagen_id, score_similitud, posicion) VALUES
(1,  1,  NULL, 0.9241, 1),
(1,  11, NULL, 0.8873, 2),
(2,  5,  NULL, 0.9512, 1),
(2,  6,  NULL, 0.9103, 2),
(3,  3,  NULL, 0.8432, 1),
(5,  10, NULL, 0.8765, 1),
(6,  NULL, 4,  0.8902, 1),
(6,  NULL, 5,  0.8541, 2),
(7,  NULL, 2,  0.8234, 1),
(8,  NULL, 4,  0.9123, 1),
(9,  1,  NULL, 0.9634, 1),
(10, 7,  NULL, 0.9301, 1);

-- ============================================================
-- Fin del script
-- ============================================================
