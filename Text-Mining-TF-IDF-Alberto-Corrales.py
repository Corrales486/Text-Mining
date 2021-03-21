import re, pprint, os, numpy
import nltk
import pandas as pd
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.cluster import adjusted_rand_score
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from google_trans_new import google_translator

# Se añade la librería textblob para identificar el idioma de cada texto y poder dividirlos.
import spacy
from spacy.pipeline import EntityRuler

# Se añade la librería spacy para realizar el procesamiento de los textos.
# Se llama a los paquetes prentrenados de spacy tanto en español como en inglés.

nlp_es = spacy.load('es_core_news_lg')
nlp_en = spacy.load('en_core_web_lg')

# 2. RECONOCIMIENTO TEXTOS.
if __name__ == "__main__":
# Primero cargamos los textos y les realizamos una pequeña visualización para obtener una idea de su temática.
    folder = "../CorpusNoticiasTexto_ANSI"

# En la carpeta Corpus NoticiasTexto_ANSI encontramos los 24 textos que queremos procesar y clasificar
# Para ello recorremos la carpeta y sacamos los textos pudiéndolos visualizar.

    listing = os.listdir(folder)
# Se crean tres listas para guardar por un lado los textos en inglés, por otro lado los textos en español
# y por último los textos en conjunto.
    t_en = []
    t_esp = []
    texts = []
# Recorremos nuestra de carpeta de textos
    for count, file in enumerate(listing, start=1):
        if file.endswith(".txt"):
            url = folder + "/" + file
            f = open(url, encoding="latin-1", errors='ignore');
            raw = f.read()
            raw = re.sub('\s', ' ', raw)
            # Se apreció que al sacar las EN en Inglés surgían diversos problemas
            # Ya que introducía como EN saltos de linea (\n) o \t
            # Por ello se deciden eliminar para evitar ruido.
            f.close()
            # Evaluamos el idioma de cada texto y se añade a cada lista según se identifique.
            # Se utiliza excepciones de Python por si la librería Textblob no funciona o da fallo
            # utilizar la librería google translator
            try:
                text_det = TextBlob(raw)
                if text_det.detect_language() == 'en':
                    #print(f'Texto número {count}\n{raw}')
                    print((text_det.detect_language(), url))
                    t_en.append(raw)
                if text_det.detect_language() == 'es':
                    print((text_det.detect_language(), url))
                    t_esp.append(raw)
                    #print(f'Texto número {count}\n{raw}')
            except ValueError:
            # Se añade una nueva forma de detección de nuestros textos por si acaso la
            # librería TextBlob falla.
                detector = google_translator()
                try:
                    detect_result = detector.detect(raw)
                    if detect_result[0] == 'en':
                        print('Ingles:', url)
                        # print(f'Texto número {count}\n{raw}')
                        t_en.append(raw)
                    elif detect_result[0] == 'es':
                        print('Español:', url)
                        t_esp.append(raw)
                except TypeError:
                    if count == 2:
                        print('Español:', url)
                        t_esp.append(raw)
                    elif count == 3:
                        print('Inglés:', url)
                        t_en.append(raw)
                    elif count == 23:
                        print('Inglés:', url)
                        t_en.append(raw)

    # Así nos encontramos con dos listas una para textos en español y otra para textos en inglés.
    print(f'Nº textos inglés: {len(t_en)} \nNº textos español: {len(t_esp)}')
    # Donde se cuentan con 16 textos en español y 8 textos en inglés.
    print("Podemos acceder a los textos en español t_esp[0] - t_esp[" + str(len(t_esp) - 1) + "]")
    print("They can be accessed using t_en[0] - t_en[" + str(len(t_en) - 1) + "]")


def identity_tokenizer(text):
    return text

# 3. PROCESAMIENTOS TEXTOS EN ESPAÑOL.
# Se utilizan las funciones originales del código para calcular el TF así como realizar el clustering.

def cluster_texts_es(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Se ha creado una colección de", len(collection), "términos.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Términos únicos textos en español: ", len(unique_terms))

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=identity_tokenizer, lowercase=False)
    # Realizamos los vectores con los términos únicos de nuestros textos
    # Creados directamente con Tfidf
    vectors = tfidf.fit_transform(texts)
    vectors_dense = vectors.todense()
    print(pd.DataFrame(vectors_dense, columns=tfidf.vocabulary_))
    # Se crean 16 vectores (uno por cada texto en español) y cado uno de ellos con 194
    # palabras correspondientes a los términos únicos de nuestros textos.
    print("Vectors created.")

    clusterer = AgglomerativeClustering(n_clusters=clustersNumber, linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors_dense)

    return clusters


# 3.1.1 TOKENIZACIÓN ERRÓNEA
# Spacy permite modificar y añadir nuevos conjuntos de reglas a las ya existentes.
# Al haber caracteres especiales o palabras importantes que no habían sido correctamente identificadas.
# Se decide modificar este conjunto de reglas para mejorar la tokenización de nuestros textos.

prefixes_es = nlp_es.Defaults.prefixes + ("", "", "", "","¡", "¿", "«",'"',"-", "\.","Ecuador", "Draghi", "europeísta", "85%", "15,98%",
                                       "15%", "populista", "Internacional", "intereses", "disponibles", "Corresponsal", "empresas",)
prefix_regex_es = spacy.util.compile_prefix_regex(prefixes_es)
nlp_es.tokenizer.prefix_search = prefix_regex_es.search

suffixes = nlp_es.Defaults.suffixes + ["", "", "", "", "", "", "»", "/", '"', "    ", "-", "La", "Ausentismo", "Internacional",
                                       "Agentes","Mario", "Máster", "Traficar", "Descubre", 'Suscríbete', "Necesitamos", "Vivimos", "Te",]
suffix_regex = spacy.util.compile_suffix_regex(suffixes)
nlp_es.tokenizer.suffix_search = suffix_regex.search

infixes = nlp_es.Defaults.infixes + [r"\.", "", "", "\?","¿", '"', "!", "¡","   ", "\(", "\)", "\)-",]
infix_regex = spacy.util.compile_infix_regex(infixes)
nlp_es.tokenizer.infix_finditer = infix_regex.finditer

ruler_es = EntityRuler(nlp_es, overwrite_ents=True) # Sobrescriba el patrón que le indicamos
patterns_es = [{'label': 'PER', 'pattern':'Loujain al Hathloul'}, {'label': 'PER', 'pattern':'Loujain'},
               {'label': 'ORG', 'pattern':'Twitter'}, {'label': 'PER', 'pattern':'Lina'},
               {'label': 'PER', 'pattern':'Lina al Hathloul'}, {'label': 'PER', 'pattern':'Loujain Al Hathloul'},
               {'label': 'PER', 'pattern':'Amina'},{'label': 'ORG', 'pattern':'Polícia Nacional'},
               {'label': 'MISC', 'pattern':'IAB'}, {'label': 'MISC', 'pattern':'Usted'},
               {'label': 'ORG', 'pattern':'MotoGP'}, {'label': 'ORG', 'pattern':'Mundial de Formula 1'},
               {'label': 'ORG', 'pattern':'Alpine'}, {'label': 'PER', 'pattern':'Riccione'},
               {'label': 'ORG', 'pattern':'Formula 1'}, {'label': 'MISC', 'pattern':'Regístrate'},
               {'label': 'MISC', 'pattern':'William Hill'}, {'label': 'ORG', 'pattern':'Mundial de F1'},
               {'label': 'MISC', 'pattern':'Hazte'}, {'label': 'LOC', 'pattern':'Campo de Gibraltar'},
               {'label': 'ORG', 'pattern':'Guardia Civil'}, {'label': 'ORG', 'pattern':'Guardia Costera'},
               {'label': 'PER', 'pattern':'Riley Beecher'}, {'label': 'ORG', 'pattern':'BCE'},
               {'label': 'ORG', 'pattern':'Banco Central Europeo'}, {'label': 'ORG', 'pattern':'M5E'},
               {'label': 'PER', 'pattern':'Draghi'}, {'label': 'PER', 'pattern':'Mario Draghi'},
               {'label': 'PER', 'pattern':'Anna Buj'}, {'label': 'PER', 'pattern':'Yaku'},
               {'label': 'PER', 'pattern':'Yaku Pérez'}, {'label': 'PER', 'pattern':'Rafael Correa'},
               {'label': 'PER', 'pattern':'Andrés Arauz'}, {'label': 'PER', 'pattern':'Arauz'},
               {'label': 'PER', 'pattern':'Vito Crimi'}, {'label': 'MISC', 'pattern':'Cable News Network'},
               {'label': 'MISC', 'pattern':'CNN Sans'},{'label': 'MISC', 'pattern':'All Rights Reserved'},
               {'label': 'PER', 'pattern':'Correa'},{'label': 'LOC', 'pattern':'Ecuador'}]
# Especificamos los distintos patrones que queremos ejecutar
ruler_es.add_patterns(patterns_es)
# Le añadimos a nuestro Entity ruler los patrones que queremos buscar
nlp_es.add_pipe(ruler_es, before='ner') # Le decimos donde tiene que ir nuestro patrón

# 3.1 TOKENIZACIÓN.

docs_es = list(nlp_es.pipe(t_esp))
# Con esta función aplicamos el objeto nlp a todos los textos que conforman la lista de textos en
# español y les aplica el pipeline de procesamiento a todos ellos.
text_esp = []


for count, doc in enumerate(docs_es, start=1):
    # Recorremos todos los objetos doc creados, uno para cada texto
    # También se puede comprobar la correcta forma del Gold standard.

    #print(f'DOCUMENTO Nº {count} en español\n', doc)

    # Sacamos los tokens de cada texto y los añadimos a una lista creando para todos los textos en español
    # una lista de listas.
    # Evitamos los espacios a derecha e izquierda que puedan provocar que se determinen
    # como palabras diferentes palabras que realmente son iguales.

# 3.2 ELIMINACIÓN  SIGNOS DE PUNTUACIÓN

    # Ahora además de seguir tokenizando nuestros documentos se evita incluir en nuestro lista de tokens
    # aquellos que sean signos de puntuación

# 3.3 ELIMINACIÓN STOPWORDS
    # Se evita añadir las stopwords a nuestras lista de tokens ya que son términos muy frecuentes que pueden
    # provocar confusión a la hora de agrupar los textos en base a Term frequency

# 3.4 LEMATIZACIÓN
    # El proceso de lematización consiste en sustituir las palabras por su raíz.
    # Obteniendo todas aquellas palabras con la misma palabra raiz el mismo lema no importando
    # ni tiempos verbales, ni género ni número.

# 3.5 POS-TAGGING
    # Las diferentes categorías gramaticales de un texto proporcionan distinta cantidad de información sobre dicho documento.
    # Primero se comprueba si los verbos y adverbios recogen gran cantidad de información para
    # realizar el agrupamiento de nuestros textos aunque se comprueba que no.

    # text_esp.append([token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
    #                  and (token.pos_ == 'VERB' or token.pos_ == 'ADV')])

    # La segunda comprobación muestra la cantidad de información que aportan los adjetivos para realizar el agrupamiento.
    # comprobando como sí mejora el rendimiento de nuestro algoritmo
    # text_esp.append([token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
    #                   and (token.pos_ == 'ADJ')])

    # Por último se comprueba si son los sustantivos y nombres propios los que mayor cantidad de información
    # aportan para diferenciar la temática de nuestros documentos.
    # texto = [token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
    #  and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ')]
    #
    # text_esp.append(texto)


    # 3.6 WordCloud
    # Una manera de mostrar gráficamente que palabras son las más frecuentes en nuestros textos es
    # a través de las nubes de palabras las cuáles permiten observar si son los términos más informativos
    # los más frecuentes en nuestros documentos.

    #print(f'DOCUMENTO Nº {count} en español con {len(texto)} términos, siendo únicos {len(set(texto))}')
    # texto_join = ' '.join(texto)
    #
    # wordcloud = WordCloud(max_words=70, background_color="white", width=500,
    #            height=500,random_state=42).generate(texto_join)
    #
    # plt.figure(figsize=(30, 20))
    # plt.axis("off")
    # plt.imshow(wordcloud, interpolation='bilinear')
    # #plt.show()
    # wordcloud.to_file(f".\Texto-{count}español.png")

    #3.7 ENTIDADES NOMBRADAS
    # Las E.N son importantes a la hora de aporta información sobre la tematica de una noticia
    # ya que pueden contener datos acerca del suceso o evento ocurrido, de los protagonistas así
    # como de la localización donde se produjo.

    entidades = [ents.text.strip() for ents in doc.ents if ents.label_ != 'MISC']
    #print(f'DOCUMENTO Nº {count} en español con {len(entidades)} entidades, siendo únicas {len(set(entidades))}')

    text_esp.append(entidades)

print(text_esp)

# Al evaluar los 16 textos escritos en español únicamente, se deben dividir en 6 grupos.
distanceFunction ="cosine"

test_esp = cluster_texts_es(text_esp,6,distanceFunction)
print("test: ", test_esp)
# Gold Standard
# 0 activista Loujain
# 1 accidente Alonso
# 2 Icautación cocaína
# 3 Rescate cubanos
# 4 Gobierno de Italia
# 5 Elecciones Ecuador

reference_esp =[0, 0, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 2, 0, 3, 3]
print("Reference español: ", reference_esp)

# Evaluation
print("Rand_score español: ", adjusted_rand_score(reference_esp,test_esp))


# 4. PROCESAMIENTO TEXTOS INGLÉS
# Contamos con 8 textos en inglés los cuáles están almacenados en t_en

def cluster_texts_en(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=identity_tokenizer, lowercase=False)
    # Realizamos los vectores con los términos únicos de nuestros textos
    # Creados directamente con Tfidf
    vectors = tfidf.fit_transform(texts)
    vectors_dense = vectors.todense()
    print(pd.DataFrame(vectors_dense, columns=tfidf.vocabulary_))
    # Se crean 8 vectores (uno por cada texto en inglés) cado uno de ellos con 166 palabras
    # correspondientes a los términos únicos.
    print("Vectors created.")

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber, linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors_dense)

    return clusters


# 4.1.1 TOKENIZACIÓN MODIFICADA INGLÉS
# Spacy permite modificar y añadir nuevos conjuntos de reglas a las ya existentes.


prefixes_en = nlp_en.Defaults.prefixes + ('"',"Filters", "\.","Allowed", "", "", "", "", "", '"', "U.S.",)
prefix_regex_en = spacy.util.compile_prefix_regex(prefixes_en)
nlp_en.tokenizer.prefix_search = prefix_regex_en.search

suffixes_en = nlp_en.Defaults.suffixes + ("Biden", "President", "Most", "Clear", "Different", "", "", "/",
                                          "",'"', "-", "Emily","âre",)
suffix_regex_en = spacy.util.compile_suffix_regex(suffixes_en)
nlp_en.tokenizer.suffix_search = suffix_regex_en.search

infixes_en = nlp_en.Defaults.infixes + ( r"\.", "/", "","", "","", "", "",r"\?",)
infix_regex_en = spacy.util.compile_infix_regex(infixes_en)
nlp_en.tokenizer.infix_finditer = infix_regex_en.finditer

docs_en = list(nlp_en.pipe(t_en))

# Con esta función aplicamos el objeto nlp a todos los textos que conforman la lista de textos en
# inglés y se les aplica el pipeline de procesamiento a todos ellos.

text_en = []

# 4.1 TOKENIZACIÓN.

for count, doc in enumerate(docs_en, start=1):
    # Recorremos todos los objetos doc creados, uno para cada texto

    #print(f'DOCUMENTO Nº {count} en inglés')

    # Sacamos los tokens de cada texto y los añadimos a una lista creando para todos los textos en inglés
    # una lista de listas.
    # Evitamos los espacios a derecha e izquierda que puedan provocar que se determinen
    # como palabras diferentes palabras que realmente son iguales.

    # 4.3 ELIMINACIÓN SIGNOS DE PUNTUACIÓN Y STOPWORDS.

    # Ahora además de seguir tokenizando nuestros documentos se evita incluir en nuestro lista de tokens
    # aquellos que sean signos de puntuación y las stopwords a nuestras lista de tokens ya que son
    # términos muy frecuentes que pueden provocar confusión a la hora de agrupar los textos en base a TF

    # 4.4 LEMATIZACIÓN
    # El proceso de lematización consiste en sustituir las palabras por su raíz.
    # Obteniendo todas aquellas palabras con la misma palabra raiz el mismo lema no importando
    # ni tiempos verbales, ni género ni número.

    # 4.5 POS-TAGGING
    # Como vimos anteriormente las diferentes categorías gramaticales proporcionan distinta cantidad
    # de información siendo los Nombres propios y sustantivos los que mayor cantidad de información aportan
    # también los verbos y los adjetivo.

    # Al no poder mejorar la métrica de la agrupación se escogen los Nombres propios, sustantivos y adjetivos
    # en esta fase ya que fueron los que mejorar la agrupación en el procesamiento de textos en español.

    # texto_en = [token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
    #  and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ')]
    # text_en.append(texto_en)

    # 4.6 WordCloud
    # Una manera de mostrar gráficamente que palabras son las más frecuentes en nuestros textos es
    # a través de las nubes de palabras las cuáles permiten observar si son los términos más informativos
    # los más frecuentes en nuestros documentos.

    # print(f'DOCUMENTO Nº {count} en inglés con {len(texto_en)} términos, siendo únicos {len(set(texto_en))}')
    # texto_en_join = ' '.join(texto_en)
    #
    # wordcloud_en = WordCloud(max_words=70, background_color="white", width=500,
    #            height=500,random_state=42).generate(texto_en_join)
    #
    # plt.figure(figsize=(30, 20))
    # plt.axis("off")
    # plt.imshow(wordcloud_en, interpolation='bilinear')
    # #plt.show()
    # wordcloud_en.to_file(f".\Texto-{count}english.png")

    # 4.7 ENTIDADES NOMBRADAS
    # Las E.N son importantes a la hora de aporta información sobre la tematica de una noticia
    # ya que pueden contener datos acerca del suceso o evento ocurrido, de los protagonistas así
    # como de la localización donde se produjo.

    entidades = [ents.text.strip() for ents in doc.ents if not ents.text.isspace() and (ents.label_ == 'PERSON'
                                or ents.label_ == 'ORG' or ents.label_ == 'GPE' or ents.label_ == 'LOC')]
    #print(f'DOCUMENTO Nº {count} en inglés cuenta con {len(entidades)} entidades, siendo únicas {len(set(entidades))}')

    text_en.append(entidades)

print(text_en)

# Si lo evaluamos sobre la métrica dada debemos agrupar nuestros textos en inglés en 3 grupos.
# Con la distancia coseno podemos evaluar dicha métrica
distanceFunction ="cosine"
#distanceFunction = "euclidean"
test_en = cluster_texts_en(text_en,3,distanceFunction)
print("test: ", test_en)
# Gold Standard
# 0 accidente Alonso
# 1 activista Loujain
# 2 Wall Mexico

reference_en =[0, 0, 2, 1, 1, 1, 2, 2]
print("Reference english: ", reference_en)

# Evaluation
print("Rand_score english: ", adjusted_rand_score(reference_en,test_en))


# 5 PROCESAMIENTO TEXTOS CONJUNTOS.

# Se unen los términos sacados para los documento en los distintos procesamientos.
texts = text_esp + text_en
print(texts)
print("Prepared ", len(texts), " documents...")
# Se pueden acceder a las distintas listas de los documentos de la siguiente forma.
print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")


def cluster_texts(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a total collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms (All texts) found: ", len(unique_terms))

    tfidf = TfidfVectorizer(analyzer='word', tokenizer=identity_tokenizer, lowercase=False)
    # Realizamos los vectores con los términos únicos de nuestros textos
    # Creados directamente con Tfidf
    vectors = tfidf.fit_transform(texts)
    vectors_dense = vectors.todense()
    print(pd.DataFrame(vectors_dense, columns=tfidf.vocabulary_))
    # Se crean 24 vectores (uno por cada texto) cado uno de ellos con 349 palabras
    # correspondientes a los términos únicos.
    print("Vectors created.")

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                      linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors_dense)

    return clusters


distanceFunction ="cosine"

test = cluster_texts(texts,7,distanceFunction)
print("test: ", test)

#Gold Standard Español
# 0 activista Loujain
# 1 accidente Alonso
# 2 Icautación cocaína
# 3 Rescate cubanos
# 4 Gobierno de Italia
# 5 Elecciones Ecuador
# reference_esp =[0, 0, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 2, 0, 3, 3]

# Gold Standard English
# 0 accidente Alonso
# 1 activista Loujain
# 2 Wall Mexico
# reference_en =[0, 0, 2, 1, 1, 1, 2, 2]

# Gold Standar conjunto.
# 0 activista Loujain
# 1 accidente Alonso
# 2 Icautación cocaína
# 3 Rescate cubanos
# 4 Gobierno de Italia
# 5 Elecciones Ecuador
# 6 Wall Mexico
# Primero se coge el orden de los textos en español y se agregan los textos en inglés
# Modificando el número asignado para que coincidan según temática.

reference =[0, 0, 1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 2, 0, 3, 3, 1, 1, 6, 0, 0, 0, 6, 6]

print("Reference total: ", reference)

# Evaluation
print("Rand_score total: ", adjusted_rand_score(reference,test))

