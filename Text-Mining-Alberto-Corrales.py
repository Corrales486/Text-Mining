import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from textblob import TextBlob
from spacy.matcher import Matcher
import unicodedata

# Se añade la librería textblob para identificar el idioma de cada texto y poder dividirlos.

import spacy

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
            f.close()
# Evaluamos el idioma de cada texto y se añade a cada lista según se identifique.
# También visualizamos los textos para entender su estructura y su temática.
            text_det = TextBlob(raw)
            if text_det.detect_language() == 'en':
                #print(f'Texto número {count}\n{raw}')
                t_en.append(raw)
            if text_det.detect_language() == 'es':
                t_esp.append(raw)
                #print(f'Texto número {count}\n{raw}')


    # Así nos encontramos con dos listas una para textos en español y otra para textos en inglés.
    print(f'Nº textos inglés: {len(t_en)} \nNº textos español: {len(t_esp)}')
    # Donde se cuentan con 16 textos en español y 8 textos en inglés.
    print("Podemos acceder a los textos en español t_esp[0] - t_esp[" + str(len(t_esp) - 1) + "]")
    print("They can be accessed using t_en[0] - t_en[" + str(len(t_en) - 1) + "]")

# 3. PROCESAMIENTOS TEXTOS EN ESPAÑOL.
# Se utilizan las funciones originales del código para calcular el TF así como realizar el clustering.

def cluster_texts_es(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Se ha creado una colección de", len(collection), "términos.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Términos únicos textos en español: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
    print("Vectores creados.")

    clusterer = AgglomerativeClustering(n_clusters=clustersNumber, linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf

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
    text_esp.append([token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
                       and (token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ')])

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
print("reference: ", reference_esp)

# Evaluation
print("rand_score: ", adjusted_rand_score(reference_esp,test_esp))



'''
    texts = t_esp + t_en
    print(texts)
    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")


def cluster_texts(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
    print("Vectors created.")

    # initialize the clusterer
    clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
                                      linkage="average", affinity=distanceFunction)
    clusters = clusterer.fit_predict(vectors)

    return clusters

# Function to create a TF vector for one document. For each of
# our unique words, we have a feature which is the tf for that word
# in the current document
def TF(document, unique_terms, collection):
    word_tf = []
    for word in unique_terms:
        word_tf.append(collection.tf(word, document))
    return word_tf


distanceFunction ="cosine"
#distanceFunction = "euclidean"
test = cluster_texts(texts,7,distanceFunction)
print("test: ", test)
# Gold Standard
# 0 activista Loujain
# 1 accidente Alonso
# 2 Muro frontera México
# 3 Icautación cocaína
# 4 Rescate cubanos
# 5 Gobierno de Italia
# 6 Elecciones Ecuador
reference =[0, 0, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 3, 0, 0, 0, 4, 4, 0, 2, 2]
print("reference: ", reference)

# Evaluation
print("rand_score: ", adjusted_rand_score(reference,test))

'''





