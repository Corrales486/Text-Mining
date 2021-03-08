import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
from textblob import TextBlob

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
                print(f'Texto número {count}\n{raw}')
                t_en.append(raw)
            if text_det.detect_language() == 'es':
                t_esp.append(raw)
                print(f'Texto número {count}\n{raw}')


    # Así nos encontramos con dos listas una para textos en español y otra para textos en inglés.
    print(f'Nº textos inglés: {len(t_en)} \nNº textos español: {len(t_esp)}')
    # Donde se cuentan con 16 textos en español y 8 textos en inglés.
    print("Podemos acceder a los textos en español t_esp[0] - t_esp[" + str(len(t_esp) - 1) + "]")
    print("They can be accessed using t_en[0] - t_en[" + str(len(t_en) - 1) + "]")

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





