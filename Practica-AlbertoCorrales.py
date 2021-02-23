import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import spacy
from textblob import TextBlob

nlp_es = spacy.load('es_core_news_sm')
nlp_en = spacy.load('en_core_web_lg')
# Cargamos las librerías preentrenadas de spacy tanto para inglés como para español


# Primero cargamos los textos y les realizamos una pequeña visualización.
folder = "../CorpusNoticiasTexto"
# Empty list to hold text documents.
# En la carpeta Corpus NoticiasTexto encontramos los 24 textos que queremos procesar y clasificar
# en distintos cluster según su temática.
# Para ello recorremos la carpeta y sacamos los textos pudiéndolos visualizar.
listing = os.listdir(folder)
t_en = []
t_esp = []
for count, file in enumerate(listing, start=1):
    if file.endswith(".txt"):
        url = folder + "/" + file
        f = open(url, encoding="utf-8", errors='ignore');
        raw = f.read()
        f.close()

        #print(f'File nº {count}: {file} cuyo contenido es: \n{raw}')

        text = TextBlob(raw)
        if text.detect_language() == 'en':
            t_en.append(raw)
        if text.detect_language() == 'es':
            t_esp.append(raw)


# Se pueden apreciar diversos saltos de línea en los textos
# Así nos encontramos con dos listas una para textos en español y otra para textos en inglés.

print(f'Nº textos inglés: {len(t_en)} \nNº textos español: {len(t_esp)}')

# Se empieza a realizar distintos preprocesados.

# PREPROCESADO TEXTOS EN ESPAÑOL

def cluster_texts_es(texts, clustersNumber, distance):
    #Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(texts)
    print("Created a collection of", len(collection), "terms.")

    #get a list of unique terms
    unique_terms = list(set(collection))
    print("Unique terms found: ", len(unique_terms))

    ### And here we actually call the function and create our array of vectors.
    vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
    print("Vectors created.")
    # vectorizer = TfidfVectorizer()
    # x = vectorizer.fit_transform(tokens)
    # print(X)
    # initialize the clusterer
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

# INICIO PREPROCESAMIENTO ESPAÑOL




docs_es = list(nlp_es.pipe(t_esp))
# Aquí se produce el preprocesado de todos los textos en español.
tokens = []

for count, doc in enumerate(docs_es, start=1):
    print(f'DOCUMENTO Nº {count} en español')
    # Tokenizamos los textos para encontrar los distintos tokens existentes en cada uno de ellos.
    # Añadimos los tokens de cada texto a una nueva lista conformando una lista de listas
    # Evitamos los saltos de línea así como los posibles espacios en nuestros tokens.
    # Ahora evitamos añadir los signos de puntuación a nuestra nueva lista.
    # Tampoco añadimos los stopwords a nuestra lista de palabras.
    # Evitando los stopwords y los signos de puntuación sacamos los lemas de las distintas palabras.
    # Añadimos a los verbos y adjetivos los Nombres para demostrar que son las categorías
    # más importantes a la hora de clasificar los textos en clusters.
    tokens.append([token.lemma_.strip() for token in doc if not token.is_punct and not token.is_stop
                   and (token.pos_ == 'PROPN' or token.pos_ == 'NOUN' or token.pos_ == 'ADJ'
                        or token.pos_ == 'VBO')])




#print(tokens)

# Si lo evaluamos sobre la métrica dada debemos agrupar nuestros textos en español en 6 grupos.
# Con la distancia coseno podemos evaluar dicha métrica
distanceFunction ="cosine"
#distanceFunction = "euclidean"
test_esp = cluster_texts_es(tokens,6,distanceFunction)
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



# nlp_es.vocab['a'].is_stop = True
# nlp_es.vocab['y'].is_stop = True
# nlp_es.vocab['o'].is_stop = True


# EXPRESIONES REGULARES.
# Buscar palabras que al realizar su extracción con Beautiful Soup se han quedado unidas cuando deberían estar
# separadas por un punto.

# expression = r"[a-z]+[A-Z]{1}[a-z\u00E0-\u00FC]"
# # substitucion = r'\g<0>' # Se queda con todo el texto encontrado
#
# substitucion = r'\g<0>' # Se queda sólo con el patrón encontrado
# #substitucion = r"[a-z]+[.][A-Z]{1}"
# # Existen tokens que están mezclados con otros. Por ejemplo mesTe, disponiblesLa
# texto = re.sub(expression, substitucion, texto)
#
# # print(texto)

# expression = r"[a-z]+[A-Z]{1}"
# # Existen tokens que están mezclados con otros. Por ejemplo mesTe, disponiblesLa
# for token in doc:
#     if re.search(expression, token.text):
#         patron = re.split("\s+|([A-Z][a-z\u00E0-\u00FC]+)", token.text)
#         print(patron)

# x = re.finditer(expression, tokens)  # Busca el patron dentro del texto
# print(x.span())