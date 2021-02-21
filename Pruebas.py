import spacy
from spacy.matcher import PhraseMatcher
from spacy.lang.en.stop_words import STOP_WORDS

import re, pprint, os, numpy
import nltk
from sklearn.metrics.cluster import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
#from google_trans_new import google_translator

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline

nlp = spacy.load('en_core_web_lg')

# def translate_en(text):
#     lang = "en"
#     t = google_translator(timeout=5)
#     translate_text = t.translate(text, lang)
#     return translate_text
# Creamos la funcion de traducir y traducimos los textos antes de que se produzca ning?n cambio
# def cluster_texts(texts, clustersNumber, distance):
#     #Load the list of texts into a TextCollection object.
#     collection = nltk.TextCollection(texts)
#     print("Created a collection of", len(collection), "terms.")
#
#     #get a list of unique terms
#     unique_terms = list(set(collection))
#     print("Unique terms found: ", len(unique_terms))
#
#     ### And here we actually call the function and create our array of vectors.
#     vectors = [numpy.array(TF(f,unique_terms, collection)) for f in texts]
#     print("Vectors created.")
#
#     # initialize the clusterer
#     clusterer = AgglomerativeClustering(n_clusters=clustersNumber,
#                                       linkage="average", affinity=distanceFunction)
#     clusters = clusterer.fit_predict(vectors)
#
#     return clusters



if __name__ == "__main__":
    folder = "CorpusNoticiasTexto"
    # Empty list to hold text documents.
    texts = []

    listing = os.listdir(folder)
    # for text in listing:
    #     traducido = translate_en(text)
    #     print(traducido)
    #     texts.append(traducido)



    # Vale con esto ya he procesado los textos y los he pasado por el pipeline.
    # Ahora lo que voy a hacer es un preprocesado r?pido donde quito todos los signos de puntuaci?n
    # y todos los stopword.

    # No conventir a string hasta el final ya que pierde el contexto las dependency labels

    # Aqu? puedo sacar los primeros procesos


    for file in listing:
        print("File: ",file)
        if file.endswith(".txt"):
            url = folder+"/"+file
            f = open(url,encoding="latin-1");
            raw = f.read()
            f.close()
        texts.append(raw)

    print(texts)
    processed_text = []
    matcher = PhraseMatcher(nlp.vocab)
    print(nlp.Defaults.stop_words)
    docs = list(nlp.pipe(texts))
    '''
    if token.is_punct != True and 
     token.is_quote != True and 
     token.is_bracket != True and 
     token.is_currency != True and 
     token.is_digit != True])'''
    for doc in docs:
        for token in doc:
            if token.is_punct or token.is_stop:
                print(token.is_stop)
                continue
            print(token.text, token.pos_)

            lexema = nlp.vocab[f'{token}']
            print(token.text, token.lemma_, lexema)

            # Lecemas no tenemos POS NI DEPENDENCY LABELS
        for ent in doc.ents:
            print(ent.text, ent.label_)
        for sentences in doc.sents:
            print(sentences.text)

    # En texts puedo empezar a realizar el procesamiento
    # Aqu? est? abierto


    # tokens = nltk.word_tokenize(raw)
    # print(tokens)
    # text = nltk.Text(tokens)
    #
    # # Lo que tenga que hacer lo meter?a aqu?. Podr?a hacer una funci?n para hacer lo mismo con
    # # todos los textos
    # print(raw)
    # print(text)
    # texts.append(text)

    print("Prepared ", len(texts), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(texts) - 1) + "]")