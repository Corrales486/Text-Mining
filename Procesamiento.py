import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import os

folder = "CorpusNoticiasTexto"
# Empty list to hold text documents.
texts_raw = []

listing = os.listdir(folder)
for file in listing:
    print("File: ",file)
    if file.endswith(".txt"):
        url = folder+"/"+file
        f = open(url,encoding="latin-1");
        raw = f.read()
        f.close()

    texts_raw.append(raw)

# Aquí en texts_raw tengo los textos separados, pero abiertos para poder procesarlos
# Para coger cada texto debemos recorrer texts_raw

print(texts_raw)



# Abrimos el pipeline de spacy con nlp cargando sólamente librería en
nlp = spacy.load('en_core_web_lg')



# Sacamos las stop words del paquete de spacy
stop_words = STOP_WORDS
puntuactions = string.punctuation
# CARGAMOS LOS PAQUETES DONDE ESTÁN ALMACENADOS TANTO LAS STOP WORDS COMO LOS SIGNOS DE PUNTUACIÓN.

# Tener en cuenta que si llamamos a token.text lo que estamos haciendo es llamar a los strings.
# Es lo último que tenemos que hacer coger los strings ya que pierden las dependencias.


# Para cada texto realizamos un doc. Lo podemos realizar con:
docs = list(nlp.pipe(texts_raw))
print(len(docs))

# Ya hemos procesado cada uno de los textos podemos iterar en ellos y realizar distintos procesamientos
sentence_doc = []
token_nostop = []
for count, doc in enumerate(docs, start=1):
    sentences = list(doc.sents)
    print(f'El texto número {count} cuenta con {len(sentences)} frases')

    sentence_doc.append(sentences)

    for token in doc:
        # Existe distintos atributos de nuestros token para los que podemos realizar procesamiento de nuestro texto
        # print(token, token.idx, token.text_with_ws, token.is_alpha, token.is_punct, token.is_space, token.shape_, token.is_stop)
        # Imprimen la palabra, el índice de inicio, token con espacion final, si es alfanuméric, si es signo de puntuación,
        # Si es un espacio, La forma de la palabra y si es stopword o no
        # Con todos estos atributos de los tokens podemos realizar un importante procesado de nuestro texto
        # Eliminando tanto los espacios como las stopword ya que no son significativas y distorsionan el análisis de frecuencias
        # del texto
        # Podemos eliminar las stop word
        if not token.is_stop and not token.is_punct:
            print(token, ', LEMA:', token.lemma_)
            # Estamos usando el diccionario en inglés. Por tanto no tiene mucho sentido analizar los lemas o los stopwords con este diccionario.
            # Una posible solución puede ser dividir al principio los textos en Español y en inglés. Posteriormente realizar una unión de textos.
            # Probar si funciona o si no funciona probar a traducir los textos al inglés con alguna librería como text bloob o alguna particular
            # Lematización permite evitar palabras duplicadas que tienen significados similares.

            print(token, token.tag_, token.pos_, spacy.explain(token.tag_))
            # Encontramos que tag_ es de grano fino
            # Encontramos que pos_ es de grano fordo
            # Podemos extraer distintas palabras según lo que necesitemos. Solo NOUN ADJ VERBOS
            # Objetivo reducir al máximo el vocabulario más importante
            if token.pos_ == 'NOUN' or token.pos_ == 'ADJ' or token.pos_ == 'VBO':
                print(token, token.pos_, token.tag_)

    tokens = [token for token in doc if not token.is_stop and not token.is_punct] # and not token.is_punct
    # Si no llamamos al atributo .text los tokens siguen teniendo los distintos atributos.
    print(tokens)
# Lo que hacemos es crear una lista con los tokens que no sean stop words. Podría hacer lo mismo con los signos de puntuación
    for x in tokens:
        print('FUNCIONA?', x.is_punct)

    token_nostop.append(tokens)

    # También puedo sacar el lema de la palabra

    for doc in docs:
        for ent in doc.ents:
            print(ent.text, ent.star_char, ent.end_char, ent.label_, spacy.explain(ent.label_), ent.iob_)

# Tambien cuenta con la definicion iob cual es inside outside y begin

# # NAMED ENTITY RECOGNITION
# Es el proceso de localizar entidades nombradas de texto y luego clasificarlas en categorías como pueden ser
# personas organizaciones, etc
# Se pueden utilizar para mejorar la busqueda de las keyword


print(sentence_doc)


# Estoy guardando todas las oraciones de cada texto en un nuevo documento.
# Cada documento cuenta con sus frases dentro de una lista.
# Cada frase a su vez es una lista. Lista de listas
# Tendrá tantas listas como textos donde almacena todas las frases que a su vez son listas
# El delimitador de las oraciones es el . aunque se podría cambiar
# Las frases sacadas no tienen mucho sentido

# Podemos tokenizar las frases. Nos permite identificar las unidades básicas en el texto
# Spacy nos permite realizar distintas actuaciones con los atributos de los tokens. Nos permite saber cuando una palabra
# Es un signo de puntuación, cuando es un dígito



# WORD FRECUENCY
# Para analizar la frecuencia de las palabras es necesario quitar signos de puntuación y stopwords ya que si no
# serán las palabras que más aparezcan


# POS
# CON LOS TOKENS SACADOS TAMBIÉN PODEMOS REALIZAR POS


# Se puede representar las dependency label de las frases.
# from spacy import displacy
# doc = nlp(text)
# displacy.serve(doc, style='dep')


# PREPROCESSING FUNCTIONS
# Crear una función que pasándole un token le realicé un conjunto de procesamientos
# Se puede coger lemma, minúsculas


# También tenemos la opcion de realizar matcher


# DEPENDENCY UTILIZANDO SPACY
# Consiste en extraer la estructura semántica de una oración. Dependencia entre las palabras principales y sus dependientes
# El nodo de la frase es aquella parte que no tiene dependencia normalmente suelen ser los verbos
# Se realiza con los atributos utilizados de los tokens.
# token.dep_ token.head.text
# displacy.serve(doc, style='dep')


# LO QUE HACEMOS ES CREAR UN PARSER
# COGEMOS UN PARSEADOR
parser = English()

# Creamos una función que tokenice nuestro texto
def tokenizer(sentence):
    # Tokens utilizado para crear documentos con anotaciones linguísticas
    tokens = parser(sentence)
    # Me parsee las frases de mi texto.


    # Podemos lematizar cada token
    tokens = [word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in tokens]

    # Quitamos los stop words.
    tokens = [word for word in tokens if word not in STOP_WORDS and word not in puntuactions]

    return tokens





#docs = list(nlp.pipe(texts_raw))
# Empezamos realizando un procesado nlp con el pipe que crea los objetos tipo doc.
# Los cuales de acuerdo a su pipeline nos devuelve con el tagger el pos_ el token.text
# Nos devuelve también las dependency labels. y las EN. cON DOC.ent ent.text ent.label_

            #
            # tokens = nltk.word_tokenize(raw)
            # text = nltk.Text(tokens)
            # texts.append(text)

print("Prepared ", len(texts_raw), " documents...")
print("They can be accessed using texts[0] - texts[" + str(len(texts_raw)-1) + "]")


# Para clasificar textos necesitamos convertir el texto en algo que se representa numéricamente
# Hay diferentes formas de convertir texto en una matriz de ocurrencia dado un documento
# Se centra en si la palabra ocurre o no en el documento y genera una matriz qu epodría verse como una BOW matrix


# GENERAR BOW MATRIX utilizando Count Vectorizer
# Definir en rango de Ngram que queremos

# bow_vector = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

# Ngram son combinaciones de palabras adyacentes en un texto dado Donde n es el nº de palabras que se incluyen en los tokens
# Unigrams son combinacions de palabras una a una
# Bigrams sería secuencias de dos palabras a la vez
# NGRAM RANGE ES UN PARÁMETRO.

# TAMBIÉN QUEREMOS UTILIZAR TF-IDF. ES UNA FORMA DE NORMALIZAR NUESTRO VECTOR DE PALABRAS
# BUSCANDO LA FREQ DE CADA PALABRA EN COMPARACIÓN CON LA FRECUENCIA EN EL DOCUMENTO
# COMO DE IMPORTANTE ES UN TÉRMINO PARTICULAR (CUANTAS VECES APARECE Y COMO DE FRECUENTE ES QUE APAREZCA EL MISMO TÉRMINO
# EN OTROS DOCUMENTOS.

# tfidf_vector = TfidfVectorizer(tokenizer = spacy_tokenizer)

