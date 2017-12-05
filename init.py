import os
import gensim
import string
import pickle
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# separador das informações
def separador():
    print(90 * "-")

# lista dos documentos
docs_list = os.listdir('docs')
# sublista dos documentos
docs_list2 = docs_list[5:10]

#print(docs_list)
#print(len(docs_list))

# read docs
docs_raw = []
for item in docs_list2:
    file = os.path.join('docs', item)
    f_object = open(file, 'r', encoding="utf8")
    docs_raw.append(f_object.read())
    f_object.close()


print("Número de documentos: {}".format(len(docs_raw)))
separador()


# lê o arquivo com as stopwords
file = os.path.join('stopwords', 'pt.txt')
f_object = open(file, 'r', encoding="utf8")
f_str = f_object.readlines()
f_str_norm = []
for item in f_str:
    tmp = item.strip('\n')
    f_str_norm.append(tmp)
f_object.close()
# elimina stopwords repetidas
stop = set(f_str_norm)
# seleciona as pontuações a serem eliminadas
exclude = set(string.punctuation)


# stopwords len
print("Número de stopwords: {}".format(len(stop)))
# punctuation len
print("Número de punctuation: {}".format(len(exclude)))

# stop lib
#print("Número de stopwords do NLTK: {}".format(len(stopwords.words("portuguese"))))
#print(stopwords.words("portuguese"))


# limpeza e preprocessamento
def clean(doc):
    # to lower doc
    doc_split = doc.lower()
    # remove punctuation
    punc_free = ''.join(ch for ch in doc_split if ch not in exclude)
    # remove stopwords
    stop_free = " ".join([i for i in punc_free.split() if i not in stop])
    # remove numbers
    numbers_free = ''.join([i for i in stop_free if not i.isdigit()])

    return numbers_free

doc_clean = [clean(doc).split() for doc in docs_raw]


dictionary = corpora.Dictionary(doc_clean)

# dictionary
print('Dicionário')
print(dictionary)

separador()

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
# doc term
#print('Doc term')
#print(doc_term_matrix)


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
n_topics = 10
ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word = dictionary, passes=50)

#print(ldamodel.print_topics(num_topics=n_topics, num_words=10))
topics = ldamodel.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=False)

# salva o modelo em binário
ldamodel.save('outcome/modelLDA')

# salva os tópicos em arquivo pickle
with open('outcome/topics.p', 'wb') as file:
    pickle.dump(topics, file)
    print("Tópicos salvos no arquivo topics.p")

separador()
print('Tópicos sem formatação')
print(topics)

#ldatmp = ldamodel.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=True)
#print('teste')
#print(ldatmp[0])

separador()
print('Tópicos formatados')
for i in range(0, n_topics):
    print('Topic {}'.format(i))
    print(ldamodel.show_topic(topicid=i, topn=10))


#tmptst = ldamodel.show_topic(topicid=7, topn=10)
#print(tmptst[6][0])


