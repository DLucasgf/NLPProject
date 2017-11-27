import os
import gensim
import string
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

docs_list = os.listdir('docs')
docs_list2 = docs_list[5:10]
outputs = "outputs"

#for i in range(1, 5):
#    arr14.append(os.listdir('docs\\2014v19\\{}'.format(i)))

#print(docs_list)
#print(len(docs_list))

# ler documentos
docs_raw = []
#for item in docs_list:
for item in docs_list2:
    file = os.path.join('docs', item)
    f_object = open(file, 'r', encoding="utf8")
    docs_raw.append(f_object.read())
    f_object.close()
#f_object = open('docs\\a_.txt', 'r')
#print(f_object.read())

print("Número de documentos: {}".format(len(docs_raw)))

# tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] for text in docs_raw]
#print(gen_docs)


# dictionary
dictionary = gensim.corpora.Dictionary(gen_docs)
#print(dictionary[5])
print("Palavra melhor: {}".format(dictionary.token2id['melhor']))
print("Número de palavras no dicionário: {}".format(len(dictionary)))
#for i in range(len(dictionary)):
    #print(i, dictionary[i])


# BoW
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
#for d in corpus:
#    print(d)

# tf idf do corpus
tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)


# similaridade dos objetos
sims = gensim.similarities.Similarity(outputs,tf_idf[corpus],num_features=len(dictionary))
print(sims)
print(type(sims))

# query de documentos e converte para idftf
query_doc = [w.lower() for w in word_tokenize("Socks are a force for good.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print(query_doc_tf_idf)


# limpeza e preprocessamento
#stop = set(stopwords.words('portuguese'))
file = os.path.join('stopwords', 'pt.txt')
f_object = open(file, 'r', encoding="utf8")
f_str = f_object.readlines()
f_str_norm = []
for item in f_str:
    tmp = item.strip('\n')
    f_str_norm.append(tmp)
print(f_str_norm[0])
stop = set(f_str_norm)
f_object.close()
print("Stopwords: {}".format(len(stop)))
print(stop)


#print("Stopwords: {}".format(len((stopwords.words('portuguese')))))
#print(stopwords.words('portuguese'))
exclude = set(string.punctuation)
#lemma = WordNetLemmatizer()
def clean(doc, stop):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return punc_free

doc_clean = [clean(doc, stop).split() for doc in docs_raw]


dictionary = corpora.Dictionary(doc_clean)

doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
n_topics = 10
ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word = dictionary, passes=50)

print(ldamodel.print_topics(num_topics=n_topics, num_words=10))
print(ldamodel.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=True))

ldatmp = ldamodel.show_topics(num_topics=n_topics, num_words=10, log=False, formatted=True)
print('teste')
print(ldatmp[0])


for i in range(0, n_topics):
    print('Topic {}'.format(i))
    print(ldamodel.show_topic(topicid=i, topn=10))