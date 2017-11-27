import os
import gensim
import string
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

docs_list = os.listdir('docs')
docs_list2 = docs_list[5:10]
outputs = "outputs"

lol = ['oi', 'aqui']

stoptst = [ 'na',
            'das',
            'esse',
            'teremos', 'terá', 'houve', 'houverem', 'somos', 'con', 'nos', 'ou', 'teria', 'tinha', 'del', 'estivéssemos', 'tive', 'tenhamos', 'já', 'aquilo', 'aquele', 'só', 'havemos', 'sejamos', 'aquela', 'tínhamos', 'tinham', 'teve', 'estivemos', 'tenho', 'um', 'esta', 'houvesse', 'estas', 'este', 'una', 'ter', 'houveram', 'houvessem', 'que', 'houvéssemos', 'teu', 'tiverem', 'nas', 'houver', 'tivermos', 'seu', 'essas', 'como', 'estava', 'seriam', 'meu', 'hão', 'da', 'las', 'pode', 'tivemos', 'houvemos', 'tiver', 'eu', 'do', 'tenha', 'tivéssemos', 'se', 'vocês', 'aos', 'houveria', 'estavam', 'meus', 'foi', 'tem', 'nós', 'o', 'a', 'você', 'estão', 'tu', 'nem', 'houveríamos', 'lhe', 'numa', 'te', 'está', 'sou', 'houverá', 'éramos', 'eram', 'fossem', 'esteve', 'tém', 'seus', 'era', 'ela', 'ele', 'houvera', 'sem', 'pelo', 'para', 'lhes', 'minhas', 'mesmo', 'dele', 'isso', 'nosso', 'estive', 'estamos', 'fôramos', 'terei', 'hei', 'é', 'haja', 'seríamos', 'no', 'tivéramos', 'de', 'ser', 'houverão', 'havia', 'fora', 'estes', 'forem', 'mas', 'muito', 'sejam', 'houvermos', 'estivermos', 'fosse', 'podem', 'estiver', 'tivera', 'estivera', 'estivesse', 'teus', 'nesse', 'esses', 'estou', 'dos', 'me', 'quem', 'seria', 'seremos', 'tiveram', 'houvéramos', 'seja', 'sua', 'suas', 'esteja', 'fui', 'estejam', 'há', 'estiverem', 'maior', 'num', 'houveremos', 'também', 'temos', 'mais', 'hajam', 'assim', 'os', 'houverei', 'quando', 'será', 'estiveram', 'qual', 'estávamos', 'p', 'estivéramos', 'terão', 'uma', 'às', 'serei', 'têm', 'foram', 'vos', 'for', 'teríamos', 'teriam', 'sobre', 'sendo', 'isto', 'estejamos', 'depois', 'pela', 'pelas', 'fôssemos', 'não', 'à', 'los', 'nossas', 'com', 'em', 'ao', 'tuas', 'estivessem', 'e', 'até', 'eles', 'minha', 'deles', 'tua', 'são', 'formos', 'nossos', 'as', 'delas', 'hajamos', 'fomos', 'tenham', 'aquelas', 'elas', 'nossa', 'serão', 'aqueles', 'por', 'essa', 'dela', 'tivesse', 'tivessem', 'entre', 'houveriam', 'pelos' ]

stoptst.sort()

print('-----------------------------------------------------------------------------')
print(stoptst)
print('-----------------------------------------------------------------------------')

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
for i in range(0, len(gen_docs)):
    file = open('outputs/doc{}.txt'.format(i), 'w', encoding="utf8")
    file.write(str(gen_docs[i]))
    file.close()


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
#stop = set(f_str_norm)
stop = set(stoptst)
f_object.close()
print("Stopwords: {}".format(len(stop)))
print(stop)
print("len f: {}".format(len(f_str_norm)))
print("len stop: {}".format(len(stop)))
#for i in range(0, len(f_str_norm)):
#    print("Algo {} : {}".format(i, f_str_norm[i]))


#print("Stopwords: {}".format(len((stopwords.words('portuguese')))))
#print(stopwords.words('portuguese'))
exclude = set(string.punctuation)
#lemma = WordNetLemmatizer()
def clean(doc, stop):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    #normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    #print("passou aqui")
    return punc_free

doc_clean = [clean(doc, stop).split() for doc in docs_raw]

for i in range(0, len(doc_clean)):
    file = open('outputs/docC{}.txt'.format(i), 'w', encoding="utf8")
    file.write(str(doc_clean[i]))
    file.close()


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


print("topic analise")
print(ldamodel.show_topic(topicid=7, topn=10))

tmptst = ldamodel.show_topic(topicid=7, topn=10)
print(tmptst[6][0])