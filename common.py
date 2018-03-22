#from pycorenlp import StanfordCoreNLP
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import itertools
import math
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import scipy
allwords = []
IDF = {}

def get_all_terms(docs,queries):
	global allwords
	global IDF
	docs = map(lambda l: l[1], docs)
	allwords = docs + map(lambda l: l[1], queries)
	allwords = list(itertools.chain.from_iterable(allwords))
	allwords = list(set(allwords))
	IDF = allwords_IDF(docs, allwords)
	print IDF['law']
	print IDF['aeroelast']
	#return (IDF,allwords)
def set_params (a,idf):
	global allwords
	global IDF
	allwords = a
	IDF = idf
	print len(allwords)
	print len(IDF)

def allwords_IDF(docs, allwords):
	N = len(docs)
	return dict(map(lambda w: (w,IDF_cal(1+sum(map(lambda d : 1 if w in d else 0, docs)),N ) ), allwords))

def IDF_cal(nt,N):
	return math.log(1 + (float(N)/float(nt)))

def read_data_docs(filename, numdocs):
	f = open(filename, 'r')
	text = f.read()
	docs = re.split(".I [0-9]+\\n",text)
	docs_text = map(lambda l: re.split(".W\\n",l)[1], docs[1:])
	docs = map(lambda l: preprocessing_nltk(l), docs_text[:numdocs])
	
	return docs

def read_queries(filename):
	f = open(filename, 'r')
	text = f.read().strip()
	ids = re.findall('.I [0-9]+',text)
	ids = map(lambda l : int(l.split(' ')[1]), ids) 
	qs = re.split(".I [0-9]+\\r\\n",text)
	qs_text = map(lambda l: re.split(".W\\r\\n",l)[1], qs[1:])
	qs = map(lambda l: preprocessing_nltk(l), qs_text)
	numqs = len(qs)
	print ids
	print len(ids)
	print len(qs)
	qs = zip(range(1,len(qs)+1), qs)
	return qs

def preprocessing_nltk(d):
	stop_words = set(stopwords.words('english'))
	sent_tokenize_list = sent_tokenize(d)
	word_punct_tokenizer = WordPunctTokenizer()
	words = map(lambda l: word_punct_tokenizer.tokenize(l), sent_tokenize_list)
	words = list(itertools.chain.from_iterable(words))
	numberpattern = re.compile("[0-9]+")
	words = filter(lambda l:(l not in ['.','/',',','(','-',')',"'"]) and (not bool(numberpattern.match(l))), words)
	words = filter(lambda l : l not in stop_words , words)  	
	porter_stemmer = PorterStemmer()
	words = map(lambda l: porter_stemmer.stem(l).encode('ascii','ignore'), words)
	words = nltk.pos_tag(words)
	wordnet_lemmatizer = WordNetLemmatizer()
	words = map(lambda l: wordnet_lemmatizer.lemmatize(l[0], pos = penn_to_wn(l[1])).encode('ascii','ignore'), words)
	return words

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return wn.NOUN

def read_query_rels(filename):
	f = open(filename,'r')
	text= f.read().strip()
	qs = re.split("\\n",text)
	qs = map(lambda l : l.split(" "),qs)
	qs = map(lambda l : filter(lambda t: t!="",l), qs)
	qrels = {}	
	for q in qs:
		if int(q[0]) in qrels.keys():
			qrels[int(q[0])] += [(int(q[1]),int(q[2]))]
		else:
			qrels[int(q[0])] = [(int(q[1]),int(q[2]))]	 
	return qrels


def calculate_AP(qrels, ret_docs):
	qrels = filter(lambda l : (l[1] <= 4 or l[1]==-1), qrels)
	qrels = map(lambda l : l[0], qrels)
	avgmap = []
	total = 0
	rel_docs = 0 
	#print qrels
	#print ret_docs
	for d in ret_docs:
		total += 1
		if d[0] in qrels:
			rel_docs += 1
			avgmap += [(float(rel_docs)/float(total))]
			#print avgmap
	if(len(avgmap)!=0):
		avgmap = float(sum(avgmap))/float(len(qrels))
		return avgmap	
	else:
		return 0
def calculate_MAP(APS):
	return float(sum(APS))/float(len(APS))


def get_rel_irrel_docs(qrels, doc_operators):
	rel_docs = filter( lambda l : l[1] <= 4 or l[1]==-1,qrels)
	rel_docs = map(lambda l : l[0],rel_docs)
	#rel_docs = [12,29,14,47,51,75,1,2,3,4,5,6]
	rel_docs += range(1,1401)
	rel_docs = list(set(rel_docs))
	docs1 = filter(lambda l: l[0] in rel_docs,doc_operators)
	#docs2 = filter(lambda l: l[0] not in rel_docs,docs[:10])
	#print len(docs1+docs2)
	return docs1

def load_sparce_mats(start,end):
	doc_operators = []
	for i in range(start, end):
		sparse_matrix = scipy.sparse.load_npz('../../cran/doc_operators/sparse_matrix'+str(i)+'.npz')
		doc_operators += [((i), sparse_matrix)]
	return doc_operators
