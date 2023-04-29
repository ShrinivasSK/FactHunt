from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine

class TfIDFSelector:
    def __init__(self,args):
        self.vect=TfidfVectorizer()
        self.args=args

    def fit(self,data):
        self.vect.fit(data)

    def transform(self,data):
        return self.vect.transform(data)
    
    def select_rows(self,fact,table_data):
        vecs=self.transform(table_data)
        query=self.transform([fact])[0]
        scores=vecs.dot(query.T)
        top=np.argsort(list(scores))[::-1][:self.args.top_k]
        return [table_data[t] for t in top]

class ContrieverSelector:
    def __init__(self,args):
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        self.model = AutoModel.from_pretrained('facebook/contriever')
        self.args=args
    
    def select_rows(self,fact,table_data):
        sentences=[fact]
        sentences.extend(table_data)
        inputs=self.tokenizer.batch_encode_plus(sentences,max_length=512,add_special_tokens=True,padding='max_length',truncation=True,return_tensors='pt')
        outputs = self.model(**inputs)
        scores=(outputs['pooler_output'] @ outputs['pooler_output'][0]).cpu().detach().numpy()
        top=np.argsort(list(scores))[::-1][:self.args.top_k]
        return [table_data[t] for t in top]
    

class LSASelector:
    def __init__(self,args) -> None:
        self.args=args
        nltk.download('wordnet')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'[a-z]+')
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        

    def fit(self,data):
        TF_IDF_matrix = self.vectorizer.fit(data)
        TF_IDF_matrix = TF_IDF_matrix.T

        U, s, VT = np.linalg.svd(TF_IDF_matrix.toarray()) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix
        K=self.args.top_k

        # Getting document and term representation
        self.terms_rep = np.dot(U[:,:K], np.diag(s[:K])) # M X K matrix where M = Vocabulary Size and N = Number of documents

    def preprocess(self,document):
        document = document.lower() # Convert to lowercase
        words = self.tokenizer.tokenize(document) # Tokenize
        words = [w for w in words if not w in self.stop_words] # Removing stopwords
        # Lemmatizing
        for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:
            words = [self.wordnet_lemmatizer.lemmatize(x, pos) for x in words]
        return " ".join(words)
    
    def get_document_reps(self,docs):
        TF_IDF_matrix = self.vectorizer.transform(docs)
        TF_IDF_matrix = TF_IDF_matrix.T

        U, s, VT = np.linalg.svd(TF_IDF_matrix.toarray()) # .T is used to take transpose and .toarray() is used to convert sparse matrix to normal matrix
        K=self.args.top_k

        # Getting document and term representation
        return np.dot(np.diag(s[:K]), VT[:K, :]).T # N x K matrix 

    def select_rows(self,fact,table_data):
        query_rep = [self.vectorizer.vocabulary_[x] for x in self.preprocess(fact).split()]
        query_rep = np.mean(self.terms_rep[query_rep],axis=0)

        docs_rep = self.get_document_reps(table_data)
        scores = [cosine(query_rep, doc_rep) for doc_rep in docs_rep]
        top=np.argsort(list(scores))[:self.args.top_k]
        
        return [table_data[t] for t in top]

        



