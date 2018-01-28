import os
from config import RES_DIR
from gensim.parsing.porter import PorterStemmer


class IndexUsage:

    def __init__(self, type, collection):
        self.collection = collection
        self.index_type = 'Index_%s' % type

    def get_id_for_term(self, term):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'terms.txt')
        with open(path, 'r') as terms:
            for line in terms:
                if line.split()[0] == term:
                    return int(line.split()[1])
        return -1

    def get_ids_for_terms(self, terms):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'terms.txt')
        term_ids = {}
        with open(path, 'r') as f_terms:
            for line in f_terms:
                if line.split()[0] in terms:
                    term_ids[line.split()[0]] = int(line.split()[1])
        return term_ids

    def get_documents_from_ids(self, doc_ids):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'documents.txt')
        doc_names = [""] * len(doc_ids)
        with open(path, 'r') as documents:
            for line in documents:
                if int(line.split()[1]) in doc_ids:
                    rank = doc_ids.index(int(line.split()[1]))
                    doc_names[rank] = line.split()[0]
        return doc_names

    def find_documents(self, term):
        raise NotImplementedError

    def get_related_documents(self, term_id):
        raise NotImplementedError

    def get_all_documents(self):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'documents.txt')
        with open(path, 'r') as f:
            return set(range(len(f.readlines())))


class DocIDIndex(IndexUsage):

    def __init__(self, collection):
        IndexUsage.__init__(self, 'DocID', collection)

    def find_documents(self, term, stemming=False):
        stemmer = PorterStemmer()
        if stemming:
            term = stemmer.stem(term)
        term_id = self.get_id_for_term(term)
        if term_id < 0:
            return set()
        docs = self.get_related_documents(term_id)
        return set(docs)

    def get_related_documents(self, term_id):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index.txt')
        with open(path, 'r') as index:
            for line in index:
                if line.split()[0] == str(term_id):
                    return map(int, line.split()[1:])


class FreqIndex(IndexUsage):

    def __init__(self, collection):
        IndexUsage.__init__(self, 'Freq', collection)

    def find_documents(self, terms):
        term_ids = self.get_ids_for_terms(terms)
        terms_index = self.get_related_documents(term_ids.values())
        return {term: terms_index[id] for term, id in term_ids.items()}

    def get_related_documents(self, term_ids):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index.txt')
        terms_index = {}

        def extract_docs_freq(str):
            return int(str.split(':')[0]), int(str.split(':')[1])

        with open(path, 'r') as index:
            for line in index:
                if int(line.split()[0]) in term_ids:
                    count = int(line.split()[1])
                    postings = dict(map(extract_docs_freq, line.split()[2:]))
                    terms_index[int(line.split()[0])] = (count, postings)
        return terms_index

    def get_related_terms(self, doc_ids):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'doc_index.txt')
        docs_index = {}

        def extract_terms_freq(str):
            return int(str.split(':')[0]), int(str.split(':')[1])

        with open(path, 'r') as index:
            for line in index:
                if int(line.split()[0]) in doc_ids:
                    terms_freq = dict(map(extract_terms_freq, line.split()[2:]))
                    docs_index[int(line.split()[0])] = terms_freq
        return docs_index

    def get_all_doc_freqs(self):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index.txt')
        doc_freqs = {}

        with open(path, 'r') as index:
            for line in index:
                term_id, doc_count = line.split()[:2]
                doc_freqs[int(term_id)] = int(doc_count)
        return doc_freqs
