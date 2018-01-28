import os
from src.compression.vb_encoding import byte_decode
from src.searching.index_reader import DocIDIndex, FreqIndex
from config import RES_DIR


def get_next_num(data, pointer):
    next_num = []
    while True:
        next_num.append(data[pointer])
        pointer += 1
        if next_num[-1] >= 128:
            return pointer, byte_decode(next_num)


def skip_next_num(data, pointer):
    while data[pointer] < 128:
        pointer += 1
    return pointer + 1


class DocIDIndexVBE(DocIDIndex):

    def __init__(self, collection):
        DocIDIndex.__init__(self, collection)
        self.index_type = 'IndexVBE_DocID'

    def get_related_documents(self, term_id):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index_VBE.txt')
        docs = []

        with open(path, "rb") as file:
            data = file.read()

        pointer = 0
        while pointer < len(data):
            pointer, num = get_next_num(data, pointer)
            pointer, count = get_next_num(data, pointer)
            if num == term_id:
                for i in range(count):
                    pointer, doc_id = get_next_num(data, pointer)
                    docs.append(doc_id)
                return docs
            else:
                for i in range(count):
                    pointer = skip_next_num(data, pointer)

        return ValueError("term not found ?")


class FreqIndexVBE(FreqIndex):

    def __init__(self, collection):
        FreqIndex.__init__(self, collection)
        self.index_type = 'IndexVBE_Freq'

    def get_related_documents(self, term_ids):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index_VBE.txt')
        terms_index = {}

        with open(path, "rb") as file:
            data = file.read()

        pointer = 0
        while pointer < len(data):
            pointer, term_id = get_next_num(data, pointer)
            pointer, count = get_next_num(data, pointer)
            if term_id in term_ids:
                postings = {}
                for i in range(count):
                    pointer, doc_id = get_next_num(data, pointer)
                    pointer, freq = get_next_num(data, pointer)
                    postings[doc_id] = freq
                terms_index[term_id] = (count, postings)
            else:
                for i in range(2 * count):
                    pointer = skip_next_num(data, pointer)
        return terms_index

    def get_related_terms(self, doc_ids):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'doc_index_VBE.txt')
        docs_index = {}

        with open(path, "rb") as file:
            data = file.read()

        pointer = 0
        while pointer < len(data):
            pointer, doc_id = get_next_num(data, pointer)
            pointer, count = get_next_num(data, pointer)
            if doc_id in doc_ids:
                docs_index[doc_id] = {}
                for i in range(count):
                    pointer, term_id = get_next_num(data, pointer)
                    pointer, freq = get_next_num(data, pointer)
                    docs_index[doc_id][term_id] = freq
            else:
                for i in range(2 * count):
                    pointer = skip_next_num(data, pointer)
        return docs_index

    def get_all_doc_freqs(self):
        path = os.path.join(RES_DIR, self.index_type, self.collection, 'index_VBE.txt')
        doc_freqs = {}

        with open(path, "rb") as file:
            data = file.read()

        pointer = 0
        while pointer < len(data):
            pointer, term_id = get_next_num(data, pointer)
            pointer, count = get_next_num(data, pointer)
            doc_freqs[term_id] = count
            for i in range(2 * count):
                pointer = skip_next_num(data, pointer)

        return doc_freqs