from threading import Lock
import os
from config import RES_DIR

from src.language_processing.processing import Collection
from src.interface import CACM, CS276


class IndexBuilder:

    def __init__(self, collection):
        self.collection = collection
        self.documents = dict()
        self.terms = dict()

        self.lock_documents = Lock()
        self.lock_terms = Lock()

    def look_for_term(self, term):
        with self.lock_terms:
            if term in self.terms:
                return self.terms[term]
            else:
                term_id = len(self.terms)
                self.terms.update({term: term_id})
                return term_id

    def look_for_document(self, document):
        with self.lock_documents:
            if document in self.documents:
                return self.documents[document]
            else:
                document_id = len(self.documents)
                self.documents.update({document: document_id})
                return document_id

    def construct_index(self):
        raise NotImplementedError


class BSBI(IndexBuilder):
    # Block Sort-Based Indexing

    def __init__(self, collection, index_type):
        IndexBuilder.__init__(self, collection)
        self.index_type = 'Index_%s' % index_type

    def prepare_folder(self):
        os.makedirs(os.path.join(RES_DIR, self.index_type), exist_ok=True)
        folder_path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name)
        os.makedirs(folder_path, exist_ok=True)
        for f in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, f))

    def construct_index(self):
        self.prepare_folder()
        blocks = self.segment_collection()

        for block_name in blocks:
            print("start processing block :", block_name, '...')
            pairs = self.parse_block(block_name)
            print("having pairs for block :", block_name, '...')
            postings = self.invert_block(pairs)
            print("having postings for block :", block_name, '...')
            self.write_block_to_disk(postings, block_name)

        print("merging blocks...")
        self.merge_blocks(blocks, 'index') # Inverted index
        self.write_dict_to_disk(self.documents, 'documents')
        self.write_dict_to_disk(self.terms, 'terms')

    def segment_collection(self):
        return self.collection.loader.blocks

    def parse_block(self, block_name):
        raise NotImplementedError

    def invert_block(self, pairs):
        raise NotImplementedError

    def write_block_to_disk(self, postings, block_name):
        raise NotImplementedError

    def merge_blocks(self, blocks, final_file):
        raise NotImplementedError

    def write_dict_to_disk(self, dictionary, file_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, file_name + '.txt')
        with open(path, "w") as file:
            for ref, id in sorted(dictionary.items()):
                file.write('%s %i\n' % (ref, id))


class MapReduce:

    def map(self, key, value):
        raise NotImplementedError

    def combine(self, pairs):
        raise NotImplementedError

    def shuffle_sort(self, all_pairs):
        raise NotImplementedError

    @staticmethod
    def reduce(key, values):
        raise NotImplementedError


if __name__ == '__main__':
    # 2.2 Indexation
    import time

    print("2.2 Indexing")
    from src.indexing.doc_index import DocBSBI
    from src.indexing.freq_index import FreqBSBI

    print("\n--- DocID Index : Collection CACM ---")
    bsbi_cacm_docid = DocBSBI(Collection(CACM()))
    start = time.time()
    bsbi_cacm_docid.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)

    print("\n--- DocId Index : Collection CS276 ---")
    bsbi_cs276_docid = DocBSBI(Collection(CS276(), tokn=False))
    start = time.time()
    bsbi_cs276_docid.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)

    print("\n--- Frequency Index : Collection CACM ---")
    bsbi_cacm_freq = FreqBSBI(Collection(CACM()))
    start = time.time()
    bsbi_cacm_freq.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)

    print("\n--- Frequency Index : Collection CS276 ---")
    bsbi_cs276_freq = FreqBSBI(Collection(CS276(), tokn=False))
    start = time.time()
    bsbi_cs276_freq.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)


    print("\n\n2.3 Compressed Indexing for CS276")
    from src.compression.index_builders import DocVBE, FreqVBE

    print("\n--- Compressed DocId Index : Collection CS276 ---")
    vbe_cs276_docid = DocVBE(Collection(CS276(), tokn=False))
    start = time.time()
    vbe_cs276_docid.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)

    print("\n--- Compressed Frequency Index : Collection CS276 ---")
    vbe_cs276_freq = FreqVBE(Collection(CS276(), tokn=False))
    start = time.time()
    vbe_cs276_freq.construct_index()
    end = time.time()
    print("Finished. Processing Time :", end - start)

