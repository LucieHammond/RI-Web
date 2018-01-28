import os
import itertools
from src.indexing.doc_index import DocBSBI
from src.indexing.freq_index import FreqBSBI
from config import RES_DIR

from src.compression.vb_encoding import byte_encode, byte_decode


class DocVBE(DocBSBI):

    def __init__(self, collection):
        DocBSBI.__init__(self, collection)
        self.index_type = "IndexVBE_DocID"

    def write_block_to_disk(self, postings, block_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '_VBE.txt')
        with open(path, "wb") as file:
            for term_id, documents in sorted(postings.items()):
                enc_term_id = byte_encode(term_id)
                enc_count = byte_encode(len(documents))
                enc_documents = itertools.chain(*map(byte_encode, documents))
                file.write(bytes(enc_term_id + enc_count + list(enc_documents)))

    def merge_blocks(self, blocks, final_file):
        indexes = list()

        # Read all blocks
        for block_name in blocks:
            path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '_VBE.txt')
            index = dict()

            all_nums = []
            with open(path, "rb") as file:
                next_num = []
                for byte in file.read():
                    next_num.append(byte)
                    if byte >= 128:
                        all_nums.append(byte_decode(next_num))
                        next_num = []

            pointer = 0
            while pointer < len(all_nums):
                term_id = all_nums[pointer]
                count = all_nums[pointer + 1]
                index[term_id] = all_nums[pointer + 2:pointer + 2 + count]
                pointer += count + 2

            indexes.append(index)
            os.remove(path)

        # Merge postings list in memory
        term_ids = set().union(*indexes)
        global_index = dict()
        for term_id in term_ids:
            postings = set().union(*[set(index.get(term_id, [])) for index in indexes])
            global_index[term_id] = sorted(list(postings))

        # Write result to disk
        self.write_block_to_disk(global_index, final_file)


class FreqVBE(FreqBSBI):

    def __init__(self, collection):
        FreqBSBI.__init__(self, collection)
        self.index_type = "IndexVBE_Freq"

    def write_block_to_disk(self, postings, block_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '_VBE.txt')
        with open(path, "wb") as file:
            for term_id, documents in sorted(postings.items()):
                enc_term_id = byte_encode(term_id)
                enc_count = byte_encode(len(documents))
                enc_doc_list = [byte_encode(doc_id) + byte_encode(freq) for doc_id, freq in sorted(documents.items())]
                enc_documents = itertools.chain(*enc_doc_list)
                file.write(bytes(enc_term_id + enc_count + list(enc_documents)))

    def add_block_to_disk(self, postings, file_name):
        path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, file_name + '_VBE.txt')
        with open(path, "ab") as file:
            for doc_id, terms in sorted(postings.items()):
                enc_term_id = byte_encode(doc_id)
                enc_count = byte_encode(len(terms))
                enc_doc_list = [byte_encode(doc_id) + byte_encode(freq) for doc_id, freq in sorted(terms.items())]
                enc_documents = itertools.chain(*enc_doc_list)
                file.write(bytes(enc_term_id + enc_count + list(enc_documents)))

    def merge_blocks(self, blocks, final_file):
        indexes = list()

        # Read all blocks
        for block_name in blocks:
            path = os.path.join(RES_DIR, self.index_type, self.collection.loader.name, block_name + '_VBE.txt')
            index = dict()

            all_nums = []
            with open(path, "rb") as file:
                next_num = []
                for byte in file.read():
                    next_num.append(byte)
                    if byte >= 128:
                        all_nums.append(byte_decode(next_num))
                        next_num = []

            pointer = 0
            while pointer < len(all_nums):
                term_id = all_nums[pointer]
                count = all_nums[pointer + 1]
                index[term_id] = {all_nums[p]: all_nums[p+1] for p in range(pointer + 2, pointer + 2 * count + 2, 2)}
                pointer += 2 * count + 2

            indexes.append(index)
            os.remove(path)

        # Merge postings list in memory
        term_ids = set().union(*indexes)
        global_index = dict()
        for term_id in term_ids:
            global_docs = dict()
            list_docs = [index.get(term_id, {}) for index in indexes]
            for doc_id in set().union(*list_docs):
                global_docs[doc_id] = sum(docs.get(doc_id, 0) for docs in list_docs)
            global_index[term_id] = global_docs

        # Write result to disk
        self.write_block_to_disk(global_index, final_file)
