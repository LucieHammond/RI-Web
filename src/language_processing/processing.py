import re
from collections import Counter
import math
import matplotlib.pyplot as plt
from threading import Lock

from gensim.parsing.porter import PorterStemmer
from src.interface import load_stop_words, CACM, CS276

STOP_WORDS = load_stop_words()


class Collection:
	"""
		This class represents a processed collection and implements different language processings methods
		to extract important information from texts.
		It also includes global statistics about the collection and methods to mesure and compute those stats
	"""

	def __init__(self, loader, tokn=True, filt=True, norm=False):
		self.loader = loader

		self.do_tokenize = tokn
		self.do_filter = filt
		self.do_normalize = norm

		self.tokens_number = 0
		self.freq_dist = Counter()
		self.tokens_lock = Lock()
		self.dist_lock = Lock()

	@staticmethod
	def tokenize(text, strong):
		if strong:
			return re.sub(r"[^A-Za-z0-9]", ' ', text).split()
		else:
			return re.sub(r"'s", ' ', text).split()

	@staticmethod
	def filter(tokens):
		return [token.lower() for token in tokens if token.isalnum() and token.lower() not in STOP_WORDS]

	@staticmethod
	def normalize(tokens):
		stemmer = PorterStemmer()
		return [stemmer.stem(word) for word in tokens]

	def process(self, text):
		tokens = self.tokenize(text, strong=self.do_tokenize)
		if self.do_filter:
			tokens = self.filter(tokens)
		if self.do_normalize:
			tokens = self.normalize(tokens)

		return tokens

	def compute_distrib(self, tokens):
		with self.tokens_lock:
			self.tokens_number += len(tokens)
		for word in tokens:
			with self.dist_lock:
				self.freq_dist[word] += 1

	def process_collection(self, percentage=1.0):
		""" Process all collection (or a percentage of it) and compute stats (distributions, size of vocabulary...) """

		def process_contents(content):
			tokens = self.process(content)
			self.compute_distrib(tokens)

		threads = self.loader.load_all_documents(process_contents, percentage)
		[thread.join() for thread in threads]

	def process_block(self, block_name, callback):
		""" Load only one block and process documents one by one in threads as soon as they are loaded """
		list_pairs = list()
		lock = Lock()

		def process_document(doc_name, content):
			tokens = self.process(content)
			with lock:
				list_pairs.append(callback(doc_name, tokens))

		threads = self.loader.load_block(block_name, process_document)
		[thread.join() for thread in threads]
		return list_pairs


if __name__ == '__main__':
	# 2.1 : Traitements linguistiques
	print("2.1 Language processing\n")

	print("--- Collection CACM ---")
	cacm = Collection(CACM())
	cacm.process_collection()

	# Q1) How many tokens in collection
	print("Nombre de tokens :", cacm.tokens_number)

	# Q2) Length of vocabulary
	print("Taille du vocabulaire :", len(cacm.freq_dist))

	# Q3) Same for half of the collection
	half_cacm = Collection(CACM())
	half_cacm.process_collection(0.5)
	print("Nombre de tokens pour la moitié de la collection :", half_cacm.tokens_number)
	print("Taille du vocabulaire pour la moitié de la collection :", len(half_cacm.freq_dist))

	# Find b and k with Rule of Heaps M = kT^b
	T1 = cacm.tokens_number
	T2 = half_cacm.tokens_number
	M1 = len(cacm.freq_dist)
	M2 = len(half_cacm.freq_dist)
	b = math.log(M1/M2) / math.log(T1/T2)
	k = math.exp(math.log(M1) - b * math.log(T1))

	print("Loi de Heaps : b =", b, ", k =", k)

	# Q4) Length of vacabulary for collection of 1 million tokens
	T3 = 10 ** 6
	M3 = k * T3**b

	print("Taille du vocabulaire pour une collection de 1 million de tokens :", round(M3))

	# Q5) Draw freq = f(range) for all tokens
	freqs = [sample[1]/cacm.tokens_number for sample in cacm.freq_dist.most_common()]
	plt.figure(figsize=(7, 5))
	plt.title("Tokens in collection CACM")
	plt.xlabel('rank')
	plt.ylabel('freq')
	plt.plot(range(1, len(freqs) + 1), freqs, 'bo', markersize=2)
	plt.show()

	# Draw log(freq) = f(log(range)) for all tokens
	freqs_log = [math.log(freq) for freq in freqs]
	ranks_log = [math.log(rank) for rank in range(1, len(freqs)+1)]
	plt.title("Tokens in collection CACM")
	plt.xlabel('log(rank)')
	plt.ylabel('log(freq)')
	plt.plot(ranks_log, freqs_log, 'bo', markersize=2)
	plt.show()


	print("\n--- Collection CS276 ---")
	cs276 = Collection(CS276(), tokn=False)
	cs276.process_collection()

	# Q1) How many tokens in collection
	print("Nombre de tokens :", cs276.tokens_number)

	# Q2) Length of vocabulary
	print("Taille du vocabulaire :", len(cs276.freq_dist))

	# Q3) Same for half of the collection
	half_cs276 = Collection(CS276(), tokn=False)
	half_cs276.process_collection(0.5)
	print("Nombre de tokens pour la moitié de la collection :", half_cs276.tokens_number)
	print("Taille du vocabulaire pour la moitié de la collection :", len(half_cs276.freq_dist))

	# Find b and k with Rule of Heaps M = kT^b
	T1 = cs276.tokens_number
	T2 = half_cs276.tokens_number
	M1 = len(cs276.freq_dist)
	M2 = len(half_cs276.freq_dist)
	b = math.log(M1 / M2) / math.log(T1 / T2)
	k = math.exp(math.log(M1) - b * math.log(T1))

	print("Loi de Heaps : b =", b, ", k =", k)

	# Q4) Length of vacabulary for collection of 1 million tokens
	T3 = 10 ** 6
	M3 = k * T3 ** b

	print("Taille du vocabulaire pour une collection de 1 million de tokens :", round(M3))

	# Q5) Draw freq = f(range) for all tokens
	freqs = [sample[1] for sample in cs276.freq_dist.most_common()]
	plt.figure(figsize=(7,5))
	plt.title("Tokens in collection CS276")
	plt.xlabel('rank')
	plt.ylabel('freq')
	plt.plot(range(1, len(freqs) + 1), freqs, 'bo', markersize=2)
	plt.show()

	# Draw log(freq) = f(log(range)) for all tokens
	freqs_log = [math.log(freq) for freq in freqs]
	ranks_log = [math.log(rank) for rank in range(1, len(freqs) + 1)]
	plt.title("Tokens in collection CS276")
	plt.xlabel('log(rank)')
	plt.ylabel('log(freq)')
	plt.plot(ranks_log, freqs_log, 'bo', markersize=2)
	plt.show()
