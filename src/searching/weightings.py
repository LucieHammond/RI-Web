import math


def custom(func, **params):
    def custom_func(*args, **kwargs):
        return func(*args, **kwargs, **params)
    return custom_func


# Weighting functions for variants of Term Frequency
def tf(freq, all_freqs):
    return freq/sum(all_freqs)


def tf_binary(freq, all_freqs):
    return int(freq > 0)


def tf_id(freq, all_freqs):
    return freq


def tf_sqrt(freq, all_freqs):
    return math.sqrt(freq)


def tf_log(freq, all_freqs, const=1, base=math.e):
    return const + math.log(freq, base) if freq > 0 else 0


def tf_log1p(freq, all_freqs):
    return math.log(1 + freq)


def tf_norm(freq, all_freqs, k=0.0):
    return k + (1-k) * freq/(max(all_freqs))


# Weighting functions for variant of Inverse Document Frequency
def idf(doc_freq, total_docs):
    return math.log(total_docs/doc_freq)


def idf_unary(doc_freq, total_docs):
    # Should be always 1
    return int(doc_freq > 0 and total_docs > 0)


def idf_log(doc_freq, total_docs, const=0, base=2.0):
    return const + math.log(total_docs / doc_freq, base)


def idf_smooth(doc_freq, total_docs):
    return math.log(1 + total_docs/doc_freq)


def idf_proba(doc_freq, total_docs):
    return max(0.0, math.log((total_docs - doc_freq)/doc_freq))


# Similarity computation models : methods for RSV
def rsv_cos(score, wq, wd):
    """
    Compute similarity between query and document with cos mesure
    :param score: the product vect(wq) * vect(wd) that has already been computed
    :param wq: list of positive weight for query (coords in query vect space)
    :param wd: list of positive weight for document (coords in document vect space)
    :return: cos score
    """
    nq = sum(map(lambda x: x**2, wq))
    nd = sum(map(lambda x: x**2, wd))
    return score / (math.sqrt(nq) * math.sqrt(nd))


def rsv_dice(score, wq, wd):
    sq = sum(wq)
    sd = sum(wd)
    return 2 * score / (sq + sd)


def rsv_jaccard(score, wq, wd):
    sq = sum(wq)
    sd = sum(wd)
    return score / (sq + sd - score)


def rsv_overlap(score, wq, wd):
    sq = sum(wq)
    sd = sum(wd)
    return score / min(sq, sd)
