import matplotlib.pyplot as plt

from src.language_processing.processing import Collection
from src.interface import read_relevance_judgments
from src.searching import bool_search as bool, vect_search as vect
from src.searching import weightings as w
from src.searching.index_reader import DocIDIndex, FreqIndex


def e_measure(precision, recall, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("Coefficient alpha must be between 0 and 1")
    if precision == 0 or recall == 0:
        return 1
    temp = alpha / precision + (1 - alpha) / recall
    return 1 - 1 / temp


def f_measure(precision, recall, beta):
    alpha = 1 / (beta**2 + 1)
    return 1 - e_measure(precision, recall, alpha)


class Evaluation:

    def __init__(self, results, relevance):
        self.results = results
        self.relevance = relevance
        self._recall_precision = None

    @property
    def recall_precision(self):
        if not self._recall_precision:
            self._recall_precision = self.global_recall_precision()
        return self._recall_precision

    def global_recall_precision(self):
        all_pr = []
        for nb, results in sorted(self.results.items()):
            tp = set(results) & set(self.relevance[nb])
            precision = float(len(tp)) / len(results) if len(results) > 0 else 1
            recall = float(len(tp)) / len(self.relevance[nb]) if len(self.relevance[nb]) > 0 else 1
            all_pr.append((recall, precision))
        return all_pr

    def global_e_measures(self, alpha=0.5):
        e_measures = []
        pr = self.recall_precision
        for (rec, prec) in pr:
            e_mes = e_measure(prec, rec, alpha)
            e_measures.append(e_mes)
        return e_measures

    def global_f_measures(self, beta=1.0):
        f_measures = []
        pr = self.recall_precision
        for (rec, prec) in pr:
            f_mes = f_measure(prec, rec, beta)
            f_measures.append(f_mes)
        return f_measures


class UnrankedResults(Evaluation):
    """ Represents evaluation of boolean research results """

    def __init__(self, query_file, collection, relevance):
        self.queries = bool.read_queries(query_file)
        self.collection = collection
        results = self.search_results()
        Evaluation.__init__(self, results, relevance)

    def search_results(self):
        """ Use boolean search """
        all_results = {}
        index = DocIDIndex(self.collection)
        for q_nb, query in enumerate(self.queries):
            results = bool.search_for_query(query, index)
            all_results[q_nb + 1] = results
        return all_results


class RankedResults(Evaluation):
    """ Represents evaluation of vectorial research results """

    def __init__(self, query_file, collection, relevance, **searchparams):
        self.queries = vect.read_queries(query_file)
        self.collection = collection
        self._recall_precision_curves = None
        results = self.search_results(**searchparams)
        Evaluation.__init__(self, results, relevance)

    @property
    def ranked_recall_precision(self):
        if not self._recall_precision_curves:
            self._recall_precision_curves = self.compute_rp_points()
        return self._recall_precision_curves

    def search_results(self, **searchparams):
        """ Use vectorial search """
        all_results = {}
        index = FreqIndex(self.collection)
        for q_nb, query in enumerate(self.queries):
            query_tokens = Collection(None).process(query)
            results = vect.search_for_query(query_tokens, index, **searchparams)
            all_results[q_nb + 1] = results
        return all_results

    def compute_rp_points(self):
        rp_curves = []
        for nb, results in sorted(self.results.items()):
            rp_points = []
            relevance = self.relevance[nb]
            corrects = 0
            if len(relevance) == 0:
                final_prec = 0 if len(results) > 0 else 1
                rp_points.append((1.0, final_prec))
            for rank, res in enumerate(results):
                if res in relevance:
                    corrects += 1
                    precision = corrects / (rank + 1)
                    recall = corrects / len(relevance)
                    rp_points.append((recall, precision))
                    if recall == 1:
                        break
            for rest in range(corrects + 1, len(relevance) + 1):
                rp_points.append((rest/len(relevance), 0))
            rp_curves.append(rp_points)
        return rp_curves

    def interpolate(self, steps):
        interpol_curves = []
        for i, query_rp in enumerate(self.ranked_recall_precision):
            new_curve = []
            for j, (rec, prec) in enumerate(query_rp):
                self.ranked_recall_precision[i][j] = (rec, max([p for (r, p) in query_rp if r >= rec]))
            for s in range(1, steps + 1):
                prec = max([p for (r, p) in query_rp if r >= s/steps])
                new_curve.append(prec)
            interpol_curves.append(new_curve)
        return interpol_curves

    def recall_precision_curves(self, steps):
        interpol_curves = self.interpolate(steps)
        avg_curve = []
        for s in range(steps):
            avg_s = sum([curve[s] for curve in interpol_curves]) / len(interpol_curves)
            avg_curve.append(avg_s)
        return interpol_curves, avg_curve

    def r_precisions(self):
        r_precisions = []
        for nb, results in sorted(self.results.items()):
            r = len(self.relevance[nb])
            if r == 0:
                precision = 0 if len(results) > 0 else 1
            else:
                precision = len(set(results[:r]) & set(self.relevance[nb])) / r
            r_precisions.append(precision)
        return r_precisions

    def mean_average_precision(self):
        sum_avg_precisions = 0
        for query_rp in self.ranked_recall_precision:
            avg_precision = sum([prec for (rec, prec) in query_rp]) / len(query_rp)
            sum_avg_precisions += avg_precision
        return sum_avg_precisions / len(self.ranked_recall_precision)


def global_pertinence(eval):

    # Recall-Precision
    recalls, precisions = zip(*eval.recall_precision)
    print("Global average precision:", sum(precisions) / len(precisions))
    print("Global average recall:", sum(recalls) / len(recalls))

    plt.title("Recall-Precision for all queries (scatter plot)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recalls, precisions, '.')
    plt.show()

    # E-measure, F-measure
    e05 = eval.global_e_measures(alpha=0.5)
    e02 = eval.global_e_measures(alpha=0.2)
    e08 = eval.global_e_measures(alpha=0.8)
    f1 = eval.global_f_measures(beta=1)
    f2 = eval.global_f_measures(beta=2)
    f05 = eval.global_f_measures(beta=0.5)

    print("\nAverage E-measure (for alpha=0.5):", sum(e05) / len(e05))
    print("Average F1-measure:", sum(f1) / len(f1))
    print("Average F2-measure:", sum(f2) / len(f2))
    print("Average F0.5-measure:", sum(f05) / len(f05), '\n')

    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    x = range(1, len(e05) + 1)
    plt.xlim([0, len(e05) + 1])
    plt.plot(x, f1, 'b', label='F_1')
    plt.plot(x, f2, 'g', label='F_2')
    plt.plot(x, f05, 'r', label='F_0.5')
    plt.legend()

    plt.subplot(212)
    plt.xlim([0, len(e05) + 1])
    plt.xlabel("Queries")
    plt.plot(x, e05, 'c', label='E_0.5')
    plt.plot(x, e02, 'y', label='E_0.2')
    plt.plot(x, e08, 'm', label='E_0.8')
    plt.legend()
    plt.show()


def pertinence_with_rankings(ranked_eval, steps):
    # Recall-Precision curve
    rp_curves, avg_rp_curve = ranked_eval.recall_precision_curves(steps)
    x = [(s + 1) / steps for s in range(steps)]

    plt.title("Recall-Precision curves for all queries")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    for curve in rp_curves:
        plt.plot(x, curve)
    plt.show()

    plt.title("Recall-Precision curve (average on all queries)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1])
    plt.plot(x, avg_rp_curve)
    plt.show()

    # R-precision
    r_precisions = ranked_eval.r_precisions()

    print("Average R-Precision:", sum(r_precisions) / len(r_precisions))

    plt.title("R-Precisions for all queries")
    plt.ylabel("Number of queries")
    plt.xlabel("R-Precisions")
    plt.hist(r_precisions, 20)
    plt.show()

    # Mean Average Precision
    map = ranked_eval.mean_average_precision()
    print("Mean Average Precision:", map)


def comparison_search_params(steps):
    x = [(s + 1) / steps for s in range(steps)]
    all_tf = [w.tf_log, w.tf, w.tf_binary, w.tf_id, w.tf_sqrt, w.tf_log1p, w.tf_norm]
    all_idf = [w.idf, w.idf_unary, w.idf_log, w.idf_smooth, w.idf_proba]
    all_rsv = [w.rsv_cos, w.rsv_dice, w.rsv_jaccard, w.rsv_overlap]
    plt.figure(figsize=(8, 5))

    def make_eval(name, **params):
        param_eval = RankedResults('query.text', 'CACM', rel_judgments, **params)
        rp_curves, avg_rp_curve = param_eval.recall_precision_curves(steps)
        r_precisions = param_eval.r_precisions()
        map = param_eval.mean_average_precision()
        plt.plot(x, avg_rp_curve, label=name)
        print('%s :\t %.6f \t\t\t %.6f' % (name.ljust(10), sum(r_precisions) / len(r_precisions), map))

    make_eval('default', tf=all_tf[0], idf=all_idf[0], rsv=all_rsv[0])

    print("\nFUNCTION TF: \tR-Precision \tMean Average Precision")
    print('%s :\t  default \t\t\t default' % all_tf[0].__name__.ljust(10))
    for tf in all_tf[1:]:
        make_eval(tf.__name__, tf=tf)

    print("\nFUNCTION IDF: \tR-Precision \tMean Average Precision")
    print('%s :\t  default \t\t\t default' % all_idf[0].__name__.ljust(10))
    for idf in all_idf[1:]:
        make_eval(idf.__name__, idf=idf)

    print("\nFUNCTION RSV: \tR-Precision \tMean Average Precision")
    print('%s :\t  default \t\t\t default' % all_rsv[0].__name__.ljust(10))
    for rsv in all_rsv[1:]:
        make_eval(rsv.__name__, rsv=rsv)

    plt.title("Comparison of Recall-Precision curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1.3])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 2.3.1 Evaluation de la pertinence pour la collection CACM
    print("2.3.2 Evaluation of pertinence for CACM\n")

    rel_judgments = read_relevance_judgments()

    print("---------- Boolean Research Model (results without ranking) ----------")
    bool_eval = UnrankedResults('query_bool.text', 'CACM', rel_judgments)
    global_pertinence(bool_eval)

    print("---------- Vectorial Research Model (results with ranking) ----------")
    vect_eval = RankedResults('query.text', 'CACM', rel_judgments)

    print("\n# Global pertinence (for all results whatever their rank)")
    global_pertinence(vect_eval)

    print("\n# Pertinence evaluation considering rankings")
    pertinence_with_rankings(vect_eval, 100)

    print("\n# Comparison of different weightings for vectorial search")
    comparison_search_params(100)
