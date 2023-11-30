import faiss

class FaissIndex(object):
    def __init__(self, corpus_vecs, dim, l2_normalize=False, index_name="IndexFlatIP"):
        self.l2_normalize = l2_normalize
        if self.l2_normalize:
            faiss.normalize_L2(corpus_vecs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(corpus_vecs.astype("float32"))

    def search(self, query_vecs, topk=50):
        if self.l2_normalize:
            faiss.normalize_L2(query_vecs)
        topk_scores, topk_indices = self.index.search(query_vecs.astype("float32"), topk)
        return topk_scores, topk_indices


        