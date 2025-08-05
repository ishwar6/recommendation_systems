import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, documents):
        """Initializes the recommender with documents."""
        self.documents = documents
        self.tf_idf_matrix = self._create_tf_idf_matrix()

    def _create_tf_idf_matrix(self):
        """Creates TF-IDF matrix from documents."""
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(self.documents)

    def recommend(self, index, top_n=5):
        """Recommends top N documents similar to the given index."""
        cosine_similarities = linear_kernel(self.tf_idf_matrix[index:index+1], self.tf_idf_matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-top_n-1:-1]
        return related_docs_indices

if __name__ == '__main__':
    sample_documents = [
        'Deep learning for content recommendation.',
        'Natural language processing and machine learning.',
        'Content-based filtering with TF-IDF.',
        'Machine learning techniques in educational applications.',
        'Building a recommender system using Python.'
    ]
    recommender = ContentBasedRecommender(sample_documents)
    recommendations = recommender.recommend(0)
    print('Recommended documents indices:', recommendations)