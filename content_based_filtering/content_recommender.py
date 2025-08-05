import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, documents):
        """Initializes the recommender with a list of documents."""
        self.documents = documents
        self.tfidf_matrix = None
        self.cosine_similarities = None
        self.vectorizer = TfidfVectorizer()

    def fit(self):
        """Fits the model by computing the TF-IDF matrix."""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

    def recommend(self, document_index, top_n=5):
        """Recommends top N similar documents for a given document index."""
        self.cosine_similarities = linear_kernel(self.tfidf_matrix[document_index], self.tfidf_matrix).flatten()
        related_docs_indices = self.cosine_similarities.argsort()[-top_n-1:-1][::-1]
        return related_docs_indices

    def get_recommendations(self, document_index, top_n=5):
        """Returns recommended documents based on the index of the document."""
        recommended_indices = self.recommend(document_index, top_n)
        return [(index, self.documents[index]) for index in recommended_indices]

if __name__ == '__main__':
    sample_documents = [
        'Deep learning is revolutionizing artificial intelligence.',
        'Natural language processing enables machines to understand human language.',
        'Content-based filtering is a recommendation system technique.',
        'Machine learning is a subset of artificial intelligence.',
        'Deep learning models can process unstructured data effectively.'
    ]
    recommender = ContentBasedRecommender(sample_documents)
    recommender.fit()
    recommendations = recommender.get_recommendations(0, top_n=3)
    print('Recommendations for document 0:')
    for index, doc in recommendations:
        print(f'Document index {index}: {doc}')