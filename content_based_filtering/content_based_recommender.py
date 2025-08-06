import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, data):
        """
        Initializes the recommender with the provided data.
        
        Args:
            data (pd.DataFrame): DataFrame containing the items to recommend.
        """
        self.data = data
        self.tfidf_matrix = None
        self.indices = None

    def fit(self):
        """
        Fits the TF-IDF model on the item descriptions to create a TF-IDF matrix.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.data['description'])
        self.indices = pd.Series(self.data.index, index=self.data['title']).drop_duplicates()

    def recommend(self, title, top_n=5):
        """
        Provides recommendations based on the item title.
        
        Args:
            title (str): The title of the item to base recommendations on.
            top_n (int): The number of recommendations to return.
        
        Returns:
            List[str]: List of recommended item titles.
        """
        idx = self.indices.get(title)
        if idx is None:
            return []
        cosine_similarities = linear_kernel(self.tfidf_matrix[idx], self.tfidf_matrix).flatten()
        related_indices = cosine_similarities.argsort()[-top_n-1:-1][::-1]
        return self.data['title'].iloc[related_indices].tolist()

# Mock data for demonstration
if __name__ == '__main__':
    sample_data = pd.DataFrame({
        'title': ['Introduction to Machine Learning', 'Deep Learning with Python', 'Data Science Handbook', 'Machine Learning Yearning', 'Python for Data Analysis'],
        'description': [
            'A comprehensive introduction to machine learning concepts and techniques.',
            'Deep learning book that provides a hands-on approach.',
            'Essential tools and techniques for data analysis.',
            'Practical guide to machine learning methodologies.',
            'Covers the basics of data analysis with Python.'
        ]
    })
    recommender = ContentBasedRecommender(sample_data)
    recommender.fit()
    recommendations = recommender.recommend('Deep Learning with Python')
    print(recommendations)