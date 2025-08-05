import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class ContentBasedRecommender:
    def __init__(self, articles):
        """Initializes the recommender with articles data."""
        self.articles = articles
        self.tfidf_matrix = None
        self.indices = None
        self._preprocess()

    def _preprocess(self):
        """Preprocesses articles and computes TF-IDF matrix."""
        tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = tfidf.fit_transform(self.articles['content'])
        self.indices = pd.Series(self.articles.index, index=self.articles['title']).to_dict()

    def recommend(self, title, top_n=5):
        """Recommends articles based on content similarity."""
        if title not in self.indices:
            raise ValueError('Title not found in articles.')
        cosine_sim = linear_kernel(self.tfidf_matrix[self.indices[title]], self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n + 1]
        article_indices = [i[0] for i in sim_scores]
        return self.articles['title'].iloc[article_indices].tolist()

# Mock data
if __name__ == '__main__':
    mock_data = pd.DataFrame({
        'title': ['Article 1', 'Article 2', 'Article 3', 'Article 4'],
        'content': [
            'Deep learning and AI are transforming the world.',
            'Machine learning provides systems the ability to learn.',
            'Natural language processing helps computers understand human language.',
            'Content-based filtering is an important recommendation system.'
        ]
    })
    recommender = ContentBasedRecommender(mock_data)
    recommendations = recommender.recommend('Article 1')
    print('Recommendations:', recommendations)