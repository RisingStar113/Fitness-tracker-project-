# recommendation_model.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FitnessRecommendationSystem:
    def _init_(self, user_data):
        self.user_data = user_data  # user_data is a matrix of user activities and preferences

    def recommend_workout(self, user_id):
        # Similarity-based recommendation (Collaborative Filtering)
        similarity_matrix = cosine_similarity(self.user_data)
        similar_users = similarity_matrix[user_id]
        
        # Get the most similar users
        similar_user_ids = np.argsort(similar_users)[::-1]
        recommended_workouts = []

        for user in similar_user_ids:
            recommended_workouts.extend(self.user_data[user])  # Add workouts from similar users

        return recommended_workouts[:5]  # Return top 5 recommended workouts

# Example usage
user_data = np.random.rand(10, 20)  # 10 users, 20 workouts (randomized data)
recommender = FitnessRecommendationSystem(user_data)
print(recommender.recommend_workout(user_id=0))  # Recommend workouts for user 0