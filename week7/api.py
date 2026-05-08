from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(title="Movie Discovery Engine API")

class RecommendationEngine:
    def __init__(self, movies_df: pd.DataFrame, all_movie_ids: List[int]):
        # Mock model for API template purposes
        self.model = None 
        self.movies_df = movies_df
        self.all_movie_ids = all_movie_ids
        
    def get_top_k_recommendations(self, user_id: int, top_k: int = 10) -> List[str]:
        """
        Generates Top-K personalized movie recommendations for a given user.
        """
        # In a real-world scenario:
        # 1. Fetch movies the user has already interacted with.
        # 2. Filter candidate movies (all_movie_ids - interacted_movies).
        # 3. Use self.model to predict ratings for all candidate movies.
        # 4. Sort predictions in descending order and slice the top_k.
        
        # Mock implementation for structural demonstration based on phase 4 of the notebook
        mock_recommended_ids = self.all_movie_ids[:top_k]
        recommended_titles = self.movies_df[
            self.movies_df['movieId'].isin(mock_recommended_ids)
        ]['title'].tolist()
        
        return recommended_titles

# Load mock dataset for the API
try:
    movies_df = pd.read_csv('./dataset/movies.csv')
    ratings_df = pd.read_csv('./dataset/ratings.csv')
    all_movie_ids = ratings_df['movieId'].unique().tolist()
except FileNotFoundError:
    # Fallback to empty data if datasets are missing
    movies_df = pd.DataFrame(columns=['movieId', 'title'])
    all_movie_ids = [1, 2, 3, 4, 5]

# Initialize the engine
engine = RecommendationEngine(movies_df=movies_df, all_movie_ids=all_movie_ids)

@app.get("/recommendations/{user_id}", response_model=Dict[str, Any])
def read_recommendations(user_id: int, top_k: int = 5):
    try:
        recs = engine.get_top_k_recommendations(user_id, top_k)
        return {"user_id": user_id, "recommendations": recs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # To run this script: uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
