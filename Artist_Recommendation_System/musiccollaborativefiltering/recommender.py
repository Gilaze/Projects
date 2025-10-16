from pathlib import Path
from typing import Tuple, List

import implicit
import scipy

from musiccollaborativefiltering.data import load_user_artists, ArtistRetriever

# Get the directory of the current script to build robust paths
SCRIPT_DIR = Path(__file__).resolve().parent


class ImplicitRecommender:

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[n], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores


if __name__ == "__main__":

    # Construct the path to the data directory relative to this script
    DATA_DIR = SCRIPT_DIR.parent / "lastfmdata"

    # load user artists matrix
    user_artists = load_user_artists(DATA_DIR / "user_artists.dat")

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(DATA_DIR / "artists.dat")

    # instantiate ALS using implicit
    implict_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender and fit
    recommender = ImplicitRecommender(artist_retriever, implict_model)
    print("Fitting the model...")
    recommender.fit(user_artists)
    print("Model fitting complete!")

    # --- MODIFICATION FOR USER INPUT ---
    # Loop to allow multiple recommendations
    while True:
        try:
            # Prompt the user for input
            user_input = input("\nEnter a User ID to get recommendations (or 'quit' to exit): ")

            if user_input.lower() == 'quit':
                print("Exiting the recommender. Goodbye!")
                break

            user_id = int(user_input)

            # Get and print recommendations
            print(f"\nGenerating top 5 recommendations for User {user_id}...")
            artists, scores = recommender.recommend(user_id, user_artists, n=5)

            print("-" * 30)
            for artist, score in zip(artists, scores):
                print(f"{artist}: {score:.4f}")
            print("-" * 30)

        except ValueError:
            print("Invalid input. Please enter a number for the User ID.")
        except Exception as e:
            print(f"An error occurred for User ID {user_input}. They may not exist in the dataset. Please try another ID.")

