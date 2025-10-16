# This is a placeholder for your data.py file.
# The original code was not provided, but you should apply the same
# path fix here if you are running this file directly.

from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix

# --- ADD THIS FOR ROBUST PATHS ---
SCRIPT_DIR = Path(__file__).resolve().parent

def load_user_artists(user_artists_file: Path) -> csr_matrix:
    """Load the user artists file and return a user-artists matrix."""
    user_artists = pd.read_csv(user_artists_file, sep="\t")

    # The .to_sparse() method is deprecated in newer versions of pandas.
    # A more direct way to create the sparse matrix is to use the
    # scipy.sparse.coo_matrix constructor directly from the DataFrame columns.
    rows = user_artists["userID"].values
    cols = user_artists["artistID"].values
    data = user_artists["weight"].values

    coo = coo_matrix((data, (rows, cols)))

    return coo.tocsr()

class ArtistRetriever:
    """A class to retrieve artist names from their IDs."""
    def __init__(self):
        self._artists = pd.DataFrame()

    def load_artists(self, artists_file: Path) -> None:
        """Load the artists file."""
        artists = pd.read_csv(artists_file, sep="\t")
        artists.set_index("id", inplace=True)
        self._artists = artists

    def get_artist_name_from_id(self, artist_id: int) -> str:
        """Return the artist name for the given ID."""
        return self._artists.loc[artist_id, "name"]

if __name__ == "__main__":
    # --- APPLY THE PATH FIX HERE AS WELL ---
    DATA_DIR = SCRIPT_DIR.parent / "lastfmdata"

    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(DATA_DIR / "artists.dat")
    artist = artist_retriever.get_artist_name_from_id(1)
    print(f"Artist with ID 1 is: {artist}")

