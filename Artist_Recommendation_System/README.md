credit to musikalkemist and his youtube video:
Build a Spotify-Like Music Recommender System in Python
Channel: Valerio Velardo - The Sound of AI

This code implements the logic of a collaborative filtering recommender system, as explained in the video, by translating user-music preferences into a problem of linear algebra. The core idea is that a massive matrix can be created where rows represent users and columns represent artists, with the values indicating how many times a user has played an artist's music. The video explains that this sparse matrix (mostly filled with zeros) can be decomposed, using techniques like matrix factorization, into two smaller matrices: one representing latent "taste" features for each user and another representing corresponding features for each artist. By multiplying a user's feature vector with the artist feature matrix, we can predict how much that user will like any given artist, even those they've never heard, and recommend the ones with the highest predicted scores.

ex. Friends A and B both enjoy listening to RnB. If Friend A introduced famous RnB artist Keshi to Friend B, Friend B would most likely enjoy the songs produced by Keshi.