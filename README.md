# LCARS: A Spatial Item Recommender System

This repository contains my implementation of LCARS (Location-Content-Aware Recommender System) for spatial item recommendation. Please refer to the following paper for details.

Paper: [LCARS: A Spatial Item Recommender System](http://net.pku.edu.cn/~cuibin/Papers/2014%20TOIS%20-%20lcars.pdf)

Foursquare data is too large, so I'll have to truncate it before uploading. Only the truncated version of DoubanEvent data is uploaded to the `data` folder.

### Problems
1. Cold-start users and cities: How can we recommend spatial items to a new user traveling to a new city?
2. User checkin histories are locally clustered, so the traditional collaborative filtering methods would produce poor recommendations.
3. Data sparsity: User activites are very limited in distant locations.

### Main Idea
To produce a personalized spatial item recommendation, LCARS combines the querying user's interest and the local preference of the querying city, each of which is a distribution over latent topics where a latent topic is a distribution over spatial items and content words. The model assumes that items and their content words are independently conditioned on the topics. It learns topics that capture both item co-occurrence and content co-occurrence patterns.

Specifically, LCARS learns:
   - latent **topic** distribution over spatial items and that over content words
   - intrinsic **user interest** distribution over topics
   - **local preference** in a given region over topics

The `model` folder contains 2 different folders (`R` and `python`), each of which contains the following:
  - `lda.R (lda.py)`: Each user is a document and events (spatial items) visited by a user are the words in the document. The model ignores both location and content words of events and shows similar performance to user-based collaborative filtering. Each user is represented as a distribution over latent topics where a latent topic itself is a distribution over spatial items.
  - `ca_lda.R (ca_lda.py)`: This is the *content* component of the LCA-LDA model. It only considers the content words of a spatial item and ignores the location.
  - `la_lda.R`: This is the *location* component of the LDA-LDA model. It only considers the location of each spatial item, not the content words.
  - `lca_lda.R`: The final LCARS model that combines `CA-LDA.py` and `LA-LDA.py` taking into account both location and content words of each spatial item.

### Methods
The paper uses **collapsed Gibbs sampling** by only keeping track of coin flips and topic assignments for every record of a user checkin history. For the hyperparameter settings, see the paper.

First, split the data into training and test set (3:1) for every user checkin history. Fit the models above on the training set. For each user's checkin history in the test set, compute the score for events as well as all other events in a city not visited by a user.

Rank the events by sorting the scores in decreasing order. For every threshold *k* (see the paper), if an event in the test set is in top-*k*, then call that a hit. Average the scores for all test records.

### Results
1. LCA-LDA outperforms all the other models, while LDA show the worst performance. Local preference and item content information are indeed effective.
2. For the number of latent topics > 25, the increase in recommendation performance is rather small. This prompts to set the number of topics to 25 to balance out the model complexity and computational efficiency.
3. Users mainly participate in social events based on their interests, but from time to time they attend popular local events regardless of their interests. Local preference plays an important role.
4. Each latent topic captures event co-occurrence under same event category across different locations. LCA-LDA model helps reduce sparsity in user-item ratings matrix.
5. User interest, local preference, and content words are blended in a flexible, robust way to produce a very interpretable probabilistic model.
6. Further research may be needed. Can local preference be split into that of avid travelers and non-travelers, possibly a weighted sum of those two group preferences? Does correlation exist between locations and content words i.e. correlated topic models?
