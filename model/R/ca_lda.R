library(Matrix)
library(plyr)
library(data.table)
library(ggplot2)

# Content-Aware Latent-Dirichlet-Allocation for Spatial Item Recommendation

# Each user is viewed as a document and spatial items visited by a user are viewed as the words in a document
# dat <- train[, .(user_id, event_id, event_category)]
dat <- train[, .(user_id, event_id, event_location, event_category, user_location)]
vocab <- unique(dat$event_id)
users <- unique(dat$user_id)
cats <- unique(dat$event_category)
dat$uid <- match(dat$user_id, users)
dat$vid <- match(dat$event_id, vocab)
dat$cid <- match(dat$event_category, cats)
docs <- dlply(dat[,.(uid, vid, cid)], .(uid), function(x) alply(x, 1, function(y) as.numeric(y)[-1]))

# Initialize the hyperparameters
K <- 150
alpha <- 50/K
beta <- betap <- 0.01
niters <- 600
burnin <- 100
thin <- 1

# Initialize the matrices
uk <- matrix(0, length(users), K)
ck <- matrix(0, length(cats), K)
vk <- matrix(0, length(vocab), K)
ta <- llply(unique(dat$uid), function(x) rep(0, dat[uid == x, .N]))
for (d in 1:length(docs)) {
  for (r in 1:length(docs[[d]])) {
    ta[[d]][r] <- sample(K, 1)
    ti <- ta[[d]][r]
    tup <- docs[[d]][[r]]
    uk[d, ti] <- uk[d, ti] + 1
    vk[tup[1], ti] <- vk[tup[1], ti] + 1
    ck[tup[2], ti] <- ck[tup[2], ti] + 1
  }
}

# Initialize the model parameter matrices
theta <- matrix(0, length(users), K)
phi <- matrix(0, length(vocab), K)
nu <- matrix(0, length(cats), K)

# Collapsed gibbs sampling here
starttime <- Sys.time()
for (i in 1:niters) {
  for (d in 1:length(docs)) {
    for (r in 1:length(docs[[d]])) {
      t0 <- ta[[d]][r]
      tup <- docs[[d]][[r]]
      uk[d, t0] <- uk[d, t0] - 1
      vk[tup[1], t0] <- vk[tup[1], t0] - 1
      ck[tup[2], t0] <- ck[tup[2], t0] - 1
      multp <- (uk[d,] + alpha) / (sum(uk[d,]) + K * alpha) *
        (vk[tup[1],] + beta) / (colSums(vk) + length(vocab) * beta) *
        (ck[tup[2],] + betap) / (colSums(ck) + length(cats) * betap)
      tnew <- sample(K, 1, prob = multp)
      ta[[d]][r] <- tnew
      uk[d, tnew] <- uk[d, tnew] + 1
      vk[tup[1], tnew] <- vk[tup[1], tnew] + 1
      ck[tup[2], tnew] <- ck[tup[2], tnew] + 1
      # if (t0 != tnew) cat(paste0('user:', users[d], ' event:', vocab[tup[1]], ' word:', cats[tup[2]], ' topic:', t0, ' => ', tnew, '\n'))
      
      # Update the model parameter matrics
      # Later iterations get more weight
      if (i > burnin & i %% thin == 0) {
        inv_wgt <- thin / (i - burnin)
        theta[d, tnew] <- theta[d, tnew] + inv_wgt * (uk[d, tnew] + alpha) / sum(uk[d,] + alpha)
        phi[tup[1], tnew] <- phi[tup[1], tnew] + inv_wgt * (vk[tup[1], tnew] + beta) / sum(vk[, tnew] + beta)
        nu[tup[2], tnew] <- nu[tup[2], tnew] + inv_wgt * (ck[tup[2], tnew] + betap) / sum(ck[, tnew] + betap)
      }
    }
  }
}
stoptime <- Sys.time()
cat(stoptime - starttime)

# Normalize the model parameter matrices
theta1 <- t(scale(t(theta), center = FALSE, scale = colSums(t(theta))))
phi1 <- scale(phi, center = FALSE, scale = colSums(phi))
nu1 <- scale(nu, center = FALSE, scale = colSums(nu))

