
library(Matrix)
library(plyr)
library(data.table)
library(ggplot2)
library(pryr)

rm(list = ls())

options(scipen = 8)

# each user is viewed as a document
# spatial items visited by a user are viewed as the words in a document
# dat <- train[, .(user_id, event_id)]
dat <- train[, .(user_id, event_id, event_location, event_category, user_location)]
vocab <- unique(dat$event_id)
users <- unique(dat$user_id)
dat$uid <- match(dat$user_id, users)
dat$vid <- match(dat$event_id, vocab)
docs <- llply(unique(dat$uid), function(x) dat[uid == x, vid])

# initialize the parameters
# increase niters to > 25000 during the actual model fitting
#K <- 25
K <- 3 #100
#alpha <- 25/K
alpha <- 3/K #100/K
eta <- 0.01
#niters <- 12000
#burnin <- 2000
niters <- 10 #1500
burnin <- 1 #100
thin <- 1

# initialize the topics
# nk <- colSums(wt)
wt <- matrix(0, length(vocab), K)
dt <- matrix(0, length(docs), K)
nk <- rep(0, K)
ta <- sapply(docs, function(x) rep(0, length(x)))
for (d in 1:length(docs)) {
  for (w in 1:length(docs[[d]])) {
    ta[[d]][w] <- sample(K, 1)
    ti <- ta[[d]][w]
    wi <- docs[[d]][w]
    wt[wi, ti] <- wt[wi, ti] + 1
    dt[d, ti] <- dt[d, ti] + 1
    nk[ti] <- nk[ti] + 1
  }
}

# model parameter matrices
theta <- matrix(0, length(users), K)
phi <- matrix(0, length(vocab), K)

# collapsed gibbs sampling
starttime <- Sys.time()
for (i in 1:niters) {
  for (d in 1:length(docs)) {
    for (w in 1:length(docs[[d]])) {
      t0 <- ta[[d]][w]
      wid <- docs[[d]][w]
      dt[d, t0] <- dt[d, t0] - 1
      wt[wid, t0] <- wt[wid, t0] - 1
      nk[t0] <- nk[t0] - 1
      multp <- (dt[d,] + alpha) * (wt[wid,] + eta) / (nk + length(vocab) * eta)
      tnew <- sample(K, 1, prob = multp)
      ta[[d]][w] <- tnew
      dt[d, tnew] <- dt[d, tnew] + 1
      wt[wid, tnew] <- wt[wid, tnew] + 1
      nk[tnew] <- nk[tnew] + 1
      # if (t0 != tnew) cat(paste0('user:', users[d], ' token:', w, ' word:', vocab[w], ' topic:', t0, ' => ', tnew, '\n'))
      
      # update model parameters
      # later iterations get more weight
      if (i > burnin & i %% thin == 0) {
        inv_wgt <- thin / (i - burnin)
        theta[d, tnew] <- theta[d, tnew] + inv_wgt * (dt[d, tnew] + alpha) / sum(dt[d,] + alpha)
        phi[wid, tnew] <- phi[wid, tnew] + inv_wgt * (wt[wid, tnew] + eta) / sum(wt[, tnew] + eta)
      }
    }
  }
}
stoptime <- Sys.time()
cat(stoptime - starttime)

# normalize the model parameter matrices
theta1 <- t(scale(t(theta), center = FALSE, scale = colSums(t(theta))))
phi1 <- scale(phi, center = FALSE, scale = colSums(phi))

