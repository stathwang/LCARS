library(Matrix)
library(plyr)
library(data.table)
library(ggplot2)

# Location-Aware Latent-Dirichlet-Allocation for Spatial Item Recommendation

# Each user is viewed as a document and spatial items visited by a user are viewed as the words in a document
# dat <- train[, .(user_id, event_id, event_location)]
dat <- train[, .(user_id, event_id, event_location, event_category, user_location)]
vocab <- unique(dat$event_id)
users <- unique(dat$user_id)
cities <- unique(dat$event_location)
dat$uid <- match(dat$user_id, users)
dat$vid <- match(dat$event_id, vocab)
dat$pid <- match(dat$event_location, cities)
docs <- dlply(dat[,.(uid, vid, pid)], .(uid), function(x) alply(x, 1, function(y) as.numeric(y)[-1]))

# Initialize the hyperparameters
K <- 2
alpha <- alphap <- 2/K
gamma <- gammap <- 1
beta <- 0.01
niters <- 20
burnin <- 1
thin <- 1

# Initialize the matrices
u2 <- matrix(0, length(users), 2)
uk <- matrix(0, length(users), K)
pk <- matrix(0, length(cities), K)
vk <- matrix(0, length(vocab), K)
ta <- cf <- llply(unique(dat$uid), function(x) rep(0, dat[uid == x, .N]))
for (d in 1:length(docs)) {
  for (r in 1:length(docs[[d]])) {
    ta[[d]][r] <- sample(K, 1)
    cf[[d]][r] <- sample(0:1, 1)
    ti <- ta[[d]][r]
    ci <- cf[[d]][r]
    tup <- docs[[d]][[r]]
    u2[d, ci + 1] <- u2[d, ci + 1] + 1
    uk[d, ti] <- uk[d, ti] + 1
    vk[tup[1], ti] <- vk[tup[1], ti] + 1
    pk[tup[2], ti] <- pk[tup[2], ti] + 1
  }
}

# Initialize the model parameter matrices
theta <- matrix(0, length(users), K)
phi <- matrix(0, length(vocab), K)
rho <- matrix(0, length(cities), K)
lambda <- matrix(0, length(users), 2)

# Collapsed gibbs sampling here
starttime <- Sys.time()
for (i in 1:niters) {
  for (d in 1:length(docs)) {
    for (r in 1:length(docs[[d]])) {
      t0 <- ta[[d]][r]
      c0 <- cf[[d]][r]
      tup <- docs[[d]][[r]]
      u2[d, c0 + 1] <- u2[d, c0 + 1] - 1
      
      if (c0 == 1) {
        uk[d, t0] <- uk[d, t0] - 1
      } else {
        pk[tup[2], t0] <- pk[tup[2], t0] - 1
      }
      
      # c1: prob of s = 1
      m1 <- (uk[d, t0] + alpha) / (sum(uk[d,]) + K * alpha)
      m2 <- (u2[d, 2] + gamma) / (sum(u2[d,]) + gamma + gammap)
      c1 <-  m1 * m2
      
      # c0: prob of s = 0
      n1 <- (pk[tup[2], t0] + alphap) / (sum(pk[tup[2],]) + K * alphap)
      n2 <- (u2[d, 1] + gammap) / (sum(u2[d,]) + gamma + gammap)
      c0 <- n1 * n2
      
      # sample a new coin
      cnew <- sample(0:1, 1, prob = c(c0, c1))
      cf[[d]][r] <- cnew
      u2[d, cnew + 1] <- u2[d, cnew + 1] + 1
      
      vk[tup[1], t0] <- vk[tup[1], t0] - 1
      
      # If s = 1, then update using user matrix
      # else if s = 0, update using event location matrix
      if (cnew == 1) {
        multp <- (uk[d,] + alpha) / (sum(uk[d,]) + K * alpha) *
          (vk[tup[1],] + beta) / (colSums(vk) + length(vocab) * beta)
        tnew <- sample(K, 1, prob = multp)
        uk[d, tnew] <- uk[d, tnew] + 1
      } else {
        multp <- (pk[tup[2],] + alphap) / (sum(pk[tup[2],]) + K * alphap) *
          (vk[tup[1],] + beta) / (colSums(vk) + length(vocab) * beta)
        tnew <- sample(K, 1, prob = multp)
        pk[tup[2], tnew] <- pk[tup[2], tnew] + 1
      }
      ta[[d]][r] <- tnew
      vk[tup[1], tnew] <- vk[tup[1], tnew] + 1
      # if (t0 != tnew) cat(paste0('user:', users[d], ' event:', vocab[tup[1]], ' city:', cities[tup[2]], ' topic:', t0, ' => ', tnew, '\n'))
      
      # Update the model parameter matrices
      # Later iterations get more weight
      if (i > burnin & i %% thin == 0) {
        inv_wgt <- thin / (i - burnin)
        theta[d, tnew] <- theta[d, tnew] + inv_wgt * (uk[d, tnew] + alpha) / sum(uk[d,] + alpha)
        phi[tup[1], tnew] <- phi[tup[1], tnew] + inv_wgt * (vk[tup[1], tnew] + beta) / sum(vk[, tnew] + beta)
        rho[tup[2], tnew] <- rho[tup[2], tnew] + inv_wgt * (pk[tup[2], tnew] + alphap) / sum(pk[, tnew] + alphap)
        if (cnew == 1) lambda[d, 2] <- lambda[d, 2] + inv_wgt * (u2[d, 2] + gamma) / (sum(u2[d,]) + gamma + gammap)
        else lambda[d, 1] <- lambda[d, 1] + inv_wgt * (u2[d, 1] + gammap) / (sum(u2[d,]) + gamma + gammap)
      }
    }
  }
}
stoptime <- Sys.time()
cat(stoptime - starttime)

# Normalize the model parameter matrices
theta1 <- t(scale(t(theta), center = FALSE, scale = colSums(t(theta))))
phi1 <- scale(phi, center = FALSE, scale = colSums(phi))
rho1 <- scale(rho, center = FALSE, scale = colSums(rho))
lambda1 <- t(scale(t(lambda), center = FALSE, scale = colSums(t(lambda))))

