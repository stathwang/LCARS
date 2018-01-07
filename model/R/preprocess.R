library(Matrix)
library(plyr)
library(data.table)
library(ggplot2)
library(pryr)

f <- fread("/data/douban_data_trunc.tsv")

# stratified split into train and test
set.seed(321)
test <- rbindlist(llply(unique(f$user_id), function(x) {
  temp <- f[user_id == x]
  test_ind <- sample(seq_len(temp[,.N]), size = floor(0.25 * temp[,.N]))
  return(temp[test_ind])
}))
train <- setdiff(f, test)
