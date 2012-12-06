# Setup and load data
setwd("/Users/logang/Documents/Code/python/Stat240")
SandP500 = as.matrix(read.csv("data/SandP500_returns.csv",header=F))
FF = as.matrix(read.csv("data/FF_returns.csv",header=F))
dates = read.csv("data/dates.csv")
colnames(FF) = c('smlo_ret', 'smme_ret', 'smhi_ret', 'bilo_ret', 'bime_ret', 'bihi_ret')

# get number of observations and stocks
n = dim(FF)[1]
m = dim(FF)[2] 

# Regress Fama-French returns on SandP500 
fit = lm(FF~SandP500)


