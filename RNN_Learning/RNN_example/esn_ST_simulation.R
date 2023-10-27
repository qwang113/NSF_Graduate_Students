rm(list = ls())
library(ggplot2)
library(sp)
library(fields)
library(mvtnorm)
library(FRK)
# Gaussian Process with Matern Correlation
longitude <- seq(from = 0, to = 1, length.out = 50)
latitude <- seq(from = 0, to = 1, length.out = 50)
coords <- expand.grid(longitude,latitude)
paired_dist = spDists(as.matrix(coords))
sd_spatial = 1
sd_temporal = 1.5
sd_residual = 0.5

d = 0.5
cov_mat <- sd_spatial * Exponential(paired_dist, range = d)
epsilon_spatial_1 <- rmvnorm(1, mean = rep(0, nrow(coords)), sigma = cov_mat)
covariates <- matrix(rnorm(2*nrow(coords)), ncol = 2)
beta_coef <- matrix(c(3,-1), ncol = 1)
y_1 <- as.vector(epsilon_spatial_1) + as.vector(covariates%*%beta_coef) + rnorm(length(epsilon_spatial_1), sd = sd_residual)
all_dat <- cbind(y_1, coords, covariates,1)
colnames(all_dat) <- c("y", "long","lat", "x1", "x2", "time")
ar_coef <- 0.5
curr_epsilon = epsilon_spatial_1
for (i in 2:100) {
  print(paste("Now doing index",i))
  curr_epsilon <- curr_epsilon * ar_coef + rnorm(length(curr_epsilon), sd = sd_temporal)
  curr_y <- as.vector(curr_epsilon) + as.vector(covariates%*%beta_coef) + rnorm(length(curr_epsilon), sd = sd_residual)
  curr_dat <- cbind(curr_y,coords, covariates,i)
  colnames(curr_dat) <- colnames(all_dat)
  all_dat <- rbind(all_dat, curr_dat)
}

# Visulize for certain time point

time_selected <- 10

dat_selected <- all_dat[which(all_dat$time == time_selected),]



ggplot(data = dat_selected) +
  geom_contour_filled(aes(x = long, y = lat, z = y))
