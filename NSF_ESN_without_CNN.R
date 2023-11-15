rm(list = ls())
library(ggplot2)
library(sp)
library(fields)
library(mvtnorm)
library(FRK)
library(utils)
library(keras)
library(reticulate)
library(tensorflow)
library(glmnet)
library(MASS)
use_condaenv("tf_gpu")
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)

nsf_wide$long <- jitter(nsf_wide$long)
nsf_wide$lat <- jitter(nsf_wide$lat)

coords <- data.frame("long" = nsf_wide$long,"lat" = nsf_wide$lat)

# Initialize Convolutional Structure: Basis Functions

# Basis Generating
long <- coords$long
lat <- coords$lat


leak_rate <- 1 # It's always best to choose 1 here according to Mcdermott and Wille, 2017
nh <- 2000 # Number of hidden units in RNN
nx_sp <- ncol(coords) # Number of covariates
# nx_dummy <- ncol(dummy_matrix)

# The range of the standard uniform distribution of the weights
nu <- 0.9
time_step <- length(unique(nsf_long$year))

W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
# U_dummy <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
ar_col <- matrix(runif(nh,-a,a), nrow = 1)


# W <- matrix(rnorm(nh^2, sd = sig), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
# U_sp <- matrix(rnorm(nh*nx_sp, sd = sig), nrow = nx_sp, ncol = nh)
# U_dummy <- matrix(rnorm(nh*nx_dummy, sd = sig), nrow = nx_dummy, ncol = nh)
# ar_col <- matrix(rnorm(nh, sd = sig), nrow = 1)


lambda_scale <- max(abs(eigen(W)$values))

ux_sp <- coords%*%U_sp
# ux_dummy <- dummy_matrix%*%U_dummy



# curr_H <- ux_sp + ux_dummy
curr_H <- ux_sp
Y <- nsf_wide[,4]

pb <- txtProgressBar(min = 1, max = length(2:time_step), style = 3)
for (i in 2:time_step) {
  setTxtProgressBar(pb,i)
  # new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + ux_dummy + nsf_wide[,i+2]%*%ar_col, c(1,2), tanh)
  new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + nsf_wide[,i+2]%*%ar_col, c(1,2), tanh)
  Y <- c(Y, nsf_wide[,i+3])
  curr_H <- rbind(curr_H, new_H)
}

# pca_H <- prcomp(curr_H)
# pca_var <-predict(pca_H)
# write.csv(curr_H, "D:/77/UCSC/study/Research/temp/NSF_dat/CRESN_H.csv", row.names = FALSE)


glm_CRESN <- glm.nb(Y~curr_H, control = glm.control(epsilon = 1e-8, maxit = 50, trace = TRUE))

CRESN_res <- glm_CRESN$residuals

year_stack <- rep(1972:2021, each = nrow(nsf_wide))
school_ID <- rep(nsf_wide$ID,50)
long_stack <- rep(nsf_wide$long,50)
lat_stack <- rep(nsf_wide$lat,50)

res_stack <- data.frame("ID" = school_ID, "long" = long_stack, "lat" = lat_stack, "year" = year_stack, "Residuals" = CRESN_res)

write.csv(res_stack, paste("D:/77/UCSC/study/Research/temp/NSF_dat/", "ESN_res", nh, ".csv", sep = ""), row.names = FALSE)
