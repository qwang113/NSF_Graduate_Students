rm(list = ls())
library(tidyverse)
library(glmnet)
library(tscount)
library(Matrix)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])

rCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1)){
  ## This function will simulate from the cMLG distribution
  m <- length(kappa)
  w <- log(rgamma(m, shape=alpha, rate=kappa))
  sparse_H <- Matrix(H, sparse = TRUE)
  return(as.numeric(solve( crossprod(sparse_H) )%*%t(sparse_H)%*%w))
}



ESN_expansion <- function(Xin = matrix(0,), Yin, Xpred, nh=100, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1, eps = 1){
  ## Fit
  p <- ncol(Xin)
  W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,1-pw), nrow=nh)
  W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
  U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,1-pu), nrow=nh)
  Uy <- matrix(runif(nh, min = -au, max = au), nrow = nh) * matrix(rbinom(nh,1,1-pu), ncol = 1)
  H <- tmp <- matrix(0, nrow = nrow(schoolsM), ncol=nh)
  for(i in 2:ncol(Yin)){
    tmp_new <- tanh(tmp%*%W + matrix(log(Yin[,i-1] + eps), ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W + matrix( log(Yin[,ncol(Yin)] + eps), ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H[-c(1:nrow(Xin)),], "pred_h" = Hpred))
}



state_idx <- model.matrix( ~ factor(state) -1, data = schools)

# MCMC parameters
total_samples <- 1000
alpha = 1000
years_to_pred <- 46:50

#Hyper-parameters
sig_eta_inv = 10
eps = 1 # Avoid underflow, avoid log(0)

# ESN Parameters by CV
nh = 30
all_nu = c(0.9,0.1,0.9,0.1,0.9)
all_a = c(0.01,0.01,0.01,0.1,0.01)
pw = pu = 0.1
reps = 1000


# Initialization
pred_all_sep <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),total_samples))

for (years in years_to_pred) {
  
  nu <- all_nu[years-45]
  aw = au = all_a[years-45]
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:(years-1)]
  
  # Generate H
  H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
  
  # Number of times to repeat
  n <- ncol(Yin[,-1])
  # Repeat the matrix and bind by rows
  repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
  design_mat <- cbind(H$train_h, as.vector(log(schoolsM[,1:(years-2)]+1)))
  
  
  # Input Data
  nh <- dim(H$train_h)[2]
  ns <- dim(state_idx)[2]
  y_tr <- as.vector(Yin[,-1])
  
  # Posterior sample boxes
  tilde_eta <- matrix(NA, ncol = total_samples, nrow = ncol(design_mat))
  sep_eta_pred <- matrix(NA, nrow = nrow(schoolsM), ncol = total_samples)
  
  # Bayesian - separate fitting model ---------------------------------------------------------------------------
  for (idx in 1:total_samples) {
    print(paste(years,idx))
    for (i in 1:nrow(schoolsM)) {
      # curr_idx_h <- seq(from = i, to = i + nrow(schoolsM)*(nrow(H$train_h)/nrow(schoolsM) - 1), by = nrow(schoolsM))
      curr_idx_h <- i + (0:(years - 3))*nrow(schoolsM)
      curr_H <- design_mat[curr_idx_h,]
      curr_H_sep <- rbind(curr_H,alpha^{-1/2}*diag(rep(sig_eta_inv,nh+1)))
      curr_y <- Yin[i,2:(years-1)]
      curr_alpha <- matrix(c(curr_y+eps, rep(alpha,nh+1)), ncol = 1)
      curr_kappa <- matrix(c(rep(1, ncol(Yin[,-1])), rep(alpha, nh+1)), ncol = 1)
      curr_pos_eta <- rCMLG(H = curr_H_sep, alpha = curr_alpha, kappa = curr_kappa)
      sep_eta_pred[i,idx] <- exp( c(H$pred_h[i,],log(schoolsM[i,years-1]+1)) %*% curr_pos_eta)
    }
    print(mean((sep_eta_pred[,idx]-schoolsM[,years])^2))
  }
  
  pred_all_sep[years-min(years_to_pred)+1,,] <- sep_eta_pred
}
setwd(here::here())
saveRDS(pred_all_sep, file="pred_all_sep.Rda")
