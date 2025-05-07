rm(list = ls())
library(tidyverse)
library(glmnet)
library(tscount)
library(Matrix)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])

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

#Hyper-parameters
eps = 1 # Avoid underflow, avoid log(0)

# ESN Parameters
nh = 30
nu = 0.9
aw = au = 0.01
pw = pu = 0.1
reps = 1000

# Penalization parameter for Lasso/Ridge for ESN in frequentist view
lambda <- 1
years_to_pred <- 46:50
pred_all_ensemble_esn <- array(NA, dim = c(length(years_to_pred), nrow(schoolsM),reps))


for (years in years_to_pred) {
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:(years-1)]
  
  # Generate H
  
  H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
  
  # Number of times to repeat
  n <- ncol(Yin)
  # Repeat the matrix and bind by rows

  design_mat <- cbind(H$train_h)
  
  # Input Data
  nh <- dim(H$train_h)[2]
  ns <- dim(state_idx)[2]
  y_tr <- as.vector(Yin[,-1])
  
  tmp_all_schools <- rep(NA, nrow(schoolsM))
  # Frequentist - Integrated Model (Ensemble)----------------------------------------------------------------------------------------  
  for(j in 1:reps){
    H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
    for (school_idx in 1:nrow(schoolsM)) {
      
      curr_idx_h <- school_idx + (0:(years - 3))*nrow(schoolsM)
      curr_H <- H$train_h[curr_idx_h,]
      curr_y <- y_tr[curr_idx_h]
      fit_fesn <- glmnet(x = curr_H, y = curr_y, family = "poisson", alpha = 1, lambda = 1)
      tmp_all_schools[school_idx] <- predict(fit_fesn, newx = H$pred_h[school_idx,], type = "response") 
    }
    pred_all_ensemble_esn[years - min(years_to_pred) + 1,,j] <- tmp_all_schools
    print(j)
    print(mean((pred_all_ensemble_esn[years - min(years_to_pred) + 1,,j] - schoolsM[,years])^2) )
  }
}

saveRDS(pred_all_ensemble_esn, file = "pred_all_ensemble_esn.Rda")


