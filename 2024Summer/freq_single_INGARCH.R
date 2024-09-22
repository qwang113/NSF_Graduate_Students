rm(list = ls())
library(tidyverse)
library(glmnet)
library(tscount)
library(Matrix)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])

ESN_expansion <- function(Xin, Yin, Xpred, nh=100, nu=0.8, aw=0.1, pw=0.1, au=0.1, pu=0.1, eps = 1){
  ## Fit
  p <- ncol(Xin)
  W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,1-pw), nrow=nh)
  W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
  U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,1-pu), nrow=nh)
  Uy <- matrix(runif(nh, min = -au, max = au), nrow = nh) * matrix(rbinom(nh,1,1-pu), ncol = 1)
  H <- matrix(NA, nrow=nrow(Xin), ncol=nh)
  tmp <- tanh(Xin %*% t(U))
  H <- tmp
  for(i in 2:ncol(Yin)){
    tmp_new <- tanh(tmp%*%W + Xin%*%t(U) + matrix( log(Yin[,i-1] + eps), ncol = 1 ) %*% t(Uy) ) 
    tmp <- tmp_new
    H <- rbind(H, tmp_new)
  }
  Hpred <- tanh(H[(nrow(H)-nrow(tmp)+1):nrow(H), ]%*%W + Xin%*%t(U) + matrix( log(Yin[,ncol(Yin)] + eps), ncol = 1 ) %*% t(Uy)) 
  return(list("train_h" = H, "pred_h" = Hpred))
}

state_idx <- model.matrix( ~ factor(state) -1, data = schools)
years_to_pred <- 46:50

#Hyper-parameters
eps = 1 # Avoid underflow, avoid log(0)

# ESN Parameters
nh = 30
nu = 0.9
aw = au = 0.01
pw = pu = 0.1

# Penalization parameter for Lasso/Ridge for ESN in frequentist view
lambda <- 1

# Initialization
pred_all_single_esn <- matrix(NA, nrow = nrow(schoolsM), ncol = length(years_to_pred))

# INGARCH model has model uncertainty, 3 dimensions to save CI's
pred_all_ING <-  array(NA, dim = c(3,length(years_to_pred), nrow(schoolsM)))



for (years in years_to_pred) {
  # Set up hypeparameters for ESN
  Xin <- Xpred <- model.matrix( ~ factor(state) -1, data = schools)
  Yin <- schoolsM[,1:(years-1)]
  
  # Generate H
  H <- ESN_expansion(Xin = state_idx, Yin = Yin, Xpred = state_idx, nh=nh, nu=nu, aw=aw, pw=pw, au=au, pu=pu, eps = eps)
  
  # Number of times to repeat
  n <- ncol(Yin)
  # Repeat the matrix and bind by rows
  repeated_state <- do.call(rbind, replicate(n, state_idx, simplify = FALSE))
  design_mat <- cbind(H$train_h, repeated_state)
  
  
  # Input Data
  nh <- dim(H$train_h)[2]
  ns <- dim(state_idx)[2]
  y_tr <- as.vector(Yin)
  
  # Frequentist - Integrated Model (Single)----------------------------------------------------------------------------------------  
  for (school_idx in 1:nrow(schoolsM)) {
    print(paste(years,school_idx,"Single ESN"))
    curr_idx_h <- school_idx + (0:(years - 2))*nrow(schoolsM)
    curr_H <- H$train_h[curr_idx_h,]
    curr_y <- y_tr[curr_idx_h]
    fit_fesn <- glmnet(x = curr_H, y = curr_y, family = "poisson", alpha = 1, lambda = 1)
    pred_all_single_esn[school_idx,years - min(years_to_pred) + 1] <- predict(fit_fesn, newx = H$pred_h[school_idx,], type = "response") 
  }
  print(mean((pred_all_single_esn[,1] - schoolsM[,46])^2))
  
  # Frequentist - INGARCH(1,1) ----------------------------------------------------------------------------------------  
  for (k in 1:nrow(schoolsM)) {
    print(paste(years,k,"INGARCH"))
    tmp_ingarch <- tsglm(Yin[k,1:(years-1)], model = list(past_obs = 1, past_mean = 1), dist = "poisson")
    pred_all_ING[1,years - min(years_to_pred) + 1,k] <- predict(tmp_ingarch, n.ahead = 1)$pred
    pred_all_ING[2,years - min(years_to_pred) + 1,k] <- predict(tmp_ingarch, n.ahead = 1)$interval[1]
    pred_all_ING[3,years - min(years_to_pred) + 1,k] <- predict(tmp_ingarch, n.ahead = 1)$interval[2]
  }
  
}

saveRDS(pred_all_single_esn, file="pred_all_single_esn.Rda")
saveRDS(pred_all_ING, file = "pred_all_ING.Rda")


# write.csv(tilde_eta, here::here("eta.csv"), row.names = FALSE)
# write.csv(sig_xi_inv, here::here("sig_xi_inv.csv"), row.names = FALSE)
# write.csv(H$pred_h,here::here("pred_h.csv"), row.names = FALSE)
# write.csv(int_eta_pred, here::here("int_eta_pred.csv"))
# write.csv(sep_eta_pred, here::here("sep_eta_pred.csv"))
