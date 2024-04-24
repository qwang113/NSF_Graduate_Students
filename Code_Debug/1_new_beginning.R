{rm(list = ls())
  library(ggplot2)
  library(sp)
  library(fields)
  library(mvtnorm)
  library(FRK)
  library(utils)
  library(keras)
  library(glmnet)
  library(MASS)
  # use_condaenv("tf_gpu")
}
nsf_wide_car <- read.csv(here::here("nsf_final_wide_car.csv"))
nsf_y <- nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))]

adj_mat <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = nrow(nsf_wide_car))
for (i in 1:nrow(adj_mat)) {
  for (j in 1:nrow(adj_mat)) {
    adj_mat[i,j] = ifelse(nsf_wide_car$UNITID[i] == nsf_wide_car$UNITID[j], 1,0)
  }
}
adj_mat <- adj_mat - diag(1, nrow = nrow(nsf_wide_car))
# define how much information they share

weight_mat <- 0.01*adj_mat + diag(1, nrow = nrow(nsf_wide_car))
gcn_mat <- t(t(weight_mat) / rowSums(weight_mat))

nh <- 100 # Number of hidden units in RNN
min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}

pi_0 <- 0.3
num_ensemble <- 1
num_year <- length(2012:2021)
a_par <- c(1)
nu_par <- c(1)
res <- array(NA, dim = c(num_ensemble, num_year, length(a_par), length(nu_par)))
sigmoid <- function(x){return(exp(x)/(1+exp(x)))}
for (ensem_idx in 1:num_ensemble) {
  for (a_par_idx in 1:length(a_par) ) {
    a <- a_par[a_par_idx]
    for (nu_par_idx in 1:length(nu_par)) {
      for( curr_year in 2012:2012){
        W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)*rbinom(nh^2, 1, 1-pi_0)
        W_GCN <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)*rbinom(nh^2, 1, 1-pi_0)
        lambda_scale <- max(abs(eigen(W)$values))

        # Figure out the dimension of EOFs, first PCA
        prev_years <- nsf_y[,1:(curr_year-1972)]
        pc_res <- prcomp(t(prev_years))
        num_pc <- min(which(cumsum(pc_res$sdev^2)/(sum(pc_res$sdev^2)) >= 0.8 ))
        pc_use <- apply(pc_res$x[,1:num_pc],2, min_max_scale)
        U_pc <- matrix(runif(nh*num_pc, -a, a), ncol = nh)
        U_ar <- matrix(runif(nh, -a, a), ncol = 1) * rbinom(nh, 1, 1-pi_0)
        # U_sp <- matrix(runif(nh*ncol(conv_covar), -a, a), ncol = nh)
        # Ux_sp <- conv_covar %*% U_sp
        Ux_sp <- matrix(0, nrow = nrow(nsf_wide_car), ncol = nh)
        curr_H <-  sigmoid(gcn_mat%*%Ux_sp%*%W_GCN)
        Y <- nsf_y[,1]
        for (year in 2:(curr_year-1972+1)) {
          new_H <- 
            sigmoid(
            nu_par[nu_par_idx]/lambda_scale * curr_H[((year-2)*nrow(nsf_y)+1):nrow(curr_H),] %*% W + 
              (matrix( log(nsf_y[,year-1]+1))) %*% t(U_ar)
          )
          new_H <- max(0,gcn_mat%*%new_H%*%W_GCN)
          curr_H <- rbind(curr_H, new_H)
          Y <- c(Y, nsf_y[,year])
        }
        # curr_H <-cbind(curr_H, curr_H^2)
        colnames(curr_H) <- paste("node", 1:ncol(curr_H))
        obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
        pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
        print(paste("Now predicting year",curr_year))
        years_before <- curr_year - 1972
        obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
        
        # I used glm to verify the previous code when lambda = 0, it's not wrong.
        glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)), control = glm.control(trace = TRUE))
        glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))
        pois_llh <- sum(dpois(nsf_wide_car$X2012, lambda = glm_pred,log = TRUE))
        res[ensem_idx,curr_year-2011, a_par_idx, nu_par_idx] <- mean((nsf_y[,curr_year-1972+1] - glm_pred)^2)
        print(mean((nsf_y[,curr_year-1972+1] - glm_pred)^2)) 
        }
    }
  }
}

# year 2017 has something strange
# outliers_2017 <- order((glm_pred-nsf_y$X2017)^2, decreasing = TRUE)
# outliers_2020 <- order((glm_pred-nsf_y$X2020)^2, decreasing = TRUE)
# m = 1
# plot(unlist(nsf_y[outliers_2017[m],]), type = 'l', x = 1972:2021, main = nsf_wide_car$UNITID[outliers_2017[m]])
# 
# abline(v = 2017, col = "red")



# Fit gcn with weight matrix w
# with every weight matrix trained, including cnn
