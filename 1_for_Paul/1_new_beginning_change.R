{rm(list = ls())
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
  # use_condaenv("tf_gpu")
}
nsf_wide_car <- read.csv(here::here("nsf_final_wide_car.csv"))
wide_y <- nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))]
nh <- 15 # Number of hidden units in RNN
# min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}

pi_0 <- 0
num_ensemble <- 1
a_par <- c(0.001)
nu_par <- c(0.9)
# res <- array(NA, dim = c(num_ensemble, length(a_par), length(nu_par)))
a <- a_par

years_to_pred <- 2012:2021
pred_y <- matrix(NA, nrow = nrow(wide_y), ncol = length(years_to_pred))

for (curr_year in years_to_pred) {
        
        # Generate W weight matrices
        W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)
        # Calculate lambda
        lambda_scale <- max(abs(eigen(W)$values))
        # Initialize the first step covariates
        curr_x <- matrix(c(1,rep(0,nrow(wide_y))), ncol = 1)
        nx <- length(curr_x)
        # Generate random U
        U <- matrix(runif(nh*nx, -a, a), nrow = nh, ncol = nx) 
        # Calculate first step UX
        curr_ux <- U %*% curr_x
        # Activate UX
        curr_H <- tanh(curr_ux)
        H_mat <- curr_H

        for (year in 2:(curr_year-1972+1)) {
          # Define current X_t
          curr_x <- matrix(c(1, log(wide_y[,year-1]+1)), ncol = 1)
          # calculate h_t
          new_H <- tanh( nu_par/lambda_scale * W %*% curr_H + U %*% curr_x )
          # merge h_t to current H matrix
          H_mat <- cbind(H_mat,new_H)
          curr_H <- new_H
        }
        # For each school, we separately regress them
        for (school in 1:nrow(wide_y)) {
          print(paste("now doing year ", curr_year, "shcool",school))
          # Get all observations of first school
          obs_y <- t(wide_y[school,1:(curr_year-1972)])
          # Get all H matrix
          obs_h <- t(H_mat[,-c(ncol(H_mat))])
          # Generalized Ridge Regression cross validation
          # curr_model_cv <- cv.glmnet(x = obs_h, y = obs_y, nfolds = 3, alpha = 0,family = poisson(link = "log"))
          # plot(curr_model_cv)
          penalized_model <- glmnet(x = obs_h, y = obs_y, lambda = 0.1, alpha = 0,
                                     family = poisson(link = "log"), maxit = 1e5, trace.it = TRUE)
          # prediction for that certain school at certain year
          pred_y[school, curr_year-2011] <- predict(penalized_model, newx = t(H_mat[,ncol(H_mat)]), type = "response")
          
          # curr_dat <- data.frame(cbind(obs_y, obs_h))
          # 
          # model_glm <- glm(obs_y~obs_h, family = poisson(link = "log"))
          # curr_pred <- exp(sum(matrix(c(1,H_mat[,ncol(H_mat)]), ncol = 1) * coef(model_glm)))
          # pred_y[school, curr_year-2011] <- curr_pred
        }
      }
res_all <- as.matrix(wide_y)[,years_to_pred-1972+1] - pred_y
mean(res_all^2)





