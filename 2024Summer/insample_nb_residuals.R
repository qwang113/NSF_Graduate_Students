rm(list = ls())
W <- as.matrix(read.csv(here::here("W.csv")))
Uy <- read.csv(here::here("Uy.csv"))
U <- read.csv(here::here("U.csv"))
eta_rs <- read.csv(here::here("eta_rs.csv"))
rr <- read.csv(here::here("rr.csv"))
r_hat <- apply(rr,1,mean)
eta_hat <- apply(eta_rs, 1, mean)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
state_idx <- model.matrix( ~ factor(state) - 1, data = schools)
num_pred <- 1000
y_hat <- array(NA, dim = c(num_pred, dim(schoolsM)))
curr_H <- tmp <- tanh(state_idx %*% t(U))
inv_logit <- function(x){return(exp(x)/(1+exp(x)))}
eps <- 1
# n <- ncol(Yin)
for (iter in 1:num_pred) {
  # Generate current design matrix
  curr_design <- matrix(NA, nrow = nrow(schoolsM), ncol = (ncol(W)+1)*length(unique(schools$state)) )
  for (i in 1:nrow(schoolsM)) {
    curr_design[i,] <- c(as.vector(outer(curr_H[i,], state_idx[i,], "*")), state_idx[1,])
  }
  
  y_hat[iter,,1] <- rnbinom( nrow(schoolsM), size = r_hat, prob = inv_logit(curr_design %*% eta_hat) )
  
  for (years in 2:ncol(schoolsM)) {
    
    print(paste("Iter:",iter, "year",years))
    
    prev_y <- y_hat[iter, , years-1]
    new_H <- tanh( curr_H%*%W + matrix( log(prev_y + eps), ncol = 1 ) %*% t(Uy) ) 
    curr_H <- new_H
    curr_design <- matrix(NA, nrow = nrow(schoolsM), ncol = (ncol(W)+1)*length(unique(schools$state)) )
    for (i in 1:nrow(schoolsM)) {
      curr_design[i,] <- c(as.vector(outer(curr_H[i,], state_idx[i,], "*")), state_idx[1,])
    }
    y_hat[iter,,years] <- rnbinom( nrow(schoolsM), size = r_hat, prob = inv_logit(curr_design %*% eta_hat) )
  }
  
}
