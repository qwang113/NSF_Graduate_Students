
nh <- 200 # Number of hidden units in RNN

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)[,-1]
dummy_state <- model.matrix(~nsf_wide_car$state - 1)[,-1]
dummy_gss <- model.matrix(~ substr(nsf_wide_car$ID, 7, 9)  - 1)[,-1]
dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)

a <- 0.1
one_step_ahead_pred_y_ridge <- one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = length(2012:2021))
leak <- 1
year = 2012
#Initialize
nx_sp <- ncol(conv_covar)
nx_dummy <- ncol(dummy_matrix)
nu <- 1
W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
U_dummy <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
ar_col <- matrix(runif(nh,-a,a), nrow = 1)
ar_col_lag2 <- matrix(runif(nh,-a,a), nrow = 1)
lambda_scale <- max(abs(eigen(W)$values))
ux_sp <- conv_covar%*%U_sp
# ux_dummy <- dummy_matrix%*%U_dummy
curr_H <- apply(ux_sp, c(1,2), tanh)
Y <- nsf_wide_car[,10]
pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
print("Calculating Recurrent H Matrix. . .")
for (i in 2:(year-1972+1)) {
  
  setTxtProgressBar(pb,i)
  new_H <- apply( 
    nu/lambda_scale*
      curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W
    # + ux_sp
    + log(nsf_wide_car[,i+8]+1)%*%ar_col
    , c(1,2), tanh)*leak + curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]*(1-leak)
  
  Y <- c(Y, nsf_wide_car[,i+9])
  curr_H <- rbind(curr_H, new_H)
}

# sp_cnn <- matrix(rep( t(convoluted_res1), year-1972+1), nrow = nrow(curr_H), byrow = TRUE)
# curr_H <- cbind(curr_H, sp_cnn)

colnames(curr_H) <- paste("node", 1:ncol(curr_H))
obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
print(paste("Now predicting year",year))
years_before <- year - 1972
obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]


# I used glm to verify the previous code when lambda = 0, it's not wrong.
glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)))
glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))

mean((glm_pred - nsf_wide_car$X2012)^2)

# 5000 nodes: 284.93
# 200 nodes: 335