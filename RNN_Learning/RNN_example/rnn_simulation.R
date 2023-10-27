rm(list = ls())
library(ggplot2)
library(keras)
library(reticulate)

use_condaenv("tf_gpu")
#https://www.geeksforgeeks.org/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/
#The link above is a good resource to learn

# Simulate an AR(2) Data Set
ar_2 <- rep(0, 5000)
phi_1 <- 0.5
phi_2 <- -0.3


covariates_ar2 <- cbind(rnorm(5000, 20, 20), rnorm(5000, -5, 10))
beta_1 <- 15
beta_2 <- -9

ar_2[1] <- rnorm(1)
ar_2[2] <- rnorm(1) + phi_1*ar_2[1]
set.seed(1)
for (i in 3:length(ar_2)) {
  ar_2[i] <- phi_1*ar_2[i-1] + phi_2*ar_2[i-2] + rnorm(1, 0, 0.5)
}


ar_2 <- ar_2 + covariates_ar2%*%matrix(c(beta_1, beta_2), ncol = 1)

# Plot the series out
ggplot() +
  geom_path(aes(x = 1:5000, y = ar_2))

arima_res <- arima(ar_2, order = c(2,0,0), xreg = covariates_ar2)

sim_dat <- cbind(ar_2, covariates_ar2)

# Build RNN 

# Split training and testing data
tr_dat_len <- ceiling(nrow(sim_dat)*0.8)
te_dat_len <- nrow(sim_dat)-ceiling(nrow(sim_dat)*0.8)

tr_dat <- sim_dat[1:tr_dat_len,]
te_dat <- sim_dat[(tr_dat_len+1):nrow(sim_dat),]




x_tr <-  array_reshape(tr_dat[,2:ncol(tr_dat)], c(1,tr_dat_len,2)) 
y_tr <-  array_reshape(tr_dat[,1], c(1,tr_dat_len)) 
x_te <- array_reshape(te_dat[,2:ncol(te_dat)], c(1,te_dat_len,2)) 
y_te <- array_reshape(te_dat[,1], c(1,te_dat_len)) 

toy_model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 200, activation = "relu",input_shape = dim(x_tr)[2:3] )%>%
  layer_dense(units = 200) %>%
  layer_dense(units = 1, activation = "linear") %>%
  compile(loss = "mse",optimizer = "adam", metrics = list("mse"))

model_checkpoint <- callback_model_checkpoint(
  filepath = "D:/77/research/temp/best_weights.h5",
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 1
)


toy_model_tr <- toy_model %>%
  fit(x = x_tr, y = y_tr, epochs = 1000, batch_size = 500, 
      validation_data = list(x_tr, y_tr) , callbacks = list(model_checkpoint))



  

