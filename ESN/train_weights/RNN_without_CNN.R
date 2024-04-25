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
  use_condaenv("tf_gpu")
}
nsf_wide_car <- read.csv(here::here("nsf_final_wide_car.csv"))
nsf_y <- as.matrix(nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))])

#generate basis
pred_drop <- 0.1
coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
long <- coords$long
lat <- coords$lat
# carnegie_1994 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1994.csv", header = TRUE)
# carnegie_1995 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1995.csv", header = TRUE)

coordinates(coords) <- ~ long + lat
gridbasis3 <- auto_basis(mainfold = plane(), data = coords, nres = 3, type = "Gaussian", regular = 1)

basis <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis3@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis3@fn), style = 3)
for (i in 1:length(gridbasis3@fn)) {
  setTxtProgressBar(pb, i)
  basis[,i] <- gridbasis3@fn[[i]](coordinates(coords))
}
X <- basis%>%
  array_reshape( dim = c(nrow(basis),1,ncol(basis))) %>% aperm(c(1, 3, 2)) %>%
  array( dim = c(dim(basis)[1], dim(basis)[2], ncol(nsf_y))) %>% aperm( c(1,3,2)) %>% 
  abind::abind( cbind(0,log(nsf_y[,-ncol(nsf_y)]+1) ), along = 3) # use the log lag to be a covariate
Y <- nsf_y
# Reshape the matrix, use: sum(X[,1,1:ncol(basis)] == basis) == prod(dim(basis)) to verify our operation
# split train and test set
curr_year <- 2012
x_tr <- X[,1:(curr_year-1972),2002] %>% array_reshape(dim = c(nrow(nsf_wide_car), curr_year-1972, 1))
x_te <-  X[,-c(1:(curr_year-1972)),2002] %>% array_reshape(dim = c(nrow(nsf_wide_car), 2022-curr_year, 1))
y_tr <- Y[,1:(curr_year-1972)]
y_te <- Y[,-c(1:(curr_year-1972))]

train_model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 100, input_shape = dim(x_tr)[2:3], activation = "tanh", return_sequences = TRUE) %>%
  layer_simple_rnn(units = 100, activation = "tanh", return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "exponential")

train_model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam()
)
# model_checkpoint <- callback_model_checkpoint(
#   filepath = "D:/77/research/temp/soc_best_weights_dnn.h5",
#   save_best_only = TRUE,
#   monitor = "train_loss",
#   mode = "min",
#   verbose = 0
# )
# Train the model
history <- train_model %>% fit(
  x = x_tr,
  y = y_tr,
  epochs = 100,
  # callbacks = list(model_checkpoint),
  batch_size = 300
)

# train_model %>% load_model_weights_hdf5("D:/77/research/temp/soc_best_weights_dnn.h5")

# # Predict M by M
# M = 32
# pred_model <- keras_model_sequential() %>%
#   layer_simple_rnn(units = 100, batch_input_shape = c(M, NA, dim(x_tr)[3]), activation = "tanh", stateful = TRUE) %>%
#   layer_dense(units = 100,activation = "relu") %>%
#   layer_dense(units = 100,activation = "relu") %>%
#   layer_dense(units = 1, activation = "exponential")  %>% set_weights(train_model %>% get_weights()) %>% reset_states()
# # Predict groups
# max_mod <- floor(nrow(nsf_wide_car)/M)
# one_step_pred <- rep(NA, nrow(nsf_wide_car))
# predict(pred_model, x_tr[1:(M*max_mod),,])
# group_cov <- array_reshape( x_tr[ 1:(M*max_mod) ,dim(x_tr)[2],] , dim = c(M*max_mod,1,dim(x_tr)[3])) 
# one_step_pred[1:(M*max_mod)] <- predict(pred_model,group_cov)
# 
# # Predict the rest
# pred_model <- keras_model_sequential() %>%
#   layer_simple_rnn(units = 100, batch_input_shape = c( nrow(nsf_wide_car)- M*max_mod, NA, dim(x_tr)[3]), activation = "tanh", stateful = TRUE) %>%
#   layer_dense(units = 100,activation = "relu") %>%
#   layer_dense(units = 100,activation = "relu") %>%
#   layer_dense(units = 1, activation = "exponential") %>% 
#   set_weights(train_model %>% get_weights()) %>% reset_states()
# predict(pred_model, x_tr[-c(1:(M*max_mod)),,])
# res_cov <- array_reshape( x_tr[ -c(1:(M*max_mod)) ,dim(x_tr)[2],] , dim = c( nrow(nsf_wide_car)-M*max_mod,1,dim(x_tr)[3])) 
# one_step_pred[-c(1:(M*max_mod))] <- predict(pred_model, res_cov)
# mean((one_step_pred - nsf_y[,curr_year-1972+1])^2)
# 

pred_model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 100, batch_input_shape = c( 1, NA, dim(x_tr)[3]), 
                   activation = "tanh", stateful = TRUE, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 100, activation = "tanh", stateful = TRUE) %>%
  layer_dense(units = 1, activation = "exponential") %>% 
  set_weights(train_model %>% get_weights())

one_one_step_ahead <- rep(NA, nrow(nsf_wide_car))
for (i in 1:nrow(nsf_wide_car)) {
  print(i)
  pred_model%>% reset_states() 
  predict(pred_model, array_reshape(x_tr[i,,] , dim = c(1,dim(x_tr)[2:3]) )  )
  one_cov <- array_reshape( log(nsf_y[i,curr_year-1972]+1) , dim = c( 1,1,dim(x_tr)[3])) 
  one_one_step_ahead[i] <- predict(pred_model, one_cov)
}

mean((one_one_step_ahead - nsf_y[,curr_year-1972+1])^2)

