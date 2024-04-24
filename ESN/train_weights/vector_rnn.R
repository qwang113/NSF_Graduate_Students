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
nsf_y <- as.matrix(nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))])
curr_year <- 2012
tr_y <- as.matrix(nsf_y[,1:(curr_year-1972)])
te_y <- nsf_y[,curr_year-1972+1]
tr_y_use <- array_reshape(tr_y,dim = c(dim(tr_y),1)) 

rnn_input <- layer_input(batch_shape = c(7,curr_year-1972,1))


rnn_output <-rnn_input %>%
  layer_simple_rnn(units = 100, activation = "tanh", stateful = TRUE,
                   return_state = TRUE)

rnn_model <- keras_model(inputs = rnn_input, outputs = rnn_output)

dim(predict(rnn_model, tr_y_use))



%>%
  layer_dense(units = 1, activation = 'linear')


