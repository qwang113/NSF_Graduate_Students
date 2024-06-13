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
coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
long <- coords$long
lat <- coords$lat
# carnegie_1994 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1994.csv", header = TRUE)
# carnegie_1995 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1995.csv", header = TRUE)

curr_year <- 2012
prev_pc_res <- prcomp(t(log(nsf_y[ ,1:(curr_year-1972)]+1)))
num_pc <- which(cumsum(prev_pc_res$sdev^2)/sum(prev_pc_res$sdev^2) > 0.8)[1]
all_pc <- as.matrix(prev_pc_res$x[,1:num_pc])
pc_tr <- array(0, dim = c(nrow(nsf_y), curr_year-1972, num_pc))
for (i in 2:(curr_year-1972)) {
    pc_tr[,i,] <- matrix( rep(all_pc[i-1,],nrow(nsf_y)), ncol = num_pc, byrow = TRUE)
  }

lag_y <-
  abind::abind(
    array_reshape(cbind(0,log(nsf_y+1))[,1:(curr_year-1972)], dim = c(dim(nsf_y)[1],curr_year-1972, 1) ),
    pc_tr,
    along = 3
  )

lag_y_input <- layer_input(shape = c(NA,1+num_pc))
all_output = lag_y_input %>% layer_simple_rnn(units = 100, activation = "tanh", return_sequences = TRUE) %>%
  layer_simple_rnn(units = 100, activation = "tanh", return_sequences = TRUE) %>%
  layer_dense(units = 1, activation = "exponential")


train_model <- keras_model(inputs = list(lag_y_input), outputs = all_output)
train_model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam()
)

model_checkpoint <- callback_model_checkpoint(
  filepath = here::here("rnn_only.h5"),
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 1
)

history <- train_model %>% fit(
  x = list(lag_y),
  y = nsf_y[,1:(curr_year-1972)],
  validation_split = 0.1,
  epochs = 100,
  batch_size = 100,
  callbacks = model_checkpoint
)


# Prediction: build a new model and load weights
# Input of Y
lag_y_input_pred <- layer_input(batch_shape = c(1,NA,1+num_pc))


all_output_pred <- lag_y_input_pred %>% 
  layer_simple_rnn(units = 100, activation = "tanh",stateful = TRUE, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 100, activation = "tanh", stateful = TRUE) %>%
  layer_dense(units = 1, activation = "exponential")

pred_model <- keras_model(inputs = list(lag_y_input_pred),
                          outputs = all_output_pred)

one_one_step_ahead <- rep(NA, nrow(nsf_wide_car))
pred_model %>% set_weights(get_weights(train_model))
for (i in 1:nrow(nsf_wide_car)) {
  print(i)
  pred_model %>% reset_states() 
  predict(
    pred_model, 
    list(
      array_reshape(lag_y[i,,], dim = c(1,dim(lag_y)[-1]))
    )
  )
  one_cov <- list(
    array_reshape(
      c(
      log(nsf_y[i,curr_year-1972]+1)
        , all_pc[curr_year-1972,])
      , dim = c(1,1,1+num_pc)
    )
  )
  one_one_step_ahead[i] <- predict(pred_model, one_cov)
}

mean((one_one_step_ahead - nsf_y[,curr_year-1972+1])^2)



# make a year by year mse table
# Observations plot, soc prediction plots
# Traceplot for some schools, tarceplot for the mse for the all models
# Reread the docs and edit everything 
# Change omega to a fixed one, data augmentation