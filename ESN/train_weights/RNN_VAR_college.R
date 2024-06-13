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
train_pc <- array(0, dim = c(nrow(nsf_y), curr_year-1972, 1)) # First year, all school has no pc, use 0 to hold a position
pred_pc <- array(0, dim = c(nrow(nsf_y), 1, 1))

for (pca_id in unique(nsf_wide_car$UNITID)  ) {
  print(pca_id)
  curr_college <- which(nsf_wide_car$UNITID == pca_id)
  if( length(curr_college) != 1 ){# If there is only one school, no need to do pca for it
    # Vector AR as pc
    curr_pc_all <- array(0, dim = c(nrow(nsf_y),curr_year-1972,length(curr_college)))
    curr_pred_pc_all <- array(0, dim = c(nrow(nsf_y),1,length(curr_college)))
    for (i in 2:(curr_year-1972)) {
      curr_pc_all[curr_college,i,] <- log(matrix( rep(nsf_y[curr_college,i-1],length(curr_college))
                                         , ncol = length(curr_college), byrow = TRUE)+1)
    }
    curr_pred_pc_all[curr_college,1,] <- log(matrix(rep(nsf_y[curr_college,curr_year-1972],
                                                        length(curr_college)),ncol = length(curr_college), byrow = TRUE)+1)
    train_pc <- abind(train_pc, curr_pc_all, along = 3)
    pred_pc <- abind(pred_pc, curr_pred_pc_all, along = 3)
  }
}

train_pc[,,1] <- cbind(0, log(nsf_y[,1:(curr_year-1972-1)]+1))
pred_pc[,,1] <-  log(nsf_y[,(curr_year-1972)]+1)




# train_pc <- array_reshape(train_pc[,,1], c(nrow(nsf_y), curr_year-1972,1))
# pred_pc <- array_reshape(pred_pc[,,1], c(nrow(nsf_y),1,1))

train_y <- nsf_y[,1:(curr_year-1972)]


lag_y_input <- layer_input(shape = c(NA,dim(train_pc)[3]))
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
  x = list(train_pc),
  y = train_y,
  validation_split = 0.1,
  epochs = 100,
  batch_size = 100,
  callbacks = model_checkpoint
)


# Prediction: build a new model and load weights
# Input of Y
lag_y_input_pred <- layer_input(batch_shape = c(1,NA,dim(train_pc)[3]))


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
      array_reshape(train_pc[i,,], dim = c(1,curr_year-1972,dim(train_pc)[3]))
    )
  )
  one_cov <- list(
    array_reshape(
      pred_pc[i,,] , dim = c(1,1,dim(train_pc)[3])
  ))
  one_one_step_ahead[i] <- predict(pred_model, one_cov)
}

mean((one_one_step_ahead - nsf_y[,curr_year-1972+1])^2)

