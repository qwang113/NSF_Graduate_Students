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

ggplot() +
  geom_point(aes(x = jitter(long), y = jitter(lat), col = nsf_wide_car$X1972)) +
  scale_color_viridis_c(limits = c(0,200))
coordinates(coords) <- ~ long + lat

gridbasis1 <- auto_basis(mainfold = plane(), data = coords, nres = 1, type = "Gaussian", regular = 1)
gridbasis2 <- auto_basis(mainfold = plane(), data = coords, nres = 2, type = "Gaussian", regular = 1)
gridbasis3 <- auto_basis(mainfold = plane(), data = coords, nres = 3, type = "Gaussian", regular = 1)

show_basis(gridbasis3) +
  coord_fixed() +
  xlab("Longitude") +
  ylab("Latitude")



basis_1 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis1@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis1@fn), style = 3)
for (i in 1:length(gridbasis1@fn)) {
  setTxtProgressBar(pb, i)
  basis_1[,i] <- gridbasis1@fn[[i]](coordinates(coords))
}

basis_2 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis2@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis2@fn), style = 3)
for (i in 1:length(gridbasis2@fn)) {
  setTxtProgressBar(pb, i)
  basis_2[,i] <- gridbasis2@fn[[i]](coordinates(coords))
}

basis_3 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis3@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis3@fn), style = 3)
for (i in 1:length(gridbasis3@fn)) {
  setTxtProgressBar(pb, i)
  basis_3[,i] <- gridbasis3@fn[[i]](coordinates(coords))
}


# Redefine three layers of basis images
basis_use_1_2d <- basis_1
basis_use_2_2d <- basis_3[,(ncol(basis_1)+1):ncol(basis_2)]
basis_use_3_2d <- basis_3[,(ncol(basis_2)+1):ncol(basis_3)]


# First resolution
shape_row_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 2 ]))
shape_col_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 1 ]))
basis_arr_1 <- array(NA, dim = c(nrow(coords@coords), shape_row_1, shape_col_1))

for (i in 1:nrow(coords@coords)) {
  basis_arr_1[i,,] <- matrix(basis_use_1_2d[i,], nrow = shape_row_1, ncol = shape_col_1, byrow = T)
}
basis_arr_1 <- array_reshape(basis_arr_1,c(dim(basis_arr_1),1))

# Second resolution
shape_row_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 2 ]))
shape_col_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 1 ]))
basis_arr_2 <- array(NA, dim = c(nrow(coords@coords), shape_row_2, shape_col_2))
for (i in 1:nrow(coords@coords)) {
  basis_arr_2[i,,] <- matrix(basis_use_2_2d[i,], nrow = shape_row_2, ncol = shape_col_2, byrow = T)
}
basis_arr_2 <- array_reshape(basis_arr_2,c(dim(basis_arr_2),1))

# Third resolution
shape_row_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 2 ]))
shape_col_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 1 ]))
basis_arr_3 <- array(NA, dim = c(nrow(coords@coords), shape_row_3, shape_col_3))
for (i in 1:nrow(coords@coords)) {
  basis_arr_3[i,,] <- matrix(basis_use_3_2d[i,], nrow = shape_row_3, ncol = shape_col_3, byrow = T)
}
basis_arr_3 <- array_reshape(basis_arr_3,c(dim(basis_arr_3),1))

pred_y <- matrix(NA, nrow = nrow(nsf_y), ncol = 10)

for (curr_year in 2019:2021) {
  lag_y <- array_reshape(cbind(0,log(nsf_y+1))[,1:(curr_year-1972)], dim = c(dim(nsf_y)[1],curr_year-1972, 1) )
  
  num_pc <- 0 
  
  basis_1t <- basis_arr_1 %>% array( dim = c(dim(basis_arr_1),curr_year-1972)) %>% aperm(c(1,5,2,3,4))
  basis_2t <- basis_arr_2 %>% array( dim = c(dim(basis_arr_2),curr_year-1972)) %>% aperm(c(1,5,2,3,4))
  basis_3t <- basis_arr_3 %>% array( dim = c(dim(basis_arr_3),curr_year-1972)) %>% aperm(c(1,5,2,3,4))
  
  
  
  # Build Time Distributed CNN
  # Input of basis 1
  basis_input_1 <- layer_input(shape = c(NA, dim(basis_arr_1)[-1] ))
  basis_output_1 <- basis_input_1 %>% time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  # Input of basis 2
  basis_input_2 <- layer_input(shape = c(NA, dim(basis_arr_2)[-1] ))
  basis_output_2 <- basis_input_2 %>% time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  # Input of basis 3
  basis_input_3 <- layer_input(shape = c(NA, dim(basis_arr_3)[-1] ))
  basis_output_3 <- basis_input_3 %>% time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  # Input of Y
  lag_y_input <- layer_input(shape = c(NA,1+num_pc))
  
  all_concatenate <- layer_concatenate(list(basis_output_1, basis_output_2, basis_output_3, lag_y_input))
  
  all_output = all_concatenate %>% layer_simple_rnn(units = 100, activation = "tanh", return_sequences = TRUE) %>%
    layer_simple_rnn(units = 100, activation = "tanh", return_sequences = TRUE) %>%
    layer_dense(units = 1, activation = "exponential")
  
  
  train_model <- keras_model(inputs = list(basis_input_1,basis_input_2,basis_input_3, lag_y_input), outputs = all_output)
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
    x = list(basis_1t, basis_2t, basis_3t, lag_y),
    y = nsf_y[,1:(curr_year-1972)],
    validation_split = 0.1,
    epochs = 100,
    batch_size = 100,
    callbacks = model_checkpoint
  )
  
  
  # Prediction: build a new model and load weights
  basis_input_1_pred <- layer_input(batch_shape = c(1, NA, dim(basis_arr_1)[-1] ))
  basis_output_1_pred <- basis_input_1_pred %>% 
    time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  
  # Input of basis 2
  basis_input_2_pred <- layer_input(batch_shape = c(1, NA, dim(basis_arr_2)[-1] ))
  basis_output_2_pred <- basis_input_2_pred %>% 
    time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  
  # Input of basis 3
  basis_input_3_pred <- layer_input(batch_shape = c(1, NA, dim(basis_arr_3)[-1] ))
  basis_output_3_pred <- basis_input_3_pred %>% 
    time_distributed(layer_conv_2d(filters = 16, kernel_size = c(2,2), activation = "tanh")) %>%
    time_distributed(layer_flatten()) %>%
    time_distributed(layer_dense(units = 100))
  
  # Input of Y
  lag_y_input_pred <- layer_input(batch_shape = c(1,NA,1+num_pc))
  
  all_concatenate_pred <- layer_concatenate(list(basis_output_1_pred, basis_output_2_pred, basis_output_3_pred, lag_y_input_pred))
  
  
  all_output_pred <- all_concatenate_pred %>% 
    layer_simple_rnn(units = 100, activation = "tanh",stateful = TRUE, return_sequences = TRUE) %>%
    layer_simple_rnn(units = 100, activation = "tanh", stateful = TRUE) %>%
    layer_dense(units = 1, activation = "exponential")
  
  pred_model <- keras_model(inputs = list(basis_input_1_pred, basis_input_2_pred, basis_input_3_pred, lag_y_input_pred),
                            outputs = all_output_pred)
  
  one_one_step_ahead <- rep(NA, nrow(nsf_wide_car))
  pred_model %>% set_weights(get_weights(train_model))
  for (i in 1:nrow(nsf_wide_car)) {
    print(i)
    pred_model %>% reset_states() 
    predict(
      pred_model, 
      list(
        array_reshape(basis_1t[i,,,,], dim = c(1,dim(basis_1t)[2:5])),
        array_reshape(basis_2t[i,,,,], dim = c(1,dim(basis_2t)[2:5])),
        array_reshape(basis_3t[i,,,,], dim = c(1,dim(basis_3t)[2:5])),
        array_reshape(lag_y[i,,], dim = c(1,dim(lag_y)[-1]))
      )
    )
    one_cov <- list(
      array_reshape(basis_1t[i,curr_year-2011,,,], dim = c(1,1,dim(basis_1t)[3:5])),
      array_reshape(basis_2t[i,curr_year-2011,,,], dim = c(1,1,dim(basis_2t)[3:5])),
      array_reshape(basis_3t[i,curr_year-2011,,,], dim = c(1,1,dim(basis_3t)[3:5])),
      array_reshape(
        log(nsf_y[i,curr_year-1972]+1) , dim = c(1,1,1+num_pc)
      )
    )
    one_one_step_ahead[i] <- predict(pred_model, one_cov)
  }
  pred_y[,curr_year-2011] <- one_one_step_ahead
}

res_y <- nsf_y[,41:50] - pred_y
write.csv( as.data.frame(pred_y), "D:/77/UCSC/study/Research/temp/NSF_dat/crnn_pred.csv", row.names = FALSE)
write.csv( as.data.frame(res_y), "D:/77/UCSC/study/Research/temp/NSF_dat/crnn_res.csv", row.names = FALSE)





# mean((one_one_step_ahead - nsf_y[,curr_year-1972+1])^2)
