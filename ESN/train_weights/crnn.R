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

curr_year <- 2012
tr_y <- nsf_y[,1:(curr_year-1972)]
te_y <- nsf_y[,curr_year-1972+1]

#generate basis
pred_drop <- 0.1
pred_drop_layer <- layer_dropout(rate=pred_drop)
coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
long <- coords$long
lat <- coords$lat
# carnegie_1994 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1994.csv", header = TRUE)
# carnegie_1995 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1995.csv", header = TRUE)

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

# # Second resolution
# shape_row_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 2 ]))
# shape_col_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 1 ]))
# basis_arr_2 <- array(NA, dim = c(nrow(coords@coords), shape_row_2, shape_col_2))
# for (i in 1:nrow(coords@coords)) {
#   basis_arr_2[i,,] <- matrix(basis_use_2_2d[i,], nrow = shape_row_2, ncol = shape_col_2, byrow = T)
# }
# basis_arr_2 <- array_reshape(basis_arr_2,c(dim(basis_arr_2),1))
# 
# # Third resolution
# shape_row_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 2 ]))
# shape_col_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 1 ]))
# basis_arr_3 <- array(NA, dim = c(nrow(coords@coords), shape_row_3, shape_col_3))
# for (i in 1:nrow(coords@coords)) {
#   basis_arr_3[i,,] <- matrix(basis_use_3_2d[i,], nrow = shape_row_3, ncol = shape_col_3, byrow = T)
# }
# basis_arr_3 <- array_reshape(basis_arr_3,c(dim(basis_arr_3),1))





#define input layers
input_basis_1 <- layer_input(shape = c(shape_row_1, shape_col_1, 1))
# input_basis_2 <- layer_input(shape = c(shape_row_2, shape_col_2, 1))
# input_basis_3 <- layer_input(shape = c(shape_row_3, shape_col_3, 1))



resolution_1_conv <- input_basis_1 %>%
  layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu')



lag1logy <- cbind(0,as.matrix(log(tr_y+1)))[,-1]
lag1logy <- array_reshape(lag1logy, dim = c(1799,curr_year-1972,1))
input_cov <- layer_input(shape = c(curr_year-1972,1))
lay_y <- input_cov %>% layer_dense(units = 1, activation = 'linear')



full_model <- layer_concatenate(list(resolution_1_conv,lay_y)) %>%
  layer_simple_rnn(units = 200, activation = "tanh") %>%
  layer_dense(units = 1, activation = "exponential")


final_model <- keras_model(inputs = list(input_basis_1, input_cov), outputs = full_model) %>%
  compile(loss = "poisson",optimizer = "adam", metrics = list("poisson"))


model_checkpoint <- callback_model_checkpoint(
  filepath = "D:/77/research/temp/best_weights.h5",
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 1
)


final_model_train <- final_model %>%
  fit(x = list(basis_arr_1, lag1logy), y = as.matrix(tr_y), epochs = 1000, batch_size = 500,  validation_split = 0.1,
      callbacks = list(model_checkpoint)) 


ylag_pred <- array(rep(log(te_y+1), curr_year-1972), dim = c(nrow(nsf_wide_car), curr_year-1972,1 ))


y_pred <- predict(final_model, list(basis_arr_1,  ylag_pred))[,1]

mean((y_pred-nsf_y$X2012)^2)
  # layer_reshape(input_shape = c(1799,100), target_shape = 1799,100)
  # 





resolution_2_conv <- input_basis_2 %>%
  layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu')

resolution_3_conv <- input_basis_3 %>%
  layer_conv_2d(filters = 128, kernel_size = c(2,2), activation = 'relu') %>%
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu')

cov_model <- input_cov %>%
  layer_dense(units = 100, activation = 'relu') %>% 
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) %>%
  layer_dense(units = 100, activation = 'relu') %>%
  layer_batch_normalization() %>%
  pred_drop_layer(training = T) 


all_cnn_model <- layer_concatenate(list(resolution_1_conv, resolution_2_conv, resolution_3_conv, cov_model))







