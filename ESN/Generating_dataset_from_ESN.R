rm(list = ls())
library(sp)
library(fields)
library(mvtnorm)
library(FRK)
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
use_condaenv("tf_gpu")

num_obs <- 100
coords_long <- rnorm(100)
coords_lat <- rnorm(100)


nh <- 100

coords <- data.frame("long" = coords_long, "lat" = coords_lat)
long <- coords$long
lat <- coords$lat

########################################Simulation for spatial CNN output

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
basis_use_1_2d <- scale(basis_1)
basis_use_2_2d <- scale(basis_3[,(ncol(basis_1)+1):ncol(basis_2)])
basis_use_3_2d <- scale(basis_3[,(ncol(basis_2)+1):ncol(basis_3)])


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


for (seed in 1:1e10) {
  set.seed(seed)
  
  a <- 0.1
  
  my_custom_initializer <- function(shape, dtype = NULL) {
    return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
  }
  
  # my_custom_initializer <- function(shape, dtype = NULL) {
  #   return(tf$random$normal(shape,mean = 0, stddev = 0.1, dtype = dtype))
  # }
  
  
  num_filters <- 64
  
  st_model_res_1 <- keras_model_sequential() %>%
    layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                  input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
    layer_flatten()%>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "sigmoid")
  
  
  st_model_res_2 <- keras_model_sequential() %>%
    layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                  input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
    layer_flatten()%>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "sigmoid")
  
  
  
  st_model_res_3 <- keras_model_sequential() %>%
    layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                  input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
    layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "sigmoid")
  
  convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
  convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
  convoluted_res3 <- predict(st_model_res_3,basis_arr_3)
  
  # conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,,,],convoluted_res2[1,,,],convoluted_res3[1,,,])))
  conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,],convoluted_res2[1,],convoluted_res3[1,])))
  
  pb <- txtProgressBar(min = 1, max = length(long), style = 3)
  for (i in 1:length(long)) {
    setTxtProgressBar(pb, i)
    # conv_covar[i,] <- c(as.vector(convoluted_res1[i,,,]),as.vector(convoluted_res2[i,,,]),as.vector(convoluted_res3[i,,,]))
    conv_covar[i,] <- c(as.vector(convoluted_res1[i,]),as.vector(convoluted_res2[i,]),as.vector(convoluted_res3[i,]))
  }
  
  # set.seed(1)
  a = 0.1
  nx_sp <- ncol(conv_covar)
  nu <- 1
  W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
  U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
  ar_col <- matrix(runif(nh,-a,a), nrow = 1)
  lambda_scale <- max(abs(eigen(W)$values))
  ux_sp <- conv_covar%*%U_sp
  curr_H <- apply(ux_sp, c(1,2), tanh)
  gamma_h_sim <- runif(nh,-1, 1)
  sim_y <- matrix(NA, nrow = 100, ncol = 50)
  sim_y[,1] <- rpois(100,exp(curr_H%*%matrix(gamma_h_sim, ncol = 1)))
  for (sim_idx in 2:50) {
    new_H <- apply( 
      nu/lambda_scale*
        curr_H%*%W
      + ux_sp
      + sim_y[,sim_idx-1]%*%ar_col
      , c(1,2), tanh)
    sim_y[,sim_idx] <- rpois(100,exp(new_H%*%matrix(gamma_h_sim, ncol = 1)))
    curr_H <- new_H
  }
  
  
  st_sim_dat <- cbind(1:100, coords_long, coords_lat, sim_y)
  colnames(st_sim_dat) <- c("ID","long","lat", 1:50)
  st_sim_dat <- data.frame(st_sim_dat)
  
  if(max(st_sim_dat)>100){break}
  
}
# write.csv(st_sim_dat, here::here("esn_sim.csv"),row.names = FALSE)
 