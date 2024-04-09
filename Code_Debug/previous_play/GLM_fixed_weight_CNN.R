rm(list = ls())
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

# Simulate a spatial dataset from GP

# Pois + GP
# num_obs <- 3000
# coords <- data.frame("long" = runif(num_obs),"lat" = runif(num_obs))
# paired_dist <- as.matrix(dist(coords, diag = TRUE, upper = TRUE))
# phi <- 0.5
# sig <- 1
# cov_mat <-  exp(-paired_dist/phi)*sig^2
# lam <- as.vector(rmvnorm(1, mean = rep(3, num_obs), sigma = cov_mat))
# y <- rpois(num_obs, lambda = exp(lam))

# MVGaussian + ceiling
num_obs <- 3000
coords <- data.frame("long" = runif(num_obs),"lat" = runif(num_obs))
paired_dist <- as.matrix(dist(coords, diag = TRUE, upper = TRUE))
phi <- 0.5
sig <- 50
cov_mat <-  exp(-paired_dist/phi)*sig^2
y <- ceiling(as.vector(rmvnorm(1, mean = rep(300, num_obs), sigma = cov_mat)))


ggplot() +
  geom_point(aes(x = coords$long, y = coords$lat, color = y)) +
  scale_color_viridis_c()


long <- coords$long
lat <- coords$lat


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

a <- 0.1

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}

# my_custom_initializer <- function(shape, dtype = NULL) {
#   return(tf$random$normal(shape,mean = 0, stddev = 0.1, dtype = dtype))
# }


num_filters <- 200

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten()  %>% layer_dense(units = 200, kernel_initializer = my_custom_initializer, activation = "tanh")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten()  %>% layer_dense(units = 200, kernel_initializer = my_custom_initializer, activation = "tanh")



st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten()  %>% layer_dense(units = 200, kernel_initializer = my_custom_initializer, activation = "tanh")

convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
convoluted_res3 <- predict(st_model_res_3,basis_arr_3)

conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,],convoluted_res2[1,],convoluted_res3[1,])))
pb <- txtProgressBar(min = 1, max = length(long), style = 3)
for (i in 1:length(long)) {
  setTxtProgressBar(pb, i)
  conv_covar[i,] <- c(as.vector(convoluted_res1[i,]),as.vector(convoluted_res2[i,]),as.vector(convoluted_res3[i,]))
}



all_dat <- cbind(y, conv_covar)
colnames(all_dat) <- c("y",1:ncol(conv_covar))
all_dat <- as.data.frame(all_dat)
tr_idx <- sample(1:num_obs, size = 0.8*num_obs)
tr_dat <- all_dat[tr_idx,]
te_dat <- all_dat[-tr_idx,]

cv_model <- cv.glmnet(x = conv_covar[tr_idx,], y = y[tr_idx], alpha = 0, nfolds = 10, trace.it = 1, family = poisson(link = "log"))
curr_model <- glmnet(x = conv_covar[tr_idx,], y = y[tr_idx], alpha = 0, family = poisson(link = "log"), trace.it = 1)
oo_pred <- predict(curr_model, conv_covar[-tr_idx,], type = "response")

pois_glm <- glm(y~., data = tr_dat, family = poisson(link = "log"))
oo_pred <- predict(pois_glm, te_dat[,-1], type = "response")


# cv_model <- cv.glmnet(x = conv_covar[tr_idx,], y = sqrt(y[tr_idx]), alpha = 0, nfolds = 10, trace.it = 1)
# no_pois <- glmnet(x = conv_covar[tr_idx,], y = sqrt(y[tr_idx]), alpha = 0, lambda = cv_model$lambda.min)
# oo_pred <- (predict(no_reg, as.matrix(te_dat[,-1])))^2


oo_mse <- mean((y[-tr_idx]-oo_pred)^2)
oo_var <- var(y[-tr_idx])
oo_mse
oo_var
1-oo_mse/oo_var
# Result: Even if poisson cannot handle GP simulated Count data set. So what kind of data can it handle?
# But we can fit log(y)