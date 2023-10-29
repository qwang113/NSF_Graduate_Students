rm(list = ls())
library(ggplot2)
library(sp)
library(fields)
library(mvtnorm)
library(FRK)
library(utils)
library(keras)
library(reticulate)
use_condaenv("tf_gpu")
# Gaussian Process with Matern Correlation
longitude <- seq(from = 0, to = 1, length.out = 40)
latitude <- seq(from = 0, to = 1, length.out = 40)
coords <- expand.grid(longitude,latitude)
colnames(coords) <- c("long","lat")
paired_dist = spDists(as.matrix(coords))
sd_spatial = 1
sd_temporal = 1.5
sd_residual = 0.5

d = 0.5
cov_mat <- sd_spatial * Exponential(paired_dist, range = d)
epsilon_spatial_1 <- rmvnorm(1, mean = rep(0, nrow(coords)), sigma = cov_mat)
covariates <- matrix(rnorm(2*nrow(coords)), ncol = 2)
beta_coef <- matrix(c(3,-1), ncol = 1)
y_1 <- as.vector(epsilon_spatial_1) + as.vector(covariates%*%beta_coef) + rnorm(length(epsilon_spatial_1), sd = sd_residual)
all_dat <- cbind(y_1, coords, covariates,1)
colnames(all_dat) <- c("y", "long","lat", "x1", "x2", "time")
ar_coef <- 0.5
curr_epsilon = epsilon_spatial_1
for (i in 2:100) {
  print(paste("Now doing index",i))
  curr_epsilon <- curr_epsilon * ar_coef + rnorm(length(curr_epsilon), sd = sd_temporal)
  curr_y <- as.vector(curr_epsilon) + as.vector(covariates%*%beta_coef) + rnorm(length(curr_epsilon), sd = sd_residual)
  curr_dat <- cbind(curr_y,coords, covariates,i)
  colnames(curr_dat) <- colnames(all_dat)
  all_dat <- rbind(all_dat, curr_dat)
}

# Visulize for certain time point
time_selected <- 50

dat_selected <- all_dat[which(all_dat$time == time_selected),]



ggplot(data = dat_selected) +
  geom_contour_filled(aes(x = long, y = lat, z = y)) +
  labs( title = paste("Contour Plot at Time Step",time_selected))

# Initialize Convolutional Structure: Basis Functions

# Basis Generating
long <- all_dat$long
lat <- all_dat$lat
y <- all_dat$y
timestep <- all_dat$time
#coords_long <- coords$long
#coords_lat <- coords$lat

coordinates(all_dat) <- ~ long + lat
#coordinates(coords) <- ~ long + lat



gridbasis1 <- auto_basis(mainfold = plane(), data = all_dat, nres = 1, type = "Gaussian", regular = 1)
gridbasis2 <- auto_basis(mainfold = plane(), data = all_dat, nres = 2, type = "Gaussian", regular = 1)
gridbasis3 <- auto_basis(mainfold = plane(), data = all_dat, nres = 3, type = "Gaussian", regular = 1)

show_basis(gridbasis3) + 
  coord_fixed() +
  xlab("Longitude") +
  ylab("Latitude")



basis_1 <- matrix(NA, nrow = nrow(all_dat), ncol = length(gridbasis1@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis1@fn), style = 3)
for (i in 1:length(gridbasis1@fn)) {
  setTxtProgressBar(pb, i)
  basis_1[,i] <- gridbasis1@fn[[i]](coordinates(all_dat))
}

basis_2 <- matrix(NA, nrow = nrow(all_dat), ncol = length(gridbasis2@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis2@fn), style = 3)
for (i in 1:length(gridbasis2@fn)) {
  setTxtProgressBar(pb, i)
  basis_2[,i] <- gridbasis2@fn[[i]](coordinates(all_dat))
}

basis_3 <- matrix(NA, nrow = nrow(all_dat), ncol = length(gridbasis3@fn))
pb <- txtProgressBar(min = 1, max = length(gridbasis3@fn), style = 3)
for (i in 1:length(gridbasis3@fn)) {
  setTxtProgressBar(pb, i)
  basis_3[,i] <- gridbasis3@fn[[i]](coordinates(all_dat))
}


# Redefine three layers of basis images
basis_use_1_2d <- basis_1
basis_use_2_2d <- basis_3[,(ncol(basis_1)+1):ncol(basis_2)]
basis_use_3_2d <- basis_3[,(ncol(basis_2)+1):ncol(basis_3)]


# First resolution
shape_row_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 2 ]))
shape_col_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 1 ]))
basis_arr_1 <- array(NA, dim = c(nrow(all_dat), shape_row_1, shape_col_1))

for (i in 1:nrow(all_dat)) {
  basis_arr_1[i,,] <- matrix(basis_use_1_2d[i,], nrow = shape_row_1, ncol = shape_col_1, byrow = T)
}
basis_arr_1 <- array_reshape(basis_arr_1,c(dim(basis_arr_1),1))

# Second resolution
shape_row_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 2 ]))
shape_col_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 1 ]))
basis_arr_2 <- array(NA, dim = c(nrow(all_dat), shape_row_2, shape_col_2))
for (i in 1:nrow(all_dat)) {
  basis_arr_2[i,,] <- matrix(basis_use_2_2d[i,], nrow = shape_row_2, ncol = shape_col_2, byrow = T)
}
basis_arr_2 <- array_reshape(basis_arr_2,c(dim(basis_arr_2),1))

# Third resolution
shape_row_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 2 ]))
shape_col_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 1 ]))
basis_arr_3 <- array(NA, dim = c(nrow(all_dat), shape_row_3, shape_col_3))
for (i in 1:nrow(all_dat)) {
  basis_arr_3[i,,] <- matrix(basis_use_3_2d[i,], nrow = shape_row_3, ncol = shape_col_3, byrow = T)
}
basis_arr_3 <- array_reshape(basis_arr_3,c(dim(basis_arr_3),1))

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -1, maxval = 1, dtype = dtype))
}

st_model_res_1 <- keras_model_sequential() %>%
layer_conv_2d(filters = 200, kernel_size = c(2, 2), 
                                 input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer)
st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 200, kernel_size = c(2, 2), 
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer)
st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 200, kernel_size = c(2, 2), 
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer)

convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
convoluted_res2 <- predict(st_model_res_2,basis_arr_2)


