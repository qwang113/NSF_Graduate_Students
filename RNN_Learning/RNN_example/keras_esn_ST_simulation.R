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
use_condaenv("tf_gpu")
# Gaussian Process with Matern Correlation
longitude <- seq(from = 0, to = 1, length.out = 50)
latitude <- seq(from = 0, to = 1, length.out = 50)
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

ar_coef <- 0.5
curr_epsilon = epsilon_spatial_1
time_step <- 100

st_dat <- vector("list", length = time_step)
st_dat[[1]] <- cbind(as.matrix(coords),covariates,y_1)
colnames(st_dat[[1]]) <- c("long","lat","x1","x2","y")

for (i in 2:100) {
  print(paste("Now doing index",i))
  curr_epsilon <- curr_epsilon * ar_coef + rnorm(length(curr_epsilon), sd = sd_temporal)
  curr_y <- as.vector(curr_epsilon) + as.vector(covariates%*%beta_coef) + rnorm(length(curr_epsilon), sd = sd_residual)
  curr_dat <- cbind(as.matrix(coords),covariates,curr_y)
  colnames(curr_dat) <- colnames(st_dat[[i-1]])
  st_dat[[i]] <- curr_dat
}

# Visulize for certain time point
time_selected <- 50

dat_selected <- as.data.frame(st_dat[[time_selected]])

ggplot(data = dat_selected) +
  geom_contour_filled(aes(x = long, y = lat, z = y)) +
  labs( title = paste("Contour Plot at Time Step",time_selected))

# Initialize Convolutional Structure: Basis Functions

# Basis Generating
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

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -0.1, maxval = 0.1, dtype = dtype))
}

num_filters <- 200

st_model_res_1 <- keras_model_sequential() %>%
layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer)
st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer)
st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer)

convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
convoluted_res3 <- predict(st_model_res_3,basis_arr_3)

conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,,,],convoluted_res2[1,,,],convoluted_res3[1,,,])))

for (i in 1:length(long)) {
  conv_covar[i,] <- c(as.vector(convoluted_res1[i,,,]),as.vector(convoluted_res2[i,,,]),as.vector(convoluted_res3[i,,,]))
}

# Begin recurrent part

min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}
relu <- function(x){if(x <=0){return(0)}else{return(x)}}
scaled_cov <- apply(cbind(covariates, conv_covar), 2, min_max_scale) 

leak_rate <- 1 # It's always best to choose 1 here according to Mcdermott and Wille, 2017
nh <- 200 # Number of hidden units in RNN
nx <- ncol(covariates) + ncol(scaled_cov) # Number of covariates
a <- 0.1# The range of the standard uniform distribution of the weights






