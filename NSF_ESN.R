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
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)

nsf_wide$long <- jitter(nsf_wide$long)
nsf_wide$lat <- jitter(nsf_wide$lat)

coords <- data.frame("long" = nsf_wide$long,"lat" = nsf_wide$lat)

# Initialize Convolutional Structure: Basis Functions

# Basis Generating
long <- coords$long
lat <- coords$lat
school_ID <-  substr(nsf_wide$ID,1,6) 
dummy_matrix <- model.matrix(~ school_ID - 1)
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

a <- 1

# my_custom_initializer <- function(shape, dtype = NULL) {
#   return(tf$random$uniform(shape, minval = -0.1, maxval = 0.1, dtype = dtype))
# }
sig <- 0.8

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$normal(shape,mean = 0, stddev = 0.1, dtype = dtype))
}


num_filters <- 200

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer)%>%
  layer_flatten() %>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")%>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu", kernel_initializer = my_custom_initializer)%>%
  layer_flatten() %>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")%>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")


st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "relu", kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")%>%
  layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "relu")


convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
convoluted_res3 <- predict(st_model_res_3,basis_arr_3)

conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,],convoluted_res2[1,],convoluted_res3[1,])))
pb <- txtProgressBar(min = 1, max = length(long), style = 3)
for (i in 1:length(long)) {
  setTxtProgressBar(pb, i)
  conv_covar[i,] <- c(as.vector(convoluted_res1[i,]),as.vector(convoluted_res2[i,]),as.vector(convoluted_res3[i,]))
}

rm(basis_1,basis_2, basis_3,basis_arr_1,basis_arr_2,basis_arr_3, basis_use_1_2d, basis_use_2_2d, basis_use_3_2d, convoluted_res1,convoluted_res2,convoluted_res3)
# Begin recurrent part

zero_col <- which(colSums(conv_covar)==0)
conv_covar <- conv_covar[,-zero_col]

min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}


leak_rate <- 1 # It's always best to choose 1 here according to Mcdermott and Wille, 2017
nh <- 2000 # Number of hidden units in RNN
nx_sp <- ncol(conv_covar) # Number of covariates
nx_dummy <- ncol(dummy_matrix)

# The range of the standard uniform distribution of the weights
nu <- 0.9
time_step <- length(unique(nsf_long$year))

# W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
# U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
# U_dummy <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
# ar_col <- matrix(runif(nh,-a,a), nrow = 1)


W <- matrix(rnorm(nh^2, sd = sig), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
U_sp <- matrix(rnorm(nh*nx_sp, sd = sig), nrow = nx_sp, ncol = nh)
U_dummy <- matrix(rnorm(nh*nx_dummy, sd = sig), nrow = nx_dummy, ncol = nh)
ar_col <- matrix(rnorm(nh, sd = sig), nrow = 1)


lambda_scale <- max(abs(eigen(W)$values))

ux_sp <- conv_covar%*%U_sp
ux_dummy <- dummy_matrix%*%U_dummy



curr_H <- ux_sp + ux_dummy

Y <- nsf_wide[,4]

pb <- txtProgressBar(min = 1, max = length(2:time_step), style = 3)
for (i in 2:time_step) {
  setTxtProgressBar(pb,i)
  new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + ux_dummy + nsf_wide[,i+2]%*%ar_col, c(1,2), tanh)
  # new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + nsf_wide[,i+2]%*%ar_col, c(1,2), tanh)
  Y <- c(Y, nsf_wide[,i+3])
  curr_H <- rbind(curr_H, new_H)
}

# pca_H <- prcomp(curr_H)
# pca_var <-predict(pca_H)
 # write.csv(curr_H, "D:/77/UCSC/study/Research/temp/NSF_dat/CRESN_H.csv", row.names = FALSE)


glm_CRESN <- glm.nb(Y~curr_H, control = glm.control(epsilon = 1e-8, maxit = 50, trace = TRUE))

CRESN_res <- glm_CRESN$residuals

year_stack <- rep(1972:2021, each = nrow(nsf_wide))
school_ID <- rep(nsf_wide$ID,50)
long_stack <- rep(nsf_wide$long,50)
lat_stack <- rep(nsf_wide$lat,50)

res_stack <- data.frame("ID" = school_ID, "long" = long_stack, "lat" = lat_stack, "year" = year_stack, "Residuals" = CRESN_res)

# write.csv(res_stack, "D:/77/UCSC/study/Research/temp/NSF_dat/300filters_res2000_complex+univ_added+norm01.csv", row.names = FALSE)
