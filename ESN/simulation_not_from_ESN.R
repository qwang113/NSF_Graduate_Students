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
range = 1
sill = 1


paired_dist <- as.matrix(dist(cbind(coords_long, coords_lat), diag = TRUE, upper = TRUE))
conv_mat <- exp(-paired_dist/range) * sill^2
#Initialization
log_lambda_vec = y_vec <- matrix(NA, ncol = 50, nrow = 100)


log_lambda_vec[,1] <- rmvnorm(1, mean = rep(0, ncol(conv_mat)), sigma = conv_mat)


y_vec[,1] <- rpois(100, lambda = exp(log_lambda_vec[,1]))

for (time in 2:50) {
  log_lambda_vec[,time] = rmvnorm(1, mean = log(y_vec[,time-1]+1)*0.9, sigma = conv_mat)
  y_vec[,time] <- rpois(100, lambda = exp(log_lambda_vec[,time]))
  }

st_sim_dat <- cbind(1:100, coords_long, coords_lat, y_vec)
colnames(st_sim_dat) <- c("ID","long","lat", 1:50)

st_sim_dat <- data.frame(st_sim_dat)

coords <- data.frame("long" = st_sim_dat$long, "lat" = st_sim_dat$lat)
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

nh <- 200

a <- 0.1

num_ensemble <- 1
one_step_ahead_pred_y <- array(NA, dim = c(num_ensemble, num_obs, length(41:50)) )
for (ensemble_idx in 1:num_ensemble) {
  for (year in 41:50) {
    print(paste("Ensemble",ensemble_idx, "year",year ))
    #Initialize
    nx_sp <- ncol(conv_covar)
    nu <- 1
    W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
    U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
    ar_col <- matrix(runif(nh,-a,a), nrow = 1)
    lambda_scale <- max(abs(eigen(W)$values))
    ux_sp <- conv_covar%*%U_sp
    curr_H <- apply(ux_sp, c(1,2), tanh)
    Y <- st_sim_dat[,4]
    pb <- txtProgressBar(min = 1, max = year, style = 3)
    print("Calculating Recurrent H Matrix. . .")
    for (i in 2:(year)) {
      setTxtProgressBar(pb,i)
      new_H <- apply( 
        nu/lambda_scale*
          curr_H[(nrow(curr_H)-nrow(st_sim_dat)+1):nrow(curr_H),]%*%W
        # + ux_sp
        + log(st_sim_dat[,i+2]+1)%*%ar_col*5
        , c(1,2), tanh)
      
      Y <- c(Y, st_sim_dat[,i+3])
      curr_H <- rbind(curr_H, new_H)
    }
    colnames(curr_H) <- paste("node", 1:ncol(curr_H))
    obs_H <- curr_H[-c((nrow(curr_H)-nrow(st_sim_dat)+1):nrow(curr_H)),]
    pred_H <- curr_H[c((nrow(curr_H)-nrow(st_sim_dat)+1):nrow(curr_H)),]
    print(paste("Finding best regularization term for year",year))
    years_before <- year - 1
    obs_y <- Y[1:(years_before*nrow(st_sim_dat))]
    one_step_ahead_model <- glm(obs_y~., family = poisson(link="log"), data = data.frame(cbind(obs_y, obs_H)),
                                control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
    one_step_ahead_pred_y[ensemble_idx,,year-40] <- predict(one_step_ahead_model, newdata = data.frame(pred_H), type = "response")
    # cv <- cv.glmnet(obs_H, obs_y, family = poisson(link="log"))
    # one_step_ahead_model <- glmnet(obs_H, obs_y, family = poisson(link="log"), alpha = 1, lambda = cv$lambda.min,
    #                                control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
    # one_step_ahead_pred_y[ensemble_idx,,year-40] <- predict(one_step_ahead_model, pred_H, type = "response")
  }
}


ensemble_mean <- apply(one_step_ahead_pred_y, 1, mean)
one_step_ahead_res <- st_sim_dat[,(ncol(st_sim_dat)-9):ncol(st_sim_dat)] - ensemble_mean
mean(unlist(as.vector(one_step_ahead_res))^2)
var(unlist(as.vector(st_sim_dat[,(ncol(st_sim_dat)-9):ncol(st_sim_dat)])))
print(paste("Our Model explained ",100- mean(unlist(as.vector(one_step_ahead_res))^2)/var(unlist(as.vector(st_sim_dat[,(ncol(st_sim_dat)-9):ncol(st_sim_dat)])))*100," % ", sep = ""))

# glmnet_res <- glmnet::cv.glmnet(x = obs_H, y = obs_y, family = poisson(link = "log")) 
# glm_model <- glmnet(obs_H, obs_y)
# Total var: 2389.072
# Without sp: 1514.139
# With sp:


# cv <- cv.glmnet(obs_H, obs_y, family = poisson(link = "log") )
# one_step_ahead_model <- glmnet(obs_H, obs_y, family = poisson(link="log"), alpha = 1, lambda = cv$lambda.min,
#                             control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
# predict(one_step_ahead_model, pred_H, type = "response")