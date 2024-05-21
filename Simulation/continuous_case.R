{
rm(list = ls())
sim_wide <- read.csv("D:/77/UCSC/study/Research/temp/norm_sim_wide.csv")
sim_long <- read.csv("D:/77/UCSC/study/Research/temp/norm_sim.csv")
wide_y <- as.matrix(sim_wide[,-c(1,2)])
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
  # use_condaenv("tf_gpu")
}

coords <- data.frame("long" = sim_wide$long, "lat" = sim_wide$lat)
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

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}




num_filters <- 64

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")



st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")

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



nh <- 50 # Number of hidden units in RNN
min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}

pi_0 <- 0
num_ensemble <- 1
a <- c(1e-1)
nu_par <- c(0.9)
# res <- array(NA, dim = c(num_ensemble, length(a_par), length(nu_par)))
# sigmoid <- function(x){return(1/(1+exp(-x)))}
osh_pred_spesn <- matrix(NA, nrow = nrow(wide_y), ncol= length(46:50))

for (ensem_idx in 1:num_ensemble) {
      for(curr_year in 46:50){
        
        # Generate W weight matrices
        W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)
        
        lambda_scale <- max(abs(eigen(W)$values))
        
        curr_x <- matrix( c(rep(0,nrow(wide_y)), as.vector(conv_covar) ), ncol = 1 )
        
        nx <- length(curr_x)
        
        U <- matrix(runif(nh*nx, -a, a), nrow = nh, ncol = nx)
        
        curr_H <- tanh(U %*% curr_x)
        H_mat <- curr_H
        
        for (year in 2:(curr_year)) {
          
          curr_x <- matrix( c(wide_y[,year-1], as.vector(conv_covar)), ncol = 1 )
          new_H <- tanh( nu_par/lambda_scale * W %*% curr_H + U %*% curr_x)
          H_mat <- cbind(H_mat,new_H)
          curr_H <- new_H
          
        }
        
        for (school in 1:nrow(wide_y)) {
          print(paste("now doing year ", curr_year, "shcool",school))
          obs_y <- matrix(wide_y[school,1:(curr_year-1)],ncol = 1)
          obs_h <- t(H_mat[,-ncol(H_mat)])
          cv_las <- cv.glmnet(x = obs_h, y = obs_y, alpha = 0)
          curr_model_lasso <- glmnet(x = obs_h, y = obs_y, lambda = cv_las$lambda.min, alpha = 0)
          osh_pred_spesn[school, curr_year-45] <- predict(curr_model_lasso, newx = t(H_mat[,ncol(H_mat)]))
        }
        
      }
    }

mean((osh_pred_spesn - as.matrix(wide_y[,46:50]))^2)
var(as.vector(unlist(wide_y[,46:50])))

# 
# pred_idx <- which(sim_long$time>45)
# df_pred <- data.frame(
#   long = sim_long[pred_idx, 1],
#   lat = sim_long[pred_idx, 2],
#   time = sim_long[pred_idx, 3],
#   value = as.vector(osh_pred_spesn)
# )
# 
# p1 <-
#   ggplot(df_pred, aes(x = long, y = lat, fill = value)) +
#   geom_tile() +
#   facet_wrap(~time) +
#   scale_fill_viridis_c() +
#   theme_minimal() +
#   labs(title = "Spatio-Temporal Data Heatmap",
#        x = "Longitude",
#        y = "Latitude",
#        fill = "Value")
# 
# 
# 
# # Reshape y to a data frame
# df_obs <- data.frame(
#   long = sim_long[pred_idx, 1],
#   lat = sim_long[pred_idx, 2],
#   time = sim_long[pred_idx, 3],
#   value = sim_long[pred_idx, 4]
# )
# 
# # Plot
# p2 <-
#   ggplot(df_obs, aes(x = long, y = lat, fill = value)) +
#   geom_tile() +
#   facet_wrap(~time) +
#   scale_fill_viridis_c() +
#   theme_minimal() +
#   labs(title = "Spatio-Temporal Data Heatmap",
#        x = "Longitude",
#        y = "Latitude",
#        fill = "Value")
# 
# 
# cowplot::plot_grid(p1,p2)
# 
