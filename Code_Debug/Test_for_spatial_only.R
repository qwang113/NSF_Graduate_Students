{
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
  ################ Spefify the virtual environment ##############################
  use_condaenv("tf_gpu")
}
# I have uploaded the dataset to the github root folder, so you don't need to change the file path.
nsf_wide_car <- read.csv(here::here("nsf_final_wide_car.csv"))

# Generate basis functions
coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
long <- coords$long
lat <- coords$lat
coordinates(coords) <- ~ long + lat
gridbasis1 <- auto_basis(mainfold = plane(), data = coords, nres = 1, type = "Gaussian", regular = 1)
gridbasis2 <- auto_basis(mainfold = plane(), data = coords, nres = 2, type = "Gaussian", regular = 1)
gridbasis3 <- auto_basis(mainfold = plane(), data = coords, nres = 3, type = "Gaussian", regular = 1)

#Specify basis functions of different resolutions

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

# Define the range of the random weight's distribution, Unif[-a,a)]
a <- 1
my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}

# Specify the number of filters
num_filters <- 64

# Build the structure of each resolution

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten()%>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")


st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")

# Input basis functions and get the output from CNN with random weights
convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
convoluted_res3 <- predict(st_model_res_3,basis_arr_3)

conv_covar <- cbind(convoluted_res1, convoluted_res2, convoluted_res3)

nh <- 200 # Number of hidden units in RNN

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)[,-1]
dummy_state <- model.matrix(~nsf_wide_car$state - 1)[,-1]
dummy_gss <- model.matrix(~ substr(nsf_wide_car$ID, 7, 9)  - 1)[,-1]
dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)

a <- 0.1
one_step_ahead_pred_y_ridge <- one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = length(2012:2021))
sp_cv <- seq(from = 0, to = 0.1, length.out = 10)
tm_cv <- seq(from = 0, to = 5, length.out = 10)
leak <- 1
year = 2012
#Initialize
nx_sp <- ncol(conv_covar)
nx_dummy <- ncol(dummy_matrix)
nu <- 1
W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
U_dummy <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
ar_col <- matrix(runif(nh,-a,a), nrow = 1)
ar_col_lag2 <- matrix(runif(nh,-a,a), nrow = 1)
lambda_scale <- max(abs(eigen(W)$values))
ux_sp <- conv_covar%*%U_sp

# res_cv <- matrix(NA, length(sp_cv),length(tm_cv))
# for (spcv in 1:length(sp_cv)) {
#   for (tmcv in 1:length(tm_cv)) {
    
    
    
    # print(paste("Now doing sptm cv for",spcv,tmcv))
    
    
    curr_H <- apply(ux_sp, c(1,2), tanh)
    Y <- nsf_wide_car[,10]
    pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
    print("Calculating Recurrent H Matrix. . .")
    for (i in 2:(year-1972+1)) {
      
      setTxtProgressBar(pb,i)
      new_H <- apply( 
        nu/lambda_scale*
          curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W
        # + ux_sp * sp_cv[spcv]
        + log(nsf_wide_car[,i+8]+1)%*%ar_col*2
        , c(1,2), tanh)*leak + curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]*(1-leak)
      
      Y <- c(Y, nsf_wide_car[,i+9])
    curr_H <- rbind(curr_H, new_H)
  }

  # sp_cnn <- matrix(rep( t(convoluted_res1), year-1972+1), nrow = nrow(curr_H), byrow = TRUE)
  # curr_H <- cbind(curr_H, sp_cnn)
  
  colnames(curr_H) <- paste("node", 1:ncol(curr_H))
  obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
  pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
  print(paste("Now predicting year",year))
  years_before <- year - 1972
  obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
  
  
  # I used glm to verify the previous code when lambda = 0, it's not wrong.
  glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)))
  glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))
  mean((glm_pred - nsf_wide_car$X2012)^2)
  # res_cv[spcv,tmcv] <- mean((glm_pred - nsf_wide_car$X2012)^2)
# }
# 
# }

#no sp:336.5021
#with sp:527.9727