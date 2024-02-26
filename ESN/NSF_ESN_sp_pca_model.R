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
# out.rm = 1
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
  # if(out.rm)
  # {
  #   # Remove outliers
  #   nh = 200
  #   osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
  #   annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
  #   nsf_wide_car <- nsf_wide_car[-annoying_cases[1:100],]
  # }
  
  
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
  
  a <- 1
  
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
    layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "sigmoid")
  
  
  st_model_res_2 <- keras_model_sequential() %>%
    layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                  input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
    layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "sigmoid")
  
  
  
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
  
  # rm(basis_1,basis_2, basis_3,basis_arr_1,basis_arr_2,basis_arr_3, basis_use_1_2d, basis_use_2_2d, basis_use_3_2d, convoluted_res1,convoluted_res2,convoluted_res3)
  # Begin recurrent part
  
  # If we only want the certain resolution
  conv_covar <- conv_covar[,201:300]
  
  # zero_col <- which(colSums(conv_covar)==0)
  # conv_covar <- conv_covar[,-zero_col]
  
  min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}
  # conv_covar <- apply(conv_covar, 2, min_max_scale)
  # write.csv(conv_covar, "D:/77/UCSC/study/Research/temp/NSF_dat/conv_basis.csv")
  
  nh <- 200 # Number of hidden units in RNN
  
  dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
  dummy_state <- model.matrix(~nsf_wide_car$state - 1)
  dummy_gss <- model.matrix(~ substr(nsf_wide_car$ID, 7, 9)  - 1)
  dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)
  
  a <- 0.1
  one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = length(2012:2021))
  for (year in 2012:2021) {
    #Initialize
    nx_sp <- ncol(conv_covar)
    nx_dummy <- ncol(dummy_matrix)
    nu <- 1
    W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
    U_sp <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
    U_dummy <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
    ar_col <- matrix(runif(nh,-a,a), nrow = 1)
    lambda_scale <- max(abs(eigen(W)$values))
    ux_sp <- conv_covar%*%U_sp
    # ux_sp <- conv_covar%*%U_sp/max(abs(eigen(U_sp%&%t(U_sp))$values))
    # ux_dummy <- dummy_matrix%*%U_dummy
    ux_dummy <- dummy_matrix%*%U_dummy 
    curr_H <- apply(ux_sp, c(1,2), tanh)
    prev_year <- nsf_wide_car[,10:(year-1972+9)]
    pc_prev_raw <- prcomp(t(prev_year))$x[,1:min(which(cumsum(prcomp(t(prev_year))$sdev^2)/sum(prcomp(t(prev_year))$sdev^2)  > 0.95))]
    pc_prev <- scale(pc_prev_raw)
    nx_pc <- ncol(pc_prev)
    U_pc <- matrix(runif(nh*nx_pc, -a,a), nrow = nx_pc)

    Y <- nsf_wide_car[,10]
    pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
    print("Calculating Recurrent H Matrix. . .")
    for (i in 2:(year-1972+1)) {
      setTxtProgressBar(pb,i)
      curr_shared_pc <- matrix(rep(pc_prev[i-1,], nrow(nsf_wide_car)), nrow = nrow(nsf_wide_car), byrow = T)
      new_H <- apply( 
        nu/lambda_scale*
          curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W 
        # + ux_sp
          + ux_dummy
        + log(nsf_wide_car[,i+8]+1)%*%ar_col
        # + curr_shared_pc %*% U_pc
        , c(1,2), tanh)
      
      Y <- c(Y, nsf_wide_car[,i+9])
      curr_H <- rbind(curr_H, new_H)
    }
    colnames(curr_H) <- paste("node", 1:ncol(curr_H))
    obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
    pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
    print(paste("Now doing year",year))
    years_before <- year - 1972
    obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
    one_step_ahead_model <- glm(obs_y~., family = poisson(link="log"), data = data.frame(cbind(obs_y, obs_H)),
    control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
    one_step_ahead_pred_y[,year-2011] <- predict(one_step_ahead_model, newdata = data.frame(pred_H), type = "response")
# 
#     one_step_ahead_model <- lm(obs_y~., data = data.frame(cbind(obs_y, obs_H)))
#     one_step_ahead_pred_y[,year-2011] <- predict(one_step_ahead_model, newdata = data.frame(pred_H))
#     
  }
  
  
  one_step_ahead_res <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - one_step_ahead_pred_y
  mean(unlist(as.vector(one_step_ahead_res))^2)
  var(unlist(as.vector(nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))])))
  
  # Write csv file
  # write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""), row.names = FALSE)
  # write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""), row.names = FALSE)
  
  
  # if(out.rm == 0){
    # write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""), row.names = FALSE)
    # write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""), row.names = FALSE)
  # }else{
  #   write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_outrm_",nh, ".csv", sep = ""), row.names = FALSE)
  #   write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_outrm_",nh, ".csv", sep = ""), row.names = FALSE)
  # }
  


# Prediction file: paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = "")
#Note: Total oo var is 9199.715, 

# MSE in each case is as follows:
# a = 0.01, PC scaled, nu = 1, 200 nodes
#1. Pois with everything: 882.71  Y_{t-1} not logged
#2. Pois without sp: 777.5057  Y_{t-1} not logged
#3. Pois without sp, without EOF: 655.0542
#4. Pois without sp, without EOF, without covariates: 653.552
#5. Pois with no covariates: 9274.638 bad model
## We can't take log Y, or the prediction MSE goes exploded, bigger than 3e5

# a = 0.01, PC scaled, nu = 1, 500 nodes
#1. Pois with everything but sp: 1188.984 Y_{t-1} not logged   seems like overfitting


# a = 0.01, PC scaled, nu = 1, 300 nodes
#1. Pois with everything but sp: 980.2472 Y_{t-1} not logged   seems like overfitting


# a = 0.01, PC scaled, nu = 1, 100 nodes
#1. Pois with everything but sp: 965



# Nodes, spatial resolutions, filters, 







