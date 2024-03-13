{rm(list = ls())
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
}
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")

  
  coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
  long <- coords$long
  lat <- coords$lat
  # carnegie_1994 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1994.csv", header = TRUE)
  # carnegie_1995 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1995.csv", header = TRUE)
  
  ggplot() +
    geom_point(aes(x = jitter(long), y = jitter(lat), col = nsf_wide_car$X1972)) +
    scale_color_viridis_c(limits = c(0,200)) 
  
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
  
  # rm(basis_1,basis_2, basis_3,basis_arr_1,basis_arr_2,basis_arr_3, basis_use_1_2d, basis_use_2_2d, basis_use_3_2d, convoluted_res1,convoluted_res2,convoluted_res3)
  # Begin recurrent part
  
  # If we only want the certain resolution
  # conv_covar_pc_res_1 <- prcomp(convoluted_res1)
  # num_use_1 <- which(cumsum(conv_covar_pc_res_1$sdev^2)/sum(conv_covar_pc_res_1$sdev^2) <= 0.99)
  # pc_use_1 <- conv_covar_pc_res_1$x[,1:max(num_use_1)]
  # 
  # conv_covar_pc_res_2 <- prcomp(convoluted_res2)
  # num_use_2 <- which(cumsum(conv_covar_pc_res_2$sdev^2)/sum(conv_covar_pc_res_2$sdev^2) <= 0.99)
  # pc_use_2 <- conv_covar_pc_res_2$x[,1:max(num_use_2)]
  # 
  # conv_covar_pc_res_3 <- prcomp(convoluted_res3)
  # num_use_3 <- which(cumsum(conv_covar_pc_res_3$sdev^2)/sum(conv_covar_pc_res_3$sdev^2) <= 0.99)
  # pc_use_3 <- conv_covar_pc_res_3$x[,1:max(num_use_3)]
  # 
  # conv_covar <- cbind(pc_use_1, pc_use_2, pc_use_3)
  # min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}
  # conv_covar <- tanh(conv_covar)
  # zero_col <- which(colSums(conv_covar)==0)
  # conv_covar <- conv_covar[,-zero_col]
  
  
  # conv_covar <- apply(conv_covar, 2, min_max_scale)
  # write.csv(conv_covar, "D:/77/UCSC/study/Research/temp/NSF_dat/conv_basis.csv")
  # conv_covar <- basis_3
  nh <- 200 # Number of hidden units in RNN
  
  dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
  dummy_state <- model.matrix(~nsf_wide_car$state - 1)
  dummy_gss <- model.matrix(~ substr(nsf_wide_car$ID, 7, 9)  - 1)
  dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)
  
  a <- 0.1
  one_step_ahead_pred_y_ridge <- one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = length(2012:2021))
  # possible_lam <- seq(from = 0, to = 100, by = 1)
  # lambda_all_pred  <- array(NA, dim  = c(length(possible_lam), nrow(nsf_wide_car), length(2012:2021)))
  # lambda_all_res <- array(NA, dim  = c(length(possible_lam), nrow(nsf_wide_car), length(2012:2021)))
  leak <- 1
  for (year in 2012:2021) {
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
    # ux_dummy <- dummy_matrix%*%U_dummy
    ux_dummy <- dummy_matrix%*%U_dummy 
    curr_H <- apply(ux_sp, c(1,2), tanh)
    # prev_year <- nsf_wide_car[,10:(year-1972+9)]
    # pc_prev_raw <- prcomp(t(prev_year))$x[,1:min(which(cumsum(prcomp(t(prev_year))$sdev^2)/sum(prcomp(t(prev_year))$sdev^2)  > 0.95))]
    # pc_prev <- scale(pc_prev_raw)
    # nx_pc <- ncol(pc_prev)
    # U_pc <- matrix(runif(nh*nx_pc, -a,a), nrow = nx_pc)

    Y <- nsf_wide_car[,10]
    pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
    print("Calculating Recurrent H Matrix. . .")
    for (i in 2:(year-1972+1)) {
      
      setTxtProgressBar(pb,i)
      # if(i == 2){
        new_H <- apply( 
          nu/lambda_scale*
            curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W
          # + ux_sp
          # + ux_dummy
          + log(nsf_wide_car[,i+8]+1)%*%ar_col
          # + curr_shared_pc %*% U_pc
          , c(1,2), tanh)*leak + curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]*(1-leak)
      # }else{
      #   
      #   new_H <- apply( 
      #     nu/lambda_scale*
      #       curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W
      #     # + ux_sp
      #     # + ux_dummy
      #     + log(nsf_wide_car[,i+8]+1)%*%ar_col + log(nsf_wide_car[,i+8-1]+1)%*%ar_col_lag2*0.1
      #     # + curr_shared_pc %*% U_pc
      #     , c(1,2), tanh)*leak + curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]*(1-leak)
      #     
      # }
      # curr_shared_pc <- matrix(rep(pc_prev[i-1,], nrow(nsf_wide_car)), nrow = nrow(nsf_wide_car), byrow = T)
      
      
      Y <- c(Y, nsf_wide_car[,i+9])
      curr_H <- rbind(curr_H, new_H)
    }
    
    # sp_cnn <- matrix(rep( t(cbind(basis_use_1_2d, basis_use_2_2d, basis_use_3_2d)), year-1972+1), nrow = nrow(curr_H), byrow = TRUE)
    # curr_H <- cbind(curr_H, sp_cnn)
    colnames(curr_H) <- paste("node", 1:ncol(curr_H))
    obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
    pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
    print(paste("Now predicting year",year))
    years_before <- year - 1972
    obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
    
    # Ridge regression
    ridge_model_cv <- cv.glmnet(x = obs_H, y = obs_y, alpha = 0, family = "poisson", trace.it = 1, nfolds = 5)
    ridge_model <- glmnet(x = obs_H, y = obs_y, alpha = 0,
    trace.it = 1, lambda = c(0,ridge_model_cv$lambda.min), family = "poisson", thresh=1e-8)
    one_step_ahead_pred_y_ridge[,year-2011] <- predict(ridge_model, pred_H, type = "response", s = ridge_model_cv$lambda.min)
    one_step_ahead_pred_y[,year-2011] <- predict(ridge_model, pred_H, type = "response", s = 0)
    
    # I used glm to verify the previous code when lambda = 0, it's not wrong.
    # glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)))
    # glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))
    # All lambda model
    # all_lambda_model <- glmnet(x = obs_H, y = obs_y, alpha = 0, family = "poisson", lambda = possible_lam, trace.it = 1)
    # lambda_all_pred[,,year-2011] <- t(predict(all_lambda_model, pred_H, type = "response"))
  }

  # for (i in 1:length(possible_lam)) {
  #  lambda_all_res[i,,] <-  as.matrix(nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - lambda_all_pred[i,,])
  # }
  # lambda_res_squared <- lambda_all_res^2
  # apply(lambda_res_squared, 1, mean)
  
  one_step_ahead_res_ridge <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - one_step_ahead_pred_y_ridge
  one_step_ahead_res <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - one_step_ahead_pred_y
  print(paste("Sample Variance",  var(unlist(as.vector(nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))])))))
  print(paste("ridge prediction error:",  mean(unlist(as.vector(one_step_ahead_res_ridge))^2) ))
  print(paste("non-ridge prediction error:",  mean(unlist(as.vector(one_step_ahead_res))^2) ))

  
  # Write csv file
  # write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""), row.names = FALSE)
  # write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""), row.names = FALSE)
  
  

# Prediction file: paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = "")
#Note: Total oo var is 9199.715, 



# sp basis outside of ESN, no CNN
# sp CNN outside of ESN
# Find a way to run it more efficiently # Finished.
# Adding more details about the ESN 0

#No basis or spatial part:   652.821 / 634.047

  
#Only 1st basis ridge/non-ridge: 663.829 / 643.675
#Only 2nd basis ridge/non-ridge: 646.458 / 630.407
#Only 3rd basis ridge/non-ridge: 659.976 / 637.840
#All basis ridge/non-ridge: 665.961 / 636.969
  
#Only CNN for 1st ridge/non-ridge: 647.522 / 627.136 ------------------------- best
#Only CNN for 2nd ridge/non-ridge: 657.265 / 639.842
#Only CNN for 3rd ridge/non-ridge: 658.838 / 642.679
#All CNN ridge/non-ridge: 650.057 / 632.487
