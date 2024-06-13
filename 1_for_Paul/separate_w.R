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
  # use_condaenv("tf_gpu")
}
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
nsf_y <- nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))]

# coords <- data.frame("long" = nsf_wide_car$long, "lat" = nsf_wide_car$lat)
# long <- coords$long
# lat <- coords$lat
# # carnegie_1994 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1994.csv", header = TRUE)
# # carnegie_1995 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/1995.csv", header = TRUE)
# 
# ggplot() +
#   geom_point(aes(x = jitter(long), y = jitter(lat), col = nsf_wide_car$X1972)) +
#   scale_color_viridis_c(limits = c(0,200))
# 
# coordinates(coords) <- ~ long + lat
# 
# gridbasis1 <- auto_basis(mainfold = plane(), data = coords, nres = 1, type = "Gaussian", regular = 1)
# gridbasis2 <- auto_basis(mainfold = plane(), data = coords, nres = 2, type = "Gaussian", regular = 1)
# gridbasis3 <- auto_basis(mainfold = plane(), data = coords, nres = 3, type = "Gaussian", regular = 1)
# 
# show_basis(gridbasis3) +
#   coord_fixed() +
#   xlab("Longitude") +
#   ylab("Latitude")
# 
# 
# 
# basis_1 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis1@fn))
# pb <- txtProgressBar(min = 1, max = length(gridbasis1@fn), style = 3)
# for (i in 1:length(gridbasis1@fn)) {
#   setTxtProgressBar(pb, i)
#   basis_1[,i] <- gridbasis1@fn[[i]](coordinates(coords))
# }
# 
# basis_2 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis2@fn))
# pb <- txtProgressBar(min = 1, max = length(gridbasis2@fn), style = 3)
# for (i in 1:length(gridbasis2@fn)) {
#   setTxtProgressBar(pb, i)
#   basis_2[,i] <- gridbasis2@fn[[i]](coordinates(coords))
# }
# 
# basis_3 <- matrix(NA, nrow = nrow(coords@coords), ncol = length(gridbasis3@fn))
# pb <- txtProgressBar(min = 1, max = length(gridbasis3@fn), style = 3)
# for (i in 1:length(gridbasis3@fn)) {
#   setTxtProgressBar(pb, i)
#   basis_3[,i] <- gridbasis3@fn[[i]](coordinates(coords))
# }
# 
# 
# # Redefine three layers of basis images
# basis_use_1_2d <- basis_1
# basis_use_2_2d <- basis_3[,(ncol(basis_1)+1):ncol(basis_2)]
# basis_use_3_2d <- basis_3[,(ncol(basis_2)+1):ncol(basis_3)]
# 
# 
# # First resolution
# shape_row_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 2 ]))
# shape_col_1 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 1) , 1 ]))
# basis_arr_1 <- array(NA, dim = c(nrow(coords@coords), shape_row_1, shape_col_1))
# 
# for (i in 1:nrow(coords@coords)) {
#   basis_arr_1[i,,] <- matrix(basis_use_1_2d[i,], nrow = shape_row_1, ncol = shape_col_1, byrow = T)
# }
# basis_arr_1 <- array_reshape(basis_arr_1,c(dim(basis_arr_1),1))
# 
# # Second resolution
# shape_row_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 2 ]))
# shape_col_2 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 2) , 1 ]))
# basis_arr_2 <- array(NA, dim = c(nrow(coords@coords), shape_row_2, shape_col_2))
# for (i in 1:nrow(coords@coords)) {
#   basis_arr_2[i,,] <- matrix(basis_use_2_2d[i,], nrow = shape_row_2, ncol = shape_col_2, byrow = T)
# }
# basis_arr_2 <- array_reshape(basis_arr_2,c(dim(basis_arr_2),1))
# 
# # Third resolution
# shape_row_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 2 ]))
# shape_col_3 <- length(table(gridbasis3@df[which(gridbasis3@df$res == 3) , 1 ]))
# basis_arr_3 <- array(NA, dim = c(nrow(coords@coords), shape_row_3, shape_col_3))
# for (i in 1:nrow(coords@coords)) {
#   basis_arr_3[i,,] <- matrix(basis_use_3_2d[i,], nrow = shape_row_3, ncol = shape_col_3, byrow = T)
# }
# basis_arr_3 <- array_reshape(basis_arr_3,c(dim(basis_arr_3),1))
# 
# a <- 1
# 
# my_custom_initializer <- function(shape, dtype = NULL) {
#   return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
# }
# 
# 
# 
# 
# num_filters <- 64
# 
# st_model_res_1 <- keras_model_sequential() %>%
#   layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
#                 input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
#   layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")
# 
# 
# st_model_res_2 <- keras_model_sequential() %>%
#   layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
#                 input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
#   layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")
# 
# 
# 
# st_model_res_3 <- keras_model_sequential() %>%
#   layer_conv_2d(filters = num_filters, kernel_size = c(3, 3), activation = "tanh",
#                 input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
#   layer_flatten() %>% layer_dense(units = 100, kernel_initializer = my_custom_initializer, activation = "tanh")
# 
# convoluted_res1 <- predict(st_model_res_1,basis_arr_1)
# convoluted_res2 <- predict(st_model_res_2,basis_arr_2)
# convoluted_res3 <- predict(st_model_res_3,basis_arr_3)
# 
# # conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,,,],convoluted_res2[1,,,],convoluted_res3[1,,,])))
# conv_covar <- matrix(NA,nrow = length(long), ncol = length(c(convoluted_res1[1,],convoluted_res2[1,],convoluted_res3[1,])))
# 
# pb <- txtProgressBar(min = 1, max = length(long), style = 3)
# for (i in 1:length(long)) {
#   setTxtProgressBar(pb, i)
#   # conv_covar[i,] <- c(as.vector(convoluted_res1[i,,,]),as.vector(convoluted_res2[i,,,]),as.vector(convoluted_res3[i,,,]))
#   conv_covar[i,] <- c(as.vector(convoluted_res1[i,]),as.vector(convoluted_res2[i,]),as.vector(convoluted_res3[i,]))
# }

nh <- 30 # Number of hidden units in RNN
min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}

pi_0 <- 0
num_ensemble <- 1
a <- c(0.1)
nu <- c(0.5)
# res <- array(NA, dim = c(num_ensemble, length(a_par), length(nu_par)))
sigmoid <- function(x){return(exp(x)/(1+exp(x)))}

pred_y <- matrix(NA, nrow = nrow(nsf_y), ncol = 10)

for (ensem_idx in 1:num_ensemble) {
      for (curr_year in 2012:2021) {
        # Generate W weight matrices
        W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)*rbinom(nh^2, 1, 1-pi_0)
        lambda_scale <- max(abs(eigen(W)$values))
        nu_par = 1
        # Figure out the dimension of EOFs, first PCA
        prev_years <- nsf_y[,1:(curr_year-1972)]
        pc_res <- prcomp(t(prev_years))
        num_pc <- min(which(cumsum(pc_res$sdev^2)/(sum(pc_res$sdev^2)) >= 0.8 ))
        pc_use <- apply(pc_res$x[,1:num_pc],2, min_max_scale)
        # pc_use <- pc_res$x[,1:num_pc]
        U_pc <- matrix(runif(nh*num_pc, -a, a), ncol = nh)
        U_ar <- matrix(runif(nh, -a, a), ncol = 1) * rbinom(nh, 1, 1-pi_0)
        # U_sp <- matrix(runif(nh*ncol(conv_covar), -a, a), ncol = nh)
        cc <- runif(nh, -a, a)
        # Ux_sp <- conv_covar %*% U_sp
        Ux_sp <- matrix(0, nrow = nrow(nsf_wide_car), ncol = nh)
        curr_H <- tanh(Ux_sp)
        Y <- nsf_y[,1]
        for (year in 2:(curr_year-1972+1)) {
          new_H <- tanh(
            nu/lambda_scale * curr_H[((year-2)*nrow(nsf_y)+1):nrow(curr_H),] %*% W 
            + (matrix( log(nsf_y[,year-1]+0.1))) %*% t(U_ar) + cc
            # + Ux_sp 
            # + matrix(rep(pc_use[year-1,], nrow(nsf_y)), ncol = num_pc, byrow = TRUE) %*% U_pc
          )
          curr_H <- rbind(curr_H, new_H)
          Y <- c(Y, nsf_y[,year])
        }
        colnames(curr_H) <- paste("node", 1:ncol(curr_H))
        for (school in 1:nrow(nsf_y)) {
          print(paste("Now doing year",curr_year, "School", school))
          obs_H <- curr_H[seq(from = school, by = nrow(nsf_y), length.out = curr_year-1972),]
          pred_H <- matrix(new_H[school,],nrow = 1)
          colnames(pred_H) <- colnames(obs_H)
          obs_y <- Y[seq(from = school, by = nrow(nsf_y), length.out = curr_year-1972)]
          
          # glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)))
          # glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))
          # pred_y[school,curr_year-2011] <- glm_pred
          # cv_model <- cv.glmnet(x = obs_H, y = obs_y, family = poisson(link = "log"), alpha = 1, nlambda = 100, type.measure = "mse")
          # plot(cv_model)
          penal_model <- glmnet(x = obs_H, y = obs_y, family = poisson(link = "log"), alpha = 1, lambda = .8)
          pred_y[school,curr_year-2011] <- predict(penal_model, type = "response", newx = pred_H)
        }
  }
}

# print(nsf_y$X2012 - glm_pred)

res_all <- as.matrix(nsf_y)[,41:50] - pred_y
apply(res_all^2, 2, mean)
mean(na.omit(res_all)^2)
# write.csv( as.data.frame(pred_y), "D:/77/UCSC/study/Research/temp/NSF_dat/pcesn_pred.csv", row.names = FALSE)
# write.csv( as.data.frame(res_all), "D:/77/UCSC/study/Research/temp/NSF_dat/pcesn_res.csv", row.names = FALSE)
# penalized separate
# X2012     X2013     X2014     X2015     X2016     X2017     X2018     X2019     X2020     X2021 
# 423.6692  548.4714  405.9747  398.5617  435.6500 2184.1975  383.0301  659.0441 1323.0543  396.9642

# share parameter
# X2012     X2013     X2014     X2015     X2016     X2017     X2018     X2019     X2020     X2021 
# 285.1488  349.0803  296.0272  314.3570  391.3869 2254.9935  256.3499  543.0569 1488.3717  380.1291


