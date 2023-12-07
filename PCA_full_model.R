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
UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")

coords <- data.frame("long" = nsf_wide$long, "lat" = nsf_wide$lat)
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

a <- 0.1

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}
sig <- 0.9

# my_custom_initializer <- function(shape, dtype = NULL) {
#   return(tf$random$normal(shape,mean = 0, stddev = 0.1, dtype = dtype))
# }


num_filters <- 200

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer)  %>%
  layer_flatten() %>%
  layer_dense(units = 2000, kernel_initializer = my_custom_initializer, activation = "sigmoid")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid", kernel_initializer = my_custom_initializer)  %>%
  layer_flatten() %>%
  layer_dense(units = 2000, kernel_initializer = my_custom_initializer, activation = "sigmoid")


st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid", kernel_initializer = my_custom_initializer)%>%
  layer_flatten() %>%
  layer_dense(units = 2000, kernel_initializer = my_custom_initializer, activation = "sigmoid")


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

rm(basis_1,basis_2, basis_3,basis_arr_1,basis_arr_2,basis_arr_3, basis_use_1_2d, basis_use_2_2d, basis_use_3_2d, convoluted_res1,convoluted_res2,convoluted_res3)
# Begin recurrent part



# zero_col <- which(colSums(conv_covar)==0)
# conv_covar <- conv_covar[,-zero_col]

min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}
# conv_covar <- apply(conv_covar, 2, min_max_scale)
# write.csv(conv_covar, "D:/77/UCSC/study/Research/temp/NSF_dat/conv_basis.csv")

num_prev <- 30
nu <- 0.9
time_step <- 50

leak_rate <- 1 # It's always best to choose 1 here according to Mcdermott and Wille, 2017
nh <- 2000 # Number of hidden units in RNN
dummy_school <- model.matrix(~nsf_wide_car$UNITID - 1)
dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
dummy_matrix <- cbind(dummy_school, dummy_car)

# PCA, we can choose to scale and certer or not
res_y <- t(nsf_wide[,-c(1:(5+num_prev))])
pca_res <- prcomp(res_y , scale. = FALSE, center = FALSE)
# Select the number of components that can explain over 95% variance
num_comp <- which(cumsum(pca_res$sdev^2)/sum(pca_res$sdev^2) > 0.95)[1]
pc_y <-  pca_res$x[,1:num_comp]


nx_sp <- ncol(conv_covar) # Number of covariates
nx_dummy <- ncol(dummy_matrix)

W <- vector("list")
U_sp <- vector("list")
U_dummy <- vector("list")
ar_col <- vector("list")
lambda_scale <- rep(NA, num_comp)
ux_sp <- vector("list")
ux_dummy <- vector("list")

for (i in 1:num_comp) {
  W[[i]] <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
  U_sp[[i]] <- matrix(runif(nh*nx_sp, -a,a), nrow = nx_sp, ncol = nh)
  U_dummy[[i]] <- matrix(runif(nh*nx_dummy, -a,a), nrow = nx_dummy, ncol = nh)
  ar_col[[i]] <- matrix(runif(nh,-a,a), nrow = 1)
  lambda_scale[i] <- max(abs(eigen(W[[i]])$values))
  ux_sp[[i]] <- conv_covar%*%U_sp[[i]]
  ux_dummy[[i]] <- dummy_matrix%*%U_dummy[[i]]
}


lambda_scale <- max(abs(eigen(W)$values))

# ux_full <- cbind(conv_covar,dummy_school)%*%rbind(U_sp,U_dummy)


curr_H <- apply(ux_sp + ux_dummy, c(1,2), tanh)
# curr_H <- ux_sp

Y <- nsf_wide[,6]

pb <- txtProgressBar(min = 1, max = length(2:time_step), style = 3)
for (i in 2:time_step) {
  setTxtProgressBar(pb,i)
  new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + ux_dummy + log(nsf_wide[,i+4] + 1)%*%ar_col, c(1,2), tanh)
  # new_H <- apply( nu/lambda_scale*curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W + ux_sp + nsf_wide[,i+2]%*%ar_col, c(1,2), tanh)
  Y <- c(Y, nsf_wide[,i+5])
  curr_H <- rbind(curr_H, new_H)
}

curr_H_scaled <- apply(curr_H, 2, min_max_scale)

colnames(curr_H_scaled) <- paste("node", 1:ncol(curr_H_scaled))

# curr_H_scaled <- as.data.frame(curr_H_scaled)

# pca_H <- prcomp(curr_H)
# pca_var <-predict(pca_H)
# write.csv(curr_H, "D:/77/UCSC/study/Research/temp/NSF_dat/CRESN_full_model_H.csv", row.names = FALSE)

# cv_model <-  cv.glmnet(x = curr_H, y = Y, family = "poisson",alpha = 0)
# ridge_cresn <- glmnet(x= curr_H, y = Y, alpha = 0, lambda = cv_model$lambda.min, family = "poisson")
# y_pred <- predict(ridge_cresn, newx = curr_H, s = cv_model$lambda)
one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide), ncol = length(2001:2021))

for (i in 2001:2021) {
  years_before <- i - 1972
  prev_y <- Y[1:(years_before*nrow(nsf_wide))]
  prev_H <- data.frame(curr_H_scaled[1:(years_before*nrow(nsf_wide)), ])
  one_step_ahead_model <- glm.nb(prev_y~., data = cbind(prev_y, prev_H), control = glm.control(epsilon = 1e-8, maxit = 1000, trace = TRUE))
  pred_H <- curr_H_scaled[c( (years_before*nrow(nsf_wide)+1):((years_before+1)*nrow(nsf_wide)) ), ]
  # estim_params <- one_step_ahead_model$coefficients
  # pred_y <- cbind(1,pred_H) %*% matrix(estim_params, ncol = 1)
  predict(one_step_ahead_model, newdata = data.frame(pred_H))
  one_step_ahead_pred_y[,i-2000] <- exp(pred_y)
  
}

one_step_ahead_res <- nsf_wide[,c(ncol(nsf_wide)-20):ncol(nsf_wide)] - one_step_ahead_pred_y





write.csv( as.data.frame(one_step_ahead_pred_y), "D:/77/UCSC/study/Research/temp/NSF_dat/oo_one_step_ahead_pred.csv", row.names = FALSE)









glm_CRESN_scaled <- glm.nb(Y~curr_H_scaled, control = glm.control(epsilon = 1e-8, maxit = 50, trace = TRUE))
glm_CRESN <- glm.nb(Y~curr_H, control = glm.control(epsilon = 1e-8, maxit = 50, trace = TRUE))

CRESN_res <- glm_CRESN$residuals
CRESN_res <- Y - y_pred

year_stack <- rep(1972:2021, each = nrow(nsf_wide))
school_ID <- rep(nsf_wide$ID,50)
long_stack <- rep(nsf_wide$long,50)
lat_stack <- rep(nsf_wide$lat,50)

res_stack <- data.frame("ID" = school_ID, "long" = long_stack, "lat" = lat_stack, "year" = year_stack, "Residuals" = CRESN_res)

write.csv(res_stack, paste("D:/77/UCSC/study/Research/temp/NSF_dat/", "Full_Model+state+Car_res+Ridge", nh, ".csv", sep = ""), row.names = FALSE)
