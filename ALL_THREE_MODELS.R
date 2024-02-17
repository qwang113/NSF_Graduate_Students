rm(list = ls())
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)

out.rm <- 0

if(out.rm)
{
  # Remove outliers
  nh = 200
  osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
  annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
  nsf_wide <- nsf_wide[-annoying_cases[1:100],]
}


UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
wide_y <- nsf_wide_car[,-c(1:9, ncol(nsf_wide_car))]

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
# dummy_matrix <- cbind(dummy_school, dummy_car)
dummy_matrix <- dummy_school
###################################################################### PCA Method#########################
#Assume Beta changes across the schools.
#No need to add covariate if they doesn't change aross the time.

individual_beta_pred <- matrix(NA, nrow = nrow(nsf_wide), ncol = length(2012:2021))
for (i in 2012:2021) {
  
  years_before <- i - 1972
  prev_y <- t(wide_y)[1:years_before,]
  prev_pc <- prcomp(prev_y)
  num_pc <- length(which(cumsum(prev_pc$sdev^2)/sum(prev_pc$sdev^2) <= 0.8 ))
  prev_pc_use <- prev_pc$x[,1:num_pc]
  
  for (j in 1:nrow(nsf_wide)) {
    print(paste(  print(paste("Now doing year",i)), "Now doing school",j))
    #Each school has its own slope but shared across the time, loop across the schoools
    curr_y <- wide_y[j,2:years_before]
    curr_pc_cov <- prev_pc_use[1:(nrow(prev_pc_use)-1),] # Use the previous year's principal component
    curr_dat <- cbind(t(curr_y), curr_pc_cov, t(wide_y[j,1:(years_before-1)]))
    colnames(curr_dat) <- c("y",colnames(curr_pc_cov),"lag")
    curr_model <- glm(y~., data = data.frame(curr_dat), family =poisson(link = "log"),  control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = FALSE))
    pred_x <- cbind(matrix(prev_pc_use[nrow(prev_pc_use),], nrow = 1), wide_y[j,years_before])
    colnames(pred_x) <- colnames(curr_dat)[-1]
    prediction <- predict(curr_model, data.frame(pred_x), type = "response")
    individual_beta_pred[j,i-2011] <- prediction
  }
  
}

individual_beta_res <- nsf_wide_car[,50:59] - individual_beta_pred

if(nrow(nsf_wide) == 1799){
  write.csv( as.data.frame(individual_beta_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv", row.names = FALSE)
  write.csv( as.data.frame(individual_beta_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv", row.names = FALSE)
}else{
  write.csv( as.data.frame(individual_beta_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred_outrm.csv", row.names = FALSE)
  write.csv( as.data.frame(individual_beta_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_res_outrm.csv", row.names = FALSE)
}





######################ESN###########################
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

if(out.rm)
{
  # Remove outliers
  nh = 200
  osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
  annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
  nsf_wide <- nsf_wide[-annoying_cases[1:100],]
}

UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")[,-c(1:4,6:9)]
gss <- substr(nsf_wide$ID,7,9)

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

a <- 0.01

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}

# my_custom_initializer <- function(shape, dtype = NULL) {
#   return(tf$random$normal(shape,mean = 0, stddev = 0.1, dtype = dtype))
# }


num_filters <- 200

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 1000, kernel_initializer = my_custom_initializer)


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten()%>% layer_dense(units = 1000, kernel_initializer = my_custom_initializer)



st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 1000, kernel_initializer = my_custom_initializer)

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
conv_covar <- apply(conv_covar, 2, min_max_scale)
# write.csv(conv_covar, "D:/77/UCSC/study/Research/temp/NSF_dat/conv_basis.csv")

nh <- 200 # Number of hidden units in RNN

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
dummy_state <- model.matrix(~nsf_wide_car$state - 1)
dummy_gss <- model.matrix(~gss - 1)
dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)

a <- 0.01
one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide), ncol = length(2012:2021))
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
  # ux_sp <- conv_covar%*%U_sp
  ux_sp <- conv_covar%*%U_sp/max(abs(eigen(U_sp%&%t(U_sp))$values))
  # ux_dummy <- dummy_matrix%*%U_dummy
  ux_dummy <- dummy_matrix%*%U_dummy /max(svd(U_dummy)$d)
  curr_H <- apply(ux_sp + ux_dummy, c(1,2), tanh)
  prev_year <- nsf_wide_car[,2:(year-1972+1)]
  pc_prev <- prcomp(t(prev_year), scale. = TRUE)$x[,1:min(which(cumsum(prcomp(t(prev_year))$sdev^2)/sum(prcomp(t(prev_year))$sdev^2)  > 0.95))]
  nx_pc <- ncol(pc_prev)
  U_pc <- matrix(runif(nh*nx_pc, -a,a), nrow = nx_pc)
  curr_H <- apply(ux_sp + ux_dummy, c(1,2), tanh)
  Y <- nsf_wide_car[,2]
  pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
  for (i in 2:(year-1972+1)) {
    setTxtProgressBar(pb,i)
    curr_shared_pc <- matrix(rep(pc_prev[i-1,], nrow(nsf_wide_car)), nrow = nrow(nsf_wide_car), byrow = T)
    new_H <- apply( 
      nu/lambda_scale*
        curr_H[(nrow(curr_H)-nrow(nsf_wide)+1):nrow(curr_H),]%*%W 
      # + ux_sp
      # + ux_dummy
      + nsf_wide_car[,i]%*%ar_col
      # + curr_shared_pc%*% U_pc
      , c(1,2), tanh)
    
    Y <- c(Y, nsf_wide_car[,i+1])
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
  # one_step_ahead_model <- glm.nb(obs_y~.,  data = data.frame(cbind(obs_y, obs_H)),
  #                             control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
  
  one_step_ahead_pred_y[,year-2011] <- predict(one_step_ahead_model, newdata = data.frame(pred_H), type = "response")
}


one_step_ahead_res <- nsf_wide_car[,c((2012-1972+2):(ncol(nsf_wide_car)-1))] - one_step_ahead_pred_y
mean(unlist(as.vector(one_step_ahead_res))^2)
var(unlist(as.vector(nsf_wide_car[,c((2012-1972+2):(ncol(nsf_wide_car)-1))])))




if(nrow(nsf_wide) == 1799){
  write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""), row.names = FALSE)
  write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""), row.names = FALSE)
}else{
  
  write.csv( as.data.frame(one_step_ahead_pred_y), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_outrm_",nh, ".csv", sep = ""), row.names = FALSE)
  write.csv( as.data.frame(one_step_ahead_res), paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_outrm_",nh, ".csv", sep = ""), row.names = FALSE)
  
}


################################# Pois AutoReg ############################################

# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)

if(out.rm)
{
  # Remove outliers
  nh = 200
  osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
  annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
  nsf_wide <- nsf_wide[-annoying_cases[1:100],]
}





carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
wide_y <- nsf_wide_car[,-c(1:9, ncol(nsf_wide_car))]

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
# dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
# dummy_matrix <- cbind(dummy_school, dummy_car)
dummy_matrix <- dummy_car

osh_pred <- matrix(NA, nrow = nrow(wide_y), ncol = length(2012:2021))

#All schools pooled together
for (year in 2012:2021) {
  long_yt <- unlist(as.vector(wide_y[,2:(year-1972)]))
  long_yt_lag <- unlist(as.vector(wide_y[,1:(year-1972-1)]))
  curr_dummy <- matrix(rep(t(dummy_matrix), year-1972-1), ncol = ncol(dummy_matrix), byrow = T)[,-1]
  obs_dat <- data.frame(cbind(long_yt, long_yt_lag, curr_dummy))
  curr_model <- glm(long_yt~., family = poisson(link = "log"), data = obs_dat)
  pred_yt_lag <- wide_y[,year-1972]
  pred_dummy <- dummy_matrix[,-1]
  pred_mat <- cbind(pred_yt_lag, pred_dummy)
  colnames(pred_mat) <- colnames(obs_dat)[-1]
  pred_dat <- data.frame(pred_mat)
  osh_pred[,year-2011] <- predict(curr_model, newdata = pred_dat, type = "response")
}

osh_res <- wide_y[,41:50] - osh_pred
mean(unlist(as.vector(osh_res^2)))

#Individually regression: AR1
for (year in 2012:2021) {
  for (school in 1:nrow(wide_y)) {
    print(paste("now doing year", year, "school", school))
    prev_obs <- unlist(wide_y[school,2:(year-1972)]) 
    lag_obs <- unlist(wide_y[school, 1:(year-1972-1)])
    model <- glm(prev_obs ~ lag_obs, family = poisson(link = "log"))
    osh_pred[school, year-2011] <- predict(model, newdata = data.frame("lag_obs"=unlist(wide_y[school, year-1972])),
                                           type = "response")
  }
}

osh_res <- wide_y[,41:50] - osh_pred


if(nrow(nsf_wide) == 1799){
  print("Full dataset")
  write.csv( as.data.frame(osh_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv", row.names = FALSE)
  write.csv( as.data.frame(osh_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv", row.names = FALSE)
}else{
  print("Not Full dataset")
  write.csv( as.data.frame(osh_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred_outrm.csv", row.names = FALSE)
  write.csv( as.data.frame(osh_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res_outrm.csv", row.names = FALSE)
}
