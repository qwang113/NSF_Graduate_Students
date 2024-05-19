# Poisson AR
rm(list = ls())
sim_wide <- read.csv("D:/77/UCSC/study/Research/temp/sim_wide.csv")
sim_long <- read.csv("D:/77/UCSC/study/Research/temp/sim.csv")
wide_y <- sim_wide[,-c(1,2)]
# osh_pred_pois <- matrix(NA, nrow = nrow(wide_y), ncol = length(11:15))
# 
# for (year in 11:15) {
#   for (school in 1:nrow(wide_y)) {
#     print(paste("now doing year", year, "school", school))
#     prev_obs <- unlist(wide_y[school,2:(year-1)]) 
#     lag_obs <- unlist(wide_y[school, 1:(year-2)])
#     model <- glm(prev_obs ~ lag_obs, family = poisson(link = "log"))
#     osh_pred_pois[school, year-10] <- predict(model, newdata = data.frame("lag_obs" = unlist(wide_y[school, year-1])),
#                                            type = "response")
#   }
# }
# mean((osh_pred_pois - as.matrix(wide_y[,11:15]))^2)




{
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

a <- .1

my_custom_initializer <- function(shape, dtype = NULL) {
  return(tf$random$uniform(shape, minval = -a, maxval = a, dtype = dtype))
}




num_filters <- 64

st_model_res_1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_1, shape_col_1, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 20, kernel_initializer = my_custom_initializer, activation = "sigmoid")


st_model_res_2 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_2, shape_col_2, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 20, kernel_initializer = my_custom_initializer, activation = "sigmoid")



st_model_res_3 <- keras_model_sequential() %>%
  layer_conv_2d(filters = num_filters, kernel_size = c(2, 2), activation = "sigmoid",
                input_shape = c(shape_row_3, shape_col_3, 1), kernel_initializer = my_custom_initializer) %>%
  layer_flatten() %>% layer_dense(units = 20, kernel_initializer = my_custom_initializer, activation = "sigmoid")

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

hist(conv_covar)

nh <- 50 # Number of hidden units in RNN
min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}

pi_0 <- 0
num_ensemble <- 1
a_par <- c(.1)
nu_par <- c(0.5)
res <- array(NA, dim = c(num_ensemble, length(a_par), length(nu_par)))
sigmoid <- function(x){return(exp(x)/(1+exp(x)))}
osh_pred_spesn <- matrix(NA, nrow = nrow(wide_y), ncol= length(11:15))

for (ensem_idx in 1:num_ensemble) {
  for (a_par_idx in 1:length(a_par) ) {
    a <- a_par[a_par_idx]
    for (nu_par_idx in 1:length(nu_par)) {
      for(curr_year in 11: 15){
      # Generate W weight matrices
      W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)*rbinom(nh^2, 1, 1-pi_0)
      lambda_scale <- max(abs(eigen(W)$values))
      U_ar <- matrix(runif(nh, -a, a), ncol = 1) * rbinom(nh, 1, 1-pi_0)
      U_sp <- matrix(runif(nh*ncol(conv_covar), -a, a), ncol = nh) *  rbinom(nh*ncol(conv_covar), 1, 1-pi_0)
      Ux_sp <- conv_covar %*% U_sp
      # Ux_sp <- matrix(0, nrow = nrow(wide_y), ncol = nh)
      curr_H <- sigmoid(Ux_sp)
      Y <- wide_y[,1]
      for (year in 2:(curr_year-1+1)) {
        new_H <- sigmoid(
          nu_par[nu_par_idx]/lambda_scale * curr_H[((year-2)*nrow(wide_y)+1):nrow(curr_H),] %*% W 
          + (  matrix( log(wide_y[,year-1]+1)) ) %*% t(U_ar)
          + Ux_sp
        )
        curr_H <- rbind(curr_H, new_H)
        Y <- c(Y, wide_y[,year])
      }
      colnames(curr_H) <- paste("node", 1:ncol(curr_H))
      obs_H <- curr_H[-c((nrow(curr_H)-nrow(wide_y)+1):nrow(curr_H)),]
      pred_H <- curr_H[c((nrow(curr_H)-nrow(wide_y)+1):nrow(curr_H)),]
      print(paste("Now predicting year",year))
      years_before <- curr_year - 1
      obs_y <- Y[1:(years_before*nrow(wide_y))]
      
      # I used glm to verify the previous code when lambda = 0, it's not wrong.
      glm_model <- glm(obs_y ~ ., family = poisson(link = "log"), data = as.data.frame(cbind(obs_y, obs_H)),
                       control = glm.control(epsilon = 1e-8, maxit = 25, trace = TRUE))
      glm_pred <- predict(glm_model, type = "response", newdata = as.data.frame(pred_H))
      osh_pred_spesn[,curr_year-10] <- glm_pred
      }
    }
  }
}

mean((osh_pred_spesn - as.matrix(wide_y[,11:15]))^2)
var(as.vector(unlist(wide_y[,11:15])))



pred_idx <- which(sim_long$time>10)
df_pred <- data.frame(
  long = sim_long[pred_idx, 1],
  lat = sim_long[pred_idx, 2],
  time = sim_long[pred_idx, 3],
  value = as.vector(osh_pred_spesn) 
)

p1 <-
  ggplot(df_pred, aes(x = long, y = lat, fill = value)) +
  geom_tile() +
  facet_wrap(~time) +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(title = "Spatio-Temporal Data Heatmap",
       x = "Longitude",
       y = "Latitude",
       fill = "Value")



# Reshape y to a data frame
df_obs <- data.frame(
  long = sim_long[pred_idx, 1],
  lat = sim_long[pred_idx, 2],
  time = sim_long[pred_idx, 3],
  value = sim_long[pred_idx, 4]
)

# Plot
p2 <-
ggplot(df_obs, aes(x = long, y = lat, fill = value)) +
  geom_tile() +
  facet_wrap(~time) +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(title = "Spatio-Temporal Data Heatmap",
       x = "Longitude",
       y = "Latitude",
       fill = "Value")


cowplot::plot_grid(p1,p2)





