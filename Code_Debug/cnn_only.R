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

# If we need dummy matrix for : 
#car: Carnegie
#State: State
#gss: indicator of school major, such as engineering, social science,..
dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
dummy_state <- model.matrix(~nsf_wide_car$state - 1)
dummy_gss <- model.matrix(~ substr(nsf_wide_car$ID, 7, 9)  - 1)
dummy_matrix <- cbind(dummy_car, dummy_gss, dummy_state)

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


#################################### Build SP-ESN model ##############################
leak = 1
nh <- 200 
# for (year in 2012:2021) {
year = 2012
# Need to repeat the basis array!
num_rep <- year-1972

basis_tr_1 <- array(NA, dim = c(num_rep*nrow(nsf_wide_car), shape_row_1, shape_col_1,1) )
basis_tr_2 <- array(NA, dim = c(num_rep*nrow(nsf_wide_car), shape_row_2, shape_col_2,1) )
basis_tr_3 <- array(NA, dim = c(num_rep*nrow(nsf_wide_car), shape_row_3, shape_col_3,1) )
for (repidx in 1:num_rep) {
  basis_tr_1[ ((repidx-1)*nrow(nsf_wide_car)+1) : (repidx*nrow(nsf_wide_car)) ,,,] <- basis_arr_1
  basis_tr_2[ ((repidx-1)*nrow(nsf_wide_car)+1) : (repidx*nrow(nsf_wide_car)),,,] <- basis_arr_2
  basis_tr_3[ ((repidx-1)*nrow(nsf_wide_car)+1) : (repidx*nrow(nsf_wide_car)),,,] <- basis_arr_3
}

years_before <- year - 1972
Y <- nsf_wide_car[,10]
pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
print("Calculating Recurrent H Matrix. . .")
for (i in 2:(year-1972+1)) {
  Y <- c(Y, nsf_wide_car[,i+9])

}

obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]


input_basis_1 <- layer_input(shape = c(shape_row_1, shape_col_1, 1))
input_basis_2 <- layer_input(shape = c(shape_row_2, shape_col_2, 1))
input_basis_3 <- layer_input(shape = c(shape_row_3, shape_col_3, 1))
input_esn <- layer_input(shape = nh)


resolution_1_conv <- input_basis_1 %>%
  layer_conv_2d(filters = 64, kernel_size = c(2,2), activation = 'tanh') %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh')

resolution_2_conv <- input_basis_2 %>%
  layer_conv_2d(filters = 64, kernel_size = c(2,2), activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh')

resolution_3_conv <- input_basis_3 %>%
  layer_conv_2d(filters = 64, kernel_size = c(2,2), activation = 'tanh') %>%
  layer_batch_normalization() %>%
  layer_flatten() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh') %>% 
  layer_batch_normalization() %>%
  layer_dense(units = 100, activation = 'tanh')


cnn_model <- layer_concatenate(list(resolution_1_conv, resolution_2_conv, resolution_3_conv))

output_layer <- cnn_model %>%  layer_dense(units = 1, activation = 'exponential')


model_sp_esn <- keras_model(inputs = list(input_basis_1, input_basis_2, input_basis_3), outputs = output_layer)

model_sp_esn <- 
  model_sp_esn %>% compile(
    optimizer = "adam",
    loss = "poisson"
  )


model_checkpoint <- callback_model_checkpoint(
  filepath = "D:/77/research/temp/nsf_best_weights_ck.h5",
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 0
)


mod_train_sp_esn<- model_sp_esn %>% fit(
  x = list(basis_tr_1, basis_tr_2, basis_tr_3),
  y = obs_y,
  epochs=50,
  batch_size=1000,
  validation_data=list(list(basis_arr_1, basis_arr_2, basis_arr_3), as.numeric(nsf_wide_car$X2012) ),
  callbacks = model_checkpoint, shuffle = TRUE
)


model_sp_esn %>% load_model_weights_hdf5("D:/77/research/temp/nsf_best_weights_ck.h5")


pred_spesn <- predict(model_sp_esn, list(basis_arr_1, basis_arr_2, basis_arr_3))

print(mean((nsf_wide_car$X2012 - pred_spesn)^2))
var(nsf_wide_car$X2012)
# }

