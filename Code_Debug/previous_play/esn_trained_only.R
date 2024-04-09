#Initialize
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
nh = 200
nu <- 1
a=0.1
year = 2012
leak = 1
W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit

ar_col <- matrix(runif(nh,-a,a), nrow = 1)

lambda_scale <- max(abs(eigen(W)$values))

curr_H <- apply(matrix(0, nrow = nrow(nsf_wide_car), ncol = nh), c(1,2), tanh)


Y <- nsf_wide_car[,10]
pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
print("Calculating Recurrent H Matrix. . .")
for (i in 2:(year-1972+1)) {
  
  setTxtProgressBar(pb,i)
  
  new_H <- apply( 
    nu/lambda_scale*
      curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W
    # + ux_sp
    # + ux_dummy
    + log(nsf_wide_car[,i+8]+1)%*%ar_col
    , c(1,2), tanh)*leak + curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]*(1-leak)
  
  Y <- c(Y, nsf_wide_car[,i+9])
  curr_H <- rbind(curr_H, new_H)
}
colnames(curr_H) <- paste("node", 1:ncol(curr_H))
obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
print(paste("Now predicting year",year))
years_before <- year - 1972
obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]

esn_dnn <- keras_model_sequential() %>%
  layer_dense(units = 200, activation = 'relu', input_shape = ncol(obs_H)) %>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = 200, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'exponential') 

esn_dnn <- 
  esn_dnn %>% compile(
    optimizer = "adam",
    loss = "poisson"
  )

model_checkpoint <- callback_model_checkpoint(
  filepath = "D:/77/research/temp/best_weights.h5",
  save_best_only = TRUE,
  monitor = "val_loss",
  mode = "min",
  verbose = 1
)

model_tr <- esn_dnn %>%
  fit(x = obs_H, y = obs_y, epochs = 100, batch_size = 500, 
      validation_data = list(pred_H, nsf_wide_car$X2012) , callbacks = list(model_checkpoint))

esn_dnn %>% load_model_weights_hdf5("D:/77/research/temp/best_weights.h5")

pred_spesn <- predict(esn_dnn, pred_H)

print(mean((nsf_wide_car$X2012 - pred_spesn)^2))
var(nsf_wide_car$X2012)
