# Load required libraries
library(magrittr)  # For %>%
library(tsibble)   # For tsibble object
library(keras)
library(abind)
# Set seed for reproducibility
set.seed(123)

# Define parameters
n_samples <- 5   # Number of samples
n_obs <- 100     # Number of observations per sample

# AR model parameters
phi1 <- 0.9

# Covariates parameters
beta1 <- 0.5
beta2 <- -0.2

# Generate covariates data (e.g., random normal)
covariate1 <- matrix(rnorm(n_obs * n_samples, mean = 0, sd = 1), nrow = n_samples, ncol = n_obs)
covariate2 <- matrix(rnorm(n_obs * n_samples, mean = 0, sd = 1), nrow = n_samples, ncol = n_obs)

Y <- matrix(NA, nrow = n_samples, ncol = n_obs)
Y[,1] <- rnorm(n_samples) + covariate1[,1]*beta1 + covariate2[,1]*beta2
for (i in 2:n_obs) {
  Y[,i] <- Y[,i-1]*phi1 + covariate1[,i]*beta1 + covariate2[,i]*beta2
}

X <- abind(covariate1, covariate2, along = 3)

x_tr <- X[,1:80,]
y_tr <- Y[,1:80]
x_te <- X[,-c(1:80),]
y_te <- Y[,-c(1:80)]
# Set up the RNN model
model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 100, input_shape = dim(x_tr)[2:3], activation = "tanh", return_sequences = TRUE) %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 1)

# Compile the model
model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam()
)



# Train the model
history <- model %>% fit(
  x = x_tr,
  y = y_tr,
  epochs = 100,
  batch_size = 32
)

pred_model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 100, batch_input_shape = c(n_samples,NA,dim(x_tr)[3]), activation = "tanh", stateful = TRUE) %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 1)

pred_model %>% set_weights(model %>% get_weights())
pred_model %>% reset_states()
predict(pred_model, x_tr) # Give initial states
# Predict for the first sequence:
seq_pred <- matrix(NA, ncol = ncol(y_te), nrow = n_samples)
for (i in 1:ncol(y_te)) {
  curr_step_cov <- array_reshape(X[,ncol(y_tr)+i,],dim = c(n_samples,1,2)) 
  seq_pred[,i] <- predict(pred_model, curr_step_cov)
}

plots <- vector("list")
i = 1
ggplot() +
geom_path(aes( x = 1:n_obs, y = Y[i,])) + 
geom_point(aes( x = 1:n_obs, y = Y[i,])) + 
geom_vline(xintercept = 81, color = "red", linetype = "dashed") + 
geom_path(aes( x = 81:100, y = seq_pred[i,]), color = "blue")



plots
  


