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

X <- abind(covariate1, covariate2,Y_lag <- cbind(0,Y[,-ncol(Y)]), along = 3)


# Set up the RNN model
model <- keras_model_sequential() %>%
  layer_simple_rnn(units = 10, input_shape = dim(X)[2:3], activation = "tanh") %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 100) %>%
  layer_dense(units = 100)

# Compile the model
model %>% compile(
  loss = 'mse',
  optimizer = optimizer_adam()
)



# Train the model
history <- model %>% fit(
  x = X,
  y = Y,
  epochs = 100,
  batch_size = 32
)

# Make predictions
predictions <- model %>% predict(X)

# Print predictions
print(predictions)

plot(Y[1,], type = 'l')
lines(predictions[1,], type = 'l', col = "red")

