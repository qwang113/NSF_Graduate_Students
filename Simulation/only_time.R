set.seed(1) # For reproducibility

gridsize = 30
long <- seq(from = 0, to = 1, length.out = gridsize)
lat <- seq(from = 0, to = 1, length.out = gridsize)



N <- gridsize^2         # Number of time series
timesize <- 50       # Length of each time series
phi <- 0.9    # AR(1) correlation parameter
sigma <- .5   # Standard deviation of the innovations
mu_vec <- rep(1, N) # Different means for each time series

# Function to simulate AR(1) process with individual means
simulate_ar1 <- function(timesize, phi, sigma, mu) {
  ar1 <- numeric(timesize)
  ar1[1] <- rnorm(1, mean = mu, sd = sigma / sqrt(1 - phi^2))
  for (t in 2:timesize) {
    ar1[t] <- phi * ar1[t - 1] + rnorm(1, mean = 0, sd = sigma)
  }
  return(ar1)
}

# Simulate AR(1) processes for each time series with individual means
log_lambda_matrix <- matrix(NA, nrow = timesize, ncol = N)
for (i in 1:N) {
  log_lambda_matrix[, i] <- simulate_ar1(timesize, phi, sigma, mu_vec[i])
}

# Transform log_lambda to lambda
lambda_matrix <- exp(log_lambda_matrix)

# Generate Poisson-distributed counts
poisson_counts <- matrix(NA, nrow = timesize, ncol = N)
for (i in 1:N) {
  poisson_counts[, i] <- rpois(timesize, lambda = lambda_matrix[, i])
}

# Print the first few rows of the Poisson counts
head(poisson_counts)
plot(poisson_counts[,1])
wide_dat <- cbind(long, lat, t(poisson_counts))
wide_y <- wide_dat[,-c(1:2)]
osh_pred_pois <- matrix(NA, nrow = nrow(wide_y), ncol = length(11:15))


for (year in 11:15) {
  for (school in 1:nrow(wide_y)) {
    print(paste("now doing year", year, "school", school))
    prev_obs <- unlist(wide_y[school,2:(year-1)])
    lag_obs <- unlist(wide_y[school, 1:(year-2)])
    model <- glm(prev_obs ~ log(lag_obs+1), family = poisson(link = "log"))
    osh_pred_pois[school, year-10] <- predict(model, newdata = data.frame("lag_obs" = unlist( log(wide_y[school, year-1] +1) )),
                                              type = "response")
  }
}
mean((osh_pred_pois - as.matrix(wide_y[,11:15]))^2)


