rm(list = ls())
library(ggplot2)
library(glmnet)
#https://www.geeksforgeeks.org/time-series-forecasting-using-recurrent-neural-networks-rnn-in-tensorflow/
#The link above is a good resource to learn

# Simulate an AR(2) Data Set
ar_2 <- rep(0, 5000)
phi_1 <- 0.8
phi_2 <- -0.1

covariates_ar2 <- cbind(runif(5000, 10, 20), runif(5000, -5, 0))
beta_1 <- 10
beta_2 <- -9

ar_2[1] <- rnorm(1)
ar_2[2] <- rnorm(1) + phi_1*ar_2[1]
set.seed(1)
for (i in 3:length(ar_2)) {
  ar_2[i] <- phi_1*ar_2[i-1] + phi_2*ar_2[i-2] + rnorm(1, 0, 0.5)
}
ar_2 <- ar_2 + covariates_ar2%*%matrix(c(beta_1, beta_2), ncol = 1)

# Plot the series out
ggplot() +
  geom_path(aes(x = 1:5000, y = ar_2))

arima_res <- arima(ar_2, order = c(2,0,0), xreg = covariates_ar2)

sim_dat <- cbind(ar_2, covariates_ar2)

X <- sim_dat[,-1]
Y <- sim_dat[,1]

min_max_scale <- function(x){return((x-min(x))/diff(range(x)))}
X <- apply(X, 2, min_max_scale) 
# set hyperparameters
leak_rate <- 1 # It's always best to choose 1 here according to Mcdermott and Wille, 2017
nh <- 200 # Number of hidden units in RNN
nx <- ncol(X) # Number of covariates
a <- 0.1# The range of the standard uniform distribution of the weights
H <- matrix(NA, nrow = length(Y), ncol = nh)
W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
U <- matrix(runif(nh*nx, -a,a), nrow = nh, ncol = nx)
H[1,] <-  sapply(U %*% X[1,], tanh)
for (i in 2:length(Y)) {
  print(i)
  H_til <- sapply(W%*%matrix(H[i-1,],ncol = 1) + U%*%X[i,], tanh) 
  H[i,] <- (1-leak_rate)*H[i-1,] + leak_rate*H_til
}
H <- apply(H, 2, min_max_scale)
ridge_lambda <- cv.glmnet(H, ar_2, alpha = 0)$lambda.min
final_model <- glmnet(H, ar_2, alpha = 0, lambda = ridge_lambda)
pred_y_ridge <- predict(final_model,H)

pred_y_lm <- predict(lm(Y~H))


# Plot the series out
p1<-
ggplot() +
  geom_path(aes(x = 1:5000, y = ar_2), color = "black")+
  ylim(100, 250)
p2<-
ggplot() +
  geom_path(aes(x = 1:5000, y = pred_y_ridge), color = "red")+
  ylim(100, 250)
p3<-
  ggplot() +
  geom_path(aes(x = 1:5000, y = pred_y_lm), color = "pink")+
  ylim(100, 250)

cowplot::plot_grid(p1,p2,p3, nrow = 1)
# Residual Plot
ggplot() +
  geom_point(aes(x = 1:5000, y = residuals(lm(Y~H))), color = "black")

summary(lm(Y~H))

acf_values <- acf(ar_2-pred_y)
# Residual ACF
acf_df <- data.frame(lag = acf_values$lag, acf = acf_values$acf)
ggplot(acf_df, aes(x = lag, y = acf)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  geom_segment(aes(xend = lag, yend = 0), linetype = "dashed", color = "gray") +
  geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
  labs(title = "Autocorrelation Function (ACF) Plot",
       x = "Lag",
       y = "Autocorrelation") +
  theme_minimal()






