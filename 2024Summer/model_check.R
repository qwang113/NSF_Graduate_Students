schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
int_pred <- readRDS("D:/77/Research/temp/pred_all_int.Rda")
sep_pred <- readRDS("D:/77/Research/temp/pred_all_sep.Rda")
ingarch_pred <- readRDS("D:/77/Research/temp/pred_all_ING.Rda")
single_esn_pred <- readRDS("D:/77/Research/temp/pred_all_single_esn.Rda")
ensemble_esn_pred <- readRDS("D:/77/Research/temp/pred_all_ensemble_esn.Rda")

int_mean <- apply(int_pred, c(1,2), mean)
sep_mean <- apply(sep_pred, c(1,2), mean)
ingarch_mean <- ingarch_pred[1,,]
single_esn_mean <- t(single_esn_pred)
ensemble_esn_mean <- apply(ensemble_esn_pred, c(1,2), mean)

alpha <- 0.05
int_up <- apply(int_pred, c(1,2), quantile, 1-alpha/2)
int_low <- apply(int_pred, c(1,2), quantile, alpha/2)

sep_up <- apply(sep_pred, c(1,2), quantile, 1-alpha/2)
sep_low <- apply(sep_pred, c(1,2), quantile, alpha/2)

ingarch_up <- ingarch_pred[2,,]
ingarch_low <- ingarch_pred[3,,]

ensemble_up <- apply(ensemble_esn_pred, c(1,2), quantile, 1-alpha/2)
ensemble_low <- apply(ensemble_esn_pred, c(1,2), quantile, alpha/2)
all_mse <- matrix(NA, nrow = 6, ncol = 5)
all_lmse <- matrix(NA, nrow = 6, ncol = 5)
for (i in 1:5) {
  curr_true <- schoolsM[,45+i]
  all_mse[1,i] <- var(curr_true)
  all_mse[2,i] <- mean((ingarch_mean[i,] - curr_true)^2)  
  all_mse[3,i] <- mean((single_esn_mean[i,] - curr_true)^2)
  all_mse[4,i] <- mean((ensemble_esn_mean[i,] - curr_true)^2)
  all_mse[5,i] <- mean((sep_mean[i,] - curr_true)^2)
  all_mse[6,i] <- mean((int_mean[i,] - curr_true)^2)
  lg_curr_true <- log(curr_true+1)
  all_lmse[2,i] <- mean((log(ingarch_mean[i,]+1) - lg_curr_true)^2)  
  all_lmse[3,i] <- mean((log(single_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[4,i] <- mean((log(ensemble_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[5,i] <- mean((log(sep_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[6,i] <- mean((log(int_mean[i,]+1) - lg_curr_true)^2)
}
