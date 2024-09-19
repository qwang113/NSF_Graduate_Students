rm(list = ls())
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")
schoolsM <- as.matrix(schools[,10:59])

int_pred <- readRDS("D:/77/Research/temp/pred_all_int.Rda")
sep_pred <- readRDS("D:/77/Research/temp/pred_all_sep.Rda")
randsl_pred <- readRDS("D:/77/Research/temp/pred_all_randsl.Rda")

ingarch_pred <- readRDS("D:/77/Research/temp/pred_all_ING.Rda")
single_esn_pred <- readRDS("D:/77/Research/temp/pred_all_single_esn.Rda")
ensemble_esn_pred <- readRDS("D:/77/Research/temp/pred_all_ensemble_esn.Rda")


int_mean <- apply(int_pred, c(1,2), mean)
sep_mean <- apply(sep_pred, c(1,2), mean)
randsl_mean <- apply(randsl_pred, c(1,2),mean)
ingarch_mean <- ingarch_pred[1,,]
single_esn_mean <- t(single_esn_pred)
ensemble_esn_mean <- apply(ensemble_esn_pred, c(1,2), mean)


alpha <- 0.05
int_up <- apply(int_pred, c(1,2), quantile, 1-alpha/2)
int_low <- apply(int_pred, c(1,2), quantile, alpha/2)

sep_up <- apply(sep_pred, c(1,2), quantile, 1-alpha/2)
sep_low <- apply(sep_pred, c(1,2), quantile, alpha/2)

randsl_up <- apply(randsl_pred, c(1,2), quantile, 1-alpha/2)
randsl_low <- apply(randsl_pred, c(1,2), quantile, alpha/2)

ingarch_up <- ingarch_pred[3,,]
ingarch_low <- ingarch_pred[2,,]

ensemble_up <- apply(ensemble_esn_pred, c(1,2), quantile, 1-alpha/2)
ensemble_low <- apply(ensemble_esn_pred, c(1,2), quantile, alpha/2)

all_mse <- matrix(NA, nrow = 7, ncol = 6)
all_lmse <- matrix(NA, nrow = 7, ncol = 6)
IS <- matrix(NA, nrow = 5, ncol = 6)
ICR <- matrix(NA, nrow = 5, ncol = 6)

int_score <- function(l,u,true_x,alpha = 0.05){
  out_1 <- u-l
  out_2 <- 2/alpha * (l-true_x) * ifelse(true_x < l, 1, 0)
  out_3 <- 2/alpha * (true_x-u) * ifelse(true_x > u, 1, 0)
  return(out_1 + out_2 + out_3)
}

all_interval_score <- function(prediction_sample, alpha = 0.05, true_x){
  num_all <- length(true_x)
  pb <- txtProgressBar(min = 0,      # Minimum value of the progress bar
                       max = num_all, # Maximum value of the progress bar
                       style = 3,    # Progress bar style (also available style = 1 and style = 2)
                       width = 50,   # Progress bar width. Defaults to getOption("width")
                       char = "=")   # Character used to cRete the bar
  
  
  all_score <- rep(NA, length(true_x))
  for (int_score_idx in 1:length(true_x)) {
    setTxtProgressBar(pb, int_score_idx)
    l <- quantile(prediction_sample[int_score_idx,], alpha/2)
    u <- quantile(prediction_sample[int_score_idx,], 1-alpha/2)
    all_score[int_score_idx] <- int_score(l = l, u = u, true_x = true_x[int_score_idx], alpha = alpha)
  }
  return(all_score)
}



for (i in 1:5) {
  curr_true <- schoolsM[,45+i]
  all_mse[1,i] <- var(curr_true)
  all_mse[2,i] <- mean((ingarch_mean[i,] - curr_true)^2)  
  all_mse[3,i] <- mean((single_esn_mean[i,] - curr_true)^2)
  all_mse[4,i] <- mean((ensemble_esn_mean[i,] - curr_true)^2)
  all_mse[5,i] <- mean((sep_mean[i,] - curr_true)^2)
  all_mse[6,i] <- mean((int_mean[i,] - curr_true)^2)
  all_mse[7,i] <- mean((randsl_mean[i,] - curr_true)^2)
  
  
  lg_curr_true <- log(curr_true+1)
  all_lmse[1,i] <- var(lg_curr_true)
  all_lmse[2,i] <- mean((log(ingarch_mean[i,]+1) - lg_curr_true)^2)  
  all_lmse[3,i] <- mean((log(single_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[4,i] <- mean((log(ensemble_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[5,i] <- mean((log(sep_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[6,i] <- mean((log(int_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[7,i] <- mean((log(randsl_mean[i,]+1) - lg_curr_true)^2)
  
  
  IS[1,i] <- mean(int_score(l = ingarch_low[i,], u = ingarch_up[i,], true_x = curr_true, alpha = alpha))
  IS[2,i] <- mean(all_interval_score(prediction_sample = ensemble_esn_pred[i,,], alpha = alpha, true_x = curr_true))
  IS[3,i] <- mean(all_interval_score(prediction_sample = sep_pred[i,,], alpha = alpha, true_x = curr_true))
  IS[4,i] <- mean(all_interval_score(prediction_sample = int_pred[i,,], alpha = alpha, true_x = curr_true))
  IS[5,i] <- mean(all_interval_score(prediction_sample = randsl_pred[i,,], alpha = alpha, true_x = curr_true))
  
  
  ICR[1,i] <- sum( curr_true>ingarch_low[i,] & curr_true<ingarch_up[i,] )/length(curr_true)
  ICR[2,i] <- sum( curr_true>ensemble_low[i,] & curr_true<ensemble_up[i,] )/length(curr_true)
  ICR[3,i] <- sum( curr_true>sep_low[i,] & curr_true<sep_up[i,] )/length(curr_true)
  ICR[4,i] <- sum( curr_true>int_low[i,] & curr_true<int_up[i,] )/length(curr_true)
  ICR[5,i] <- sum( curr_true>randsl_low[i,] & curr_true<randsl_up[i,] )/length(curr_true)
  
}

all_mse[,ncol(all_mse)] <- apply(all_mse[,-ncol(all_mse)],1,mean)
all_lmse[,ncol(all_mse)] <- apply(all_lmse[,-ncol(all_lmse)],1,mean)
IS[,ncol(IS)] <- apply(IS[,-ncol(IS)],1,mean)
ICR[,ncol(ICR)] <- apply(ICR[,-ncol(ICR)],1,mean)

rownames(all_mse) <- c("Intercept","INGARCH(1,1)","Single ESN","Ensemble ESN","Separate Bayesian ESN","Integrated Bayesian ESN","Random Slope Bayesian ESN")
rownames(all_lmse) <- c("Intercept","INGARCH(1,1)","Single ESN","Ensemble ESN","Separate Bayesian ESN","Integrated Bayesian ESN","Random Slope Bayesian ESN")
rownames(IS) <- c("INGARCH(1,1)","Ensemble ESN","Separate Bayesian ESN","Integrated Bayesian ESN","Random Slope Bayesian ESN")
rownames(ICR) <- c("INGARCH(1,1)","Ensemble ESN","Separate Bayesian ESN","Integrated Bayesian ESN","Random Slope Bayesian ESN")
colnames(all_mse) <- colnames(all_lmse) <- colnames(IS) <- colnames(ICR) <- c(2017:2021,"5 Year Average")

knitr::kable(all_mse, format = "latex", align = 'c',digits = 0)
knitr::kable(all_lmse, format = "latex", align = 'c',digits = 3)
knitr::kable(IS, format = "latex", align = 'c', digits = 0)
knitr::kable(ICR, format = "latex", align = 'c', digits = 3)

# - Random Slope Checking
y = 1
true <- schoolsM[,45+y]
pred <- randsl_mean[y,]
boxplot((true-pred)^2)
special_order <- order((true-pred)^2, decreasing = TRUE)

for (i in special_order[1:50]) {
  p1 <-
    ggplot() +
    geom_line(aes( x = 1972:2021, y = schoolsM[i,]), color = "red") +
    geom_point(aes( x = 1972:2021, y = schoolsM[i,]), color = "red") +
    # geom_line(aes( x = 2017:2021, y = randsl_mean[,special_order[1]]), color = "blue") +
    labs(title = paste(schools$state[i],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[i])])) +
    geom_vline(xintercept = 2017)
  p2 <-  ggplot() +
    geom_line(aes( x = 1972:2021, y = schoolsM[i,]), color = "red") +
    geom_point(aes( x = 1972:2021, y = schoolsM[i,]), color = "red") +
    geom_line(aes( x = 2017:2021, y = randsl_mean[,i]), color = "blue") +
    geom_point(aes( x = 2017:2021, y = randsl_mean[,i]), color = "blue") 
  p_all <- cowplot::plot_grid(p1,p2,ncol = 1)
    ggsave(paste("D:/77/Research/temp/special/",y,"_",i,".png",sep = ""), plot = p_all, width = 8, height = 6, dpi = 300,)
}

delete_idx <- unique(c(266,250,262,483,1021,745,393,1543,1381,1305,1088,1727,363,1010,661,404,1233,
                580,1438,627,639,1125,1655,267,318,404,39,69,129,359,702,649,1788,875,47,1438,244,487,297,460,
                1568,228,1066))

# 2017         2018         2019         2020 2021 5 Year Average
# Intercept                    9505.57 9.177750e+03 9.429050e+03 6.264150e+03   NA             NA
# INGARCH(1,1)                 2101.20 8.267000e+02 8.374700e+02 1.110530e+03   NA             NA
# Single ESN                   2619.77 3.235700e+02 5.867600e+02 2.421310e+03   NA             NA
# Ensemble ESN                 2426.54 3.412800e+02 5.944800e+02 1.769330e+03   NA             NA
# Separate Bayesian ESN     1395692.71 1.826928e+26 3.611812e+84 1.452277e+35   NA             NA
# Integrated Bayesian ESN      3552.89 3.556800e+02 7.091100e+02 3.927300e+03   NA             NA
# Random Slope Bayesian ESN     149.20 5.294086e+05 4.653600e+02 3.924200e+02   NA             NA
