rm(list = ls())
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")
schoolsM <- as.matrix(schools[,10:59])

int_pred <- readRDS("D:/77/Research/temp/pred_all_randsl_sch.Rda")
sep_pred <- readRDS("D:/77/Research/temp/pred_all_sep.Rda")
randsl_pred <- readRDS("D:/77/Research/temp/pred_all_randsl.Rda")

ingarch_pred <- readRDS("D:/77/Research/temp/pred_all_ING.Rda")
single_esn_pred <- readRDS("D:/77/Research/temp/pred_all_single_esn.Rda")
ensemble_esn_pred <- readRDS("D:/77/Research/temp/pred_all_ensemble_esn.Rda")

all_r <- readRDS("D:/77/Research/temp/all_rr.Rda")
arr_r <- simplify2array(all_r)
arr_r_new <- aperm(arr_r, perm = c(3, 1, 2))
# Calculate prediction means
ingarch_mean <- ingarch_pred[1,,]
single_esn_mean <- t(single_esn_pred)


int_mean <- apply(int_pred[,,-1], c(1,2), mean)
sep_mean <- apply(sep_pred, c(1,2), mean)
randsl_mean <- apply(randsl_pred, c(1,2),mean)
ensemble_esn_mean <- apply(ensemble_esn_pred, c(1,2), mean)

# Generate prediction samples
int_samples <- array(rnbinom(length(int_pred[,,-1]), size = arr_r_new[,,-1], p = arr_r_new[,,-1]/(int_pred[,,-1]+arr_r_new[,,-1])), dim = dim(int_pred))
sep_samples <- array(rpois(length(sep_pred), lambda = sep_pred), dim = dim(sep_pred))
randsl_samples <- array(rpois(length(randsl_pred), lambda = randsl_pred), dim = dim(randsl_pred))
ensemble_esn_samples <- array(rpois(length(ensemble_esn_pred), lambda = ensemble_esn_pred), dim = dim(ensemble_esn_pred))




# Calculate quantiles
alpha <- 0.05
int_up <- apply(int_samples, c(1,2), quantile, 1-alpha/2)
int_low <- apply(int_samples, c(1,2), quantile, alpha/2)

sep_up <- apply(sep_samples, c(1,2), quantile, 1-alpha/2)
sep_low <- apply(sep_samples, c(1,2), quantile, alpha/2)

randsl_up <- apply(randsl_samples, c(1,2), quantile, 1-alpha/2)
randsl_low <- apply(randsl_samples, c(1,2), quantile, alpha/2)

ingarch_up <- ingarch_pred[3,,]
ingarch_low <- ingarch_pred[2,,]
# ingarch_up <- qpois(1-alpha/2,lambda = ingarch_pred[1,,])
# ingarch_low <- qpois(alpha/2,lambda = ingarch_pred[1,,])


ensemble_up <- apply(ensemble_esn_samples, c(1,2), quantile, 1-alpha/2)
ensemble_low <- apply(ensemble_esn_samples, c(1,2), quantile, alpha/2)

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
  all_mse[1,i] <- mean((curr_true - apply(schoolsM[,(1:(44+i))],1,mean))^2)
  all_mse[2,i] <- mean((ingarch_mean[i,] - curr_true)^2)  
  all_mse[3,i] <- mean((single_esn_mean[i,] - curr_true)^2)
  all_mse[4,i] <- mean((ensemble_esn_mean[i,] - curr_true)^2)
  all_mse[5,i] <- mean((sep_mean[i,] - curr_true)^2)
  all_mse[6,i] <- mean((int_mean[i,] - curr_true)^2)
  all_mse[7,i] <- mean((randsl_mean[i,] - curr_true)^2)
  
  
  lg_curr_true <- log(curr_true+1)
  all_lmse[1,i] <- mean((lg_curr_true - apply(log(schoolsM[,(1:(44+i))]+1),1,mean))^2)
  all_lmse[2,i] <- mean((log(ingarch_mean[i,]+1) - lg_curr_true)^2)  
  all_lmse[3,i] <- mean((log(single_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[4,i] <- mean((log(ensemble_esn_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[5,i] <- mean((log(sep_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[6,i] <- mean((log(int_mean[i,]+1) - lg_curr_true)^2)
  all_lmse[7,i] <- mean((log(randsl_mean[i,]+1) - lg_curr_true)^2)
  
  
  IS[1,i] <- mean(int_score(l = ingarch_low[i,], u = ingarch_up[i,], true_x = curr_true, alpha = alpha))
  IS[2,i] <- mean(all_interval_score(prediction_sample = ensemble_esn_pred[i,,], alpha = alpha, true_x = curr_true))
  IS[3,i] <- mean(all_interval_score(prediction_sample = sep_pred[i,,], alpha = alpha, true_x = curr_true))
  IS[4,i] <- mean(all_interval_score(prediction_sample = int_pred[i,,-1], alpha = alpha, true_x = curr_true))
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

rownames(all_mse) <- c("Intercept","INGARCH(1,1)","Single ESN","Ensemble ESN","Bayesian Poisson ESN ","Bayesian Hierarchical NB ESN ","Bayesian Hierarchical Poisson ESN")
rownames(all_lmse) <- c("Intercept","INGARCH(1,1)","Single ESN","Ensemble ESN","Bayesian Poisson ESN ","Bayesian Hierarchical NB ESN ","Bayesian Hierarchical Poisson ESN")
rownames(IS) <- c("INGARCH(1,1)","Ensemble ESN","Bayesian Poisson ESN ","Bayesian Hierarchical NB ESN ","Bayesian Hierarchical Poisson ESN")
rownames(ICR) <- c("INGARCH(1,1)","Ensemble ESN","Bayesian Poisson ESN ","Bayesian Hierarchical NB ESN ","Bayesian Hierarchical Poisson ESN")
colnames(all_mse) <- colnames(all_lmse) <- colnames(IS) <- colnames(ICR) <- c(2017:2021,"5 Year Average")

knitr::kable(all_mse, format = "latex", align = 'c',digits = 0)
knitr::kable(all_lmse, format = "latex", align = 'c',digits = 3)
knitr::kable(IS, format = "latex", align = 'c', digits = 0)
knitr::kable(ICR, format = "latex", align = 'c', digits = 3)



# check IS
num_sch <- ncol(schoolsM)
ggplot() +
  geom_point(aes(x = 1:length(as.vector(schoolsM[1:num_sch,46])), y = as.vector(schoolsM[1:num_sch,46])), size = 1) + 
  geom_ribbon(aes(ymin = ingarch_low[1,1:num_sch], ymax = ingarch_up[1,1:num_sch], x = 1:length(as.vector(schoolsM[1:num_sch,46]))),
              fill = "red", alpha = 0.4) +
  geom_ribbon(aes(ymin = int_low[1,1:num_sch], ymax = int_up[1,1:num_sch], x = 1:length(as.vector(schoolsM[1:num_sch,46]))), fill = "blue", alpha = 0.4)


res <- schoolsM[,46:50] - t(randsl_mean)
boxplot(res)

mean(res/sqrt(t(randsl_mean)))
