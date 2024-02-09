nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
shared_beta_pred <- read.csv(  "D:/77/UCSC/study/Research/temp/NSF_dat/share_beta_pred.csv")
individual_beta_pred <- read.csv(  "D:/77/UCSC/study/Research/temp/NSF_dat/indi_beta_pred.csv")
cresn_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/cresn_pred_20240113.csv")
pred_y <- nsf_wide[,(ncol(nsf_wide)-6):ncol(nsf_wide)]


variance_y <- var(unlist(as.vector(pred_y)))
mse_shared <- mean(as.matrix(shared_beta_pred - pred_y)^2)
mse_individual <- mean(as.matrix(individual_beta_pred - pred_y)^2)
mse_cresn <- mean(as.matrix(cresn_pred - pred_y)^2)
