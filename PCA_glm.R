rm(list = ls())
library(ggplot2)
library(glmnet)
library(MASS)
use_condaenv("tf_gpu")
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
wide_y <- nsf_wide_car[,-c(1:5, ncol(nsf_wide_car))]

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
# dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
# dummy_matrix <- cbind(dummy_school, dummy_car)
dummy_matrix <- dummy_car



write.csv( as.data.frame(shared_beta_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/share_beta_pred.csv", row.names = FALSE)

