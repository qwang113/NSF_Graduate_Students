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
use_condaenv("tf_gpu")
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
wide_y <- nsf_wide_car[,-c(1:5, ncol(nsf_wide_car))]
prc_y <- prcomp(t(wide_y))



for (i in 2015:2021) {
  print(paste("Now doing year",i))
  years_before <- i - 1972
  prev_y <- t(wide_y)[1:years_before,]
  prev_pc <- prcomp(prev_y)
  num_pc <- length(which(cumsum(prev_pc$sdev^2)/sum(prev_pc$sdev^2) <= 0.95 ))
  prev_pc_use <- prev_pc$x[,1:num_pc]
  
  # one_step_ahead_model <- glm.nb(prev_y~., data = data.frame(cbind(prev_y, prev_H)), control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
  pred_H <- curr_H_scaled[c( (years_before*nrow(nsf_wide)+1):((years_before+1)*nrow(nsf_wide)) ), ] 
  one_step_ahead_pred_y[,i-2014] <- predict(one_step_ahead_model, newdata = data.frame(pred_H))
}

