rm(list = ls())
library(Matrix)
library(tidyverse)
library(glmnet)
library(tscount)
library(ggplot2)
library(BayesLogit)
library(forecast)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
set.seed(2)
idx <- floor(runif(4)*nrow(schoolsM))

pred_nb <- readRDS("insample_nb.Rda")
nb_mean = apply(pred_nb,1,mean)
nb_mean <- matrix(nb_mean, nrow = nrow(schoolsM))
nb_res <- nb_mean - schoolsM
curr_r <- apply(readRDS("rr.Rda"),1, mean)
pred_p <- 1/(nb_mean/curr_r + 1)
xt_var_nb <- nb_mean * 1/pred_p
st_nb <- nb_res/sqrt(xt_var_nb)
test_pvalue_nb = test_statistic_nb = rep(NA, nrow(st_nb))
for (i in 1:nrow(st_nb)) {
  test_pvalue_nb[i] <- Box.test(st_nb[i,], lag = 7, type = "Ljung-Box")$p.value
  test_statistic_nb[i] <- Box.test(st_nb[i,], lag = 7, type = "Ljung-Box")$statistic
}
bad_cases_nb <- which(test_pvalue_nb <= 0.05/nrow(schools))
mean(st_nb)
var(as.vector(st_nb))


pred_pois <- readRDS("insample_pois.Rda")
pois_mean = apply(pred_pois,1,mean)
pois_mean <- matrix(pois_mean, nrow = nrow(schoolsM))
xt_var_pois <- pois_mean
pois_res <- pois_mean - schoolsM
st_pois <- pois_res/sqrt(xt_var_pois)
test_pvalue_pois = test_statistic_pois = rep(NA, nrow(st_pois))
for (i in 1:nrow(st_pois)) {
  test_pvalue_pois[i] <- Box.test(st_pois[i,], lag = 7, type = "Ljung-Box")$p.value
  test_statistic_pois[i] <- Box.test(st_pois[i,], lag = 7, type = "Ljung-Box")$statistic
}
bad_cases_pois <- which(test_pvalue_pois <= 0.05/nrow(schools))
mean(st_pois)
var(as.vector(st_pois))
