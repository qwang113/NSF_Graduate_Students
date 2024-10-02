rm(list = ls())
library(Matrix)
library(tidyverse)
library(glmnet)
library(tscount)
library(ggplot2)
library(reshape2)
set.seed(0)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schools_pred <- readRDS(here::here("D:/77/Research/temp/pois_insample_randsl.Rda"))
schoolsM <- as.matrix(schools[,10:59])
schools_mean <- matrix(apply(schools_pred, 1, mean), nrow = nrow(schoolsM))
sch_res <- schoolsM - schools_mean

acf_list <- map(1:nrow(sch_res), ~acf(sch_res[.x, ], plot = FALSE))

# Extract the ACF values and first lag values
acf_matrix <- sapply(acf_list, function(acf_obj) acf_obj$acf)
sig_idx <- abs(acf_matrix[-1,]) > 1.96/sqrt(ncol(schoolsM))

acf_matrix_sorted <- acf_matrix[, which(colSums(sig_idx) >0 )]

# Convert sorted ACF matrix to long format for heatmap
acf_long <- melt(acf_matrix_sorted)

# Plot as a heatmap
ggplot(acf_long, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "ACF Heatmap of Residuals (Sorted by Lag 1)", x = "Lag", y = "Series") +
  theme_minimal()



