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
library(reshape2)
library(corrplot)
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
res_all <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/300filters_res5000_complex+univ_added+norm01.csv")
res_wide <- dcast(res_all, ID + long + lat ~ year, value.var = "Residuals")
school_infor <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ins_loc.csv", header = TRUE)

unique_university <- unique(substr(nsf_wide$ID,1,6))
for (i in 1:length(unique_university)) {
  university_name <- school_infor$INSTNM[which(school_infor$UNITID==unique_university[i])]
  colleges <- nsf_wide[which(substr(nsf_wide$ID,1,6)==unique_university[i]),]
  colleges_y <- t(colleges[,-c(1:3)])
  cor_matrix <- cor(colleges_y)
  curr_pic <-
  ggplot()+
  geom_tile(aes(x = expand.grid(1:nrow(colleges),1:nrow(colleges))[,1] ,y = rev(expand.grid(1:nrow(colleges),1:nrow(colleges))[,2]), fill = as.vector(cor_matrix))) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limits = c(-1,1),name=expression(rho)) +
  labs(x = "", y = "", title = paste("Colleges Correlation Plot of",university_name)) +
  theme_void() +
  theme(plot.title = element_text(hjust = 0.5))+
  theme(plot.title = element_text(color = "blue", face = "bold", size = 25)) +
  geom_text(aes(x = expand.grid(1:nrow(colleges),1:nrow(colleges))[,1], y = rev(expand.grid(1:nrow(colleges),1:nrow(colleges))[,2]) ,
                label = as.vector(round(cor_matrix,2))), vjust = 1) +
    theme(
      legend.text = element_text(size = 12, face = "bold", color = "blue"),
      legend.title = element_text(size = 14, face = "bold", color = "blue")
    )
  
  
  colleges_res <- res_wide[which(substr(res_wide$ID,1,6)==unique_university[i]),]
  colleges_res_y <- t(colleges_res[,-c(1:3)])
  corr_res <- cor(colleges_res_y)
  curr_res <-
    ggplot()+
    geom_tile(aes(x = expand.grid(1:nrow(colleges),1:nrow(colleges))[,1] ,y = rev(expand.grid(1:nrow(colleges),1:nrow(colleges))[,2]), fill = as.vector(corr_res))) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0, limits = c(-1,1),name=expression(rho)) +
    labs(x = "", y = "", title = paste("Residuals Correlation Plot of",university_name)) +
    theme_void() +
    theme(plot.title = element_text(hjust = 0.5))+
    theme(plot.title = element_text(color = "blue", face = "bold", size = 25)) +
    geom_text(aes(x = expand.grid(1:nrow(colleges),1:nrow(colleges))[,1], y = rev(expand.grid(1:nrow(colleges),1:nrow(colleges))[,2]) ,
                  label = as.vector(round(corr_res,2))), vjust = 1) +
    theme(
      legend.text = element_text(size = 12, face = "bold", color = "blue"),
      legend.title = element_text(size = 14, face = "bold", color = "blue")
    )
  
  combined_pic <- cowplot::plot_grid(curr_pic, curr_res, nrow = 1)
  
  
  
  
  ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_corr_plot/", gsub("[^A-Za-z0-9_]", "_",university_name), ".png", sep = ""),combined_pic, width = 25, height = 10)
}

