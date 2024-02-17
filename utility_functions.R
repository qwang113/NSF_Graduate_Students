nh = 200
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
rm_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)[1:100]
rm_ID <- nsf_wide$ID[rm_cases]

show_obs <- function(curr_id, out.rm = FALSE){
  # Gaussian Process with Matern Correlation
  nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
  # Remove Outliers
  if(out.rm){
    nsf_long <- nsf_long[-which(nsf_long$ID %in% rm_ID ),]
    nsf_wide <- nsf_wide[-annoying_cases[1:100],]

  }
  UNITID <- substr(nsf_wide$ID,1,6)
  nsf_wide <- cbind(UNITID,nsf_wide)
  carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
  colnames(carnegie_2021)[1] <- "UNITID"
  nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
  wide_ID_y <- nsf_wide_car[,-c(1,3:9, ncol(nsf_wide_car))]
  curr_y <- as.numeric(wide_ID_y[curr_id,-1])
  obs_trace <-
    ggplot() +
    geom_line(aes(x = 1972:2021, y = curr_y), col = "blue", linewidth = 2) +
    geom_point(aes(x = 1972:2021, y = curr_y), col = "red", size = 2) +
    labs(x = "Year", y = "Count", title = nsf_long$col_sch[min(which(nsf_wide$ID[curr_id]== nsf_long$ID))]) +
    theme(plot.title = element_text(hjust = 0.5))
  bacf <- acf(curr_y, plot = FALSE)
  bacfdf <- with(bacf, data.frame(lag, acf))
  obs_acf <-
    ggplot(data = bacfdf, mapping = aes(x = lag, y = acf)) +
    geom_hline(aes(yintercept = 0)) +
    geom_segment(mapping = aes(xend = lag, yend = 0), linewidth = 2, color = "blue")
 return(cowplot::plot_grid(obs_trace, obs_acf, ncol = 1))
}

show_pred <- function(curr_id = 1, scale_range = TRUE, out.rm = FALSE){
  if(out.rm){
    pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred_outrm.csv")
    pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred_outrm.csv")
    nh = 200
    esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_outrm_",nh, ".csv", sep = ""))
  }else{
    pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv")
    pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv")
    nh = 200
    esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""))
  }
  
  # Gaussian Process with Matern Correlation
  nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
  if(out.rm){
    osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
    annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
    nsf_long <- nsf_long[-which(nsf_long$ID %in% nsf_wide$ID[annoying_cases[1:100]] ),]
    nsf_wide <- nsf_wide[-annoying_cases[1:100],]
    
  }
  
  UNITID <- substr(nsf_wide$ID,1,6)
  nsf_wide <- cbind(UNITID,nsf_wide)
  carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
  colnames(carnegie_2021)[1] <- "UNITID"
  nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
  wide_ID_y <- nsf_wide_car[,-c(1,3:9, ncol(nsf_wide_car))]
  curr_y <- as.numeric(wide_ID_y[curr_id,-1])
  colorss <- c("Poisson" = "purple", "PCA" = "green", "ESN" = "orange")
  
  obs_trace <-
    ggplot() +
    geom_line(aes(x = 1972:2021, y = curr_y), col = "blue", linewidth = 2) +
    geom_point(aes(x = 1972:2021, y = curr_y), col = "red", size = 2) +
    labs(x = "Year", y = "Count", title = paste(nsf_long$col_sch[min(which(nsf_wide$ID[curr_id]== nsf_long$ID))],
                                                ifelse(out.rm, "(Outlier Removed)","Full Dataset")), color = "Legend") +
    theme(plot.title = element_text(hjust = 0.5)) +
    geom_line(aes(x = 2012:2021, y = as.numeric(pois_pred[curr_id,]), color = "Poisson"), lwd = 1) +
    geom_point(aes(x = 2012:2021, y = as.numeric(pois_pred[curr_id,]), color = "Poisson"), size = 2) +
    geom_line(aes(x = 2012:2021, y = as.numeric(pca_pred[curr_id,]), color = "PCA"), lwd = 1) +
    geom_point(aes(x = 2012:2021, y = as.numeric(pca_pred[curr_id,]), color = "PCA"), size = 2) +
    geom_line(aes(x = 2012:2021, y = as.numeric(esn_pred[curr_id,]), color = "ESN"), lwd = 1) +
    geom_point(aes(x = 2012:2021, y = as.numeric(esn_pred[curr_id,]), color = "ESN"), size = 2) +
    geom_vline(xintercept = 2012, color = "black", lwd = 1, linetype= "dashed") +
    scale_color_manual(values = colorss) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(title = "",override.aes = list(size = 5))) 
  
  if(scale_range){
    obs_trace <- obs_trace +  coord_cartesian(ylim =c(range(curr_y)[1], range(curr_y)[2] + sd(curr_y)) )
  }
  
  return(obs_trace)
}

for (out.rm in 0:1) {
  nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
  # Remove Outliers
  if(out.rm){
    osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
    annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
    nsf_long <- nsf_long[-which(nsf_long$ID %in% nsf_wide$ID[annoying_cases[1:100]] ),]
    nsf_wide <- nsf_wide[-annoying_cases[1:100],]
    for (i in 1:nrow(nsf_wide)) {
      cowplot::plot_grid(show_pred(i, out.rm = TRUE), show_pred(i,scale_range = FALSE, out.rm = TRUE), nrow = 2)
      ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_RMOUT/",nsf_long$col_sch[min(which(nsf_wide$ID[i]== nsf_long$ID))],".png",sep = ""))
    }
  }else{
    for (i in 1:nrow(nsf_wide)) {
      cowplot::plot_grid(show_pred(annoying_cases[i], out.rm = FALSE), show_pred(annoying_cases[i],scale_range = FALSE, out.rm = FALSE), nrow = 2)
      ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_FULL/",i,nsf_long$col_sch[min(which(nsf_wide$ID[annoying_cases[i]]== nsf_long$ID))],".png",sep = ""))
    }
  }
  
  
}
