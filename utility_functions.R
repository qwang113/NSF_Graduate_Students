show_obs <- function(curr_id){
  # Gaussian Process with Matern Correlation
  nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
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

show_pred <- function(curr_id){
  pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv")
  pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv")
  nh = 200
  esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""))
  # Gaussian Process with Matern Correlation
  nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
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
    theme(plot.title = element_text(hjust = 0.5)) +
    geom_line(aes(x = 2012:2021, y = as.numeric(pois_pred[curr_id,])), color = "green", lwd = 1) +
    geom_line(aes(x = 2012:2021, y = as.numeric(pca_pred[curr_id,])), color = "yellow", lwd = 1) +
    geom_line(aes(x = 2012:2021, y = as.numeric(esn_pred[curr_id,])), color = "pink", lwd = 1) +
    geom_vline(xintercept = 2012, color = "black", lwd = 1, linetype= "dashed")
  return(obs_trace)
}

