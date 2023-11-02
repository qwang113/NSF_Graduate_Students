library(reshape2)
library(ggplot2)
res_all <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/CRESN_res5000.csv")
res_wide <- dcast(res_all, ID + long + lat ~ year, value.var = "Residuals")

# Get the residuals for schools individually, time series residuals

for (curr_ID in res_wide$ID) {
  curr_plot <- 
  ggplot() + 
    geom_line(aes(x = 1972:2021, y = unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)])), linewidth = 2) +
    geom_point(aes(x = 1972:2021, y = unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)])), color = "red", size = 2) +
    labs(title = paste("Residual Time Series of ",curr_ID), x = "Time", y = "Residuals") +
    theme(plot.title = element_text(hjust = 0.5))
  
  acf_values <- acf(unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)]))
  # Residual ACF
  acf_df <- data.frame(lag = acf_values$lag, acf = acf_values$acf)
  curr_acf <-
  ggplot(acf_df, aes(x = lag, y = acf)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
    geom_segment(aes(xend = lag, yend = 0), linetype = "dashed", color = "gray") +
    geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
    labs(title = "Autocorrelation Function (ACF) Plot",
         x = "Lag",
         y = "Autocorrelation") +
    theme_minimal()+
    theme(plot.title = element_text(hjust = 0.5))
  
   res_plot <- cowplot::plot_grid(curr_plot, curr_acf, ncol = 1)
  ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_res_solo/",curr_ID,"_res.png", sep = ""), res_plot)
  
}
