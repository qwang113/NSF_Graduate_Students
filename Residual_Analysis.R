library(reshape2)
library(ggplot2)
library(fields)
library(sp)
library(geoR)
res_all <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/CRESN_res2000_filter100_50step.csv")
res_wide <- dcast(res_all, ID + long + lat ~ year, value.var = "Residuals")
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)

# Get the residuals for schools individually, time series residuals
# 
# for (curr_ID in res_wide$ID) {
#   curr_plot <- 
#   ggplot() + 
#     geom_line(aes(x = 1972:2021, y = unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)])), linewidth = 2) +
#     geom_point(aes(x = 1972:2021, y = unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)])), color = "red", size = 2) +
#     labs(title = paste("Residual Time Series of ",curr_ID), x = "Time", y = "Residuals") +
#     theme(plot.title = element_text(hjust = 0.5))
#   
#   acf_values <- acf(unlist(res_wide[which(res_wide$ID == curr_ID),-c(1:3)]))
#   # Residual ACF
#   acf_df <- data.frame(lag = acf_values$lag, acf = acf_values$acf)
#   curr_acf <-
#   ggplot(acf_df, aes(x = lag, y = acf)) +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
#     geom_segment(aes(xend = lag, yend = 0), linetype = "dashed", color = "gray") +
#     geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
#     labs(title = "Autocorrelation Function (ACF) Plot",
#          x = "Lag",
#          y = "Autocorrelation") +
#     theme_minimal()+
#     theme(plot.title = element_text(hjust = 0.5))
#   
#    res_plot <- cowplot::plot_grid(curr_plot, curr_acf, ncol = 1)
#   ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_res_2000_filter100/",curr_ID,"_res.png", sep = ""), res_plot)
#   
# }
# 
# 
# nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
# nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
# 
# 
# for (curr_ID in nsf_wide$ID) {
#   school_col <- nsf_long$col_sch[which(nsf_long$ID==curr_ID)][1]
#   school_col = gsub("[^A-Za-z0-9_]", "_", school_col)
#   curr_plot <- 
#     ggplot() + 
#     geom_line(aes(x = 1972:2021, y = unlist(nsf_wide[which(nsf_wide$ID == curr_ID),-c(1:3)])), linewidth = 2) +
#     geom_point(aes(x = 1972:2021, y = unlist(nsf_wide[which(nsf_wide$ID == curr_ID),-c(1:3)])), color = "red", size = 2) +
#     labs(title = paste("Time Series of ",school_col), x = "Time", y = "Count") +
#     theme(plot.title = element_text(hjust = 0.5))
#   
#   acf_values <- acf(unlist(nsf_wide[which(nsf_wide$ID == curr_ID),-c(1:3)]))
#   # Residual ACF
#   acf_df <- data.frame(lag = acf_values$lag, acf = acf_values$acf)
#   curr_acf <-
#     ggplot(acf_df, aes(x = lag, y = acf)) +
#     geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
#     geom_segment(aes(xend = lag, yend = 0), linetype = "dashed", color = "gray") +
#     geom_bar(stat = "identity", fill = "blue", alpha = 0.7) +
#     labs(title = paste("ACF of ",school_col),
#          x = "Lag",
#          y = "Autocorrelation") +
#     theme_minimal()+
#     theme(plot.title = element_text(hjust = 0.5))
#   
#   res_plot <- cowplot::plot_grid(curr_plot, curr_acf, ncol = 1)
#   ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_TS_solo/",curr_ID,"_res.png", sep = ""), res_plot)
#   
# }
# 
# for (curr_year in 1972:2021) {
#   us_map <- map_data("state")
#   res_sur <-
#     ggplot() + theme_bw() +
#     theme(legend.title=element_blank(),legend.key.width=unit(2,'cm'),legend.position='bottom') +
#     geom_point(aes(x=jitter(res_wide$long, amount = 0.5),y=jitter(res_wide$lat, amount = 0.5),colour=res_wide[,4+curr_year-1972]),size=1,shape=19) +
#     scale_colour_gradientn(colours = hcl.colors(10)) +
#     geom_path(data = us_map, aes(x = long, y = lat, group = group), color = "red") +
#     coord_fixed(ratio = 1.1) +
#     labs(title = paste("Spatial Map of Residuals of Year",curr_year), x = "Longitude",y = "Latitude") + 
#     theme(plot.title = element_text(hjust = 0.5)) 
#   ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_res_map_2000_filter100/",curr_year,"_res.png", sep = ""), res_sur)
# }

# Variogram

curr_year <- 1972
for (curr_year in 1973:2021) {
  par(mfrow = c(1,2))
  nsf_coords <- data.frame("long" = nsf_wide$long, "lat" = nsf_wide$lat)
  curr_y <- nsf_wide[,curr_year-1968]
  prev_y <- nsf_wide[,curr_year-1969]
  plot(variog(coords = nsf_coords, data = curr_y-prev_y), main = curr_year)
  plot(variog4(coords = jitter(as.matrix(nsf_coords)), data = curr_y-prev_y)) 
}







