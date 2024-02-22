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

show_pred <- function(curr_id = 1
                      # , scale_range = TRUE, out.rm = FALSE
                      ){
  # pois_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")^2,1, sum) ), decreasing = TRUE)
  # pca_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")^2,1, sum) ), decreasing = TRUE)
  # esn_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_200.csv")^2,1, sum) ), decreasing = TRUE)
  pois_res <- unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")^2,1, sum))
  pca_res <- unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")^2,1, sum) )
  esn_res <- unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_200.csv")^2,1, sum) )
  # if(out.rm){
  #   pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred_outrm.csv")
  #   pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred_outrm.csv")
  #   nh = 200
  #   esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_outrm_",nh, ".csv", sep = ""))
  # }else{
    pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv")
    pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv")
    nh = 200
    esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""))
  # }
  
  nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv", header = TRUE)
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
  # if(out.rm){
  #   # Remove outliers
  #   nh = 200
  #   osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
  #   annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
  #   nsf_wide_car <- nsf_wide_car[-annoying_cases[1:100],]
  # }
  
  curr_y <- as.numeric(nsf_wide_car[curr_id,-c(1:9, ncol(nsf_wide_car))])
  colorss <- c("Poisson" = "purple", "PCA" = "green", "ESN" = "orange")
  
  obs_trace <-
    ggplot() +
    geom_line(aes(x = 1972:2021, y = curr_y), col = "blue", linewidth = 2) +
    geom_point(aes(x = 1972:2021, y = curr_y), col = "red", size = 2) +
    labs(x = "Year", y = "Count", title = paste(nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[curr_id]))]),
                                                # ifelse(out.rm, "(Outlier Removed)","Full Dataset")), 
         color = "Legend") +
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
    guides(color = guide_legend(title = "",override.aes = list(size = 5))) +
    annotate("text", x= 1980, y = range(curr_y,pois_pred[curr_id,],pca_pred[curr_id,],esn_pred[curr_id,])[2], label= paste("Poisson SSE:", round(pois_res[curr_id],2)) ) +
    annotate("text", x= 1980, y = 0.9*range(curr_y,pois_pred[curr_id,],pca_pred[curr_id,],esn_pred[curr_id,])[2], label= paste("PCA SSE:", round(pca_res[curr_id],2)) ) +
    annotate("text", x= 1980, y = 0.8*range(curr_y,pois_pred[curr_id,],pca_pred[curr_id,],esn_pred[curr_id,])[2], label= paste("ESN SSE:", round(esn_res[curr_id],2)) ) 
  # + 
    # annotate("text", x = 4, y=13000, label = "ship") +
    # annotate("text", x=8, y=13000, label= "boat") 
  
  # 
  # if(scale_range){
  #   obs_trace <- obs_trace +  coord_cartesian(ylim =c(range(curr_y,pois_pred[curr_id,],pca_pred[curr_id,],esn_pred[curr_id,])[1],
  #                                                     range(curr_y,pois_pred[curr_id,],pca_pred[curr_id,],esn_pred[curr_id,])[2] + sd(curr_y)) )
  # }else{
  #   obs_trace +  coord_cartesian(ylim = range(curr_y))
  # }
  
  return(obs_trace)
}
nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", header = TRUE)
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv", header = TRUE)
# for (out.rm in 0:1) {
#   if(out.rm){
#     for (i in 1:nrow(nsf_wide_car)) {
#       cowplot::plot_grid(show_pred(i, out.rm = TRUE), show_pred(i,scale_range = FALSE, out.rm = TRUE), nrow = 2)
#       ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_RMOUT/",nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[i]))],".png",sep = ""))
#     }
#   }else{
  pois_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")^2,1, sum) ), decreasing = TRUE)
  pca_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")^2,1, sum) ), decreasing = TRUE)
  esn_annoying_order <- order(unlist(apply(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_200.csv")^2,1, sum) ), decreasing = TRUE)
  annoying_order <- rbind(pois_annoying_order, pca_annoying_order, esn_annoying_order)
    for (i in 1:nrow(nsf_wide_car)) {
     show_pred(pois_annoying_order[i])
      ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_FULL/pois_perform_decreasing/",i,"_",nsf_wide_car$ID[pois_annoying_order[i]],
                   nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[pois_annoying_order[i]]))],".png",sep = ""))
    }
  for (i in 1:nrow(nsf_wide_car)) {
  show_pred(pca_annoying_order[i])
    ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_FULL/pca_perform_decreasing/",i,"_",nsf_wide_car$ID[pca_annoying_order[i]],
                 nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[pca_annoying_order[i]]))],".png",sep = ""))
  }
  for (i in 1:nrow(nsf_wide_car)) {
   show_pred(esn_annoying_order[i])
    ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Pois_PCA_ESN_FULL/esn_perform_decreasing/",i,"_",nsf_wide_car$ID[esn_annoying_order[i]],
                 nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[esn_annoying_order[i]]))],".png",sep = ""))
  }
  
  # }
  
  
# }


pois_res <- rowSums(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")^2)
pca_res <- rowSums(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")^2)
nh = 200
esn_res <- rowSums(read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))^2)
all_res <- rbind(pois_res, pca_res, esn_res)
pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv")
pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv")
nh = 200
esn_pred <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_pred_",nh, ".csv", sep = ""))


mean(unlist(pois_res^2))
mean(unlist(pca_res^2))
mean(unlist(esn_res^2))




#150136907
#166629801
#











