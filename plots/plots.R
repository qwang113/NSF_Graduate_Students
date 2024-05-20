rm(list = ls())
library(ggplot2)

nsf_wide_car <- read.csv(here::here("nsf_final_wide_car.csv"))
nsf_y <- as.matrix(nsf_wide_car[,-c(1:9,ncol(nsf_wide_car))])

pois_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv")
pois_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")
pca_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv")
pca_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")
esn_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/esn_pred.csv")
esn_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/esn_res.csv")
crnn_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/crnn_pred.csv")
crnn_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/crnn_res.csv")
spesn_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/spesn_pred.csv")
spesn_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/spesn_res.csv")
pcesn_pred <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pcesn_pred.csv")
pcesn_res <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pcesn_res.csv")

pois_res_by_year <- colMeans(pois_res^2)
pca_res_by_year <- colMeans(pca_res^2)
esn_res_by_year <- colMeans(esn_res^2)
crnn_res_by_year <- colMeans(crnn_res^2)
spesn_res_by_year <- colMeans(spesn_res^2)
pcesn_by_year <- colMeans(pcesn_res^2)
var_by_year <- apply(nsf_y[,41:50], 2, var)


res_all <- sqrt(rbind(pois_res_by_year, pca_res_by_year, esn_res_by_year,
                      crnn_res_by_year, spesn_res_by_year, pcesn_by_year, var_by_year))
# rownames(res_all) <- c("Poisson","PCA","ESN")

p <- ggplot() +
  geom_path(aes(x = 2012:2021, y = res_all[1,], color = "Model 1"), linewidth = 1) +
  geom_path(aes(x = 2012:2021, y = res_all[2,], color = "Model 2"), linewidth = 1) +
  geom_path(aes(x = 2012:2021, y = res_all[3,], color = "Model 3"), linewidth = 1) +
  geom_point(aes(x = 2012:2021, y = res_all[1,], color = "Model 1")) +
  geom_point(aes(x = 2012:2021, y = res_all[2,], color = "Model 2")) +
  geom_point(aes(x = 2012:2021, y = res_all[3,], color = "Model 3")) +
  # geom_path(aes(x = 2012:2021, y = res_all[4,], color = "Model 4"), linewidth = 1.5, linetype = "dotted") +
  # geom_path(aes(x = 2012:2021, y = res_all[5,], color = "Model 5"), linewidth = 1.5, linetype = "dotted") +
  # geom_path(aes(x = 2012:2021, y = res_all[6,], color = "Model 6"), linewidth = 1.5, linetype = "dotted") +
  geom_path(aes(x = 2012:2021, y = res_all[7,], color = "Model 4"), linewidth = 1) +
  geom_point(aes(x = 2012:2021, y = res_all[7,], color = "Model 4")) +
  coord_cartesian(ylim = c(0, 110)) +
  scale_color_manual(name = "Model",  # Legend title
                     # values = viridis(4),
                       values =  c("lightpink",  "lightblue","gray","black"),  # Custom legend colors

                     labels = c(paste("Poisson AR:",round(mean(sqrt(pois_res_by_year)),2)),
                                paste("PC AR:",round(mean(sqrt(pca_res_by_year)),2)),
                                paste("ESN:",round(mean(sqrt(esn_res_by_year)),2)),
                                paste("Intercept:",round(mean(sqrt(var_by_year)),2)))) +  # Custom legend labels
  labs(x = "Year", y = "RMSE") +
  theme(legend.position = "bottom",panel.background = element_blank(),
        axis.line = element_line(color = "black"))+
  scale_x_continuous(breaks = 2012:2021)

p
all_res_by_year <-res_all
rownames(all_res_by_year) <- c("Poisson", "PC-AR", "ESN", "CRNN", "SP-ESN","PC-ESN" ,"Intercept")
all_res_by_year <- cbind(all_res_by_year, rowMeans(all_res_by_year))
colnames(all_res_by_year)[ncol(all_res_by_year)] <- "Average"

knitr::kable(t(all_res_by_year), format = "latex", align = 'c', digits = 2)

# set some schools
ins_loc <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ins_loc.csv")
names <- data.frame("UNITID" = ins_loc$UNITID,"College Name" = ins_loc$INSTNM)
nsf_names <- merge( data.frame("UNITID" = nsf_wide_car$UNITID), names, by = "UNITID")


mrs <- function(x){return(round(mean(sqrt(unlist(x^2))),2))}

cowplot::plot_grid(ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[132,], color = "Model 1"), linewidth = 2)+ 
                     geom_path(aes(x = 2012:2021, y = unlist(pois_pred[132,]), color = "Model 2"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(pca_pred[132,]), color = "Model 3"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(spesn_pred[132,]), color = "Model 4"), linewidth = 1)+
                     scale_color_manual(name = "Model and RMSE",  # Legend title
                                        values = c("black", "red",  "blue","purple" ),  # Custom legend colors

                                        labels = c("True", paste("Poisson",mrs(pois_res[132,])) ,
                                                   paste("PC-AR",mrs(pca_res[132,])),
                                                   paste("SP-ESN",mrs(spesn_res[132,]))

                                                   )) +  # Custom legend labels
                     labs(x = "Year", y = "Y", title = paste(nsf_names[132,2], " (",substr(nsf_wide_car$ID[132], 7,9),")", sep = "")) +
                     geom_vline(xintercept = 2012, linewidth = 2, linetype = "dashed", color = "pink") +
                     theme(plot.title = element_text(hjust = 0.5),legend.position = "bottom"),
                   
                   ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[1722,], color = "Model 1"), linewidth = 2) + 
                     geom_path(aes(x = 2012:2021, y = unlist(pois_pred[1722,]), color = "Model 2"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(pca_pred[1722,]), color = "Model 3"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(spesn_pred[1722,]), color = "Model 4"), linewidth = 1)+
                     scale_color_manual(name = "Model and RMSE",  # Legend title
                                        values = c("black", "red",  "blue","purple" ),  # Custom legend colors

                                        labels = c("True", paste("Poisson",mrs(pois_res[1722,])) ,
                                                   paste("PC-AR",mrs(pca_res[1722,])),
                                                   paste("SP-ESN",mrs(spesn_res[1722,]))

                                        )) +
                     labs(x = "Year", y = "Y", title = paste(nsf_names[1722,2], " (",substr(nsf_wide_car$ID[1722], 7,9),")", sep = "")) +
                     geom_vline(xintercept = 2012, linewidth = 2, linetype = "dashed", color = "pink") +
                     theme(plot.title = element_text(hjust = 0.5),legend.position = "bottom"),
                   
                   ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[218,], color = "Model 1"), linewidth = 2) + 
                     geom_path(aes(x = 2012:2021, y = unlist(pois_pred[218,]), color = "Model 2"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(pca_pred[218,]), color = "Model 3"), linewidth = 1) +
                     geom_path(aes(x = 2012:2021, y = unlist(spesn_pred[218,]), color = "Model 4"), linewidth = 1)+
                     scale_color_manual(name = "Model and RMSE",  # Legend title
                                        values = c("black", "red",  "blue","purple" ),  # Custom legend colors

                                        labels = c("True", paste("Poisson",mrs(pois_res[218,])) ,
                                                   paste("PC-AR",mrs(pca_res[218,])),
                                                   paste("SP-ESN",mrs(spesn_res[218,]))

                                        )) +
                     labs(x = "Year", y = "Y", title = paste(nsf_names[218,2], " (",substr(nsf_wide_car$ID[218], 7,9),")", sep = "")) +
                     geom_vline(xintercept = 2012, linewidth = 2, linetype = "dashed", color = "pink") +
                     theme(plot.title = element_text(hjust = 0.5),legend.position = "bottom")
                   
                   
                   
                   , ncol = 1)


cowplot::plot_grid(ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[132,]), linewidth = 2, color = "black") +
                     labs(x = "Year", y = "Y", title = paste(nsf_names[132,2], " (",substr(nsf_wide_car$ID[132], 7,9),")", sep = "")) +
                     theme(plot.title = element_text(hjust = 0.5))+
                     coord_cartesian(xlim = c(1971,2021)),
                   
                   ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[1722,]), linewidth = 2, color = "black") + 
                     labs(x = "Year", y = "Y", title = paste(nsf_names[1722,2], " (",substr(nsf_wide_car$ID[1722], 7,9),")", sep = "")) +
                     theme(plot.title = element_text(hjust = 0.5))+
                     coord_cartesian(xlim = c(1971,2021)),
                   
                   ggplot() +
                     geom_path(aes(x = 1972:2021, y = nsf_y[218,]), linewidth = 2, color = "black") + 
                     labs(x = "Year", y = "Y", title = paste(nsf_names[218,2], " (",substr(nsf_wide_car$ID[218], 7,9),")", sep = "")) +
                     theme(plot.title = element_text(hjust = 0.5))+
                     coord_cartesian(xlim = c(1971,2021))
                   
                   
                   
                   , ncol = 1)





