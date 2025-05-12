library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")
nb_pred <- readRDS("D:/77/Research/temp/pred_all_randsl_sch.Rda")
set.seed(2)

n <- 4
idx <- floor(runif(n) * nrow(schoolsM)) 

years <- 1972:2021
df <- data.frame(
  Year = rep(years, n),
  Counts = as.vector(t(schoolsM[idx, ])),
  group = rep(
    sapply(idx, function(i) {
      schools_name$INSTNM[which(schools_name$UNITID == schools$UNITID[i])]
    }),
    each = length(years)
  )
)

# Plot with ggplot
ggplot(df, aes(x = Year, y = Counts, color = group)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c("red", "lightpink", "lightblue", "lightgreen")) +
  labs(color = "") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))

acf1 <- acf(df$Counts[1:50])
acf2 <- acf(df$Counts[51:100])
acf3 <- acf(df$Counts[101:150])
acf4 <- acf(df$Counts[151:200])

acf_to_df <- function(acf_object) {
  data.frame(
    lag = acf_object$lag[,1,1],
    acf = acf_object$acf[,1,1]
  )
}
 
acf1_df <- acf_to_df(acf1)[2:10,]
acf2_df <- acf_to_df(acf2)[2:10,]
acf3_df <- acf_to_df(acf3)[2:10,]
acf4_df <- acf_to_df(acf4)[2:10,]

plot_acf <- function(acf_df, title, col = "skyblue") {
  ggplot(acf_df, aes(x = lag, y = acf)) +
    geom_bar(stat = "identity", fill = col) +
    theme_bw() +
    ggtitle(title) +
    ylab("ACF") +
    xlab("Lag") +
    scale_x_continuous(breaks = seq(0, max(acf_df$lag), by = 1)) +
    theme(plot.title = element_text(hjust = 0.5),panel.grid = element_blank() ) # Center the title
}

p1 <- plot_acf(acf1_df, 
                schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])],"red") +
      geom_hline(yintercept = 1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red") +  
      geom_hline(yintercept = -1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red")   
p2 <- plot_acf(acf2_df, 
                schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])],"lightpink")+
  geom_hline(yintercept = 1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red") +  
  geom_hline(yintercept = -1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red")
p3 <- plot_acf(acf3_df, 
                schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])],"lightblue")+
  geom_hline(yintercept = 1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red") +  
  geom_hline(yintercept = -1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red")
p4 <- plot_acf(acf4_df, 
               schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])],"lightgreen")+
  geom_hline(yintercept = 1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red") +  
  geom_hline(yintercept = -1.96/sqrt(ncol(schoolsM)), linetype = "dashed", color = "red")

cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)



set.seed(21)
n <- 100  
idx <- factor(sample(1:nrow(schoolsM),n))

years <- 1972:2021
df <- data.frame(
  Year = rep(years, n),
  Residuals = as.vector(t(st_nb[idx, ])),
  group = factor(rep(idx, each = length(years)))
)


# Plot with ggplot
ggplot(df, aes(x = Year, y = Residuals, color = group)) +
  geom_line(linewidth = 0.5,alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 0.6) +
  labs(color = "") +
  theme(legend.position = "") +
  guides(color = guide_legend(nrow = 2))




