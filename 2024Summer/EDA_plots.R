library(ggplot2)
library(dplyr)
library(purrr)
library(tidyr)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")

set.seed(2)
idx <- floor(runif(4)*nrow(schoolsM))
df <- data.frame(
  year = rep(1972:2021, 4),
  value = c(schoolsM[idx[1],], schoolsM[idx[2],], schoolsM[idx[3],], schoolsM[idx[4],]),
  group = rep(c(   schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])]
                ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])]
                ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])]
                ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])]
                ), each = length(1972:2021)
              )
)

# Plot with ggplot
ggplot(df, aes(x = year, y = value, color = group)) +
  geom_line(linewidth = 1.5) +
  scale_color_manual(values = c("red", "lightpink", "lightblue", "lightgreen")) +
  labs(color = "") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))

acf1 <- acf(df$value[1:50])
acf2 <- acf(df$value[51:100])
acf3 <- acf(df$value[101:150])
acf4 <- acf(df$value[151:200])

acf_to_df <- function(acf_object) {
  data.frame(
    lag = acf_object$lag[,1,1],
    acf = acf_object$acf[,1,1]
  )
}
 
acf1_df <- acf_to_df(acf1)
acf2_df <- acf_to_df(acf2)
acf3_df <- acf_to_df(acf3)
acf4_df <- acf_to_df(acf4)

plot_acf <- function(acf_df, title, col = "skyblue") {
  ggplot(acf_df, aes(x = lag, y = acf)) +
    geom_bar(stat = "identity", fill = col) +
    theme_bw() +
    ggtitle(title) +
    ylab("ACF") +
    xlab("Lag") +
    theme(plot.title = element_text(hjust = 0.5),panel.grid = element_blank() ) # Center the title
}

p1 <- plot_acf(acf1_df, 
               paste(schools$state[idx[1]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])],"-",substr(schools$ID[idx[1]],7,9)))
p2 <- plot_acf(acf2_df, 
               paste(schools$state[idx[2]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])],"-",substr(schools$ID[idx[2]],7,9)))
p3 <- plot_acf(acf3_df, 
               paste(schools$state[idx[3]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])],"-",substr(schools$ID[idx[3]],7,9)))
p4 <- plot_acf(acf4_df, 
               paste(schools$state[idx[4]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])],"-",substr(schools$ID[idx[4]],7,9)))

cowplot::plot_grid(p1, p2, p3, p4, ncol = 1)
