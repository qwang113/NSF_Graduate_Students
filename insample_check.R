rm(list = ls())
library(Matrix)
library(tidyverse)
library(glmnet)
library(tscount)
library(ggplot2)
library(BayesLogit)
library(forecast)
years <- 1972:2021
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
set.seed(2)
idx <- floor(runif(4)*nrow(schoolsM))
setwd("D:/77/Research/temp/")
pred_nb <- readRDS("insample_nb.Rda")
nb_mean = apply(pred_nb,1,mean)
nb_mean <- matrix(nb_mean, nrow = nrow(schoolsM))
nb_res <- nb_mean - schoolsM
curr_r <- apply(readRDS(here::here("rr.Rda")),1, mean)
pred_p <- 1/(nb_mean/curr_r + 1)
xt_var_nb <- nb_mean * 1/pred_p
st_nb <- nb_res/sqrt(xt_var_nb)
test_pvalue_nb = test_statistic_nb = rep(NA, nrow(st_nb))
for (i in 1:nrow(st_nb)) {
  test_pvalue_nb[i] <- Box.test(st_nb[i,], lag = 7, type = "Ljung-Box")$p.value
  test_statistic_nb[i] <- Box.test(st_nb[i,], lag = 7, type = "Ljung-Box")$statistic
}
bad_cases_nb <- which(test_pvalue_nb <= 0.05/nrow(schools))
mean(st_nb)
var(as.vector(st_nb))


pred_pois <- readRDS("insample_pois.Rda")
pois_mean = apply(pred_pois,1,mean)
pois_mean <- matrix(pois_mean, nrow = nrow(schoolsM))
xt_var_pois <- pois_mean
pois_res <- pois_mean - schoolsM
st_pois <- pois_res/sqrt(xt_var_pois)
test_pvalue_pois = test_statistic_pois = rep(NA, nrow(st_pois))
for (i in 1:nrow(st_pois)) {
  test_pvalue_pois[i] <- Box.test(st_pois[i,], lag = 7, type = "Ljung-Box")$p.value
  test_statistic_pois[i] <- Box.test(st_pois[i,], lag = 7, type = "Ljung-Box")$statistic
}
bad_cases_pois <- which(test_pvalue_pois <= 0.05/nrow(schools))
mean(st_pois)
var(as.vector(st_pois))

sch_res <- data.frame("Poisson" = apply(st_pois,1,var), "Negative Binomial" = apply(st_nb,1,var))
library(tidyr)
library(ggplot2)
library(dplyr)
library(patchwork)

long_data <- pivot_longer(sch_res, cols = everything(), names_to = "group", values_to = "value")
full_plot <- ggplot(long_data, aes(x = group, y = value, fill = group)) +
  geom_boxplot() +
  labs(
    title = "Variance of Pearson Residual for Each School",
    y = "",
    x = ""
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),  legend.position = "none"
  )

# Truncated boxplot with visible range -5 to 5
zoomed_plot <- ggplot(long_data, aes(x = group, y = value, fill = group)) +
  geom_boxplot() +
  coord_cartesian(ylim = c(0, 5)) + 
  labs(
    title = "Variance of Pearson Residual for Each School (Zoomed)",
    y = "",
    x = ""
  ) +
  theme(
    plot.title = element_text(hjust = 0.5),  legend.position = "none"
  )

# Combine the two plots into one figure
combined_plot <-zoomed_plot

# Display the combined plot
combined_plot



schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")
df <- data.frame(
  year = rep(1973:2021, 4),
  value = c(st_nb[idx[1],-1], st_nb[idx[2],-1], st_nb[idx[3],-1], st_nb[idx[4],-1]),
  group = rep(c(   schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])]
  ), each = length(1973:2021)
  )
)

ggplot(df, aes(x = year, y = value, color = group)) +
  geom_line(linewidth = 1) +
  scale_color_manual(values = c("red", "lightpink", "lightblue", "lightgreen")) +
  labs(color = "", y = 'Residuals') +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))


df <- data.frame(
  year = rep(1972:2021, 4),
  value = c(st_nb[idx[1],], st_nb[idx[2],], st_nb[idx[3],], st_nb[idx[4],]),
  group = rep(c(   schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])]
                   ,  schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])]
  ), each = length(1972:2021)
  )
)



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


# More series

n <- 100  
idx <- factor(sample(1:nrow(st_nb),n))

df <- data.frame(
  Year = rep(years, n),
  Counts = as.vector(t(st_nb[idx, ])),
  group = rep(idx,each = length(years))
)

ggplot(df, aes(x = Year, y = Counts, group = group)) +
  geom_line(color = "black", linewidth = 0.5, alpha = 0.15) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  theme_bw() +
  theme(legend.position = "none")



resis_diff <- schoolsM[,46:50] - schoolsM[,45:49]
dd <- matrix(rpois(length(schoolsM[,46:50]), schoolsM[,46:50]), nrow = nrow(schoolsM), ncol = 5)

apply((dd - schoolsM[,45:49])^2, 2, mean)


resis_ldiff <- log(schoolsM[,46:50]+1) - log(schoolsM[,45:49]+1)
apply(resis_ldiff^2, 2, mean)



# ---------------------------------------------- Outliers check
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")
idx <- c(1558,388) 
years <- 1972:2021
df <- data.frame(
  Year = rep(years, 2),
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
  scale_color_manual(values = c("lightpink", "lightblue")) +
  labs(color = "") +
  theme(legend.position = "bottom") +
  guides(color = guide_legend(nrow = 2))


