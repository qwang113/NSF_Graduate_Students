library(ggplot2)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
schools_name <- read.csv("D:/77/Research/temp/ins_loc.csv")

set.seed(7)
idx <- floor(runif(6)*1799)
df <- data.frame(
  year = rep(1972:2021, 4),
  value = c(schoolsM[idx[1],], schoolsM[idx[2],], schoolsM[idx[3],], schoolsM[idx[4],]),
  group = rep(c(  paste(schools$state[idx[1]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[1]])])
                , paste(schools$state[idx[2]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[2]])])
                , paste(schools$state[idx[3]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[3]])])
                , paste(schools$state[idx[4]],":", schools_name$INSTNM[which(schools_name$UNITID==schools$UNITID[idx[4]])])
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
