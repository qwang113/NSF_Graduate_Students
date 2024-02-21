nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
asu_907 <- nsf_wide[which(nsf_wide$ID=="104151907"),-c(1:8)] 
