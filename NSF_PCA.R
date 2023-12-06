nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
nsf_y <- nsf_wide[,-c(1:4)]
pca_res <- prcomp(nsf_y,rank = 9)


par(mfrow = c(3,3))
for (i in 1:9) {
  plot(pca_res$x[,i], ylab = "weight", xlab = "component")
}
