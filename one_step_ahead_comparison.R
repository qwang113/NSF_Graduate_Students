source(here::here("utility_functions.R"))


pois_res <- unlist(as.matrix(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv")))
pca_res <- unlist(as.matrix(read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv")))
nh = 200
esn_res <- unlist(as.matrix(read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))))


annoying_cases <- order(apply(esn_res^2,1, mean), decreasing = TRUE)
ordered_mse <- rowMeans(esn_res^2)[annoying_cases]
how_annoying <- 2
show_obs(annoying_cases[how_annoying])
show_pred(annoying_cases[how_annoying])
ordered_mse[how_annoying]
sqrt(ordered_mse[how_annoying])

scaled <- apply(unlist(as.matrix(wide_y)), 1, scale)
scaled_range <- apply(scaled,2,range)

special_cases <- which(scaled_range[2,]- scaled_range[1,] >= quantile(scaled_range[2,]- scaled_range[1,], 0.98))
show_obs(special_cases[6])
