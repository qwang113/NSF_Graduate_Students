rm(list = ls())
# Gaussian Process with Matern Correlation
nsf_wide <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", header = TRUE)
UNITID <- substr(nsf_wide$ID,1,6)
nsf_wide <- cbind(UNITID,nsf_wide)
carnegie_2021 <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_Carnegie/2021.csv", header = TRUE)[,c(1,4)]
colnames(carnegie_2021)[1] <- "UNITID"
nsf_wide_car <- merge(nsf_wide, carnegie_2021, by = "UNITID")
wide_y <- nsf_wide_car[,-c(1:9, ncol(nsf_wide_car))]

dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
# dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
# dummy_matrix <- cbind(dummy_school, dummy_car)
dummy_matrix <- dummy_car

osh_pred <- matrix(NA, nrow = nrow(wide_y), ncol = length(2012:2021))

#All schools pooled together
for (year in 2012:2021) {
  long_yt <- unlist(as.vector(wide_y[,2:(year-1972)]))
  long_yt_lag <- unlist(as.vector(wide_y[,1:(year-1972-1)]))
  curr_dummy <- matrix(rep(t(dummy_matrix), year-1972-1), ncol = ncol(dummy_matrix), byrow = T)[,-1]
  obs_dat <- data.frame(cbind(long_yt, long_yt_lag, curr_dummy))
  curr_model <- glm(long_yt~., family = poisson(link = "log"), data = obs_dat)
  pred_yt_lag <- wide_y[,year-1972]
  pred_dummy <- dummy_matrix[,-1]
  pred_mat <- cbind(pred_yt_lag, pred_dummy)
  colnames(pred_mat) <- colnames(obs_dat)[-1]
  pred_dat <- data.frame(pred_mat)
  osh_pred[,year-2011] <- predict(curr_model, newdata = pred_dat, type = "response")
}

osh_res <- wide_y[,41:50] - osh_pred
mean(unlist(as.vector(osh_res^2)))

#Individually regression: AR1
for (year in 2012:2021) {
  for (school in 1:nrow(wide_y)) {
  print(paste("now doing year", year, "school", school))
   prev_obs <- unlist(wide_y[school,2:(year-1972)]) 
   lag_obs <- unlist(wide_y[school, 1:(year-1972-1)])
   model <- glm(prev_obs ~ lag_obs, family = poisson(link = "log"))
   osh_pred[school, year-2011] <- predict(model, newdata = data.frame("lag_obs"=unlist(wide_y[school, year-1972])),
                                          type = "response")
  }
}

osh_res <- wide_y[,41:50] - osh_pred


write.csv( as.data.frame(osh_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv", row.names = FALSE)

write.csv( as.data.frame(osh_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv", row.names = FALSE)


# File path: "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv"

mean(unlist(as.vector(osh_res^2)))
var(unlist(as.vector( wide_y[,41:50])))
source(here::here("utility_functions.R"))
annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
ordered_mse <- rowMeans(unlist(as.matrix(osh_res^2)))[annoying_cases]
how_annoying <- 1799
show_obs(annoying_cases[how_annoying])
show_pred(annoying_cases[how_annoying])
ordered_mse[how_annoying]
sqrt(ordered_mse[how_annoying])


# Delete some bad observations
# CNN with/without
# Visualilze the prediction trace of different models
# Median/Median sqared error for ESN, PCA, PoisAutoReg with/without special caeses





