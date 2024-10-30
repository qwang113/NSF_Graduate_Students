
pred_all_insample <- readRDS("insample_nb.Rda")
pred_mean = apply(pred_all_insample,1,mean)
pred_mean <- matrix(pred_mean, nrow = nrow(schoolsM))
pred_res <- pred_mean - schoolsM
curr_r <- read.csv("rr.csv")
pred_p <- 1/(pred_mean/curr_r + 1)
xt_var <- pred_mean * 1/pred_p

st <- pred_res/sqrt(xt_var)