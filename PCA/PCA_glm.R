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
dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
# dummy_matrix <- cbind(dummy_school, dummy_car)
dummy_matrix <- dummy_school

#Assume Beta changes across the schools.
#No need to add covariate if they doesn't change aross the time.

individual_beta_pred <- matrix(NA, nrow = nrow(nsf_wide), ncol = length(2012:2021))
for (i in 2012:2021) {

  years_before <- i - 1972
  prev_y <- t(wide_y)[1:years_before,]
  prev_pc <- prcomp(prev_y)
  num_pc <- length(which(cumsum(prev_pc$sdev^2)/sum(prev_pc$sdev^2) <= 0.8 ))
  prev_pc_use <- prev_pc$x[,1:num_pc]

  for (j in 1:nrow(nsf_wide)) {
    print(paste(  print(paste("Now doing year",i)), "Now doing school",j))
    #Each school has its own slope but shared across the time, loop across the schoools
    curr_y <- wide_y[j,2:years_before]
    curr_pc_cov <- prev_pc_use[1:(nrow(prev_pc_use)-1),] # Use the previous year's principal component
    curr_dat <- cbind(t(curr_y), curr_pc_cov, t(wide_y[j,1:(years_before-1)]))
    colnames(curr_dat) <- c("y",colnames(curr_pc_cov),"lag")
    curr_model <- glm(y~., data = data.frame(curr_dat), family =poisson(link = "log"),  control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = FALSE))
    pred_x <- cbind(matrix(prev_pc_use[nrow(prev_pc_use),], nrow = 1), wide_y[j,years_before])
    colnames(pred_x) <- colnames(curr_dat)[-1]
    prediction <- predict(curr_model, data.frame(pred_x), type = "response")
    individual_beta_pred[j,i-2011] <- prediction
  }

}

individual_beta_res <- nsf_wide_car[,50:59] - individual_beta_pred
sqrt(mean(unlist(as.vector(individual_beta_res^2))))

write.csv( as.data.frame(individual_beta_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_pred.csv", row.names = FALSE)

write.csv( as.data.frame(individual_beta_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pca_res.csv", row.names = FALSE)


# File name: "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv"

# 
# #Assume Beta does not change across the schools.
# shared_beta_pred <- matrix(NA, nrow = nrow(nsf_wide), ncol = length(2012:2021))
# for (i in 2012:2021) {
#   print(paste("Now doing year",i))
#   years_before <- i - 1972
#   prev_y <- t(wide_y)[1:years_before,]
#   prev_pc <- prcomp(prev_y)
#   num_pc <- length(which(cumsum(prev_pc$sdev^2)/sum(prev_pc$sdev^2) <= 0.95 ))
#   prev_pc_use <- prev_pc$x[,1:num_pc]
# 
#   j = 1
# 
#   #Each school has its own slope but shared across the time, loop across the schoools
#   curr_y <- wide_y[j,2:years_before]
#   curr_pc_cov <- prev_pc_use[1:(nrow(prev_pc_use)-1),] # Use the previous year's principal component
#   curr_cov <- dummy_matrix[j,]
#   curr_dummy <- matrix(rep(curr_cov, nrow(curr_pc_cov)), byrow = TRUE, nrow = nrow(curr_pc_cov))
#   curr_x <- cbind(curr_pc_cov, curr_dummy)
#   curr_dat <- cbind(t(curr_y), curr_x)
#   use_dat <- curr_dat
#   
#   for (j in 2:nrow(nsf_wide)) {
# 
#     #Each school has its own slope but shared across the time, loop across the schoools
#     curr_y <- wide_y[j,2:years_before]
#     curr_pc_cov <- prev_pc_use[1:(nrow(prev_pc_use)-1),] # Use the previous year's principal component
#     curr_cov <- dummy_matrix[j,]
#     curr_dummy <- matrix(rep(curr_cov, nrow(curr_pc_cov)), byrow = TRUE, nrow = nrow(curr_pc_cov))
#     curr_x <- cbind(curr_pc_cov, curr_dummy)
#     curr_dat <- cbind(t(curr_y), curr_x)
#     use_dat <- rbind(use_dat, curr_dat)
#   }
#   
#   final_dat <- use_dat[-1,]
#   colnames(final_dat) <- c("y",colnames(curr_pc_cov),colnames(dummy_matrix))
#   curr_model <- glm.nb(y~., data = data.frame(final_dat))
#   pred_pc <- matrix( rep(prev_pc_use[nrow(prev_pc_use),],nrow(nsf_wide)), nrow = nrow(nsf_wide), byrow = TRUE)
#   pred_cov <- cbind(pred_pc, dummy_matrix)
#   colnames(pred_cov) <- colnames(final_dat)[-1]
#   shared_beta_pred[,i-2011] <- predict(curr_model, data.frame(pred_cov), type = "response")
# }
# 
# shared_beta_res <- nsf_wide_car[,50:59] - shared_beta_pred
# 
# write.csv( as.data.frame(shared_beta_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/share_beta_pred.csv", row.names = FALSE)
# 
