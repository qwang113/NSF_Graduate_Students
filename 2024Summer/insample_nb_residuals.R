W <- read.csv(here::here("W.csv"))
Uy <- read.csv(here::here("Uy.csv"))
eta_rs <- read.csv(here::here("eta_rs.csv"))
rr <- read.csv(here::here("rr.csv"))
r_hat <- apply(rr,1,mean)
eta_hat <- apply(eta_rs, 1, mean)
schools <- read.csv(here::here("nsf_final_wide_car.csv"))
schoolsM <- as.matrix(schools[,10:59])
state_idx <- model.matrix( ~ factor(state) - 1, data = schools)


