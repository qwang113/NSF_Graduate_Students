nsf_dat <- read.csv(here::here("nsf_final_wide_car.csv"))
# Statewise ANOVA
all_pvalue <- rep(NA, 50)
for (i in 10:59) {
  selected_dat <- data.frame("state" = nsf_dat$state, "counts" = nsf_dat[,i])
  anova_res <- aov(counts ~ state, data = selected_dat)
  all_pvalue[i-9] <- anova(anova_res)$`Pr(>F)`[1]
}
# Statewise Moran's I test
library(splm)
library(spdep)
moran_pvalue <- rep(0,50)
adj_mat <- ifelse(usaww == 0, 0, 1)
adj_mat_states <- colnames(adj_mat)
all_adj <- rbind(cbind(adj_mat,0),0) 
dc_nb <- which(adj_mat_states %in% c("MARYLAND","VIRGINIA"))
all_adj[dc_nb,ncol(all_adj)] <- 1
all_adj[nrow(all_adj),dc_nb] <- 1
colnames(all_adj) <- rownames(all_adj) <- c(unique(nsf_dat$state)[-which(unique(nsf_dat$state)=="DC")],"DC")
listw <- mat2listw(all_adj, style="W")

for (i in 10:59) {
  selected_dat <- data.frame("state" = nsf_dat$state, "counts" = nsf_dat[,i])
  curr_state_mean <- aggregate(counts ~ state, selected_dat, mean)
  reordered_state <- rbind(curr_state_mean[-7,],curr_state_mean[7,])
  moran_test <- moran.test(reordered_state[,2], listw)
  moran_pvalue[i-9] <- moran_test$p.value
}
