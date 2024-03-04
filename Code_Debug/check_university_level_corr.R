{
  library(spdep)
  library(sp)
  library(ape)
  }
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
agg_dat <- aggregate( as.matrix(nsf_wide_car[,10:59]) ~ nsf_wide_car$UNITID + nsf_wide_car$long + nsf_wide_car$lat, FUN = mean )
long <- agg_dat$`nsf_wide_car$long`
lat <- agg_dat$`nsf_wide_car$lat`
pair_dist <- spDists(cbind(long,lat))
inv_dist <- 1/pair_dist
diag(inv_dist) <- 0

moran_res <- vector("list")
for (colnum in 4:ncol(agg_dat)) {
  moran_res[[colnum]] = unlist(Moran.I(agg_dat[,colnum], inv_dist))
}


