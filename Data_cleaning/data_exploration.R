rm(list = ls())
library(openxlsx)
library(dplyr)
library(tidyr)
library(nlme)
library(ggplot2)
library(maps)
library(gifski)
library(gganimate)
library(reshape2)

var_name_old <- "ft_tot_all_races_v"
var_name_new <- "ft_tot_all_races_v"
sheet <- "Race"

# Since there is a data collection method change in 2017, so var_name_old should be the previous variable name, and the var_name_new should be the name after 2017. There is a sheet recording the equivalent names for them.
ID_variables <- c(
  "institution_id","institution_state","school_id","UNITID","year", "Institution_Name","hdg_inst","toc_code", "institution_state", "hbcu_flag","land_grant_flag","carnegie_code_1994","carnegie_code_2005","carnegie_code_2010","carnegie_code_2015","full_school_name","school_name","school_zip","school_type_code","gss_code","hdg_code"
)

all_sheet<- vector("list")
selected_mat <-  vector("list")
for (i in 1:50) {
  print( paste("Now retrieving data of year: ",i+1971))
  all_sheet[[i]] <- read.xlsx( paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_grad/gss",1971+i,"_Code.xlsx", sep = ""), sheet = sheet)
  if(1971+i <= 2016){selected_mat[[i]] <- all_sheet[[i]][,c(ID_variables, var_name_old)]}
  else{selected_mat[[i]] <- all_sheet[[i]][,c(ID_variables, var_name_new)]}
}

# Add coordinates
final_mat <- bind_rows(selected_mat)
col_coords <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/ins_loc.csv", header = TRUE)
id_coords <- data.frame("UNITID" = as.character(col_coords$UNITID) , "long" = as.numeric(col_coords$LONGITUDE), "lat" = as.numeric(col_coords$LATITUDE))
out <- left_join(final_mat, id_coords, by = "UNITID")
NA_sch <- out$Institution_Name[which(is.na(out$long))]

dat <- list("DAT" = out, "NAschools" = unique(NA_sch))

use_dat <- dat$DAT[-which(dat$DAT$UNITID==999999),c("UNITID","Institution_Name","institution_state","carnegie_code_1994","carnegie_code_2005","carnegie_code_2010","carnegie_code_2015", "year","gss_code","ft_tot_all_races_v","long","lat")]
tem = spBayes::pointsInPoly(as.matrix(map_data("usa", region = "main")[,1:2]),cbind(use_dat$long, use_dat$lat))
# summed_dat <- aggregate( ft_tot_all_races_v ~  school_id + gss_code + year, data = use_dat, FUN = function(x) sum(x, na.rm = TRUE))
use_dat <- use_dat[tem,]
sch_gss <- as.numeric(paste(as.character(use_dat$UNITID), as.character(use_dat$gss_code), sep = ""))
final_dat <- data.frame("ID" = sch_gss,"state" = use_dat$institution_state,"year" = as.numeric(use_dat$year), "y" = as.numeric(use_dat$ft_tot_all_races_v),
                        "long" = use_dat$long, "lat" = use_dat$lat)


final_dat <- aggregate(data = final_dat, y~., sum)

all_schools <- unique(final_dat$ID)
stay <- rep(NA, length(all_schools))

for (i in 1:length(all_schools)) {
  print(paste("Now doing index",i,"out of",length(all_schools)))
  stay[i] <- (length( unique( final_dat$year[which(final_dat$ID==all_schools[i])] ) ) == 50)
}

stay_schools <- unique(final_dat$ID)[stay]
# double_check <- rep(NA, length(stay_schools))
stay_dat_long <- final_dat[final_dat$ID %in% stay_schools, ]
# 
# 
# for (i in 1:length(stay_schools)) {
#   double_check[i] <- nrow(stay_dat[which(stay_dat$ID == stay_schools[i]),]) != 50
# }
# 
# double_check_schools <- stay_schools[double_check]
# checked_dat <- stay_dat[stay_dat$ID %in% double_check_schools, ]
# 
# details <- paste(stay_dat$ID, stay_dat$year, sep = "_") %in% 
#   rownames(table(paste(checked_dat$ID, checked_dat$year, sep = "_"))>1 )[which(table(paste(checked_dat$ID, checked_dat$year, sep = "_"))>1)]
# 
# detail_error <- stay_dat[details,]
# detail_error_sort <- detail_error[order(detail_error[,1]),]
# 
# stay_dat <- aggregate(data = stay_dat, y~., sum)

univ_id <- substr(stay_dat_long$ID,1,6)
sch_id <- substr(stay_dat_long$ID,7,9)
ID <- paste(univ_id, sch_id, sep = "")
stay_dat_long <- cbind(univ_id, sch_id, ID, stay_dat_long[,-1])
sch_name <- data.frame("univ_id" = col_coords$UNITID, "univ_name" = col_coords$INSTNM)
stay_dat_long <- merge(sch_name, stay_dat_long, by = "univ_id")

stay_dat_wide <- dcast(as.data.frame(stay_dat_long), ID + univ_name + long + lat ~ year, value.var = "y")

agg <- aggregate(data = stay_dat_long, y ~ ID, sum)
exclude_0 <- stay_dat_wide[stay_dat_wide$ID %in% agg$ID[which(agg$y > 50)],]

special_cases <- which( rowSums(exclude_0[,-c(1:4)]==0)>5 )
special_mat <- as.matrix(exclude_0[special_cases,])


write.csv(as.data.frame(exclude_0), "D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv", row.names = FALSE)
write.csv(as.data.frame(stay_dat_wide), "D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide.csv", row.names = FALSE)



for (i in 1:nrow(special_mat)) {
  
  curr_plot <- 
    ggplot() +
    geom_path(aes(x = 1972:2021, y = as.numeric(special_mat[i,-c(1:4)])), linewidth = 2) +
    labs(x = "Year", y = "Number of Graduate Students", title = paste(special_mat[i,1], special_mat[i,2])) +
    theme(plot.title = element_text(hjust = 0.5)) + 
    scale_x_continuous(limits = c(1972,2021), breaks = seq(1972, 2021, 2)) + 
    theme(plot.title = element_text(face = "bold", size = 25)) 
  
  
    ggsave(paste("D:/77/UCSC/study/Research/temp/NSF_dat/NSF_data_cleaning/", gsub("[^A-Za-z0-9_]", "_",paste(special_mat[i,1], special_mat[i,2])), ".png", sep = ""),curr_plot, width = 25, height = 15)
}
# table(rowSums(stay_dat_wide[,-c(1:4)]==0))
# exist_0_ID <- stay_dat_wide$ID[which( rowSums( stay_dat_wide[,-c(1:3)] == 0 ) != 0 )]
# na_omit_rows <- stay_dat_wide$ID[which( is.na(rowSums( stay_dat_wide[,-c(1:3)] )) )]











