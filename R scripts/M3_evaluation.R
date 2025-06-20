library(data.table)
library(openxlsx)
library(magrittr)
library(Mcomp)
library(stringr)
library(DescTools)
print(R.home())
########################## Actuals ##########################
## Load (raw) actuals for M3 and M4
## We collect actuals for both the test sets and training sets
## Training actuals can be used for MASE/RMSSE scaling later on

#############################################################
# setwd(dirname(file.choose()))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dataset_subset <- 'Monthly' #'Other' 
dataset_subset_indicator <- 'M' #'O'
FH <- 18#8#18
FL <- 6#2#6
FO <- 13#7#13
dataset_selection <- 'M3' #'M3'
#############################################################
#6.5x5

if (dataset_selection == 'M3'){
  
  MS <- subset(M3, dataset_subset)
  # item_id M1385/N2786 --> one observation in test set is -1200 in M3 data via raw source vs 1200 in M3 via R package
  
  MS_actuals_train <- data.frame()
  max_l <- 0
  max_n <- 0
  for (i in c(1:length(MS))) {
    if (length(MS[[i]]$x) > max_l) {max_l <- length(MS[[i]]$x)}
    if (MS[[i]]$n > max_n) {max_n <- MS[[i]]$n}
  }
  if (max_l != max_n) print('TS length error')
  
  MS_actuals_test <- data.frame()
  
  counter <- 1
  for (i in c(1:length(MS))) {
    ts <- MS[[i]]
    
    MS_actuals_train[i,1] <- MS_actuals_test[i,1] <- ts$st
    MS_actuals_train[i,2] <- MS_actuals_test[i,2] <- ts$sn
    MS_actuals_train[i,3:c(max_l+2)] <- c(ts$x, rep(NA, max_l-length(ts$x)))
    MS_actuals_test[i,3:c(FH+2)] <- ts$xx
    
    counter <- counter+1
  }
  
  MS_actuals_train <- data.table(MS_actuals_train)
  MS_actuals_test <- data.table(MS_actuals_test)
  
  setnames(MS_actuals_train, names(MS_actuals_train), c('item_id1', 'item_id2', 1:max_l))
  setnames(MS_actuals_test, names(MS_actuals_test), c('item_id1', 'item_id2', 1:FH))
  MS_actuals_train[, item_id := as.numeric(str_remove(item_id1, dataset_subset_indicator))]
  MS_actuals_test[, item_id := as.numeric(str_remove(item_id1, dataset_subset_indicator))]
  MS_actuals_train[, item_id1 := NULL]
  MS_actuals_train[, item_id2 := NULL]
  MS_actuals_test[, item_id1 := NULL]
  MS_actuals_test[, item_id2 := NULL]
  setcolorder(MS_actuals_train, c('item_id'))
  setorder(MS_actuals_train, item_id)
  setcolorder(MS_actuals_test, c('item_id'))
  setorder(MS_actuals_test, item_id)
  
  rm(MS, ts)
  
} else if (dataset_selection == 'M4') {
  
  MS_actuals_train <- fread(paste0(dataset_subset, '-train.csv'))
  MS_actuals_test <- fread(paste0(dataset_subset, '-test.csv'))
  
  max_l <- ncol(MS_actuals_train)-1
  
  setnames(MS_actuals_train, names(MS_actuals_train), c('item_id', 1:max_l))
  setnames(MS_actuals_test, names(MS_actuals_test), c('item_id', 1:FH))
  MS_actuals_train[, item_id := as.numeric(str_remove(item_id, dataset_subset_indicator))]
  MS_actuals_test[, item_id := as.numeric(str_remove(item_id, dataset_subset_indicator))]
  
}

MS_actuals_all <- merge.data.table(MS_actuals_train, MS_actuals_test, by = 'item_id')
setnames(MS_actuals_all, names(MS_actuals_all), c('item_id', 1:c(ncol(MS_actuals_train)-1+FH)))
MS_actuals_all <- melt.data.table(MS_actuals_all, id.vars = 'item_id')
MS_actuals_all <- MS_actuals_all[!is.na(value)]
setorder(MS_actuals_all, variable)
MS_actuals_all[, variable := c(1:.N) , by = list(item_id)]
MS_actuals_all[, variable := as.numeric(variable)]
MS_actuals_all[, diff_sq := (value - shift(value))^2, by = item_id]

MS_actuals_test[, fc_origin := 1]
MS_actuals_test <- melt.data.table(MS_actuals_test,
                                    id.vars = c('item_id', 'fc_origin'),
                                    variable.name = 'forecast_horizon',
                                    value.name = 'value')
MS_actuals_test[, forecast_horizon := as.numeric(forecast_horizon)]

MS_A <- data.table()
for (origin in c(1:FO)) {
  MS_A_fc_origin <- MS_actuals_test[forecast_horizon %in% c(origin:(origin+(FL-1)))]
  MS_A_fc_origin[, fc_origin := origin]
  MS_A_fc_origin[, forecast_horizon := (forecast_horizon - origin + 1)]
  
  n_obs_remove <- FH - (origin-1)
  MS_actuals_all_subset <- data.table(MS_actuals_all)
  MS_actuals_all_subset[, max_obs := (max(variable) - n_obs_remove), by = list(item_id)]
  MS_actuals_all_subset <- MS_actuals_all_subset[variable <= max_obs]
  
  MS_dfs <- MS_actuals_all_subset[, list(discount_factor = mean(diff_sq, na.rm = T)), by = item_id]
  MS_A_fc_origin <- merge.data.table(MS_A_fc_origin, MS_dfs, by = 'item_id')
  
  # MS_dfs <- MS_actuals_all_subset[, list(discount_factor = mean(sqrt(diff_sq), na.rm = T)), by = item_id]
  # MS_A_fc_origin <- merge.data.table(MS_A_fc_origin, MS_dfs, by = 'item_id')
  
  MS_A <- rbind(MS_A, MS_A_fc_origin)
}
setorder(MS_A, item_id, fc_origin, forecast_horizon)

########################## Evaluation forecasts DT ##########################
# n_methods <- 12
# n_items <- unique(MS_A$item_id)
# MS_F <- matrix(NA, nrow = length(n_items), ncol = 1 + n_methods) %>% data.table()
# setnames(MS_F, c('ID', 
#                  'ETS', 'ARIMA', 'THETA', 
#                  'NBEATS', 'NBEATS_WD', 'NBEATS_DROP', 'NBEATS_TE', 'NBEATS_OE',
#                  'NBEATSS', 'NBEATSS_WD', 'NBEATSS_TE', 'NBEATSS_OE'))
# MS_F[, ID := unique(MS_A$item_id)]

n_methods <- 15
n_items <- unique(MS_A$item_id)
MS_F <- matrix(NA, nrow = length(n_items), ncol = 1 + n_methods) %>% data.table()
setnames(MS_F, c('ID',
                 "ETS","ARIMA", "THETA","NBEATS", 'NBEATSS low',"NBEATSS high" ,"Gradnorm","weighted gcossim",
                 "gcossim", "RW", "TARW high","TARW low","UW","NashMTL",
                 "AuxiNash"))
 MS_F[, ID := unique(MS_A$item_id)]


MS_SMAPE <- data.table(MS_F)
MS_SMAPC <- data.table(MS_F)
MS_RMSSE <- data.table(MS_F)
MS_RMSSC <- data.table(MS_F)

########################## Evaluation metrics and functions ##########################
SMAPE <- function(actuals, forecasts){
  200 * mean(abs(forecasts - actuals)/(abs(actuals) + abs(forecasts)))
}

SMAPC <- function(forecasts, forecasts_previous){
  200 * mean(abs(forecasts - forecasts_previous)/(abs(forecasts) + abs(forecasts_previous)), na.rm = T)
}

RMSSE <- function(actuals, forecasts, discount_factor){
  sqrt(mean((actuals - forecasts)^2/discount_factor))
}

RMSSC <- function(forecasts, forecasts_previous, discount_factor){
  sqrt(mean((forecasts - forecasts_previous)^2/discount_factor, na.rm = T))
}

MS_METHOD_PREP <- function(MS_METHOD){
  MS_METHOD <- melt.data.table(MS_METHOD,
                            id.vars = c('item_id', 'fc_origin'),
                            variable.name = 'forecast_horizon',
                            value.name = 'forecast')
  MS_METHOD[, forecast_horizon := as.numeric(forecast_horizon)]
  setorder(MS_METHOD, item_id, fc_origin, forecast_horizon)
  MS_METHOD[forecast < 0, forecast := 0]
  MS_METHOD[, forecast_previous := shift(forecast, (FL-1))]
  MS_METHOD[fc_origin == 1, forecast_previous := NA]
  MS_METHOD[forecast_horizon == FL, forecast_previous := NA]
  MS_METHOD[, actual := MS_A$value]
  MS_METHOD[, discount_factor := MS_A$discount_factor]
  return(MS_METHOD)
}

MS_METHOD_RESULTS <- function(MS_METHOD_PREP, METHOD){
  MS_METHOD_ITEM_ORIGIN <- MS_METHOD_PREP[, 
                                     list(SMAPE = SMAPE(actual, forecast),
                                          SMAPC = SMAPC(forecast, forecast_previous),
                                          RMSSE = RMSSE(actual, forecast, unique(discount_factor)),
                                          RMSSC = RMSSC(forecast, forecast_previous, unique(discount_factor))), 
                                     by = list(fc_origin, item_id)]
  MS_METHOD_ITEM <- MS_METHOD_ITEM_ORIGIN[,
                                          list(SMAPE = mean(SMAPE),
                                               SMAPC = mean(SMAPC, na.rm = T),
                                               RMSSE = mean(RMSSE),
                                               RMSSC = mean(RMSSC, na.rm = T)), 
                                          by = list(item_id)]
  MS_SMAPE[, (METHOD) := MS_METHOD_ITEM$SMAPE]
  MS_SMAPC[, (METHOD) := MS_METHOD_ITEM$SMAPC]
  MS_RMSSE[, (METHOD) := MS_METHOD_ITEM$RMSSE]
  MS_RMSSC[, (METHOD) := MS_METHOD_ITEM$RMSSC]
}
  

########################## Forecast methods ##########################


# ETS #####
#if (dataset_selection=='M4'){
#  MS_ETS <- fread(paste0(MS_datafolder_baselines, 'non_prob/', 
#                         paste0(dataset_selection, dataset_subset_indicator,
#                                '_ETS.csv')))
#} else {
#  MS_ETS <- fread(paste0(MS_datafolder_baselines, 
#                         paste0(dataset_selection, dataset_subset_indicator,
#                                '_ETS_probabilistic.csv')))
#}
if (dataset_selection=='M4'){
  MS_ETS <- fread(paste0("M4M_ETS.csv"))
}else {
  MS_ETS <- fread(paste0("M3M_ETS_probabilistic.csv"))
}
MS_ETS[, item_id := as.integer(as.factor(item_id))]
MS_ETS <- MS_ETS[type == 'mean_forecast']
MS_ETS[, type := NULL]
MS_ETS <- MS_METHOD_PREP(MS_ETS)
MS_METHOD_RESULTS(MS_ETS, 'ETS')

# ARIMA #####
#if (dataset_selection=='M4'){
#  MS_ARIMA <- fread(paste0(MS_datafolder_baselines, 'non_prob/',
#                           paste0(dataset_selection, dataset_subset_indicator,
#                                  '_arima.csv')))
#} else {
#  MS_ARIMA <- fread(paste0(MS_datafolder_baselines, 
#                           paste0(dataset_selection, dataset_subset_indicator,
#                                  '_arima_probabilistic.csv')))
#}
if (dataset_selection=='M4'){
  MS_ARIMA <- fread(paste0("M4M_arima.csv"))
}else {
  MS_ARIMA <- fread(paste0("M3M_arima_probabilistic.csv"))
}
MS_ARIMA[, item_id := as.integer(as.factor(item_id))]
MS_ARIMA <- MS_ARIMA[type == 'mean_forecast']
MS_ARIMA[, type := NULL]
MS_ARIMA <- MS_METHOD_PREP(MS_ARIMA)
MS_METHOD_RESULTS(MS_ARIMA, 'ARIMA')

# THETA #####
#if (dataset_selection=='M4'){
#  MS_THETA <- fread(paste0(MS_datafolder_baselines, 'non_prob/',
#                           paste0(dataset_selection, dataset_subset_indicator,
#                                  '_theta.csv')))
#} else {
#  MS_THETA <- fread(paste0(MS_datafolder_baselines, 
#                           paste0(dataset_selection, dataset_subset_indicator,
#                                  '_theta_probabilistic.csv')))
#}
if (dataset_selection=='M4'){
  MS_THETA <- fread(paste0("M4M_theta.csv"))
}else {
  MS_THETA <- fread(paste0("M3M_theta_probabilistic.csv"))
}
MS_THETA[, item_id := as.integer(as.factor(item_id))]
MS_THETA <- MS_THETA[type == 'mean_forecast']
MS_THETA[, type := NULL]
MS_THETA <- MS_METHOD_PREP(MS_THETA)
MS_METHOD_RESULTS(MS_THETA, 'THETA')

# NBEATS #####
#}
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_NBEATS_test/")
}else {
  datafolder <- paste0("M3_Monthly_NBEATS_test/")
}
MS_NBEATS_files <- list.files(datafolder)
MS_NBEATS_files <- paste0(datafolder, MS_NBEATS_files)
MS_NBEATS <- lapply(MS_NBEATS_files, fread) %>% rbindlist()
MS_NBEATS <- MS_NBEATS[type == 'forecast']
MS_NBEATS[, type := NULL]
MS_NBEATS <- MS_NBEATS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_NBEATS <- MS_METHOD_PREP(MS_NBEATS)
MS_METHOD_RESULTS(MS_NBEATS, 'NBEATS')

# NBEATSS #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_NBEATSS_low_test/")
}else {
  datafolder <- paste0("M3_Monthly_NBEATSS_low_test/")
}
MS_NBEATSS_files <- list.files(datafolder)
MS_NBEATSS_files <- paste0(datafolder, MS_NBEATSS_files)
MS_NBEATSS <- lapply(MS_NBEATSS_files, fread) %>% rbindlist()
MS_NBEATSS <- MS_NBEATSS[type == 'forecast']
MS_NBEATSS[, type := NULL]
MS_NBEATSS <- MS_NBEATSS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_NBEATSS <- MS_METHOD_PREP(MS_NBEATSS)
MS_METHOD_RESULTS(MS_NBEATSS, 'NBEATSS low')

# NBEATSS #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_NBEATSS_high_test/")
}else {
  datafolder <- paste0("M3_Monthly_NBEATSS_high_test/")
}
MS_NBEATSS_files <- list.files(datafolder)
MS_NBEATSS_files <- paste0(datafolder, MS_NBEATSS_files)
MS_NBEATSS <- lapply(MS_NBEATSS_files, fread) %>% rbindlist()
MS_NBEATSS <- MS_NBEATSS[type == 'forecast']
MS_NBEATSS[, type := NULL]
MS_NBEATSS <- MS_NBEATSS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_NBEATSS <- MS_METHOD_PREP(MS_NBEATSS)
MS_METHOD_RESULTS(MS_NBEATSS, 'NBEATSS high')





# random weighting #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_rw_test/")
}else {
  datafolder <- paste0("M3_Monthly_rw_test/")
}
MS_RW_files <- list.files(datafolder)
MS_RW_files <- paste0(datafolder, MS_RW_files)
MS_RW <- lapply(MS_RW_files, fread) %>% rbindlist()
MS_RW <- MS_RW[type == 'forecast']
MS_RW[, type := NULL]
MS_RW <- MS_RW[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_RW <- MS_METHOD_PREP(MS_RW)
MS_METHOD_RESULTS(MS_RW, 'RW')


# random weighting #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_TARW_high_test/")
#  datafolder <- paste0("My_TARW/")
  
}else {
  datafolder <- paste0("M3_Monthly_TARW_high_test/")

}
MS_RWCAP_files <- list.files(datafolder)
MS_RWCAP_files <- paste0(datafolder, MS_RWCAP_files)
MS_RWCAP <- lapply(MS_RWCAP_files, fread) %>% rbindlist()
MS_RWCAP <- MS_RWCAP[type == 'forecast']
MS_RWCAP[, type := NULL]
MS_RWCAP <- MS_RWCAP[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_RWCAP <- MS_METHOD_PREP(MS_RWCAP)
MS_METHOD_RESULTS(MS_RWCAP, 'TARW high')


# random weighting #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_TARW_low_test/")
  #  datafolder <- paste0("My_TARW/")
  
}else {
  datafolder <- paste0("M3_Monthly_TARW_low_test/")
  
}
MS_RWCAP_files <- list.files(datafolder)
MS_RWCAP_files <- paste0(datafolder, MS_RWCAP_files)
MS_RWCAP <- lapply(MS_RWCAP_files, fread) %>% rbindlist()
MS_RWCAP <- MS_RWCAP[type == 'forecast']
MS_RWCAP[, type := NULL]
MS_RWCAP <- MS_RWCAP[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_RWCAP <- MS_METHOD_PREP(MS_RWCAP)
MS_METHOD_RESULTS(MS_RWCAP, 'TARW low')


# Gradnorm #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_gradnorm_test/")
}else {
  datafolder <- paste0("M3_Monthly_gradnorm_test/")
}
MS_GRAD_files <- list.files(datafolder)
MS_GRAD_files <- paste0(datafolder, MS_GRAD_files)
MS_GRAD <- lapply(MS_GRAD_files, fread) %>% rbindlist()
MS_GRAD <- MS_GRAD[type == 'forecast']
MS_GRAD[, type := NULL]
MS_GRAD <- MS_GRAD[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_GRAD <- MS_METHOD_PREP(MS_GRAD)
MS_METHOD_RESULTS(MS_GRAD, 'Gradnorm')

# Unweighted gcosim #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_gcossim_test/")
}else {
  datafolder <- paste0("M3_Monthly_gcossim_test/")
}
MS_UWGCOS_files <- list.files(datafolder)
MS_UWGCOS_files <- paste0(datafolder, MS_UWGCOS_files)
MS_UWGCOS <- lapply(MS_UWGCOS_files, fread) %>% rbindlist()
MS_UWGCOS <- MS_UWGCOS[type == 'forecast']
MS_UWGCOS[, type := NULL]
MS_UWGCOS <- MS_UWGCOS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_UWGCOS <- MS_METHOD_PREP(MS_UWGCOS)
MS_METHOD_RESULTS(MS_UWGCOS, 'gcossim')

# weighted gcosim #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_weighted gcossim_test/")
}else {
  datafolder <- paste0("M3_Monthly_weighted gcossim_test/")
}
MS_WGCOS_files <- list.files(datafolder)
MS_WGCOS_files <- paste0(datafolder, MS_WGCOS_files)
MS_WGCOS <- lapply(MS_WGCOS_files, fread) %>% rbindlist()
MS_WGCOS <- MS_WGCOS[type == 'forecast']
MS_WGCOS[, type := NULL]
MS_WGCOS <- MS_WGCOS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_WGCOS <- MS_METHOD_PREP(MS_WGCOS)
MS_METHOD_RESULTS(MS_WGCOS, 'weighted gcossim')

if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_uw_test/")
}else {
  datafolder <- paste0("M3_Monthly_uw_test/")
}

MS_WGCOS_files <- list.files(datafolder)
MS_WGCOS_files <- paste0(datafolder, MS_WGCOS_files)
MS_WGCOS <- lapply(MS_WGCOS_files, fread) %>% rbindlist()
MS_WGCOS <- MS_WGCOS[type == 'forecast']
MS_WGCOS[, type := NULL]
MS_WGCOS <- MS_WGCOS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_WGCOS <- MS_METHOD_PREP(MS_WGCOS)
MS_METHOD_RESULTS(MS_WGCOS, 'UW')

if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_nashmtl_test/")
}else {
  datafolder <- paste0("M3_Monthly_nashmtl_test/")
}

MS_NashMTL_files <- list.files(datafolder)
MS_NashMTL_files <- paste0(datafolder,MS_NashMTL_files)
MS_NashMTL <- lapply(MS_NashMTL_files, fread) %>% rbindlist()
MS_NashMTL <- MS_NashMTL[type == 'forecast']
MS_NashMTL[, type := NULL]
MS_NashMTL <- MS_NashMTL[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_NashMTL <- MS_METHOD_PREP(MS_NashMTL)
MS_METHOD_RESULTS(MS_NashMTL, 'NashMTL')

if (dataset_selection=='M4'){
  datafolder <- paste0("M4_Monthly_auxinash_test/")
}else {
  datafolder <- paste0("M3_Monthly_auxinash_test/")
}

MS_AuxiNash_files <- list.files(datafolder)
MS_AuxiNash_files <- paste0(datafolder,MS_AuxiNash_files)
MS_AuxiNash <- lapply(MS_AuxiNash_files, fread) %>% rbindlist()
MS_AuxiNash <- MS_AuxiNash[type == 'forecast']
MS_AuxiNash[, type := NULL]
MS_AuxiNash <- MS_AuxiNash[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_AuxiNash <- MS_METHOD_PREP(MS_AuxiNash)
MS_METHOD_RESULTS(MS_AuxiNash, 'AuxiNash')


setnames(MS_SMAPE, 
         names(MS_SMAPE),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_SMAPC, 
         names(MS_SMAPC),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_RMSSE, 
         names(MS_RMSSE),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_RMSSC, 
         names(MS_RMSSC),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
MS_SMAPE %>% colMeans() %>% round(2)
MS_SMAPC %>% colMeans() %>% round(2)
MS_RMSSE %>% colMeans() %>% round(3)
MS_RMSSC %>% colMeans() %>% round(3)
# Step 1: Compute column means and round
sMAPE <- MS_SMAPE %>% colMeans() %>% round(2)
sMAPC <- MS_SMAPC %>% colMeans() %>% round(2)
RMSSE <- MS_RMSSE %>% colMeans() %>% round(3)
RMSSC <- MS_RMSSC %>% colMeans() %>% round(3)

# Step 2: Convert each to a one-row data frame
df_sMAPE <- as.data.frame(t(sMAPE))
df_sMAPC <- as.data.frame(t(sMAPC))
df_RMSSE <- as.data.frame(t(RMSSE))
df_RMSSC <- as.data.frame(t(RMSSC))

# Step 3: Write each to a separate CSV
write.csv(df_sMAPE, "tables/M3sMAPE.csv", row.names = FALSE)
write.csv(df_sMAPC, "tables/M3sMAPC.csv", row.names = FALSE)
write.csv(df_RMSSE, "tables/M3RMSSE.csv", row.names = FALSE)
write.csv(df_RMSSC, "tables/M3RMSSC.csv", row.names = FALSE)


########################## Statistical comparison ##########################
library(tsutils)

# Replace NA values by base model for rankings

# MS_SMAPE[, NBEATS_DROP := NBEATS]
# MS_SMAPC[, NBEATS_DROP := NBEATS]

# MS_SMAPE[, NBEATS_WD := NBEATS]
# MS_SMAPC[, NBEATS_WD := NBEATS]

# MS_SMAPE[, NBEATSS_WD := NBEATSS]
# MS_SMAPC[, NBEATSS_WD := NBEATSS]

# MS_RMSSE[, NBEATS_DROP := NBEATS]
# MS_RMSSC[, NBEATS_DROP := NBEATS]





setnames(MS_SMAPE, 
         names(MS_SMAPE),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_SMAPC, 
         names(MS_SMAPC),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_RMSSE, 
         names(MS_RMSSE),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))
setnames(MS_RMSSC, 
         names(MS_RMSSC),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S low',"N-BEATS-S high",
           "GradNorm","Weighted GCosSim","GCosSim", "RW", "TARW high","TARW low", 
           "UW","NashMTL",'AuxiNash'))

# Define width and height once
fig_width <- 6
fig_height <- 6

# Create directory if it doesn't exist
if (!dir.exists("figures")) dir.create("figures")

########################## Save Nemenyi plots to PDF ##########################

# sMAPE
pdf(paste0("figures/", dataset_selection, "_MCB_SMAPE.pdf"), width = fig_width, height = fig_height)
MCB_SMAPE <- nemenyi(as.matrix(MS_SMAPE[, 2:16]), plottype = 'vmcb')
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
dev.off()

# sMAPC
pdf(paste0("figures/", dataset_selection, "_MCB_SMAPC.pdf"), width = fig_width, height = fig_height)
MCB_SMAPC <- nemenyi(as.matrix(MS_SMAPC[, 2:16]), plottype = 'vmcb')
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
dev.off()

# RMSSE
pdf(paste0("figures/", dataset_selection, "_MCB_RMSSE.pdf"), width = fig_width, height = fig_height)
MCB_RMSSE <- nemenyi(as.matrix(MS_RMSSE[, 2:16]), plottype = 'vmcb')
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
dev.off()

# RMSSC
pdf(paste0("figures/", dataset_selection, "_MCB_RMSSC.pdf"), width = fig_width, height = fig_height)
MCB_RMSSC <- nemenyi(as.matrix(MS_RMSSC[, 2:16]), plottype = 'vmcb', cex = 1.2)
grid(nx = NULL, ny = NULL, col = "gray", lty = "dotted")
dev.off()

# print(MCB_SMAPE$intervals)
# 
# # Convert MCB results to data frames
# smape_results <- as.data.frame(MCB_SMAPE$means)
# print("l")
# print(smape_results)# Adjust this based on the actual structure of MCB_SMAPE
# smapc_results <- as.data.frame(MCB_SMAPC$means)
# print(smapc_results)# e.g., MCB_SMAPE$rank, MCB_SMAPE$p.value if available
# rmsse_results <- as.data.frame(MCB_RMSSE$means)
# rmssc_results <- as.data.frame(MCB_RMSSC$means)
# 
# smape_results <- smape_results[order(rownames(smape_results)), , drop = FALSE]
# smapc_results <- smapc_results[order(rownames(smapc_results)), , drop = FALSE]
# rmsse_results <- rmsse_results[order(rownames(rmsse_results)), , drop = FALSE]
# rmssc_results <- rmssc_results[order(rownames(rmssc_results)), , drop = FALSE]
# print(smape_results)
# print(smapc_results)
# 
# # Combine SMAPE, RMSSE, RMSSC into a data frame for exporting
# mcb_summary_data <- data.frame(
#   Method = rownames(smape_results),  # Assuming row names contain method names
#   SMAPE_Rank = smape_results,     # Replace 'V1' with the actual column name for ranks
#   SMAPC_Rank = smapc_results,
#   RMSSE_Rank = rmsse_results,
#   RMSSC_Rank = rmssc_results
# )
# 
# # Save as CSV for loading into Python
# 
# 
# # Conditional saving based on the value of M4
# if (dataset_selection == "M4") {
#   write.csv(mcb_summary_data, "M4_summary_data.csv", row.names = FALSE)
# } else {
#   write.csv(mcb_summary_data, "M3_summary_data.csv", row.names = FALSE)
# }
# 
# 
# 
# convert_intervals_to_list <- function(interval_matrix, metric_name) {
#   # Get the method names (column names in the interval matrix)
#   method_names <- colnames(interval_matrix)
#   
#   # Initialize an empty list to store intervals as lists in each cell
#   interval_lists <- lapply(method_names, function(method) {
#     # Each element in the list is an interval for a method (as a vector)
#     list(interval_matrix[, method])
#   })
#   
#   # Create a data frame with method names and list of intervals
#   df <- data.frame(Method = method_names, Intervals = I(interval_lists))
#   # Rename intervals column with metric name for identification
#   names(df)[2] <- metric_name
#   return(df)
# }
# 
# # Convert each MCB interval matrix into a data frame with lists
# smape_intervals_df <- convert_intervals_to_list(MCB_SMAPE$intervals, "SMAPE_Intervals")
# smapc_intervals_df <- convert_intervals_to_list(MCB_SMAPC$intervals, "SMAPC_Intervals")
# rmsse_intervals_df <- convert_intervals_to_list(MCB_RMSSE$intervals, "RMSSE_Intervals")
# rmssc_intervals_df <- convert_intervals_to_list(MCB_RMSSC$intervals, "RMSSC_Intervals")
# 
# print(smape_intervals_df)
# # Sort each data frame by Method
# smape_intervals_df <- smape_intervals_df[order(smape_intervals_df$Method), ]
# smapc_intervals_df <- smapc_intervals_df[order(smapc_intervals_df$Method), ]
# rmsse_intervals_df <- rmsse_intervals_df[order(rmsse_intervals_df$Method), ]
# rmssc_intervals_df <- rmssc_intervals_df[order(rmssc_intervals_df$Method), ]
# 
# # Merge all interval data frames by the 'Method' column to create one data frame
# mcb_intervals_summary <- Reduce(function(x, y) merge(x, y, by = "Method"), 
#                                 list(smape_intervals_df, smapc_intervals_df, rmsse_intervals_df, rmssc_intervals_df))
# 
# # Print the combined and sorted intervals summary
# print(mcb_intervals_summary)
# 
# # Save as a CSV (but note that CSV may not preserve the list structure well)
# # Conditional saving based on the value of M4
# if (dataset_selection == "M4") {
#   write.csv(mcb_intervals_summary, "M4_mcb_intervals_summary.csv", row.names = FALSE)
# } else {
#   write.csv(mcb_intervals_summary, "M3_mcb_intervals_summary.csv", row.names = FALSE)
# }
# png(file = paste0(MS_datafolder_NBEATSS, 
#                   dataset_selection, 
#                   dataset_subset_indicator,'/',
#                   dataset_selection, 
#                   dataset_subset_indicator,
#                   '_MCB_SMAPE.png'),
#     units = "in", width = 6, height = 4, res = 400)
# nemenyi(as.matrix(MS_SMAPE[,2:13]), plottype = 'vmcb')
# dev.off()
# 
# png(file = paste0(MS_datafolder_NBEATSS, 
#                   dataset_selection, 
#                   dataset_subset_indicator,'/',
#                   dataset_selection, 
#                   dataset_subset_indicator,
#                   '_MCB_SMAPC.png'),
#     units = "in", width = 6, height = 4, res = 400)
# nemenyi(as.matrix(MS_SMAPC[,2:13]), plottype = 'vmcb')
# dev.off()

# # Hourly
# MS_SMAPE[, NBEATS_DROP := NBEATS]
# MS_SMAPC[, NBEATS_DROP := NBEATS]
# MS_SMAPE[, NBEATS_TE := NULL]
# MS_SMAPC[, NBEATS_TE := NULL]
# MS_SMAPE[, NBEATSS_WD := NBEATSS]
# MS_SMAPC[, NBEATSS_WD := NBEATSS]
# MS_SMAPE[, NBEATSS_TE := NULL]
# MS_SMAPC[, NBEATSS_TE := NULL]
# 
# setnames(MS_SMAPE, 
#          names(MS_SMAPE),
#          c('ID',
#            'ETS','ARIMA','THETA',
#            'N-BEATS', 'N-BEATS weight decay', 'N-BEATS dropout', 'N-BEATS origin ensemble',
#            'N-BEATS-S', 'N-BEATS-S weight decay', 'N-BEATS-S origin ensemble'))
# setnames(MS_SMAPC, 
#          names(MS_SMAPC),
#          c('ID',
#            'ETS','ARIMA','THETA',
#            'N-BEATS', 'N-BEATS weight decay', 'N-BEATS dropout', 'N-BEATS origin ensemble',
#            'N-BEATS-S', 'N-BEATS-S weight decay', 'N-BEATS-S origin ensemble'))
# 
# png(file = paste0(MS_datafolder_NBEATSS, 
#                   dataset_selection, 
#                   dataset_subset_indicator,'/',
#                   dataset_selection, 
#                   dataset_subset_indicator,
#                   '_MCB_SMAPE.png'),
#     units = "in", width = 6, height = 4, res = 400)
# nemenyi(as.matrix(MS_SMAPE[,2:11]), plottype = 'vmcb')
# dev.off()
# 
# png(file = paste0(MS_datafolder_NBEATSS, 
#                   dataset_selection, 
#                   dataset_subset_indicator,'/',
#                   dataset_selection, 
#                   dataset_subset_indicator,
#                   '_MCB_SMAPC.png'),
#     units = "in", width = 6, height = 4, res = 400)
# nemenyi(as.matrix(MS_SMAPC[,2:11]), plottype = 'vmcb')
# dev.off()


