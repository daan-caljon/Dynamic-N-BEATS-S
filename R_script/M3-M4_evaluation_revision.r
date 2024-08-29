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
setwd(dirname(file.choose()))
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
  
  #MS_actuals_train <- fread(paste0('/Users/jentevanbelle/Documents/NBeats stability/M4DataSet/', dataset_subset, '-train.csv'))
  #MS_actuals_test <- fread(paste0('/Users/jentevanbelle/Documents/NBeats stability/M4DataSet/', dataset_subset, '-test.csv'))
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

n_methods <- 11
n_items <- unique(MS_A$item_id)
MS_F <- matrix(NA, nrow = length(n_items), ncol = 1 + n_methods) %>% data.table()
setnames(MS_F, c('ID',
                 "ETS","ARIMA", "THETA","NBEATS", 'NBEATSS',"Gradnorm","weighted gcossim",
                 "unweighted gcossim", "random weighting", "random weighting cap","UW"))
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
  datafolder <- paste0("M4NBEATS/")
}else {
  datafolder <- paste0("M3NBEATS/")
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
  datafolder <- paste0("M4NBEATSS/")
}else {
  datafolder <- paste0("M3NBEATSS/")
}
MS_NBEATSS_files <- list.files(datafolder)
MS_NBEATSS_files <- paste0(datafolder, MS_NBEATSS_files)
MS_NBEATSS <- lapply(MS_NBEATSS_files, fread) %>% rbindlist()
MS_NBEATSS <- MS_NBEATSS[type == 'forecast']
MS_NBEATSS[, type := NULL]
MS_NBEATSS <- MS_NBEATSS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_NBEATSS <- MS_METHOD_PREP(MS_NBEATSS)
MS_METHOD_RESULTS(MS_NBEATSS, 'NBEATSS')


# random weighting #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4Random weighting/")
}else {
  datafolder <- paste0("M3Random weighting/")
}
MS_RW_files <- list.files(datafolder)
MS_RW_files <- paste0(datafolder, MS_RW_files)
MS_RW <- lapply(MS_RW_files, fread) %>% rbindlist()
MS_RW <- MS_RW[type == 'forecast']
MS_RW[, type := NULL]
MS_RW <- MS_RW[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_RW <- MS_METHOD_PREP(MS_RW)
MS_METHOD_RESULTS(MS_RW, 'random weighting')


# random weighting #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4Random weighting cap/")
}else {
  datafolder <- paste0("M3Random weighting cap/")
}
MS_RWCAP_files <- list.files(datafolder)
MS_RWCAP_files <- paste0(datafolder, MS_RWCAP_files)
MS_RWCAP <- lapply(MS_RWCAP_files, fread) %>% rbindlist()
MS_RWCAP <- MS_RWCAP[type == 'forecast']
MS_RWCAP[, type := NULL]
MS_RWCAP <- MS_RWCAP[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_RWCAP <- MS_METHOD_PREP(MS_RWCAP)
MS_METHOD_RESULTS(MS_RWCAP, 'random weighting cap')

# Gradnorm #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4gradnorm/")
}else {
  datafolder <- paste0("M3 gradnorm/")
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
  datafolder <- paste0("M4unweighted gcossim/")
}else {
  datafolder <- paste0("M3 unweighted gcossim/")
}
MS_UWGCOS_files <- list.files(datafolder)
MS_UWGCOS_files <- paste0(datafolder, MS_UWGCOS_files)
MS_UWGCOS <- lapply(MS_UWGCOS_files, fread) %>% rbindlist()
MS_UWGCOS <- MS_UWGCOS[type == 'forecast']
MS_UWGCOS[, type := NULL]
MS_UWGCOS <- MS_UWGCOS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_UWGCOS <- MS_METHOD_PREP(MS_UWGCOS)
MS_METHOD_RESULTS(MS_UWGCOS, 'unweighted gcossim')

# weighted gcosim #####
if (dataset_selection=='M4'){
  datafolder <- paste0("M4weighted gcossim/")
}else {
  datafolder <- paste0("M3weighted gcossim/")
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
  datafolder <- paste0("M4UW/")
}else {
  datafolder <- paste0("M3UW/")
}

MS_WGCOS_files <- list.files(datafolder)
MS_WGCOS_files <- paste0(datafolder, MS_WGCOS_files)
MS_WGCOS <- lapply(MS_WGCOS_files, fread) %>% rbindlist()
MS_WGCOS <- MS_WGCOS[type == 'forecast']
MS_WGCOS[, type := NULL]
MS_WGCOS <- MS_WGCOS[, lapply(.SD, median) , by = list(item_id, fc_origin), .SDcols = as.character(c(1:FL))]
MS_WGCOS <- MS_METHOD_PREP(MS_WGCOS)
MS_METHOD_RESULTS(MS_WGCOS, 'UW')


MS_SMAPE %>% colMeans() %>% round(2)
MS_SMAPC %>% colMeans() %>% round(2)
# MS_RMSSE %>% colMeans() %>% round(3)
# MS_RMSSC %>% colMeans() %>% round(3)

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
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S',"GradNorm","Weighted GCosSim",
           "GCosSim", "RW", "TARW","UW"))
setnames(MS_SMAPC, 
         names(MS_SMAPC),
         c('ID',
           "ETS","ARIMA", "THETA","N-BEATS", 'N-BEATS-S',"GradNorm","Weighted GCosSim",
           "GCosSim", "RW", "TARW","UW"))

MCB_SMAPE <- nemenyi(as.matrix(MS_SMAPE[,2:12]), plottype = 'vmcb')
MCB_SMAPC <- nemenyi(as.matrix(MS_SMAPC[,2:12]), plottype = 'vmcb')
# nemenyi(as.matrix(MS_RMSSE[,2:13]), plottype = 'vmcb')
# nemenyi(as.matrix(MS_RMSSC[,2:13]), plottype = 'vmcb')

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


