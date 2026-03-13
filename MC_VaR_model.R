library(dplyr)
library(tidyr)
library(MASS)

# Load positions
pos_data <- read.csv("positions.csv")

# Without currency hedges:
# pos_data <- pos_data[pos_data$AccountType == "TRADE", ]

pos_data <- pos_data %>%
  dplyr::select(ContractCode, InvestedValueInFundCurrency)

# Load market data
hist_data$adjprice <- hist_data$close + hist_data$adjoffset

hist_data <- hist_data %>%
  dplyr::select(date, contractcode, adjprice)

# Compute log returns
current_assets <- unique(pos_data$ContractCode)

returns_data <- hist_data %>%
  filter(contractcode %in% current_assets) %>%
  arrange(contractcode, date) %>%
  group_by(contractcode) %>%
  mutate(ret = log(adjprice / lag(adjprice))) %>%
  filter(!is.na(ret)) %>%
  ungroup() %>%
  dplyr::select(date, contractcode, ret)

returns_matrix_wide <- returns_data %>%
  pivot_wider(names_from = contractcode, values_from = ret) %>%
  drop_na()

matrix_values   <- as.matrix(returns_matrix_wide[, -1])
final_asset_order <- colnames(matrix_values)

# Aggregate exposures and align to returns matrix
pos_agg <- pos_data %>%
  group_by(ContractCode) %>%
  summarise(exposure = sum(InvestedValueInFundCurrency, na.rm = TRUE)) %>%
  ungroup()

portfolio_aligned <- data.frame(contractcode = final_asset_order) %>%
  left_join(pos_agg, by = c("contractcode" = "ContractCode"))

exposures <- portfolio_aligned$exposure

# Guard against missing exposures (assets in returns but not positions)
if (any(is.na(exposures))) {
  missing <- final_asset_order[is.na(exposures)]
  stop(paste(
    "The following assets have no position data and would corrupt the VaR:",
    paste(missing, collapse = ", ")
  ))
}

# EWMA covariance 
n_obs  <- nrow(matrix_values)
lambda <- 0.94                          # decay factor 

weights        <- lambda^((n_obs - 1):0)
weights        <- weights / sum(weights)
ew_mu          <- colSums(matrix_values * weights)
centered_rets  <- sweep(matrix_values, 2, ew_mu)
ew_cov         <- t(centered_rets) %*% diag(weights) %*% centered_rets

# Regularise the covariance matrix 
n_assets <- ncol(ew_cov)
ew_cov   <- ew_cov + diag(1e-6, n_assets)

#           Zero out the mean for VaR simulation
#           Daily expected returns are negligible vs. volatility and
#           including them slightly understates downside risk.

sim_mu <- rep(0, n_assets)

# Monte Carlo simulation
set.seed(123) 
n_sim <- 10000

sim_returns      <- mvrnorm(n = n_sim, mu = sim_mu, Sigma = ew_cov)
portfolio_pnl_sim <- sim_returns %*% exposures

# VaR
confidence_level <- 0.99
var_99 <- quantile(portfolio_pnl_sim, 1 - confidence_level)

print(paste("1-Day Monte Carlo VaR (99%):", round(var_99, 2)))

