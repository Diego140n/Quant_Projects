library(dplyr)
library(ggplot2)
library(factoextra)
library(dbscan)
library(scatterplot3d)
library(cluster)
library(tidyr)

set.seed(123)
years <- seq(1980, 2020, by=1)
n <- length(years) * 30

nba_data <- data.frame(
  Year = rep(years, each=30),
  Team = rep(paste("Team", 1:30), times=n/30),
  FGM = rnorm(n, mean=40, sd=3) - (0.005 * (rep(years, each=30) - 1980)),
  `3PM` = rnorm(n, mean=2, sd=1) + (0.1 * (rep(years, each=30) - 1980)),
  `3PA` = rnorm(n, mean=5, sd=2) + (0.3 * (rep(years, each=30) - 1980)),
  `FTM` = rnorm(n, mean=20, sd=2) - (0.01 * (rep(years, each=30) - 1980)),
  `FTA` = rnorm(n, mean=25, sd=3) - (0.2 * (rep(years, each=30) - 1980)),
  `TOV` = rnorm(n, mean=15, sd=2) - (0.05 * (rep(years, each=30) - 1980)),
  `DRB` = rnorm(n, mean=12, sd=2) - (0.4 * (rep(years, each=30) - 1980)),
  `PPG` = rnorm(n, mean=100, sd=5) - (0.2 * (rep(years, each=30) - 1980))
)

dim(nba_data)


standardize <- function(x) {
  return ((x - mean(x)) / sd(x))
}

data_standarized <- nba_data %>%
  mutate(across(FGM:PPG, standardize))

data_standarized <- data_standarized[, 3:10]


pca_result <- prcomp(data_standarized, scale = TRUE)

summary(pca_result)

pca_df <- as.data.frame(pca_result$x[, 1:2])


dbscan_result_24 <- dbscan(data_standarized, eps = 2.4, minPts = 10)
table(dbscan_result_24$cluster)


dbscan_result_45 <- dbscan(data_standarized, eps = 4.5, minPts = 10)
table(dbscan_result_45$cluster)



kmeans_result <- kmeans(pca_df[, 1:2], centers = 2, nstart = 25)

pca_df$Clusters <- as.factor(kmeans_result$cluster)

sil_scores <- sapply(2:9, function(k) {
  km <- kmeans(data_standarized, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(data_standarized))
  mean(ss[, 3])
})

silhouette_table <- data.frame(Clusters = 2:9, Silhouette_Score = sil_scores)
print(silhouette_table)


ggplot(pca_df, aes(x = PC1, y = PC2, color = Clusters)) +
  geom_point(size = 2, alpha = 0.7) +
  theme_minimal()

pca_df$PC3 <- pca_result$x[, 3]

scatterplot3d(pca_df$PC1, pca_df$PC2, pca_df$PC3, 
              color = as.numeric(pca_df$Clusters),
              main = "", pch = 16, xlab = "PC1", ylab = "PC2", zlab = "PC3")



data_standardize <- nba_data %>%
  mutate(across(FGM:PPG, standardize))


aggregated_data <- data_standardize %>%
  group_by(Year) %>%
  summarise(across(FGM:PPG, mean, na.rm = TRUE))


plot_data <- pivot_longer(aggregated_data, cols = FGM:PPG, 
                          names_to = "Attribute", values_to = "Value")


ggplot(plot_data, aes(x = Year, y = Value, color = Attribute)) +
  geom_line() +
  labs(title = "",
       x = "Year",
       y = "Normalized Value",
       color = "Attribute") +
  theme_minimal()

