shh <- suppressMessages
shh(library(raster)); shh(library(rgdal)); shh(library(dismo));
shh(library(maps)); shh(library(maptools)); shh(library(mapdata))

# Get bioclimatic data
env <- suppressWarnings(shh(getData("worldclim", var = "bio", res = 10)))

# Get occurrence records for S. commersonii
species <- shh(gbif("Solanum", "commersonii"))

# Remove NAs and duplicate occurrences
species <- subset(species, !is.na(lat) & !is.na(lon))
dup_rm <- duplicated(species[, c("lat", "lon")])
species <- species[!dup_rm, ]

# Remove occurrence in ocean
species <- subset(species, lon > -90)

# Crop environment to reasonable extent
model_env <- crop(env,
                  extent(min(species$lon) - 10, max(species$lon) + 10,
                         min(species$lat) - 10, max(species$lat) + 10)
)

# Normalize bioclimatic features (rescale to [0, 1])
normalize <- function(x) {
  return( ((x - min(x, na.rm = T)) / (max(x, na.rm = T) - min(x, na.rm = T))) )
}
values(model_env) <- apply(values(model_env), 2, normalize)

# Save environemnt data to .csv
model_env_data <- data.frame(values(model_env))
model_env_data <- model_env_data[complete.cases(model_env_data), ]
write.csv(model_env_data, "Data/norm_env_data.csv", row.names = F)

# Extract bioclimatic features for our presences
presence_features <- extract(model_env, species[, c("lon", "lat")])

# Check for NAs and remove from both occurrence and climatic data
idx_rm <- which(is.na(presence_features[, 3]))
presence_features <- presence_features[-idx_rm, ]
species <- species[-idx_rm, ]

# Create random background points/occurrences
set.seed(1)
bg <- randomPoints(model_env, nrow(species))
absence_features <- extract(model_env, bg)

# Combine into one data frame
sdm_data <- data.frame(
  Presence = c(rep(1, nrow(presence_features)), rep(-1, nrow(absence_features)))
)
sdm_data$lon <- c(species$lon, bg[, 1])
sdm_data$lat <- c(species$lat, bg[, 2])
sdm_data <- cbind(sdm_data, rbind(presence_features, absence_features))

# Save SDM data to .csv
write.csv(sdm_data, "Data/S_commersonii_sdm_data.csv", row.names = F)
