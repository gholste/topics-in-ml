## Quick script to get trait data for some crop wild relatives from BIEN ##
## Author: Greg Holste                                                   ##
## Last modified: 9/28/19

library(BIEN)
library(mosaic)

options(stringsAsFactors = F)

# Read in CWR occurrence records (from previous research)
tmp <- read.csv("Data/CWR_species_by_cell.csv")

# Get species names, remove underscores
CWRs <- unique(tmp$Species)
CWRs <- gsub("_", " ", CWRs)
rm(tmp)

# Get trait data for these CWRs
data <- BIEN_trait_species(CWRs)
str(data)

# Which traits most common?
sort(tally(data$trait_name), increasing = T)

# Find species with 3 common continuous traits
traits_of_interest <- c("leaf area per leaf dry mass", "whole plant height", "seed mass")
s1 <- unique(with(data, scrubbed_species_binomial[trait_name == traits_of_interest[1]])); s1
s2 <- unique(with(data, scrubbed_species_binomial[trait_name == traits_of_interest[2]])); s2
s3 <- unique(with(data, scrubbed_species_binomial[trait_name == traits_of_interest[3]])); s3

species_of_interest <- intersect(s1, intersect(s2, s3)); length(species_of_interest)

# Subset trait data for these ~60 CWRs
mydata <- subset(data,
                 (scrubbed_species_binomial %in% species_of_interest &
                  trait_name %in% traits_of_interest),
                 select = c("scrubbed_species_binomial", "trait_name", "trait_value"))
mydata$trait_value <- as.numeric(mydata$trait_value)
mydata <- mydata[complete.cases(mydata), ]

# Often multiple measurements of a trait for each species, so reduce to mean
tmp1 <- aggregate(with(mydata, trait_value[trait_name == traits_of_interest[1]]),
          by = list(with(mydata, scrubbed_species_binomial[trait_name == traits_of_interest[1]])),
          FUN = mean)
tmp2 <- aggregate(with(mydata, trait_value[trait_name == traits_of_interest[2]]),
          by = list(with(mydata, scrubbed_species_binomial[trait_name == traits_of_interest[2]])),
          FUN = mean)
tmp3 <- aggregate(with(mydata, trait_value[trait_name == traits_of_interest[3]]),
          by = list(with(mydata, scrubbed_species_binomial[trait_name == traits_of_interest[3]])),
          FUN = mean)

df <- merge(tmp1, tmp2, by = "Group.1")
df <- merge(df, tmp3, by = "Group.1")
colnames(df) <- c("Species", "Leaf Area per Leaf Dry Mass",
                  "Whole Plant Height", "Seed Mass")

write.csv(df, "Data/CWR_trait_data.csv", row.names = F)
