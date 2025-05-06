
# Set seed for reproducibility
set.seed(1000000)

getwd()


# Please set your working directory before running
#setwd("/PUT/YOUR/PROJECT/PATH/HERE")

# Create folders if missing
dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# Load/install required packages
packages <- c(
  "dplyr", "ggplot2", "tidyverse", "tidyr", "vegan", 
  "stringr", "circlize", "here", "viridis", "readr", "RColorBrewer"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# Set paths
data_path <- "data"
results_path <- "results"

# Load data
motu <- read_csv(file.path(data_path, "motu_filtered.csv"))
environmental <- read_csv(file.path(data_path, "geochemical.csv"))
Z <- read_csv(file.path(data_path, "Z.csv"))

# Drop unwanted OTU column if present
motu <- motu[, !names(motu) %in% c("...1", "otu23987")]

# Clean and merge Z
Z_clean <- Z %>%
  rename(biome = pelagicBiome_) %>%
  select(id, biome)

# Merge with environmental data
env <- left_join(environmental, Z_clean, by = "id")

# Clean ocean region if needed
env$OceanRegionClean <- sub(".*\\] (.*?) \\(.*", "\\1", env$Ocean.region)

# Merge with OTU table
merged <- inner_join(env, motu, by = "id")

# Identify OTU columns
motu_cols <- grep("^otu", names(merged), value = TRUE)

# Calculate richness (count of non-zero OTUs per sample)
merged$richness <- rowSums(merged[, motu_cols] > 0, na.rm = TRUE)

# Calculate Shannon and Evenness
otu_table <- merged[, motu_cols]
merged$shannon <- diversity(otu_table, index = "shannon")
merged$evenness <- ifelse(merged$richness > 0, 
                          merged$shannon / log(merged$richness), 
                          NA)


###############################################################################

# NMDS 

# Normalize OTU table (relative abundance)
otu_table <- otu_table[rowSums(otu_table) > 0, ]

otu_relabund <- sweep(otu_table, 1, rowSums(otu_table), "/")

# Bray-Curtis distance and NMDS
bray_dist <- vegdist(otu_relabund, method = "bray")
nmds <- metaMDS(bray_dist, k = 2, trymax = 100)
nmds_scores <- as.data.frame(scores(nmds))

# Add metadata to NMDS results
nmds_scores$polar <- merged$polar
nmds_scores$biome <- merged$biome
nmds_scores$Layer <- merged$Layer

# Define shared color palette (for both Layer and Biome)
shared_colors <- brewer.pal(4, "Set2")

# NMDS plot by Layer
nmds_plot_layer <- ggplot(nmds_scores, aes(x = NMDS1, y = NMDS2, color = Layer, shape = polar)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = shared_colors) +
  theme_minimal()

ggsave(file.path(results_path, "nmds_motu_layer.jpeg"), 
       plot = nmds_plot_layer, width = 10, height = 6, dpi = 300, units = "in")

# PERMANOVA by Layer
adonis_result <- adonis2(bray_dist ~ polar + Layer, data = merged, permutations = 999)
print(adonis_result)

# NMDS plot by Biome
nmds_plot_biome <- ggplot(nmds_scores, aes(x = NMDS1, y = NMDS2, color = biome, shape = polar)) +
  geom_point(size = 3, alpha = 0.7) +
  scale_color_manual(values = shared_colors) +
  theme_minimal()

ggsave(file.path(results_path, "nmds_motu_biome.jpeg"), 
       plot = nmds_plot_biome, width = 10, height = 6, dpi = 300, units = "in")

# PERMANOVA by Biome
adonis_biome <- adonis2(bray_dist ~ polar + biome, data = merged, permutations = 999)
print(adonis_biome)

# Stress output
cat("NMDS Stress:", round(nmds$stress, 3), "\n")

