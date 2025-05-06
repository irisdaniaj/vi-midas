set.seed(111)
getwd()

# Please set your working directory before running
#setwd("PUT/YOUR/PROJECT/PATH/HERE")


dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# install and load all required packages
packages <- c( "dplyr", "ggplot2", "readr", "tidyverse", "tidyr",  
               "stringr", "circlize", "here", "viridis", "readr",
               "stringr", "vegan", "RColorBrewer"
               
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

data_path <- "data"
results_path <- "results"

motu <- read_csv(file.path(data_path, "count_table.csv"))
env <- read_csv(file.path(data_path, "geochemical.csv"))
Z <- read_csv(file.path(data_path, "Z.csv"))


# Drop the `...1` column
motu <- motu[, names(motu) != "...1"]

# Check if 'otu23987' exists in the column names of count_df
if ("otu23987" %in% colnames(motu)) {
  # Drop the 'otu23987' column
  count_df <- motu[, !colnames(motu) %in% "otu23987"]
}

# Clean and merge Z
Z_clean <- Z %>%
  rename(biome = pelagicBiome_) %>%  # Rename the column
  select(id, biome)  # Keep only 'id' and 'biome'

# Merge with env on 'id'
env <- left_join(envv, Z_clean, by = "id")

# Clean up Ocean Region for plotting
env$OceanRegionClean <- sub(".*\\] (.*?) \\(.*", "\\1", env$Ocean.region)

# Ensure polar is a clean string- keep "polar" and "non polar"
env$polar <- as.character(X$polar)
env$polar <- ifelse(X$polar %in% c("polar", "non polar"), X$polar, NA)

#View(env)
intersect(env$id, motu$id)
length(intersect(env$id, motu$id))

merged <- inner_join(env, motu, by = "id")
dim(merged)
dim(motu)
dim(env)

motu_cols <- grep("^otu", names(merged), value = TRUE)



# Sanity check: should return hundreds/thousands of OTU columns
length(motu_cols)  # Should be > 0

# Now calculate richness
merged$richness <- rowSums(merged[, motu_cols] > 0, na.rm = TRUE)
summary(merged$richness)

# Relative abundance matrix (skip 'id' column)
otu_matrix <- merged[, motu_cols]
otu_rel <- otu_matrix / rowSums(otu_matrix)

# Shannon and richness
library(vegan)
merged$shannon <- diversity(otu_rel, index = "shannon")
merged$evenness <- merged$shannon / log(merged$richness)

# Define custom colors for your depth layers
layer_colors <- c(
  "DCM" = "#4DAF4A",   # green
  "MES" = "#1B9E77",   # teal green
  "MIX" = "#377EB8",   # blue
  "SRF" = "#74ADD1"    # light blue
)


# Set factor levels to control the order
merged$Layer <- factor(merged$Layer, levels = c("SRF", "DCM", "MIX", "MES"))


# Richness Plot by Depth Layer
# Plot
rich_dl <- ggplot(merged, aes(x = polar, y = richness, fill = Layer)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = layer_colors) +
  theme(
    legend.position = "top", 
    axis.text.x = element_text(angle = 45, hjust = 0.5),
    plot.title = element_text(hjust = 0.5)
  )
ggsave(file.path(results_path, "rich_dl.jpeg"), 
       plot = rich_dl,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")

# Richness Plot by Biome
rich_biome <- ggplot(merged, aes(x = polar, y = richness, fill = biome)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_brewer(palette = "Blues") + 
  theme(
    legend.position = "top", 
    axis.text.x = element_text(angle = 45, hjust = 0.5),
    plot.title = element_text(hjust = 0.5)
)
ggsave(file.path(results_path, "rich_biome.jpeg"), 
       plot = rich_biome,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")

even_dl <- ggplot(merged, aes(x = polar, y = evenness, fill = Layer)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_manual(values = layer_colors) +
  labs(y = "Evenness") +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 0.5)
  )

ggsave(file.path(results_path, "evenness_dl.jpeg"), 
       plot = even_dl,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")

even_biome <- ggplot(merged, aes(x = polar, y = evenness, fill = biome)) +
  geom_boxplot() +
  theme_minimal() +
  scale_fill_brewer(palette = "Blues") +  
  labs(y = "Evenness") +
  theme(
    legend.position = "top",
    axis.text.x = element_text(angle = 45, hjust = 0.5)
  )

ggsave(file.path(results_path, "evenness_biome.jpeg"), 
       plot = even_biome,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")

