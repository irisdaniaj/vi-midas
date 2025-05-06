set.seed(888)

getwd()
# Please set your working directory before running
#setwd("PUT/YOUR/PROJECT/PATH/HERE")


dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)


# install and load all required packages
packages <- c(
  "dplyr", "ggplot2", "readr", "pheatmap", "ComplexHeatmap",
  "vegan", "RColorBrewer", "circlize"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

data_path <- "data"
results_path <- "results"

otu <- read_csv(file.path(data_path, "Y1.csv"))
metadata <- read_csv(file.path(data_path, "Z.csv"))

# Set the working directory manually
if (!file.exists("data/Y1.csv")) {
  stop("Working directory is not set correctly. Please use setwd() to point to the project root.")
}

# Sample IDs as rownames
otu <- as.data.frame(otu)
metadata <- as.data.frame(metadata)
rownames(otu) <- rownames(metadata) <- metadata$id
all(rownames(otu) == rownames(metadata))


#  Log-transform OTU counts 
otu_clean <- otu[, sapply(otu, is.numeric)]
otu_log <- log1p(otu_clean) #log(1+W) abundance


metadata$Layer <- metadata$EnvFeature
row_anno <- data.frame(
  Layer = metadata$Layer,
  Biome = metadata$pelagicBiome_
)
rownames(row_anno) <- rownames(otu_log)

# define custom annotation colors
layer_colors <- c("SRF" = "#377EB8", "DCM" = "#4DAF4A", "MES" = "#FF7F00", "MIX" = "#984EA3")
biome_colors <- c(
  "Polar Biome" = "#CAB2D6",
  "Coastal Biome" = "#B15928",
  "Westerlies Biome" = "#A6CEE3",
  "Trades Biome" = "#FB9A99"
)

ha <- HeatmapAnnotation(
  df = row_anno,
  col = list(
    Layer = layer_colors,
    Biome = biome_colors
  ),
  which = "row"
)

otu_log_filtered <- otu_log


# Bray–Curtis dissimilarity on log-transformed mOTU table
bray_dist <- vegdist(otu_log_filtered, method = "bray")

# Hierarchical clustering (average linkage)
sample_clust <- hclust(bray_dist, method = "average")

# Create heatmap with custom sample clustering
ht <- Heatmap(
  as.matrix(otu_log_filtered),
  name = "Log(1 + W) Abundance",
  cluster_rows = as.dendrogram(sample_clust),  # << this is the key change
  cluster_columns = TRUE,
  show_row_names = FALSE,
  show_column_names = FALSE,
  left_annotation = ha,
  col = colorRamp2(c(0, 2, 4, 6, 8, 10), c("blue", "cyan", "green", "yellow", "orange", "red")),
  column_title = "mOTUs",
  row_title = "Samples"
)


jpeg(
  filename = file.path(results_path, "motu_heatmap.jpeg"),
  width = 2500,      # pixels
  height = 2000,     # pixels
  res = 300          # dots per inch (DPI)
)
draw(ht) # must be inside jpeg() context
dev.off() # closes and writes the file (image)

file.exists(file.path(results_path, "motu_heatmap.jpeg")) 
