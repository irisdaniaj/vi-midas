set.seed(1000)

getwd()
# Please set your working directory before running
#setwd("PUT/YOUR/PROJECT/PATH/HERE")



dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# install and load all required packages
packages <- c( "dplyr", "ggplot2", "maps", "tidyverse", "tidyr",  "stringr", "circlize", "here", "viridis", "readr",
                "gridExtra", "ggpubr"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

data_path <- "data"
results_path <- "results"

df <- read_csv(file.path(data_path, "best_per_variable_imputed_data.csv"))
tara <- read_csv(file.path(data_path, "map_tara.csv"))
y1 <- read_csv(file.path(data_path, "Y1_old.csv"))
z <-  read_csv(file.path(data_path, "Z.csv"))



df1 <- df[, c("id", "Latitude", "Longitude", "Ocean.region", "Layer")] %>%
  left_join(
    z %>% select(id, pelagicBiome_) %>% rename(Biome = pelagicBiome_),
    by = "id"
  )

# Clean the 'Ocean.region' column
df1$Ocean.region <- gsub("\\[.*?\\]", "", df1$Ocean.region)  # Remove the code inside brackets
df1$Ocean.region <- gsub("\\(.*?\\)", "", df1$Ocean.region)  # Remove the code inside parentheses
df1$Ocean.region <- trimws(df1$Ocean.region)  # Trim any extra whitespace

# Load world map data (from the 'maps' package)
world_map <- map_data("world")


################################################################################

# New Data 
# Create the plot - by Ocean Region 

plot <- ggplot() +
  # Plot the world map in light grey, with no borders (set lwd to 0)
  geom_map(data = world_map, map = world_map, aes(map_id = region), 
           fill = "lightgrey", color = NA, size = 0) +
  
  # Plot your sample points with custom colors for each ocean region
  geom_point(data = df1, aes(x = Longitude, y = Latitude, color = Ocean.region), size = 3) +
  
  # Custom color scale for each ocean region
  scale_color_manual(values = c(
    "North Atlantic Ocean" = "#FFB200",
    "Mediterranean Sea" = "#255F38",
    "Red Sea" = "#562B08",
    "Indian Ocean" = "#A5B68D", 
    "South Atlantic Ocean" = "#9B3922",
    "Southern Ocean" = "#E19898",
    "South Pacific Ocean" = "#87CEFA",
    "North Pacific Ocean" = "#20B2AA", 
    "Arctic Ocean" = "#003092"
  )) +
  
  # Map and point adjustments
  theme_minimal() +
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    panel.grid = element_blank(),  # Remove grid lines
    axis.text = element_blank(),   # Remove axis text (latitude/longitude labels)
    axis.ticks = element_blank(),  # Remove axis ticks (latitude/longitude ticks)
    axis.title = element_blank(),  # Remove axis title (latitude/longitude)
    plot.title = element_blank()   # Remove plot title
  ) +
  labs(color = "Ocean Region") +  # Set the legend title
  coord_fixed(ratio = 1, xlim = c(-180, 180), ylim = c(-90, 90))  # Ensure correct map aspect ratio

# Save the plot using ggsave
ggsave(file.path(results_path, "new_samples_by_ocean.jpeg"), 
       plot = plot,  # Pass the plot object to ggsave
       width = 10,           
       height = 7,            
       dpi = 300,             
       units = "in")         

################################################################################

# New Data 
# Create the plot - by Biome 

# Create the plot colored by Biome
plot <- ggplot() +
  # Plot the world map in light grey, with no borders
  geom_map(data = world_map, map = world_map, aes(map_id = region), 
           fill = "lightgrey", color = NA, size = 0) +
  
  # Plot sample points colored by Layer
  geom_point(data = df1, aes(x = Longitude, y = Latitude, color = Biome), size = 3) +
  
  # Use default color palette or viridis for categorical Layer variable
  scale_color_viridis_d(option = "D") +
  
  # Map and point adjustments
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    plot.title = element_blank()
  ) +
  labs(color = "Biome") +
  coord_fixed(ratio = 1, xlim = c(-180, 180), ylim = c(-90, 90))

# Save the plot
ggsave(file.path(results_path, "new_samples_by_biome.jpeg"), 
       plot = plot,
       width = 10,
       height = 7,
       dpi = 300,
       units = "in")

################################################################################

# Original Data 
# Create the plot - by Ocean Region

# Clean up the 'internal_sample_name' column in the 'tara' dataframe
tara$internal_sample_name <- gsub("_METAG$", "", tara$internal_sample_name)

# Merge 'tara' with 'df1' using left_join
df1 <- df1 %>%
  left_join(tara %>% select(internal_sample_name, sample_material), by = c("id" = "internal_sample_name"))

# Replace the hyphen with a period in df1$sample_material to match y1$Id
df1$sample_material <- gsub("-", ".", df1$sample_material)

# Check if each sample_material in df1 exists in y1$Id after the replacement
df1 <- df1 %>%
  mutate(old_data = ifelse(sample_material %in% y1$Id, 1, 0))  # 1 if match, 0 if no match

# Filter df1 to include only rows where old_data == 1
df1_filtered <- df1 %>% filter(old_data == 1)


plot1 <- ggplot() +
  # Plot the world map in light grey, with no borders (set lwd to 0)
  geom_map(data = world_map, map = world_map, aes(map_id = region), 
           fill = "lightgrey", color = NA, size = 0) +
  
  # Plot only the sample points where old_data == 1 (filtered data)
  geom_point(data = df1_filtered, aes(x = Longitude, y = Latitude, color = Ocean.region), size = 3) +
  
  # Custom color scale for each ocean region
  scale_color_manual(values = c(
    "North Atlantic Ocean" = "#FFB200",
    "Mediterranean Sea" = "#255F38",
    "Red Sea" = "#562B08",
    "Indian Ocean" = "#A5B68D", 
    "South Atlantic Ocean" = "#9B3922",
    "Southern Ocean" = "#E19898",
    "South Pacific Ocean" = "#87CEFA",
    "North Pacific Ocean" = "#20B2AA", 
    "Arctic Ocean" = "#003092"
  )) +
  
  # Map and point adjustments
  theme_minimal() +
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    panel.grid = element_blank(),  # Remove grid lines
    axis.text = element_blank(),   # Remove axis text (latitude/longitude labels)
    axis.ticks = element_blank(),  # Remove axis ticks (latitude/longitude ticks)
    axis.title = element_blank(),  # Remove axis title (latitude/longitude)
    plot.title = element_blank()   # Remove plot title
  ) +
  labs(color = "Ocean Region") +
  coord_fixed(ratio = 1, xlim = c(-180, 180), ylim = c(-90, 90))  # Ensure correct map aspect ratio and no unnecessary limits

# Save the plot
ggsave(file.path(results_path, "original_samples_by_ocean.jpeg"), 
       plot = plot1,
       width = 10,
       height = 7,
       dpi = 300,
       units = "in")


################################################################################

# Original Data 
# Create the plot - by Biome

plot1 <- ggplot() +
  geom_map(data = world_map, map = world_map, aes(map_id = region), 
           fill = "lightgrey", color = NA, size = 0) +
  
  geom_point(data = df1_filtered, aes(x = Longitude, y = Latitude, color = Biome), size = 3) +
  
  scale_color_viridis_d(option = "D") +  # Optional: switch to custom colors if preferred
  
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid = element_blank(),
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.title = element_blank(),
    plot.title = element_blank()
  ) +
  labs(color = "Biome") +
  coord_fixed(ratio = 1, xlim = c(-180, 180), ylim = c(-90, 90))

ggsave(file.path(results_path, "original_samples_by_biome.jpeg"), 
       plot = plot1,
       width = 10,
       height = 7,
       dpi = 300,
       units = "in")

################################################################################



