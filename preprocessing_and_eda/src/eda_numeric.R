set.seed(100000)

getwd()
# Please set your working directory before running
#setwd("PUT/YOUR/PROJECT/PATH/HERE")


dir.create("data", showWarnings = FALSE)
dir.create("results", showWarnings = FALSE)

# install and load all required packages
packages <- c( "dplyr", "ggplot2", "tidyverse", "tidyr",  "stringr", "circlize", "here", "viridis", "readr",
               "gridExtra", "ggpubr", "corrplot", 'patchwork'
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

data_path <- "data"
results_path <- "results"

motu <- read_csv(file.path(data_path, "motu_filtered.csv"))
env <- read_csv(file.path(data_path, "geochemical.csv"))
climatology <- read_csv(file.path(data_path, "climatology.csv"))
sample <-  read_csv(file.path(data_path, "sample_data.csv"))

################################################################################

# Missingness Percentage

merged_df <- left_join(sample, climatology, by = "id")

selected_columns <- c("Temperature", "Salinity", "Oxygen", 
                      "NO2", "PO4", "NO2NO3", "Si", "SST", 
                      "ChlorophyllA", "Carbon.total", "Fluorescence", "Chl", "PAR", 
                      "mld", "wind", "EKE", "Rrs490", "Rrs510", "Rrs555")

smaller_df <- merged_df %>% select(all_of(selected_columns))


# calculating missing percentage 
missing_pct <- smaller_df %>%
  summarise(across(everything(), ~mean(is.na(.)) * 100)) %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "MissingPercent") %>%
  mutate(Variable = fct_reorder(Variable, MissingPercent, .desc = TRUE)) 

missing <- ggplot(missing_pct, aes(x = Variable, y = MissingPercent)) +
  geom_bar(stat = "identity", fill = "#377EB8") +
  labs(
    #title = "Missing Data Percentage by Variable",
    x = NULL,
    y = "Missing Data (%)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(results_path, "missing_data_percent.jpeg"),
       plot = missing,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")


################################################################################


# Environmental covariates boxplots : Polar vs Non-polar

X <- env %>%
  select("id","Temperature", "Salinity", "Oxygen", 
         "NO2", "PO4", "NO2NO3", "Si", "SST", 
         "ChlorophyllA", "Carbon.total", "Layer", "polar", "Ocean.region")

# Ensure polar is a clean string- keep "polar" and "non polar"
X$polar <- as.character(X$polar)
X$polar <- ifelse(X$polar %in% c("polar", "non polar"), X$polar, NA)

# Define variable names with units
var_labels <- c(
  "Temperature" = "Temperature (°C)",
  "Salinity" = "Salinity (PSU)",
  "Oxygen" = "Oxygen (µmol/kg)",
  "NO2" = "NO2 (µmol/L)",
  "PO4" = "PO4 (µmol/L)",
  "NO2NO3" = "NO2NO3 (µmol/L)",
  "Si" = "Silicate (µmol/L)",
  "SST" = "SST (°C)",
  "ChlorophyllA" = "ChlorophyllA (mg/m³)",
  "Carbon.total" = "Carbon.total (µmol/L)"
)

# Define the variable order
var_order <- names(var_labels)

# Pivot and plot
X_long <- X %>%
  select(polar, all_of(var_order)) %>%
  pivot_longer(cols = all_of(var_order), names_to = "Variable", values_to = "Measurement") %>%
  mutate(
    Variable = factor(Variable, levels = var_order),
    polar = factor(polar, levels = c("non polar", "polar"))  # keep space version
  )


p <- ggplot(X_long, aes(x = polar, y = Measurement, fill = polar)) +
  geom_boxplot() +
  facet_wrap2(
    ~Variable,
    scales = "free_y",
    nrow = 2,
    labeller = labeller(Variable = var_labels),
    strip = strip_vanilla(),       # keep default facet look
    axes = "all"                   # this repeats axis ticks and labels!
  ) +
  scale_fill_manual(values = c("polar" = "#4DAF4A", "non polar" = "#377EB8")) +
  theme_minimal() +
  theme(
    legend.position = "none",                  # remove legend
    strip.text = element_text(face = "bold", size = 12),  # bold facet titles
    axis.title = element_blank(),              # no x/y labels
    plot.title = element_blank()               # no title
  )

ggsave(file.path(results_path, "env_covariates_boxplots.jpeg"), 
       plot = p,
       width = 10,
       height = 6,
       dpi = 300,
       units = "in")

################################################################################

# Correlation matrix : pre (complete case) and post-imputation

# 1. Variable list
vars <- c("Temperature", "Salinity", "Oxygen", "NO2", "PO4", "NO2NO3", 
          "Si", "SST", "ChlorophyllA", "Carbon.total", "Fluorescence", 
          "Chl", "PAR", "mld", "wind", "EKE", "Rrs490", "Rrs510", "Rrs555")

# 2. Extract from env
env_selected <- env %>%
  select(all_of(vars)) %>%
  mutate(across(everything(), as.numeric))

# 3. Merge climatology + sample on ID, then select same rows from env
clim_sample_merged <- inner_join(climatology, sample, by = "id")

clim_sample_combined <- env %>%
  filter(id %in% clim_sample_merged$id) %>%
  select(all_of(vars)) %>%
  mutate(across(everything(), as.numeric))

# 4. correlation before imputatuon
corr_before_impute = cor(smaller_df, use = "pairwise.complete.obs")
corrplot(corr_before_impute, method = 'color', tl.col = "black")

# Correlation before imputation
png(file.path(results_path, "correlation_before_imputation.png"), width = 1000, height = 1000)
corrplot(corr_before_impute, method = 'color', tl.col = "black")
dev.off()


# 5. correlation after imputation
env_corr <- env %>% select(all_of(selected_columns))
corr_after_impute = cor (env_corr)
corrplot(corr_after_impute, method = 'color', tl.col = "black")

# Correlation after imputation
png(file.path(results_path, "correlation_after_imputation.png"), width = 1000, height = 1000)
corrplot(corr_after_impute, method = 'color', tl.col = "black")
dev.off()

