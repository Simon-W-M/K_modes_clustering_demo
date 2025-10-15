# R Script for K-Modes clustering demo

# Load necessary libraries
library(klaR)
library(tidyverse)
library(janitor)
library(RKaggle)
library(cowplot)
library(gtsummary)
library(hopkins)
library(caret)


# details of this sample dtaset can be found here
# https://www.kaggle.com/datasets/prasad22/healthcare-dataset

data <- get_dataset("prasad22/healthcare-dataset")

# clean those column names
data <- clean_names(data)

# going to slect only a few columns for demo
# also going to convert them to factor
# (don't need to as Klar does this in background,
#  but for demo makes it easier to use summary and to plot)

# we could also one hot encode our data into binary variables

sample_data <- data |>
  dplyr::select(gender,
                blood_type,
                medical_condition,
                admission_type,
                medication) |>
  mutate(across(where(is.character), as.factor))

# lets look at our data
summary(sample_data)

# or
sample_data |> tbl_summary()

# this is a great dataset to work with as it looks as if all our features 
# are just spread across our patients - seamingly 'normally'?

# Hopkins Test!

# if we one hot encode we can run a hopkins test
one_hot <- dummyVars(~., data = sample_data)

sample_data_oh <- predict(one_hot, newdata = sample_data)

hopkins(  sample_data_oh,
          m = 500,
          method = "simple")

# this does return a 1, which I need to double check, but on face of it
# means data is highly clusterable
# this is usually used for numeric data and so need to understand if this
# is a feature of one hot encoding or not


# How many clusters?

# As per first manual example - I just aribtrarily chose the number of clusters
# this is an unsupervised method and so the number of clusters to split your data
# into is not an exact science, especially when dealing with categorical only data

# lets look at a common method of determining the number of clusters
# the elbow method

# Define the maximum number of clusters (K) to test
max_K <- 10

# Initialize a vector to store the total within-cluster dissimilarity (cost)
dissimilarity_scores <- numeric(max_K)

# --- 3. Iteratively Run K-Modes and Calculate Dissimilarity ---

# Loop through K values from 1 to max_K
for (k in 1:max_K) {
  # Run kmodes clustering
  # 'modes = k' sets the number of clusters
  # 'nstart = 5' ensures better initialization by running 5 times and picking the best result
  # 'iter.max = 10' sets the max iterations
  tryCatch({
    km_result <- kmodes(sample_data_oh, modes = k, iter.max = 10)
    
    # The dissimilarity metric (cost) is stored in the 'withindiff' component
    # We sum up the within-cluster dissimilarities for all clusters to get the total cost.
    dissimilarity_scores[k] <- sum(km_result$withindiff)
    
    cat(sprintf("K = %d, Dissimilarity = %.2f\n", k, dissimilarity_scores[k]))
    
  }, error = function(e) {
    # This handles potential errors, though kmodes is generally robust.
    # Often necessary if k is close to the number of unique points.
    dissimilarity_scores[k] <- NA
    cat(sprintf("Error running kmodes for K = %d: %s\n", k, e$message))
  })
}





# The elbow curve shows the decay of the dissimilarity scores
# what we are looking for is an elbow in the plot
# signifying where there is a significant change in the disimmilarity
# (There are other packages and methods that do this)

# Create a data frame for plotting
elbow_data <- data.frame(
  K = 1:max_K,
  Dissimilarity = dissimilarity_scores
)

# Remove NA values if any cluster failed to run
elbow_data <- na.omit(elbow_data)

# Generate the Elbow Plot
elbow_plot <- elbow_data  |> 
  ggplot() +
  aes(x = K, 
      y = Dissimilarity) +
  geom_line(color = "#007BFF", 
            linewidth = 1.2) +
  geom_point(color = "#DC3545", 
             size = 3, 
             shape = 21, 
             fill = "white", 
             stroke = 1.5) +
  geom_vline(xintercept = elbow_data$K[which.min(diff(diff(elbow_data$Dissimilarity))) + 2], 
             linetype = "dashed", 
             color = "darkgreen", 
             linewidth = 0.8) +
  labs(
    title = "Elbow Method for K-Modes Clustering",
    subtitle = paste0("Total Dissimilarity (Cost) across K = 1 to ", max_K),
    x = "Number of Clusters (K)",
    y = "Total Within-Cluster Dissimilarity (Cost)"
  ) +
  scale_x_continuous(breaks = seq(1, 
                                  max_K, 
                                  by = 1)) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", 
                              hjust = 0.5, 
                              size = 16),
    plot.subtitle = element_text(hjust = 0.5),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    panel.grid.major.x = element_line(linetype = "dotted"),
    panel.grid.minor.x = element_blank()
  )

elbow_plot

# Note on interpretation: The 'elbow' is the point where the rate of decrease 
# in dissimilarity sharply changes, indicating diminishing returns for adding 
# more clusters. In the plot, look for that sharp bend.

# So now lets run the clustering with a K of 5

k_5_clustering <- kmodes(sample_data_oh, modes = 5)


# lets now assign a cluster column back to our main dataset 
sample_data <- sample_data |>
  mutate(cluster = k_5_clustering$cluster)

# lets see how many patients are in each cluster
table(sample_data$cluster)

# or plop that in a graph
sample_data |>
  ggplot(aes(x = cluster)) + 
  geom_bar() + 
  coord_flip() +
  theme_minimal()

# just going to do a little house keeping as I like to order 
# my clusters from largest to smallest - adds a level of context

sample_data <- sample_data |>
  mutate(cluster_size = n(),
         .by = cluster) |>
  mutate(ordered_cluster = dense_rank(-cluster_size))


# lets knock up a quick function
# to plot a feature by cluster

plot_feature <- function (feature) {
  
  sample_data |>
    ggplot(aes(x = !!sym(feature))) + 
    geom_bar() + 
    facet_grid(~ordered_cluster) +
    coord_flip() +
    theme_minimal()
}

# try a few out
plot_feature('gender')
plot_feature('medication')


p1 <- plot_feature('gender')
p2 <- plot_feature('blood_type')
p3 <- plot_feature('medical_condition')
p4 <- plot_feature('admission_type')
p5 <- plot_feature('medication')

# but this is not about look at indiviual features,
# lets explore the interaction of features in our clusters

plot_grid(p1, p2, p3, p4, p5, ncol=1)

# in conclusion
# 
# the magic happens in just one line of code
# k-modes is the most simple and intuitive method of clustering
# it is not the best - but it is useful
# loads more methods that can produce far more robust results and allow
# you to work with mixed datasets of categories and values
# a whole world of clustering to explore
# 
# lets make reporting patient centric!