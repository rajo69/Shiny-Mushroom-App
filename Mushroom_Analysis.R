# Install Packages
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(plotly)) install.packages("plotly")
if (!require(DT)) install.packages("DT")
if (!require(rpart)) install.packages("rpart")
if (!require(randomForest)) install.packages("randomForest")
if (!require(rpart.plot)) install.packages("rpart.plot")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")

# Import Libraries
library(ggplot2)
library(plotly)
library(DT)
library(rpart)
library(randomForest)
library(rpart.plot)
library(caret)
library(pROC)

# Load the mushroom dataset
mushroom_data <- read.csv("mushrooms.csv")
mushroom_data <- data.frame(lapply(mushroom_data, as.factor))  # Convert all columns to factors

# Split the dataset into training and testing sets
set.seed(1234)
n <- nrow(mushroom_data)
smp_size <- floor(0.80 * n)
train_ind <- sample(seq_len(n), size = smp_size)
mushrooms_train <- mushroom_data[train_ind, ]
mushrooms_test <- mushroom_data[-train_ind, ]

# Pick a feature
feature <- 'Odor'

# Extract data for the feaure
feature_data <- data.frame(Feature = mushroom_data[[feature]], Edible = mushroom_data$Edible)

# Create the ggplot
p <- ggplot(feature_data, aes(x = Feature, fill = Edible)) +
  geom_bar(position = "dodge") +
  labs(title = paste("Distribution of", feature, "by Edibility"),
       x = feature, y = "Count", fill = "Edibility") +
  theme_minimal()

# Convert ggplot to an interactive plotly plot
ggplotly(p)

# Decision Tree Model
decision_tree <- rpart(Edible ~ ., data = mushrooms_train, method = "class")
save(decision_tree, file = 'dt_rpart.Rdata')

# Plot vanilla decision tree
rpart.plot::rpart.plot(decision_tree)

# Tuning Decision Tree with cp values
results <- data.frame(cp = double(), Accuracy = double())
for (cp in seq(0.0001, 0.01, by = 0.0001)) {
  model <- rpart(Edible ~ ., data = mushrooms_train, method = "class", control = rpart.control(cp = cp))
  pred <- predict(model, mushrooms_test, type = "class")
  accuracy <- sum(pred == mushrooms_test$Edible) / nrow(mushrooms_test)
  results <- rbind(results, data.frame(cp = cp, Accuracy = accuracy))
}

write.csv(results, "tuning_result_dt.csv")

p <- ggplot(results, aes(x = cp, y = Accuracy)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Effect of cp Parameter on Decision Tree Accuracy", x = "Complexity Parameter (cp)", y = "Accuracy")
ggplotly(p)

# Tuned Decision Tree Model
decision_tree_tuned <- rpart(Edible ~ ., data = mushrooms_train, method = "class", control = rpart.control(cp = 0.0022))
save(decision_tree_tuned, file = 'dt_rpart_tuned.Rdata')
rpart.plot::rpart.plot(decision_tree_tuned)

# Random Forest Model
rf_model <- randomForest(Edible ~ ., data = mushrooms_train, ntree = 100)
save(rf_model, file = 'rf_rf.Rdata')

# Tuning Random Forest with ntree and mtry values
set.seed(12345)
results_rf <- data.frame(ntree = integer(), mtry = integer(), Accuracy = double())
for (ntree in seq(100, 500, by = 100)) {
  for (mtry in 1:5) {
    model_rf <- randomForest(Edible ~ ., data = mushrooms_train, ntree = ntree, mtry = mtry)
    pred_rf <- predict(model_rf, mushrooms_test)
    accuracy <- sum(pred_rf == mushrooms_test$Edible) / nrow(mushrooms_test)
    results_rf <- rbind(results_rf, data.frame(ntree = ntree, mtry = mtry, Accuracy = accuracy))
   }
}

write.csv(results_rf, "tuning_result_rf.csv")

p <- ggplot(results_rf, aes(x = ntree, y = Accuracy, color = factor(mtry))) +
  geom_line() +
  theme_minimal() +
  labs(title = "Effect of ntree and mtry on Random Forest Accuracy", x = "Number of Trees (ntree)", y = "Accuracy", color = "mtry")
ggplotly(p)

# Tuned Random Forest Model
rf_model_tuned <- randomForest(Edible ~ ., data = mushrooms_train, ntree = 100, mtry = 2)
save(rf_model_tuned, file = 'rf_rf_tuned.Rdata')

# Random Forest Feature Importance
rf_importance <- importance(rf_model_tuned)
rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])
rf_importance_df <- rf_importance_df[order(rf_importance_df$Importance, decreasing = TRUE), ]
  
p <- ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Random Forest Feature Importance", x = "Feature", y = "Importance")
ggplotly(p)

# Random Forest ROC Curve
rf_prob <- predict(rf_model_tuned, mushrooms_test, type = "prob")
roc_rf <- roc(mushrooms_test$Edible, rf_prob[, 2], levels = rev(levels(mushrooms_test$Edible)))
plot(roc_rf, col = "blue", main = "ROC Curve for Random Forest Model")
abline(a = 0, b = 1, lty = 2, col = "red")

# Model Comparison
dt_predictions <- predict(decision_tree_tuned, mushrooms_test, type = "class")
rf_predictions <- predict(rf_model_tuned, mushrooms_test)

tree_accuracy <- sum(dt_predictions == mushrooms_test$Edible) / nrow(mushrooms_test)
rf_accuracy <- sum(rf_predictions == mushrooms_test$Edible) / nrow(mushrooms_test)
cat("Tuned Decision Tree Accuracy:", tree_accuracy, "\n")
cat("Tuned Random Forest Accuracy:", rf_accuracy, "\n")

# Confusion Matrices
# Decision Tree Confusion Matrix
dt_cm <- confusionMatrix(dt_predictions, mushrooms_test$Edible)
dt_cm_table <- as.table(dt_cm$table)
dt_cm_df <- as.data.frame(dt_cm_table)
  
dt_plot <- ggplot(dt_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  ggtitle("Decision Tree Confusion Matrix") +
  theme_minimal()

# Random Forest Confusion Matrix
rf_cm <- confusionMatrix(rf_predictions, mushrooms_test$Edible)
rf_cm_table <- as.table(rf_cm$table)
rf_cm_df <- as.data.frame(rf_cm_table)
  
rf_plot <- ggplot(rf_cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white") +
  ggtitle("Random Forest Confusion Matrix") +
  theme_minimal()

# Cross-Validation Metrics Comparison
set.seed(1234)
train_control <- trainControl(method = "cv", number = 10)
dt_model_cv <- train(Edible ~ ., data = mushroom_data, method = "rpart", trControl = train_control, tuneGrid = data.frame(cp = 0.0022))
rf_model_cv <- train(Edible ~ ., data = mushroom_data, method = "rf", trControl = train_control, tuneGrid = data.frame(mtry = 2), ntree = 100)
save(dt_model_cv, file = 'dt_caret_cv.Rdata')
save(rf_model_cv, file = 'rf_caret_cv.Rdata')
  
dt_accuracies <- dt_model_cv$resample$Accuracy
rf_accuracies <- rf_model_cv$resample$Accuracy
  
comparison_df <- data.frame(
  Model = rep(c("Decision Tree", "Random Forest"), each = length(dt_accuracies)),
  Accuracy = c(dt_accuracies, rf_accuracies)
)

write.csv(comparison_df, "cross_val_acc.csv")

p <- ggplot(comparison_df, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Cross-Validation Accuracy Comparison Between Models", y = "Accuracy")
  
ggplotly(p)
  
# Plot both side by side
grid.arrange(dt_plot, rf_plot, ncol = 2)

# User input data for prediction
icap_shape <- ""
icap_surface <- ""
icap_color <- ""
iodor <- ""
iheight <- ""

new_data <- data.frame(
  CapShape = factor(icap_shape, levels = levels(mushroom_data$CapShape)),
  CapSurface = factor(icap_surface, levels = levels(mushroom_data$CapSurface)),
  CapColor = factor(icap_color, levels = levels(mushroom_data$CapColor)),
  Odor = factor(iodor, levels = levels(mushroom_data$Odor)),
  Height = factor(iheight, levels = levels(mushroom_data$Height))
)
pred <- predict(rf_model_tuned, new_data, type = "class")
cat("Predicted Edibility:", as.character(pred))