if (!require(shiny)) install.packages("shiny")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(plotly)) install.packages("plotly")
if (!require(DT)) install.packages("DT")
if (!require(rpart)) install.packages("rpart")
if (!require(randomForest)) install.packages("randomForest")
if (!require(rpart.plot)) install.packages("rpart.plot")
if (!require(caret)) install.packages("caret")
if (!require(pROC)) install.packages("pROC")
if (!require(gridExtra)) install.packages("gridExtra")
if (!require(rsconnect)) install.packages("rsconnect")

library(shiny)
library(ggplot2)
library(plotly)
library(DT)
library(rpart)
library(randomForest)
library(rpart.plot)
library(caret)
library(pROC)
library(gridExtra)
library(rsconnect)

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

# Define UI for the Shiny app
ui <- fluidPage(
  titlePanel("Mushroom Dataset Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      h4("About the App"),
      p("This app provides an analysis of the Mushroom dataset, including an interactive report \
and the ability to predict if a mushroom is edible or poisonous based on its characteristics."),
      width = 3
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Data Overview",
                 h3("Introduction"),
                 textOutput("text_1"),
                 h3("Data Overview"),
                 dataTableOutput("data_table"),
                 h3("Summary"),
                 h5("This dataset has 6 feature (including target variable - Edible) and 8124 observations. The dataset does not contain any missing data."),
                 verbatimTextOutput("summary_stats"),
        ),
        
        tabPanel("Interactive Report",
                 h3("Feature Analysis"),
                 selectInput("feature", "Select a feature to display the distribution:", 
                             choices = names(mushroom_data)[2:length(names(mushroom_data))]),
                 plotlyOutput("distPlot"),
                 textOutput("plotCaption"),
                 
                 h3("Model Implementation and Tuning"),
                 h4("Train Test Split"),
                 textOutput("text_2"),
                 h4("Decision Tree"),
                 textOutput("text_3"),
                 plotOutput("decision_tree_plot"),
                 uiOutput("decision_tree_caption"),
                 h4("Tuning"),
                 textOutput("text_4"),
                 plotlyOutput("cp_tuning_plot"),
                 uiOutput("cp_tuning_caption"),
                 textOutput("text_5"),
                 plotOutput("tuned_decision_tree_plot"),
                 uiOutput("tuned_decision_tree_caption"),
                 h4("Random Forest"),
                 textOutput("text_6"),
                 plotlyOutput("rf_tuning_plot"),
                 uiOutput("rf_tuning_caption"),
                 h4("Feature Importance"),
                 plotlyOutput("rf_feature_importance_plot"),
                 uiOutput("rf_feature_importance_caption"),
                 h4("ROC Curve for Model Evaluation"),
                 plotOutput("roc_curve_plot"),
                 uiOutput("roc_curve_caption"),
                 
                 h3("Model Comparison"),
                 h4("Accuracy"),
                 verbatimTextOutput("model_comparison"),
                 h4("Confusion Matrices"),
                 plotOutput("confusion_matrices_plot"),
                 
                 h4("Precision, Recall, and F1 Score Evaluation"),
                 verbatimTextOutput("precision_recall_f1_output"),
                 
                 h4("Cross-Validation Metrics Comparison"),
                 plotlyOutput("cross_validation_comparison_plot"),
                 uiOutput("cross_validation_comparison_caption"),
                 textOutput("text_7"),
                 
                 h3("Live Prediction"),
                 uiOutput("prediction_ui"),
                 verbatimTextOutput("prediction_output")
        )
      )
    )
  )
)

# Define server logic
server <- function(input, output) {
  # Tab 1: Data Overview
  output$data_table <- renderDataTable({
    datatable(mushroom_data)
  })
  
  # Texts
  output$text_1 <- renderText("In this report, we explore the predictive capabilities of decision tree and random forest classifiers on the classic mushroom dataset. This dataset, sourced from The Audubon Society Field Guide to North American Mushrooms (1981), contains categorical features like cap shape, surface, color, height, and odor, with a label for edibility (edible or poisonous). Our objective is to predict whether a mushroom is edible based on these features.")
  
  output$text_2 <- renderText("We build decision tree and random forest models to predict mushroom edibility. The data is split into training (80%) and test (20%) sets to assess model accuracy on unseen data, using a random seed of 1234.")
  
  output$text_3 <- renderText("To apply a decision tree to our dataset, we used the rpart package in R. Figure 2 displays the resulting decision tree, which uses only the 'odour' feature to classify mushroom edibility. If the odour is almond, anise, or absent, the mushroom is classified as edible; otherwise, it is classified as poisonous. This simple model uses a single feature to achieve maximum homogeneity at each split, which can be effective if that feature is highly discriminative. However, relying on one feature may lead to underfitting if the dataset's complexity requires capturing relationships among multiple features.")
  
  output$text_4 <- renderText("To capture the complexity of additional features, we can adjust the complexity parameter (cp) in the rpart package, which controls decision tree size and complexity. The cp parameter sets the threshold for the minimum improvement required for a split, determining whether a split should be made. Lowering cp allows for more splits, potentially utilizing more features, but also increases the risk of overfitting by capturing noise. To find the optimal cp value, we created multiple decision trees with cp values ranging from 0.0001 to 0.01 and evaluated their accuracy on the test set. Figure 3 illustrates the impact of cp on model performance.")
  
  output$text_5 <- renderText("In Figure 3, we observe that lower cp values yield higher test accuracy. However, extremely low cp may cause overfitting. Thus, we select the cp value at the highest accuracy bend (0.9871) to train our decision tree model (cp = 0.0022). This results in a more complex tree (see Figure 6) that considers multiple input features before determining edibility. Figure 4 shows that, like the previous tree, the decision tree classifies mushrooms without almond, anise, or no odor as poisonous. However, for mushrooms with these odors, it evaluates additional features, such as cap shape, color, and surface, instead of directly classifying them as edible. Consequently, mushrooms with almond, anise, or no odor may also be classified as poisonous, unlike in the first tree. Though height is still unused, this tree captures more complex relationships in the data.")
  
  output$text_6 <- renderText("We applied the Random Forest classifier using the randomForest package in R, tuning the mtry and ntree parameters for optimal performance. mtry defines the number of features considered at each tree split, while ntree specifies the number of trees in the forest. We tested ntree values from 100 to 1000 (in steps of 100) and mtry values from 1 to 5, given our 5 input features. Figure 5 shows the impact of these parameters on model performance. The highest accuracy (0.9877) was achieved with mtry = 2 and ntree = 100. Thus, we used these values for our final model.")
  
  output$text_7 <- renderText("As observed from Figure 8, the decision tree model has both higher average accuracy as well as lower standard deviation of accuracy from mean compared to the random forest model, hence the decision tree model is a better choice in this case.")
  
  # Captions
  output$decision_tree_caption <- renderUI({
    p("Figure 2: Decision Tree Model for Mushroom Edibility Prediction")
  })
  
  output$cp_tuning_caption <- renderUI({
    p("Figure 3: Effect of Complexity Parameter (cp) on Decision Tree Accuracy")
  })
  
  output$tuned_decision_tree_caption <- renderUI({
    p("Figure 4: Tuned Decision Tree Model for Mushroom Edibility Prediction")
  })
  
  output$rf_tuning_caption <- renderUI({
    p("Figure 5: Effect of ntree and mtry on Random Forest Accuracy")
  })
  
  output$rf_feature_importance_caption <- renderUI({
    p("Figure 6: Random Forest Feature Importance")
  })
  
  output$roc_curve_caption <- renderUI({
    p("Figure 7: ROC Curve for Random Forest Model")
  })
  
  output$cross_validation_comparison_caption <- renderUI({
    p("Figure 8: Cross-Validation Accuracy Comparison Between Models")
  })
  
  output$summary_stats <- renderPrint({
    summary(mushroom_data)
  })
  
  # Feature Analysis Plots
  output$distPlot <- renderPlotly({
    # Create a data frame for plotting
    feature_data <- data.frame(Feature = mushroom_data[[input$feature]], Edible = mushroom_data$Edible)
    
    # Create the ggplot
    p <- ggplot(feature_data, aes(x = Feature, fill = Edible)) +
      geom_bar(position = "dodge") +
      labs(title = paste("Distribution of", input$feature, "by Edibility"),
           x = input$feature, y = "Count", fill = "Edibility") +
      theme_minimal()
    
    # Convert ggplot to an interactive plotly plot
    ggplotly(p)
  })
  
  output$plotCaption <- renderText({
    paste("Figure 1: This plot shows the distribution of", input$feature, "separated by whether the mushroom is edible or poisonous.")
  })
  
  
  
  # Decision Tree Model
  decision_tree <- readRDS('dt_rpart.rds')
  output$decision_tree_plot <- renderPlot({
    req(decision_tree)
    rpart.plot::rpart.plot(decision_tree)
  })
  
  # Tuned Decision Tree Model
  decision_tree_tuned <- readRDS('dt_rpart_tuned.rds')
  output$tuned_decision_tree_plot <- renderPlot({
    req(decision_tree_tuned)
    rpart.plot::rpart.plot(decision_tree_tuned)
  })
  
  
  # Random Forest Model
  rf_model <- readRDS('rf_rf.rds')
  
  # Tuned Random Forest Model
  rf_model_tuned <- readRDS('rf_rf_tuned.rds')
  
  # Model Comparison
  dt_predictions <- predict(decision_tree_tuned, mushrooms_test, type = "class")
  rf_predictions <- predict(rf_model_tuned, mushrooms_test)
  
  output$model_comparison <- renderPrint({
    req(dt_predictions, rf_predictions)
    tree_accuracy <- sum(dt_predictions == mushrooms_test$Edible) / nrow(mushrooms_test)
    rf_accuracy <- sum(rf_predictions == mushrooms_test$Edible) / nrow(mushrooms_test)
    cat("Tuned Decision Tree Accuracy:", tree_accuracy, "\n")
    cat("Tuned Random Forest Accuracy:", rf_accuracy, "\n")
  })
  
  # Confusion Matrices
  output$confusion_matrices_plot <- renderPlot({
    req(dt_predictions, rf_predictions)
    
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
    
    # Plot both side by side
    grid.arrange(dt_plot, rf_plot, ncol = 2)
  })
  
  # Tuning Decision Tree with cp values
  output$cp_tuning_plot <- renderPlotly({
    results <- read.csv('tuning_result_dt.csv')
    p <- ggplot(results, aes(x = cp, y = Accuracy)) +
      geom_line() +
      theme_minimal() +
      labs(title = "Effect of cp Parameter on Decision Tree Accuracy", x = "Complexity Parameter (cp)", y = "Accuracy")
    ggplotly(p)
  })
  
  # Tuning Random Forest with ntree and mtry values
  set.seed(12345)
  output$rf_tuning_plot <- renderPlotly({
    results_rf <- read.csv('tuning_result_rf.csv')
    p <- ggplot(results_rf, aes(x = ntree, y = Accuracy, color = factor(mtry))) +
      geom_line() +
      theme_minimal() +
      labs(title = "Effect of ntree and mtry on Random Forest Accuracy", x = "Number of Trees (ntree)", y = "Accuracy", color = "mtry")
    ggplotly(p)
  })
  
  # Random Forest Feature Importance
  output$rf_feature_importance_plot <- renderPlotly({
    rf_importance <- importance(rf_model_tuned)
    rf_importance_df <- data.frame(Feature = rownames(rf_importance), Importance = rf_importance[, 1])
    rf_importance_df <- rf_importance_df[order(rf_importance_df$Importance, decreasing = TRUE), ]
    
    p <- ggplot(rf_importance_df, aes(x = reorder(Feature, Importance), y = Importance)) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      theme_minimal() +
      labs(title = "Random Forest Feature Importance", x = "Feature", y = "Importance")
    ggplotly(p)
  })
  
  # ROC Curve for Model Evaluation
  output$roc_curve_plot <- renderPlot({
    req(rf_model)
    rf_prob <- predict(rf_model_tuned, mushrooms_test, type = "prob")
    roc_rf <- roc(mushrooms_test$Edible, rf_prob[, 2], levels = rev(levels(mushrooms_test$Edible)))
    plot(roc_rf, col = "blue", main = "ROC Curve for Random Forest Model")
    abline(a = 0, b = 1, lty = 2, col = "red")
  })
  
  # Precision, Recall, and F1 Score Evaluation
  output$precision_recall_f1_output <- renderPrint({
    req(dt_predictions, rf_predictions)
    dt_conf_matrix <- confusionMatrix(dt_predictions, mushrooms_test$Edible)
    rf_conf_matrix <- confusionMatrix(rf_predictions, mushrooms_test$Edible)
    
    dt_precision <- dt_conf_matrix$byClass["Pos Pred Value"]
    dt_recall <- dt_conf_matrix$byClass["Sensitivity"]
    dt_f1 <- 2 * ((dt_precision * dt_recall) / (dt_precision + dt_recall))
    
    rf_precision <- rf_conf_matrix$byClass["Pos Pred Value"]
    rf_recall <- rf_conf_matrix$byClass["Sensitivity"]
    rf_f1 <- 2 * ((rf_precision * rf_recall) / (rf_precision + rf_recall))
    
    cat("Decision Tree Precision:", round(dt_precision, 4), "\n")
    cat("Decision Tree Recall:", round(dt_recall, 4), "\n")
    cat("Decision Tree F1 Score:", round(dt_f1, 4), "\n")
    cat("Random Forest Precision:", round(rf_precision, 4), "\n")
    cat("Random Forest Recall:", round(rf_recall, 4), "\n")
    cat("Random Forest F1 Score:", round(rf_f1, 4), "\n")
  })
  
  # Cross-Validation Metrics Comparison
  set.seed(1234)
  output$cross_validation_comparison_plot <- renderPlotly({
    comparison_df <- read.csv('cross_val_acc.csv')
    #rf_model_cv <- load('rf_caret_cv.rds')

    p <- ggplot(comparison_df, aes(x = Model, y = Accuracy, fill = Model)) +
      geom_boxplot() +
      theme_minimal() +
      labs(title = "Cross-Validation Accuracy Comparison Between Models", y = "Accuracy")

    ggplotly(p)
  })
  
  # Live Prediction
  output$prediction_ui <- renderUI({
    tagList(
      selectInput("cap_shape", "Cap Shape:", choices = levels(mushroom_data$CapShape)),
      selectInput("cap_surface", "Cap Surface:", choices = levels(mushroom_data$CapSurface)),
      selectInput("cap_color", "Cap Color:", choices = levels(mushroom_data$CapColor)),
      selectInput("odor", "Odor:", choices = levels(mushroom_data$Odor)),
      selectInput("height", "Height:", choices = levels(mushroom_data$Height)),
      actionButton("predict", "Predict Edibility")
    )
  })
  
  output$prediction_output <- renderPrint({
    input$predict
    isolate({
      new_data <- data.frame(
        CapShape = factor(input$cap_shape, levels = levels(mushroom_data$CapShape)),
        CapSurface = factor(input$cap_surface, levels = levels(mushroom_data$CapSurface)),
        CapColor = factor(input$cap_color, levels = levels(mushroom_data$CapColor)),
        Odor = factor(input$odor, levels = levels(mushroom_data$Odor)),
        Height = factor(input$height, levels = levels(mushroom_data$Height))
      )
      
      pred <- predict(rf_model_tuned, new_data, type = "class")
      cat("Predicted Edibility:", as.character(pred))
    })
  })
}

# Run the application 
shinyApp(ui = ui, server = server)

