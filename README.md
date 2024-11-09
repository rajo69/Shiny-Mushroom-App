# Mushroom Classification Project

This repository contains an R project focused on analyzing the **mushrooms.csv** dataset to develop reliable models that can classify mushrooms as **edible** or **poisonous** based on their various characteristics such as cap shape, odor, gill size, and more. The primary objective was to build accurate models and use the results in a Shiny app for real-time classification.

## Project Overview

This project involved several key steps:

1. **Data Analysis and Visualization**: Summarization, missing value check and visualizations were performed to understand the features of the mushrooms dataset and the relationships between them.
2. **Model Training and Tuning**: Two machine learning models, a **Decision Tree** and a **Random Forest**, were trained and tuned for better performance.
3. **Result Visualization**: Results were visualized for better understanding and comparison between the models.
4. **Model Saving**: The trained models and results were saved for use in a Shiny app.
5. **Shiny App Development**: A Shiny application (**app.R**) was created to provide an interactive interface for predicting whether a mushroom is edible or poisonous.

## Dataset

The **mushrooms.csv** dataset consists of various features that describe the physical properties of different mushrooms. These features include **cap shape**, **cap surface**, **cap color**, **odor**, among others, which are essential for determining if a mushroom is edible or poisonous.

## Models

- **Decision Tree**: A simple, interpretable model that was trained on the dataset to classify mushrooms.
- **Random Forest**: An ensemble model trained with multiple decision trees to improve classification accuracy and handle overfitting.

## Workflow

1. **Exploratory Data Analysis (EDA)**:
   - **Libraries used**: `ggplot2`
   - Performed missing value check, summary statistics, and visualizations.
   - Visualized relationships between different features using frequency distributions.

2. **Model Training and Evaluation**:
   - **Libraries used**: `caret`, `rpart`, `randomForest`
   - Split the dataset into training and testing sets.
   - Used cross-validation to tune the **Decision Tree** and **Random Forest** models.
   - Compared model performance using metrics such as **accuracy**, **precision**, and **recall**.

3. **Saving Models and Results**:
   - Saved the trained models using `saveRDS()` for later use in the Shiny app.
   - Plotted feature importance for the Random Forest model.

4. **Shiny Application**:
   - **Libraries used**: `shiny`, `rsconnect`
   - Developed a Shiny app (**app.R**) to allow users to input mushroom features and get predictions on whether a mushroom is edible or poisonous.

## Pre-Existing Code and Libraries

- **Data Wrangling and Visualization**: Made use of `ggplot2` and `plotly` libraries for interactive visualization of features and results. These libraries provided pre-existing functions for result plotting (`ggplot()`, `geom_point()`, `ggplotly()`, etc.).
- **Model Training**: The `rpart` library was used for decision tree training, while `randomForest` was employed to train the Random Forest model. The `caret` package was used for training the models during cross-validation using functions for model fitting and evaluation such as `train()` and `trainControl()` and for confusion matrix using function `confusionMatrix()`. 
- **Shiny Framework**: Used `shiny` and `rsconnect` to develop the user interface for interactive report and deployment to shinyapp.io.

## Custom Code Written

- **Data Preprocessing**: Custom code was written to check and handle missing values and split the data into training and testing sets.
- **Model Tuning**: Parameters for the Decision Tree and Random Forest models were tuned manually using custom code to identify the best parameters based on tuning result plots.
- **Custom Visualizations**: Additional visualizations were created to illustrate feature importance and the distribution of model predictions.
- **Shiny App Logic**: Implemented server-side logic to load the saved models and perform real-time predictions based on user inputs within the Shiny app.

## How to Run the Project

1. Clone this repository to your local machine.
2. Open the R project in RStudio.
3. Run the **Mushroom_Analysis.R** script to perform data analysis, visualization, train the models and save them.
4. Run the **app.R** file to launch the Shiny app for interactive report and predictions.

## Dependencies

- R (>= 4.0.0)
- Packages: `ggplot2`, `caret`, `rpart`, `randomForest`, `shiny`, `plotly`, `rpart.plot`, `pROC`, `gridExtra`, `rsconnect`

## Conclusion

The goal of this project was to develop a reliable model to predict whether a mushroom is **edible** or **poisonous** based on its physical characteristics. Both **Decision Tree** and **Random Forest** models performed well, with the Random Forest model showing higher accuracy due to its ensemble nature. The Shiny app provides an easy-to-use interface for users to classify mushrooms based on the trained models.

Feel free to contribute or reach out for any questions regarding the implementation!

