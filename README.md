# SC1015: Data Science Mini Project - predicting G3
Mini-Project for SC1015 - Introduction to Data Science and Artificial Intelligence

Lab: FCED

Group: 2

---

# About
This repository contains all the Jupyter Notebooks, dataset, video presentation and the source materials/references we have used and created as part of the Mini Project.

This README breifly highlights what we have accomplished in the project. If you want a more detailed explanation of things, please refer to the Jupyter Notebooks in this repository. They contain more in-depth descriptions and smaller details which are not mentioned here in the README. For convenience, we have divided the notenooks into smaller parts.

---

# Table of contents
1. [Problem definition](#problem-definition)
2. [Data preparation and Cleaning](#data-preparation-and-cleaning)
3. [Exploratory Data Analysis on Numeric Variables](#exploratory-data-analysis-on-numeric-variables)
4. [Exploratory Data Analysis on Categorical Variables](#exploratory-data-analysis-on-categorical-variables)
5. [Predictive modeling using Random Tree Forest](#predictive-modeling-using-random-tree-forest)
6. [Conclusion](#conclusion)
7. [References](#references)
8. [Learning Outcomes](#learning-outcomes)
   
---

# Problem Definition
Our Dataset: https://archive.ics.uci.edu/dataset/320/student+performance 

Our Question: Which predicators are the most accurate / reliable in  determining the future score (G3) ?

Success : determine future score (G3) using the predicators we found that are accurate / reliable

Rationale : By understanding which predictors have the strongest association with future scores, you can build more effective predictive models. This can lead to better allocation of resources and interventions to support students who may be at risk of achieving lower scores.

# Data Preparation and Cleaning
In this section of the project, we prepped and cleaned the dataset to help us to analyze our data better and also to help us use our data for the purposes of machine learning in the later sections.

We performed the following:
1. **Preliminary variable selection** : 8 relevant variables (3 numerical and 5 categorical) out of 33 were selected
2. **Encoding numeric variable**: the numeric variable were encoded appropriately by assigning letter grades to the respective numeric scores and adding a new column to the dataset
3. **Removing outliers for numeric variables**: using box-plot method
4. **Converting Categorical Data to Numerical Data using One-Hot Encoding**, producing a separate data frame with a column per category 
Eg. 4 Categories (A-D) , if row n has value of A, under the column for A it has a value of 1, for columns B-D is has value of 0

For full code, please refer to the Jupyter Notebook on Data Preparation and Cleaning

# Exploratory Data Analysis on Numeric Variables
Then, we performed **linear regression** on 3 numeric variables (absences, health and studytime)

In our exploratory data analysis, the decision to utilize **K-fold cross-validation** was driven by several important considerations. 
- Given the relatively small size of our dataset (~650 entries), the risk of overfitting while performing linear regression on our numeric variables (absences, health, and studytime) is significantly heightened.
- K-fold cross-validation addresses this issue effectively by dividing the entire dataset into 'K' number of subsets, thereby allowing the model to be trained and validated multiple times across different segments of data.
- This method enhances the reliability of the model by ensuring that every data point gets to be part of both the training and validation sets across the folds.
- Additionally, it improves the generalizability of our findings, making our regression model more robust against different samples of data.

By leveraging K-fold cross-validation, we can **achieve a more accurate and consistent estimation of model performance**, an essential factor in our analysis given the limited data size. 

Based on our findings, we concluded that **'studytime' is the most accurate numeric predictor of 'G3' grades within the given dataset** as it has a relatively stronger association with the target variable ('G3' grades) compared to the other numeric variables.

For further findings and explanations, please refer to the Jupyter Notebook on EDA on Numeric Variables.

# Exploratory Data Analysis on Categorical Variables
Next, we used **decision tree model** on 5 categorical variables (address, paid, activities, higher education and reason)

Based on the classification accuracies obtained, we concluded that **'paid' is the most accurate categorical predictor of 'G3' grades within the given dataset** as it demonstrates the highest accuracy among the categorical variables assessed.

For further findings and explanations, please refer to the Jupyter Notebook on EDA on Categorical Variables.

# Predictive modeling using Random Tree Forest
Here, we leveraged the **Random Forest model**, combining key numeric (without removing the outliers) and categorical variables—specifically 'studytime' and 'paid'—to investigate their collective impact on predicting the G3 score. This approach was taken based on preliminary findings that these variables individually contributed significant predictive power. By integrating both types of data into the Random Forest, we aimed to enhance the model's accuracy.

Random Forests, known for their robustness against overfitting and ability to handle diverse data types effectively, provided a strategic framework to assess the combined influence of these variables. 

Reasons for not removing the outliers in the numeric variable:
- **Model Robustness**: Since Random Forest can handle outliers as it makes use of the ensemble method, which is the utilising of, and combining multiple different models to obtain better predictive performance. This reduces the impact of any outliers, casuing the predictive method to not be significantly affected by them.
- **Valuable Insights**: Outliers may provide valuable insights into students' performance patterns. Education is not a one-size-fits all approach, and there will be students who will perform significantly better or worse than others. Hence removing outliers would be potentially overlooking any complex, non-linear relationships that might have been detected by our model.
We have noticed that the combined model did not improve the overall accuracy afterall.

For further findings and explanations, please refer to the Jupyter Notebook on Predictive Modeling.

# Conclusion
Despite the higher classification accuracy of paid, when considering R² score and MSE (which are more directly relevant to the regression task), studytime appears to have a slightly better performance compared to paid or the combined model.

Unexpectedly, combining both our best variables from both categories into a single model yielded poorer performance.

Moving forward, a possible solution could be to use more data that measures other aspects of a students learning, and combine it into 1 model.

In conclusion, studytime alone may offer a simpler and more interpretable model compared to the combined model involving categorical encoding and interactions between variables.

# References
- https://archive.ics.uci.edu/dataset/320/student+performance
- https://medium.com/@bididudy/the-essential-guide-to-k-fold-cross-validation-in-machine-learning-2bcb58c50578
- https://www.statology.org/k-fold-cross-validation-in-python/
- https://www.youtube.com/watch?v=TIgfjmp-4BA
- https://www.youtube.com/watch?v=etkxDx4gesk
- https://towardsdatascience.com/random-forest-3a55c3aca46d
- https://www.geeksforgeeks.org/random-forest-regression-in-python/
- https://www.youtube.com/watch?v=YUsx5ZNlYWc
- https://www.youtube.com/watch?v=HDVIc66vi_0

# Learning Outcomes
Embarking on this project has been a journey of discovery, expanding our knowledge beyond the confines of our course curriculum for our group. 

Two key learnings stand out prominently: the utilization of K-fold cross-validation and the application of Random Forest models for predictive modeling. 

K-fold cross-validation introduced us to a robust technique for assessing model performance, especially vital in scenarios where data is limited, as in our case. This method allowed us to maximize the utility of our dataset, ensuring that our model's performance metrics were reliable and indicative of its true capabilities. 

Additionally, our exploration into Random Forest models illuminated the power of ensemble learning techniques in handling complex datasets comprising both numerical and categorical variables. Through experimentation with this versatile algorithm, we gained insights into its strengths in mitigating overfitting and capturing intricate patterns within the data. 

These newfound skills not only enriched our understanding of machine learning methodologies but also equipped us with practical tools for tackling real-world challenges beyond the scope of our academic coursework.

---

# Contributors
- [@glynislxl](https://github.com/glynislxl) - Data Cleaning , Predictive modeling
- Wang Xin Ping - Exploratory Data Analysis on numeric variables
- Goh Qing Wen - Exploratory Data Analysis on categorical variables
