# Install and load necessary libraries
# Install packages if not already installed
if (!require(tidyverse)) install.packages("tidyverse")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(caret)) install.packages("caret")
if (!require(VIM)) install.packages("VIM")
if (!require(rpart)) install.packages("rpart")
if (!require(rpart.plot)) install.packages("rpart.plot")
if (!require(pROC)) install.packages("pROC")
if (!require(knitr)) install.packages("knitr")
if (!require(DescTools)) install.packages("DescTools")

# Load necessary libraries
library(tidyverse)
library(ggplot2)
library(caret)
library(VIM)
library(rpart)
library(rpart.plot)
library(pROC)
library(knitr)
library(DescTools)




# --------------------------------------------------------------
# Load dataset
heart_data <- read.table('C:/Users/gohqw/OneDrive/Desktop/MH3511 Data Analysis with Computer/heart+disease/processed.cleveland.data', sep=",")
# Note: You may need to adjust the file path to match your location

# Rename headers for clarity
colnames(heart_data) <- c("age", "sex", "cp", "trestbps", 
                          "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", 
                          "slope", "ca", "thal", "num")


## Check for variables with missing data and outliers
# Although the dataset had told us which variables had missing data (ca and thal), we check again.
# To handle the missing value we will check the columns of the datasets, 
# if we found some missing data inside the columns then this generates the NA values as an output, which can be not good for every model.

# --------------------------------------------------------------
## Encoding Categorical Variables
# Variables like cp (chest pain), thal, slope, sex, fbs, exang need to be factorised or one-hot encoded first before data processing.

# Convert categorical variables to factors
categorical_vars <- c("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "num")
heart_data[categorical_vars] <- lapply(heart_data[categorical_vars], as.factor)
summary(heart_data)

# -   After converting the categorical variables to factor datatype, we can see that the missing values are stored as "?" instead of "na", so we cannot use the usual is.na function
# -   Thus, we replace "?" values with "NA" so that we can use is.na
# -   From the results, we confirm that variables "ca" and "thal" have missing values

# Verify the number of "?" in each column (specifically "ca" and "thal")
missing_qmarks <- sapply(heart_data, function(x) sum(x == "?"))
print("Count of '?' in each column:")
print(missing_qmarks)

# Replace "?" with NA in the entire data frame
heart_data[heart_data == "?"] <- NA

# Drop unused factor levels
heart_data$ca <- droplevels(heart_data$ca)
heart_data$thal <- droplevels(heart_data$thal)

# Check for NA in each column after replacement
missing_nas <- sapply(heart_data, function(x) sum(is.na(x)))
print("Count of NA in each column:")
print(missing_nas)

# Now, summary() should not show "?"
summary(heart_data)

# --------------------------------------------------------------

## Method to check for distribution (in categorical variables)
par(mfrow = c(3, 3))

for (var in categorical_vars) {
  # Calculate the frequency of each category
  category_counts <- table(heart_data[[var]])
  
  # Create a bar chart
  barplot(category_counts,
          main = paste("Distribution of", var),
          xlab = var,
          ylab = "Frequency")
}

## Method to check for outliers (in numerical variables)
# Boxplots for key numerical variables
numeric_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")

par(mfrow = c(2, 3))
for (var in numeric_vars) {
  boxplot(heart_data[[var]], main = paste("Boxplot of", var), col = "lightblue")
}

# --------------------------------------------------------------

# Now, we imput NA values and outliers with the median for numerical variables
# 
# * For thalach, 
# * low-end values (~70 bpm) were initially considered as possible outliers, but further medical context suggests such values are plausible in patients with severe heart disease, so we exclude it when imputing outliers with the median. 
# * For oldpeak,
# * Higher values of ST depression (the four outliers) likely indicate a more severe degree of myocardial ischemia induced by exercise.
# * Hence, we don't remove outliers, as we may be discarding potentially crucial information about individuals with a stronger ischemic response to exercise.

# Making a copy for data cleaning as in the RMD file
heart_data1 <- heart_data

# Function to impute with median for numerical variables
# Following the RMD approach, treating outliers for selected variables only
impute_with_median <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  outliers <- x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR)
  x[outliers | is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
}

# As per the RMD file, only apply outlier treatment to these variables
# Excluding thalach and oldpeak as mentioned in the RMD
numeric_vars_for_imputation <- c("age", "trestbps", "chol")

# Apply imputation on selected numerical variables
heart_data1[numeric_vars_for_imputation] <- lapply(heart_data1[numeric_vars_for_imputation], impute_with_median)

## Histogram (in numerical variables)
# Now, we check for skewed data

# Boxplots for key numerical variables
numeric_vars <- c("age", "trestbps", "chol", "thalach", "oldpeak")

par(mfrow = c(2, 3))
for (var in numeric_vars) {
  hist(heart_data1[[var]], main = paste("Histogram of", var), col = "lightblue")
}

# --------------------------------------------------------------

## Methods to handle missing values
# Dropping Rows (Listwise Deletion):
#   
#   -   When to use: If the number of missing values is small and you believe the missingness is random (MCAR), you can remove rows with NA values.
# -   Note: This can lead to significant data loss if missingness is widespread, but currently only 6 rows out of 303 are affected
# 
# 
# Imputation (Replacing Missing Values):
#   
#   -   When to use: If dropping data would result in significant information loss, imputation is a better option.
# -   Note: Imputation can introduce bias if not done carefully.
# -   Methods:
#   -   Mean/Median Imputation: Replace NA with the mean or median of the column. (for numerical data)
# -   Mode Imputation: Replace NA with the most frequent category (mode). (for categorical data)
# -   Advanced Imputation: Use more sophisticated techniques like k-nearest neighbors (KNN) imputation or model-based imputation. 
# - We will not be doing this as it is too complicated for 6 missing values.

# As the number of rows affected by NA values are small, we drop rows affected by NA values instead of imputing wiht the mode (possibility of introducing biases)


# Drop rows with NA values as done in the RMD file
# (instead of trying to impute categorical variables)
heart_data1 <- na.omit(heart_data1)

# Check the cleaned data
cat("Data preparation completed. Created heart_data1 with", nrow(heart_data1), "rows.\n")

#--------------------------------------------------------------
# END OF DATA CLEANING
#--------------------------------------------------------------

#--------------------------------------------------------------
# START OF PART 4.1.1: Is the risk of heart disease of an individual dependent on age and cholesterol?
#--------------------------------------------------------------

#--------------------------------------------------------------
# Create binary heart disease variable for analysis
#--------------------------------------------------------------
# Convert num variable to binary (0 = No disease, 1-4 = Yes disease)
heart_df <- heart_data1 %>%
  mutate(heart_disease = ifelse(num == "0", "No", "Yes")) %>%
  mutate(heart_disease = factor(heart_disease, levels = c("No", "Yes")))

#--------------------------------------------------------------
# 1. Summary statistics for age and cholesterol by heart disease status
#--------------------------------------------------------------
age_chol_summary <- heart_df %>%
  group_by(heart_disease) %>%
  summarise(
    mean_age = mean(as.numeric(as.character(age))),
    median_age = median(as.numeric(as.character(age))),
    sd_age = sd(as.numeric(as.character(age))),
    min_age = min(as.numeric(as.character(age))),
    max_age = max(as.numeric(as.character(age))),
    mean_chol = mean(as.numeric(as.character(chol))),
    median_chol = median(as.numeric(as.character(chol))),
    sd_chol = sd(as.numeric(as.character(chol))),
    min_chol = min(as.numeric(as.character(chol))),
    max_chol = max(as.numeric(as.character(chol)))
  )

print(age_chol_summary)

#--------------------------------------------------------------
# 2. Visualize distributions - Exploratory Plots
#--------------------------------------------------------------
# Convert to numeric for plotting
heart_df$age <- as.numeric(as.character(heart_df$age))
heart_df$chol <- as.numeric(as.character(heart_df$chol))

# Age distribution by heart disease status
age_hist <- ggplot(heart_df, aes(x = age, fill = heart_disease)) +
  geom_histogram(position = "dodge", bins = 15, alpha = 0.7) +
  labs(title = "Age Distribution by Heart Disease Status",
       x = "Age",
       y = "Count") +
  theme_minimal()
print(age_hist)

# Cholesterol distribution by heart disease status
chol_hist <- ggplot(heart_df, aes(x = chol, fill = heart_disease)) +
  geom_histogram(position = "dodge", bins = 15, alpha = 0.7) +
  labs(title = "Cholesterol Distribution by Heart Disease Status",
       x = "Cholesterol (mg/dl)",
       y = "Count") +
  theme_minimal()
print(chol_hist)

#--------------------------------------------------------------
# 3. Box plots - These can be used for the report
#--------------------------------------------------------------

# FIGURE 1a (alternative): Box plot for Age by Heart Disease Status
age_box <- ggplot(heart_df, aes(x = heart_disease, y = age, fill = heart_disease)) +
  geom_boxplot() +
  labs(title = "Age by Heart Disease Status",
       x = "Heart Disease",
       y = "Age") +
  theme_minimal()
print(age_box)

# FIGURE 1b (alternative): Box plot for Cholesterol by Heart Disease Status
chol_box <- ggplot(heart_df, aes(x = heart_disease, y = chol, fill = heart_disease)) +
  geom_boxplot() +
  labs(title = "Cholesterol by Heart Disease Status",
       x = "Heart Disease",
       y = "Cholesterol (mg/dl)") +
  theme_minimal()
print(chol_box)

#--------------------------------------------------------------
# 4. Statistical tests
#--------------------------------------------------------------
# T-test for age between disease and no disease groups
age_ttest <- t.test(age ~ heart_disease, data = heart_df)
print(age_ttest)

# T-test for cholesterol between disease and no disease groups
chol_ttest <- t.test(chol ~ heart_disease, data = heart_df)
print(chol_ttest)

#--------------------------------------------------------------
# 5. FIGURE 1: Scatter plot for age vs cholesterol by heart disease status
#--------------------------------------------------------------
scatter_plot <- ggplot(heart_df, aes(x = age, y = chol, color = heart_disease)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, aes(group = heart_disease)) +
  labs(title = "Relationship between Age and Cholesterol by Heart Disease Status",
       x = "Age",
       y = "Cholesterol (mg/dl)") +
  theme_minimal() +
  scale_color_manual(values = c("No" = "blue", "Yes" = "red"))
print(scatter_plot)

#--------------------------------------------------------------
# 6. Logistic Regression - Individual Predictors
#--------------------------------------------------------------
# Age as predictor
age_model <- glm(heart_disease ~ age, data = heart_df, family = "binomial")
summary(age_model)

# Cholesterol as predictor
chol_model <- glm(heart_disease ~ chol, data = heart_df, family = "binomial")
summary(chol_model)

# Interpreting the coefficients
age_odds_ratio <- exp(coef(age_model)["age"])
age_ci <- exp(confint(age_model)["age", ])
cat("Age odds ratio:", age_odds_ratio, "95% CI:", age_ci[1], "-", age_ci[2], "\n")

chol_odds_ratio <- exp(coef(chol_model)["chol"])
chol_ci <- exp(confint(chol_model)["chol", ])
cat("Cholesterol odds ratio (per 1 unit):", chol_odds_ratio, "95% CI:", chol_ci[1], "-", chol_ci[2], "\n")
cat("Cholesterol odds ratio (per 10 units):", chol_odds_ratio^10, "95% CI:", chol_ci[1]^10, "-", chol_ci[2]^10, "\n")

#--------------------------------------------------------------
# 7. Logistic Regression - Combined Model
#--------------------------------------------------------------
# Age and cholesterol as predictors
age_chol_model <- glm(heart_disease ~ age + chol, data = heart_df, family = "binomial")
summary(age_chol_model)

# Combined model odds ratios
combined_odds <- exp(coef(age_chol_model))
combined_ci <- exp(confint(age_chol_model))
print(data.frame(
  Odds_Ratio = combined_odds,
  Lower_CI = combined_ci[, 1],
  Upper_CI = combined_ci[, 2]
))

#--------------------------------------------------------------
# 8. FIGURE 2: Create risk heatmap by age and cholesterol groups
#--------------------------------------------------------------
# Create age and cholesterol groups
heart_df <- heart_df %>%
  mutate(age_group = cut(age, breaks = c(0, 40, 55, 100), 
                         labels = c("<40", "40-55", ">55")),
         chol_group = cut(chol, breaks = c(0, 200, 240, 600), 
                          labels = c("<200", "200-240", ">240")))

# Create risk heatmap
risk_heatmap <- heart_df %>%
  group_by(age_group, chol_group) %>%
  summarise(
    total_count = n(),
    disease_count = sum(heart_disease == "Yes"),
    disease_proportion = mean(heart_disease == "Yes"),
    .groups = 'drop'
  )

# Display the risk by groups
print(risk_heatmap)

# Plot heatmap
heatmap_plot <- ggplot(risk_heatmap, aes(x = age_group, y = chol_group, fill = disease_proportion)) +
  geom_tile() +
  geom_text(aes(label = paste0(round(disease_proportion * 100, 1), "%\n(", disease_count, "/", total_count, ")")), 
            color = "white", size = 3) +
  scale_fill_gradient(low = "lightblue", high = "darkred") +
  labs(title = "Heart Disease Risk by Age and Cholesterol Groups",
       x = "Age Group",
       y = "Cholesterol Group",
       fill = "Disease %") +
  theme_minimal()
print(heatmap_plot)

#--------------------------------------------------------------
# 9. FIGURE 3: ROC Curves to evaluate predictive power
#--------------------------------------------------------------
# ROC curve for age model
age_probs <- predict(age_model, type = "response")
age_roc <- roc(heart_df$heart_disease, age_probs)
age_auc <- auc(age_roc)

# ROC curve for cholesterol model
chol_probs <- predict(chol_model, type = "response")
chol_roc <- roc(heart_df$heart_disease, chol_probs)
chol_auc <- auc(chol_roc)

# ROC curve for combined model
age_chol_probs <- predict(age_chol_model, type = "response")
age_chol_roc <- roc(heart_df$heart_disease, age_chol_probs)
age_chol_auc <- auc(age_chol_roc)

# Plot ROC curves
par(mfrow = c(1, 1))  # Reset plotting parameters
plot(age_roc, col = "blue", main = "ROC Curves for Heart Disease Prediction")
plot(chol_roc, col = "red", add = TRUE)
plot(age_chol_roc, col = "green", add = TRUE)
legend("bottomright", legend = c(paste("Age (AUC =", round(age_auc, 3), ")"),
                                 paste("Cholesterol (AUC =", round(chol_auc, 3), ")"),
                                 paste("Age + Cholesterol (AUC =", round(age_chol_auc, 3), ")")),
       col = c("blue", "red", "green"), lwd = 2)

#--------------------------------------------------------------
# 10. Compare models
#--------------------------------------------------------------
# Compare nested models
anova(age_model, age_chol_model, test = "Chisq")
anova(chol_model, age_chol_model, test = "Chisq")


#--------------------------------------------------------------
# 12. Standardized coefficients (to compare importance)
#--------------------------------------------------------------
# Scale the predictors
heart_df_scaled <- heart_df %>%
  mutate(age_scaled = scale(age),
         chol_scaled = scale(chol))

# Fit model with standardized predictors
model_scaled <- glm(heart_disease ~ age_scaled + chol_scaled, 
                    data = heart_df_scaled, family = "binomial")
summary(model_scaled)


#--------------------------------------------------------------
# 13. FIGURE 4: Enhanced Predicted Probability Visualization
#--------------------------------------------------------------
# Create a more detailed grid for smoother visualization
# Use sequence instead of expand.grid to avoid duplicates
age_seq <- seq(30, 80, by = 2)
chol_seq <- seq(150, 350, by = 5)
detailed_grid <- expand.grid(age = age_seq, chol = chol_seq)

# Calculate predicted probabilities using our combined model
detailed_grid$predicted_prob <- predict(age_chol_model, newdata = detailed_grid, type = "response")

# Create enhanced probability heatmap with simpler approach
enhanced_prob_plot <- ggplot(detailed_grid, aes(x = age, y = chol)) +
  geom_raster(aes(fill = predicted_prob)) +  # Use geom_raster instead of tile for better performance
  scale_fill_gradient2(
    low = "lightblue", 
    mid = "yellow", 
    high = "darkred", 
    midpoint = 0.5,
    name = "Probability of\nHeart Disease"
  ) +
  # Add clinical reference lines
  geom_hline(yintercept = 200, linetype = "dashed", color = "darkgreen", alpha = 0.7) +
  geom_hline(yintercept = 240, linetype = "dashed", color = "darkred", alpha = 0.7) +
  # Add annotations for cholesterol thresholds
  annotate("text", x = 32, y = 195, label = "Desirable < 200", color = "darkgreen", size = 3) +
  annotate("text", x = 32, y = 245, label = "High > 240", color = "darkred", size = 3) +
  # Add probability contours as separate overlays to avoid aesthetic conflicts
  stat_contour(aes(z = predicted_prob), 
               breaks = c(0.25, 0.5, 0.75),
               color = "white", 
               linewidth = 0.5) +
  # Improve labels and appearance
  labs(
    title = "Predicted Probability of Heart Disease",
    subtitle = "Based on logistic regression model with age and cholesterol",
    x = "Age (years)",
    y = "Cholesterol (mg/dl)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

print(enhanced_prob_plot)

# Save the figures for the report
# Save the scatter plot (Figure 1)
ggsave("figure1_age_chol_scatter.png", scatter_plot, width = 8, height = 6)

# Save the risk heatmap (Figure 2)
ggsave("figure2_risk_heatmap.png", heatmap_plot, width = 8, height = 6)

# Save the enhanced probability plot (Figure 5)
ggsave("figure5_enhanced_probability.png", enhanced_prob_plot, width = 9, height = 6)

# Save the decision tree (Figure 4)
# Note: For the ROC curve (Figure 3), you'll need to save it directly from the plot window
# or use a different approach as it's created with base R graphics, not ggplot2

#--------------------------------------------------------------
# END OF PART 4.1.1
#--------------------------------------------------------------




#--------------------------------------------------------------
# START OF PART 4.1.2: Can the risk of heart disease be detected based on the maximum heart rate?
#--------------------------------------------------------------

myoverall=data.frame(heart_data1[8],heart_data1[14])
colnames(myoverall)=c("Maximum Heart Rate","Presence of Heart Disease")

boxplot(myoverall$`Maximum Heart Rate`,main="Maximum Heart Rate",ylab="Heart Rate (BPM) ",ylim=c(50,250))
hist(myoverall$`Maximum Heart Rate`,breaks=18,main="Maximum Heart Rate",xlab="Heart Rate (BPM)")

testdata=myoverall
testdata$`Presence of Heart Disease`
length(testdata$`Presence of Heart Disease`)
for (i in 1:length(testdata$`Presence of Heart Disease`)){
  if( testdata$`Presence of Heart Disease`[i]!=0){
    testdata$`Presence of Heart Disease`[i]=1
  }
}

testdata

par(mfrow=c(1,1))
# For Presence of Heart Disease = 0
zero1=subset(testdata,testdata$`Presence of Heart Disease`==0)
zero1=zero1$`Maximum Heart Rate`
hist(zero1,main="Presence of Heart Disease",xlab="Maximum Heart Rate (BPM)")


# For Presence of Heart Disease = 1
one1=subset(testdata,testdata$`Presence of Heart Disease`==1)
one1=one1$`Maximum Heart Rate`
hist(one1,main="No presence of Heart Disease",xlab="Maximum Heart Rate (BPM)")


# Fit binary logistic regression model
model <- glm(`Presence of Heart Disease` ~ `Maximum Heart Rate`, data = testdata, family = binomial)

# View summary
summary(model)

#--------------------------------------------------------------
# END OF PART 4.1.2
#--------------------------------------------------------------




#--------------------------------------------------------------
# START OF PART 4.1.3: Is a higher level of resting blood sugar an indicator of heart disease?
#--------------------------------------------------------------

# Research Question: Is a higher level of resting blood sugar (fbs) an indicator of heart disease?

# Exploratory Data Analysis and Visualisation

# Convert heart disease to binary: 0 = no disease, 1 = presence of disease (1–4)
heart_data1$heart_disease <- ifelse(heart_data1$num == 0, 0, 1)
heart_data1$heart_disease <- as.factor(heart_data1$heart_disease)

# Check distribution of fbs and heart disease
table(heart_data1$fbs)
table(heart_data1$heart_disease)



# Frequency(contingency) table
fbs_hd_table <- table(heart_data1$fbs, heart_data1$heart_disease)
kable(fbs_hd_table, caption = "Contingency Table: Fasting Blood Sugar (fbs) vs Heart Disease")

# Proportion table
prop_table <- prop.table(fbs_hd_table, margin = 1)
kable(round(prop_table, 2), caption = "Proportion of Heart Disease Within Each fbs Group")



# Bar plot of heart disease by fbs level

ggplot(heart_data1, aes(x = fbs, fill = heart_disease)) +
  geom_bar(position = "dodge") +
  labs(
    title = "Heart Disease Count by Fasting Blood Sugar",
    x = "Fasting Blood Sugar > 120 mg/dl (fbs)",
    y = "Count",
    fill = "Heart Disease"
  ) +
  geom_text(stat = "count", aes(label = ..count..), position = position_dodge(width = 0.9), vjust = -0.3) +
  theme_minimal()

# Statistical Testing: Chi-Square Test of Independence

# Create a contingency table of fbs vs heart_disease
contingency_table <- table(heart_data1$fbs, heart_data1$heart_disease)
print("Contingency Table:")
print(contingency_table)

# Perform Chi-Square Test
chi_result <- chisq.test(contingency_table)
print("Chi-Square Test Results:")
print(chi_result)

# Check expected frequencies to validate assumptions
print("Expected Frequencies:")
print(chi_result$expected)

# Cramer's V for effect size measurement
cramers_v <- CramerV(contingency_table)
print(paste("Cramer's V:", round(cramers_v, 3)))

#--------------------------------------------------------------
# END OF PART 4.1.3
#--------------------------------------------------------------



#--------------------------------------------------------------
# START OF PART 4.1.4: Can different types of chest pain types be an indicator of heart disease?
#--------------------------------------------------------------

# Question 4: cp to num + regression (Jason)
# Just copy/paste this to the bottom of the main file

# Factoring num into binary
heart_data1$num_binary <- ifelse(heart_data1$num == 0, 0, 1)
heart_data1$num_binary

cp_num_binary <- table(heart_data1$cp, heart_data1$num_binary)
rownames(cp_num_binary) = c("cp1", "cp2", "cp3", "cp4")
colnames(cp_num_binary) = c("num0 (Absent)", "num1 (Present)")
cp_num_binary

chisq.test(cp_num_binary)

# Manual Chi-square (not in report)
colsum = matrix(colSums(cp_num_binary), ncol=2)
rowsum = matrix(rowSums(cp_num_binary), nrow=4)
expected = rowsum%*%colsum / sum(colsum)
expected
cp_num_m = matrix(cp_num_binary, ncol=2, byrow=TRUE)
chisq = sum((cp_num_m-expected)^2 / expected)
chisq
pvalue = 1-pchisq(chisq, df=12)
pvalue

# ANOVA and pairwise
aov_result = aov(num_binary ~ cp, data=heart_data1)
summary(aov_result)

pairwise.t.test(heart_data1$num_binary, heart_data1$cp, p.adjust.method = "none")

#--------------------------------------------------------------
# END OF PART 4.1.4:
#--------------------------------------------------------------


#--------------------------------------------------------------
# START  OF PART 4.2: 
#--------------------------------------------------------------

# Regression

# Remove extraneous variables (num2 was created by me)
heart_data_reg <- subset(heart_data1, select = -c(num, heart_disease))
# Create model
full_model = glm(num_binary ~ ., data=heart_data_reg)
step_model = step(full_model, direction = "backward")
summary(step_model)

# Below is code for factorized num, not used in report

summary(heart_data1$cp)
summary(heart_data1$num)
cp_num <- table(heart_data1$cp, heart_data1$num)
cp_num
rownames(cp_num) = c("cp1", "cp2", "cp3", "cp4")
colnames(cp_num) = c("num0", "num1", "num2", "num3", "num4")
help(rownames)
qqnorm(as.numeric(heart_data1$cp))
qqline(as.numeric(heart_data1$cp))

qqnorm(as.numeric(heart_data1$num))
qqline(as.numeric(heart_data1$num))

# Chi square (auto)
chisq.test(cp_num)

# Chi square (manual)
colsum = matrix(colSums(cp_num), ncol=5)
rowsum = matrix(rowSums(cp_num), nrow=4)
expected = rowsum%*%colsum / sum(colsum)
expected
cp_num_m = matrix(cp_num, ncol=5, byrow=TRUE)
chisq = sum((cp_num_m-expected)^2 / expected)
chisq
pvalue = 1-pchisq(chisq, df=12)
pvalue

# ANOVA
aov_result = aov(as.numeric(num) ~ cp, data=heart_data1)
summary(aov_result)

# Pairwise
pairwise.t.test(as.numeric(heart_data1$num), heart_data1$cp, p.adjust.method = "none")

#--------------------------------------------------------------
# END OF PART 4.2
#--------------------------------------------------------------

