library(tidyverse)
library(caret)
library(GGally)
library(car)
library(doParallel)
df <- read.csv("/Users/liyif/Desktop/Airbnb_Data.csv")
df_clean <- df %>%
  select(id, log_price, property_type, room_type, accommodates, bathrooms,
         bed_type, cancellation_policy, cleaning_fee, city, instant_bookable,
         host_identity_verified, host_has_profile_pic,
         number_of_reviews, review_scores_rating, bedrooms, beds)
df_clean <- na.omit(df_clean)#drop the row with NA value
df_clean <- df_clean %>%
  mutate(cleaning_fee = cleaning_fee == "True")#the value of it store as string, so convert to boolean first
df_clean <- df_clean %>%
  mutate(cleaning_fee = as.integer(cleaning_fee))#convert to binary so we can do it numeric
df_clean <- df_clean %>%
  mutate(
    instant_bookable = instant_bookable == "t",
    host_identity_verified = host_identity_verified == "t",
    host_has_profile_pic = host_has_profile_pic == "t"
  )#same here
df_clean <- df_clean %>%
  mutate(
    instant_bookable = as.integer(instant_bookable),
    host_identity_verified = as.integer(host_identity_verified),
    host_has_profile_pic = as.integer(host_has_profile_pic)
  )
summary(df_clean)
#drop the room type with few observation
df_clean <- df_clean %>%
  group_by(property_type) %>%
  filter(n() >= 50) %>%
  ungroup()
ggplot(df_clean, aes(x = log_price)) +
  geom_histogram(bins = 30, color = "black") +
  labs(title = "Distribution of Log Price", x = "Log Price", y = "Frequency") + theme_minimal()
ggplot(df_clean, aes(x = room_type, y = log_price)) +
  geom_boxplot(fill = "steelblue") +
  labs(title = "Log Price by Room Type", x = "Room Type", y = "Log Price") + theme_minimal()
ggplot(df_clean, aes(x = accommodates, y = log_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Log Price vs Accommodates", 
       x = "Accommodates", 
       y = "Log Price") + theme_minimal()
ggplot(df_clean, aes(x = bedrooms, y = log_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Log Price vs Bedrooms",
       x = "Number of Bedrooms",
       y = "Log Price") + theme_minimal()
ggplot(df_clean, aes(x = beds, y = log_price)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  labs(title = "Log Price vs Beds",
       x = "Number of Beds",
       y = "Log Price") + theme_minimal()
df_numeric <- df_clean %>%
  select(log_price, accommodates, bathrooms, bedrooms, beds,
         number_of_reviews, review_scores_rating)
ggpairs(df_numeric,
        title = "Pairwise Bivariate Analysis of Airbnb Numeric Variable")
#so accommodates vs bedrooms vs beds may have collinearty

log_price <- df_clean$log_price

q1 <- quantile(log_price, 0.25, na.rm = TRUE)
q3 <- quantile(log_price, 0.75, na.rm = TRUE)
iqr <- q3 - q1

lower_bound <- q1 - 1.5 * iqr
upper_bound <- q3 + 1.5 * iqr

outlier_indices <- which(log_price < lower_bound | log_price > upper_bound)

outlier_values <- log_price[outlier_indices]
length(outlier_values)

#model
lm_model <- lm(
  log_price ~ accommodates + bathrooms + 
  number_of_reviews + review_scores_rating + room_type +
  property_type + cancellation_policy + cleaning_fee +
  instant_bookable + host_identity_verified + host_has_profile_pic +
  city + bed_type,
  data = df_clean
)
summary(lm_model)
anova(lm_model)
lm_model_new <- lm(
  log_price ~ accommodates + bathrooms + number_of_reviews +
  review_scores_rating + room_type +
  property_type + cancellation_policy +
  instant_bookable +  host_has_profile_pic +
  city + bed_type,
  data = df_clean
)

vif(lm_model_new)
plot(lm_model$fitted.values, lm_model$residuals,
     xlab = "Fitted values", ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, col = "red")
fitControl <- trainControl(method = "cv", number = 10)
lm_kfold <- train(
  log_price ~ accommodates + bathrooms + number_of_reviews +
  review_scores_rating + room_type +
  property_type + cancellation_policy +
  instant_bookable +  host_has_profile_pic +
  city + bed_type,
  data = df_clean,
  method = "lm",
  trControl = fitControl
)
print(lm_kfold)
(lm_kfold$results$RMSE)^2
pred_lm <- predict(lm_kfold, newdata = df_clean)
ggplot(df_clean, aes(x = log_price, y = pred_lm)) +
  geom_point(color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(
    title = "Predicted vs. Actual: Linear Regression",
    x = "Actual Log Price",
    y = "Predicted Log Price"
  ) + theme_minimal()

#CV

cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

fitControl_rf <- trainControl(method = "cv", number = 10, allowParallel = TRUE)

rf_kfold <- train(
  log_price ~ .,
  data = df_clean,
  method = "rf",
  trControl = fitControl_rf,
  ntree = 100
)
stopCluster(cl)
rf_kfold
rf_mse <- 0.4051238^2
rf_mse
var_imp <- varImp(rf_kfold)
plot(var_imp, main = "Variable Importance (Random Forest with 10-fold CV)")
pred_rf <- predict(rf_kfold, newdata = df_clean)
ggplot(df_clean, aes(x = log_price, y = pred_rf)) +
  geom_point(color = "steelblue") +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(
    title = "Predicted vs. Actual: Random Forest Model",
    x = "Actual Log Price",
    y = "Predicted Log Price"
  ) + theme_minimal()
