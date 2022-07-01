library(tidyverse) 
library(data.table)
library(rstudioapi)
library(skimr)
library(inspectdf)
library(mice)
library(plotly)
library(highcharter)
library(recipes) 
library(caret) 
library(purrr) 
library(graphics) 
library(Hmisc) 
library(glue)
library(h2o) 


# Question 1

raw <- ggplot2::mpg

# Question 2

raw %>% inspect_na()  # NO Null values

df.num <- raw %>%
  select_if(is.numeric) %>%
  select(cty,everything())

df.chr <- raw %>%
  select_if(is.character)


# One Hot Encoding

df.chr <- dummyVars(" ~ .", data = df.chr) %>%
   predict(newdata = df.chr) %>%
   as.data.frame()

df <- cbind(df.chr,df.num) %>%
  select(cty,everything())

# Multicollinearity 

target <- 'cty'
features <- df %>% select(-cty) %>% names()

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

glm %>% summary()

coef_na <- attributes(alias(glm)$Complete)$dimnames[[1]]
features <- features[!features %in% coef_na]

f <- as.formula(paste(target, paste(features, collapse = " + "), sep = " ~ "))
glm <- glm(f, data = df)

glm %>% summary()


# VIF (Variance Inflation Factor) ----
while(glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[1] >= 1.5){
  afterVIF <- glm %>% faraway::vif() %>% sort(decreasing = T) %>% .[-1] %>% names()
  f <- as.formula(paste(target, paste(afterVIF, collapse = " + "), sep = " ~ "))
  glm <- glm(f, data = df)
}

glm %>% faraway::vif() %>% sort(decreasing = T) %>% names() -> features 

df <- df %>% select(cty,features)

# Scaling

df[,-1] <- df[,-1] %>% scale() %>% as.data.frame()


# Question 3

# Modelling 

h2o.init()
h2o_data <- df %>% as.h2o()

# Splitting the data 
h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'cty'
features <- df %>% select(-cty) %>% names()


# Fitting h2o model ----
model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))



# Question 4

target <- 'cty'
features <- df %>% select(year,cyl, displ) %>% names()

model <- h2o.glm(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  nfolds = 10, seed = 123,
  lambda = 0, compute_p_values = T)

# Question 5 

model@model$coefficients_table %>%
  as.data.frame() %>%
  dplyr::select(names,p_value) %>%
  mutate(p_value = round(p_value,3)) %>%
  .[-1,] %>%
  arrange(desc(p_value))

# 'cyl' and 'displ' variables are significant (<0.05) however, year variable is not significant.




