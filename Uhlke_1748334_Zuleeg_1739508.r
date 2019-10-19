# Loading required packages
if (!require("dplyr")) install.packages("dplyr")
if (!require("caret")) install.packages("caret")
if (!require("mlr")) install.packages("mlr")
library(dplyr)
library(caret)
library(mlr)

#setting seed for reproducable results
set.seed(1234)

#loading data
dat <- read.csv2("file:///C:/Users/pc/Desktop/Master_Statistik/Energy-Analytics/smart_meter_data.csv")
dat_survey <- read.csv2("file:///C:/Users/pc/Desktop/Master_Statistik/Energy-Analytics/survey_with_smart_meter_data.csv")

#deleting first column
dat <- dat[,-1]

# Feature extraction
Features <- function(df){
  
  #create a matrix with 7 columns for each day
  consump <- matrix(as.numeric(df))
  
  D <- data.frame(c_week = mean(consump, na.rm = T))
  
  consump <- as.data.frame(consump)
  colnames(consump) <- "consumption"
  index_max <- which(consump$consumption == max(consump$consumption))
  if (length(index_max > 1)){
    index_max <- index_max[1]
    
  }
  
  D$f15_max <- index_max
  D$sd_week <- sd(consump$consumption, na.rm = T)
  D$f_above_2kw <- length(which(consump$consumption > 2))
  D$f_above_1kw <- length(which(consump$consumption > 1))
  D$f_above_mean_sd <- length(which(consump$consumption > (mean(consump$consumption) + D$sd_week)))
  
  rm(consump)
  
  consump <- matrix(as.numeric(df), ncol = 7)
  
  # define some times
  weekday <- 1:(5*2*24)
  weekend <- (5*2*24+1):336
  night <-  (1*2+1):(6*2)
  morning <- (6*2+1):(10*2)
  noon <- (10*2+1):(14*2)
  afternoon <- (14*2+1):(18*2)
  evening <- (18*2+1):(22*2)
  
  #calculate consumption features
  D$c_night <-     mean(consump[night,     1:7], na.rm = T)
  D$c_morning <-   mean(consump[morning,   1:7], na.rm = T)
  D$c_noon <-      mean(consump[noon,      1:7], na.rm = T)
  D$c_afternoon <- mean(consump[afternoon, 1:7], na.rm = T)
  D$c_evening <-   mean(consump[evening,   1:7], na.rm = T)
  
  #calculate statistical features
  D$s_we_max <- max(consump[weekend], na.rm = T)
  D$s_we_min <- min(consump[weekend], na.rm = T)
  D$s_wd_max <- max(consump[weekday], na.rm = T)
  D$s_wd_min <- min(consump[weekday], na.rm = T)
  
  #calculate relations with max
  D$r_max_wd_we <- D$s_wd_max / D$s_we_max
  D$r_max_wd_we <- ifelse(is.na(D$r_max_wd_we), 0, D$r_max_wd_we)
  
  return(D)
}

Features(dat[2,])

features <- Features(dat[1,])
for(i in 2:nrow(dat)){
  features <- rbind(features, Features(dat[i,]))
}

# preparing our final dataset
features$ID <- dat_survey[,1]
features$ID.single <- dat_survey[,11]
features$single_dummy <- ifelse(features$ID.single=="Single",1,0)
features <- features[,-18]

# splitting our data into training and test 
s <- .8
n <- nrow(features)

features <- features %>% dplyr::mutate(single_dummy = as.factor(features$single_dummy))

train <- sample(1:n,s*n,replace=FALSE)

f_train <- features[train,] # training data
f_test <- features[-train,] # test data

#logistic regression model 
log_test <- glm(single_dummy ~ c_week + f15_max + sd_week + f_above_2kw + f_above_1kw +
                  f_above_mean_sd + c_night + c_morning + c_noon + c_afternoon +
                  c_evening + s_we_max + s_we_min + s_wd_max + s_wd_min + r_max_wd_we, data = f_train, family = binomial(link = "logit"))
summary(log_test)

# transforming coefficients for interpretation purposes
exp(log_test$coefficients)

# prediction
eval <- predict(log_test, newdata = f_test, type = "response")
length(eval)

#we have to modify our results in order to compute the following confusion matrix
new_eval <- as.numeric(eval>0.5)
new_eval <- as.factor(new_eval)

# create confusionMatrix
m <- confusionMatrix(data = new_eval, reference = f_test$single_dummy)
m

## Cross validation process
# generating 5 random subsamples 
features$index <- sample(1:5, n, replace = TRUE)
head(features)

# cross validation simulation
for (foldIndex in 1:5){
  
  train_k <- features %>% dplyr::filter(index != foldIndex) %>% dplyr::select(-index)
  nrow(train_k)
  test_k <- features %>% dplyr::filter(index == foldIndex) %>% dplyr::select(-index)
  nrow(test_k)
  glm_fit <- glm(single_dummy~c_week + f15_max + sd_week + f_above_2kw + f_above_1kw +
               f_above_mean_sd + c_night + c_morning + c_noon + c_afternoon +
               c_evening + s_we_max + s_we_min + s_wd_max + s_wd_min + r_max_wd_we, data = train_k,
               family = binomial(link = "logit"), na.action = na.pass)
  
  test_labels <- predict(glm_fit, test_k, type = "response")
  new_test_labels <- as.numeric(test_labels>0.5)
  new_test_labels <- as.factor(new_test_labels)
  
  result <- confusionMatrix(data = new_test_labels, reference = test_k$single_dummy)
  print(confusionMatrix(new_test_labels, test_k$single_dummy))
  result <- result$overall["Accuracy"]
  results[foldIndex,] <- cbind(results, result)
}

# Accuracy values of cross validation
co <- c(0.8502, 0.8302, 0.7933, 0.7979, 0.8167) 
# Mean of the cross validation values
co_mean <- mean(co) 

# calculate confidence interbal by using variance and standard deviation
v <- var(co[1:5,]) # variance
s <- sd(co[1:5]) # standard deviation
con_up <- co_mean + 1.96 * s/5^(1/2) # upper confidence interval
con_down <- co_mean - 1.96 * s/5^(1/2) # below vonfidence Interval

# plotting results
barplot(c(0.8502, 0.8302, 0.7933, 0.7979, 0.8167),
        main = "acc Barplot",col=c("slategray4","slategray3","slategray2","slategray1", "slategray"),
        beside = TRUE,ylim = c(0,1),xlab ="cross validation runs",ylab = "Acc",names = c("1","2", "3", "4", "5"))
abline(h=mean(c(0.8502, 0.8302, 0.7933, 0.7979, 0.8167)),col="red",lwd=2)
abline(h=con_up, col = "gold", lwd = 2, lty = 2)
abline(h=con_down, col = "gold", lwd = 2, lty = 2)


# random guess
number <- length(features$single_dummy)

random_gues <- sample(c(0,1), size = number, replace=TRUE) # random guess generator
random_gues <- as.factor(random_gues) 
table(random_gues)


random_cm <- confusionMatrix(random_gues, features$single_dummy)
random_cm_acc <- random_cm$overall["Accuracy"]

# biased random guess

anteil <- table(features$single_dummy)/number
anteil["0"]

anteil = c(anteil["0"], anteil["1"]) #your probabilities
biased_random = sample(c(0,1),
           size = number, #1000 times sampled from `mynumbers`
           replace = T,
           prob = anteil)

biased_random <- as.factor(biased_random)

biased_cm <- confusionMatrix(biased_random, features$single_dummy)
biased_cm_acc <- biased_cm$overall["Accuracy"]

#combined results
barplot(c(co_mean, biased_cm_acc, random_cm_acc),col=c("cornflowerblue","forestgreen","purple"),
        beside = TRUE,ylim = c(0,1),xlab ="comparasion",ylab = "Accuracy",
        names = c("logistic","biased random", "random"))
abline(h=mean(c(0.8502, 0.8302, 0.7933, 0.7979, 0.8167)),col="black",lwd=2)
abline(h=con_up, col = "red", lwd = 2, lty = 2)
abline(h=con_down, col = "red", lwd = 2, lty = 2)





