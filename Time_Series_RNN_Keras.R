library(keras)
library(dplyr)
library(hydroGOF) # for mse

airlines <-  AirPassengers
#write.csv(airlines,"air.csv")
plot.ts(airlines)

#air_mat <- as.numeric(airlines)
minx = min(airlines)
maxx = max(airlines)

range01 <- function(x){(x-min(x))/(max(x)-min(x))}
revert_range <- function(x) {x*(maxx-minx)+minx}

air_mat <- as.numeric(range01(airlines)) 
  
  

#Create Data set with target = t+1 
lookback = 1
air_data <- data.frame(first = air_mat, second = c(air_mat[(lookback+1):length(air_mat)],rep(NA,lookback)))
air_data <- air_data[-c(nrow(air_data):(nrow(air_data)-lookback+1)),]
air_data <- air_data[-c(1,2),]

train_pct = 0.67
train_index = as.integer(train_pct*nrow(air_data))
test_index = train_index+1


train <- air_data[1:train_index,,]
test <- air_data[test_index:nrow(air_data),,]


#Shape array according to input and output required
x_train <- array(train$first,dim = c(nrow(train),1,1))
#y_train <- train$second
y_train <- array(train$second,dim = c(nrow(train)))

x_test <- array(test$first,dim = c(nrow(test),1,1))

#In this case dim of x and y should be the same 
dim(x_train)
dim(y_train)

#Number of neurons
HIDDEN_SIZE = 4


#initiate model
model = keras_model_sequential()

#Define number of lstm layers and their architecture
model %>% layer_lstm(HIDDEN_SIZE,input_shape = c(1,lookback)) 

#Last layer 
model %>% layer_dense(units = 1) %>% layer_activation("sigmoid")

#Compile the model and get weights and constants
model %>% compile(
  loss = "mean_squared_error",
  optimizer = "adam"
)


#Train the model
model %>% fit(
  x = x_train,
  y = y_train,
  epochs = 100,
  batch_size = 1,
  verbose = 2
)



trainPred <- model$predict(x_train)
testPred <- model$predict(x_test)

# Change dimension to calculate mse
x_train <- array(train$first,dim = c(nrow(train),1))
mse(trainPred,x_train)
x_test <- array(test$first,dim = c(nrow(test),1))
mse(testPred,x_test)

pred_train_actual <- revert_range(trainPred)
pred_test_actual <- revert_range(testPred)
