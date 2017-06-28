input_dataset <- tf$examples$tutorials$mnist$input_data
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

sess <- tf$InteractiveSession()

x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))





K=4
L=8
M=12

W1 = tf$Variable(tf$truncated_normal(shape(5,5,1,4),stddev = 0.1))
B1 = tf$Variable(tf$ones(K)/10)

W2 = tf$Variable(tf$truncated_normal(shape(5,5,K,L),stddev = 0.1))
B2 = tf$Variable(tf$ones(L)/10)

W3 = tf$Variable(tf$truncated_normal(shape(4,4,L,M),stddev = 0.1))
B3 = tf$Variable(tf$ones(M)/10)

N=200

W4 = tf$Variable(tf$truncated_normal(shape(7*7*M,N),stddev = 0.1))
B4 = tf$Variable(tf$ones(N)/10)

W5 = tf$Variable(tf$truncated_normal(shape(N,10),stddev = 0.1))
B5 = tf$Variable(tf$zeros(10)/10)


conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides=c(1L, 1L, 1L, 1L), padding='SAME')
}

# max_pool_2x2 <- function(x) {
#   tf$nn$max_pool(
#     x, 
#     ksize=c(1L, 2L, 2L, 1L),
#     strides=c(1L, 2L, 2L, 1L), 
#     padding='SAME')
# }


keep_prob <- tf$placeholder(tf$float32)

x_image <- tf$reshape(x, shape(-1L, 28L, 28L, 1L))


Y_1 <- tf$nn$relu(conv2d(x_image,W1) + B1)
# Y1_image <-  max_pool_2x2(Y_1)



Y_2 <- tf$nn$relu(tf$nn$conv2d(Y_1, W2, strides=c(1L, 2L, 2L, 1L), padding='SAME') + B2)
Y_3 <- tf$nn$relu(tf$nn$conv2d(Y_2, W3, strides=c(1L, 2L, 2L, 1L), padding='SAME') + B3)


YY = tf$reshape(Y_3,shape(-1,7*7*M))
Y_4 = tf$nn$relu((tf$matmul(YY,W4)+B4))
Y = tf$nn$softmax(tf$matmul(Y_4,W5)+B5)



cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(Y), reduction_indices=1L))

optimizer <- tf$train$GradientDescentOptimizer(0.05)
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)


correct_prediction <- tf$equal(tf$argmax(Y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$global_variables_initializer())


# for (i in 1:20000) {
#   batches <- mnist$train$next_batch(100L)
#   batch_xs <- batches[[1]]
#   #print(head(batch_xs))
#   batch_ys <- batches[[2]]
#   #print(head(batch_ys))
#   sess$run(train_step,
#            feed_dict = dict(x = batch_xs, y_ = batch_ys))
# }


for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

test_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))


#AFter Regularization


# Y_1_drop <- tf$nn$dropout(Y_1,keep_prob)
# 
# Y_2 <- tf$nn$relu(tf$nn$conv2d(Y_1_drop, W2, strides=c(1L, 2L, 2L, 1L), padding='SAME') + B2)
# 
# Y_2_drop <- tf$nn$dropout(Y_2,keep_prob)
# 
# Y_3 <- tf$nn$relu(tf$nn$conv2d(Y_2_drop, W3, strides=c(1L, 2L, 2L, 1L), padding='SAME') + B3)
# Y_3_drop <- tf$nn$dropout(Y_3,keep_prob)
# 
# YY = tf$reshape(Y_3_drop,shape(-1,7*7*M))
# YY_drop <- tf$nn$dropout(YY,keep_prob)

Y_4 = tf$nn$relu((tf$matmul(YY,W4)+B4))
Y_4_drop <- tf$nn$dropout(Y_4,keep_prob)

Y = tf$nn$softmax(tf$matmul(Y_4_drop,W5)+B5)

cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(Y), reduction_indices=1L))

optimizer <- tf$train$GradientDescentOptimizer(0.05)
train_step <- tf$train$AdamOptimizer(1e-4)$minimize(cross_entropy)


correct_prediction <- tf$equal(tf$argmax(Y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
sess$run(tf$global_variables_initializer())


# for (i in 1:20000) {
#   batches <- mnist$train$next_batch(100L)
#   batch_xs <- batches[[1]]
#   #print(head(batch_xs))
#   batch_ys <- batches[[2]]
#   #print(head(batch_ys))
#   sess$run(train_step,
#            feed_dict = dict(x = batch_xs, y_ = batch_ys))
# }


for (i in 1:20000) {
  batch <- mnist$train$next_batch(50L)
  if (i %% 100 == 0) {
    train_accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep_prob = 1))
    cat(sprintf("step %d, training accuracy %g\n", i, train_accuracy))
  }
  train_step$run(feed_dict = dict(
    x = batch[[1]], y_ = batch[[2]], keep_prob = 0.5))
}

test_accuracy <- accuracy$eval(feed_dict = dict(
  x = mnist$test$images, y_ = mnist$test$labels, keep_prob = 1.0))
cat(sprintf("test accuracy %g", test_accuracy))
