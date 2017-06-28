library(tensorflow)
input_dataset <- tf$examples$tutorials$mnist$input_data
mnist <- input_dataset$read_data_sets("MNIST-data", one_hot = TRUE)

sess <- tf$InteractiveSession()

x <- tf$placeholder(tf$float32, shape(NULL, 784L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 10L))
W <- tf$Variable(tf$zeros(shape(784L, 10L)))
b <- tf$Variable(tf$zeros(shape(10L)))

sess$run(tf$global_variables_initializer())

y <- tf$nn$softmax(tf$matmul(x,W) + b)
cross_entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices=1L))

optimizer <- tf$train$GradientDescentOptimizer(0.05)
train_step <- optimizer$minimize(cross_entropy)

for (i in 1:1000) {
  batches <- mnist$train$next_batch(100L)
  batch_xs <- batches[[1]]
  print(head(batch_xs))
  batch_ys <- batches[[2]]
  print(head(batch_ys))
  sess$run(train_step,
           feed_dict = dict(x = batch_xs, y_ = batch_ys))
}

correct_prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct_prediction, tf$float32))
accuracy$eval(feed_dict=dict(x = mnist$test$images, y_ = mnist$test$labels))
