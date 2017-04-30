import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from sklearn import linear_model

df = pd.read_csv("data.csv")
contents = df['content']

print(contents[0])

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
data_vector = vectorizer.fit_transform(df.content)

data_contents = data_vector[:10000].toarray()
print(data_contents[0])


# Set parameters
learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2

# TF graph input
x = tf.placeholder(tf.float32, [None, 169465]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.string, [None, 11]) # 0-9 digits recognition => 10 classes

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([169465, 11]))
b = tf.Variable(tf.zeros([11]))

# Construct a linear model
model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
# Cross entropy
cost_function = -tf.reduce_sum(y*tf.log(model))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per eiteration step
        if iteration % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print ("Tuning completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))