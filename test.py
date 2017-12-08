import tensorflow as tf
import load
input = tf.Variable([[1,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]], name="counter")
result = tf.slice(input, [0,2], [2,4])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
a = sess.run(result)
print(load.train_data[0])