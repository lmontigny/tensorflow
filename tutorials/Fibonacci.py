f = [tf.constant(1),tf.constant(1)]

for i in range(2,10):
    temp = f[i-1] + f[i-2]
    f.append(temp)

with tf.Session() as sess:
    result = sess.run(f)
    print result
