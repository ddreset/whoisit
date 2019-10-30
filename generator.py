import tensorflow as tf
import numpy as np


class Sample_generator:

    def __init__(self, input_size, num_intra_class=10, num_inter_class=20):

        # input_size is usually (2,256)
        self.input_size = input_size
        self.num_intra_class = int(num_intra_class)
        self.num_inter_class = int(num_inter_class)

        self.inputs = tf.placeholder(tf.float32, [None, None, self.input_size[0], self.input_size[1]])
        self.samples = self.build_all(self.inputs)

    def build_all(self, input):
        # generate sample sets for each pic
        # in each set:
        # first (num_intra_class) pics are from the same class
        # the following (num_inter_class) pics are from other classes 
        # in training, model should predict if two pics from the same class or not
        sampled = self.build_sample_graph(input)
        # randomly shuffle blocks in sample sets for 100 times.
        # if input_size is (2,265) each pic has 2 blocks
        # (still not sure why do this, data augmentation？)
        shifted = self.build_shift_graph(tf.reshape(sampled, [-1, self.input_size[0], self.input_size[1]]))
        return tf.reshape(shifted, [-1, self.num_intra_class + self.num_inter_class, self.input_size[0], self.input_size[1]])

    def build_sample_graph(self, input):

        def body_intra(X):
            # X contains one class's data
            # tf.shape(X)[0] means number of pictures in this class
            # tf.shape(input)[1] means number of pictures in each class
            # this function chooses random pictures from current class
            selected_intra_indices = tf.random_uniform([tf.shape(input)[1] * self.num_intra_class], 0, tf.shape(X)[0], dtype=tf.int32)
            return tf.gather(X, selected_intra_indices)

        intra_sample = tf.map_fn(body_intra, input, parallel_iterations=512, back_prop=False)

        flat = tf.reshape(input, [-1, self.input_size[0], self.input_size[1]])

        def body_inter(index):
            # index is the index of current class
            # all_indices contains all pictures' indices except current class
            # this function chooses random pictures from all classes except current class
            all_indices = tf.concat([
                tf.range(index * tf.shape(input)[1]),
                tf.range((index + 1) * tf.shape(input)[1], tf.shape(input)[0] * tf.shape(input)[1])], axis=0)
            selected_indices = tf.random_uniform([tf.shape(input)[1] * self.num_inter_class], 0, tf.shape(all_indices)[0], dtype=tf.int32)
            return tf.gather(flat, tf.gather(all_indices, selected_indices))

        inter_sample = tf.map_fn(body_inter, tf.range(tf.shape(input)[0]), dtype=tf.float32, parallel_iterations=512, back_prop=False)

        # new shape: [tf.shape(input)[0]*tf.shape(input)[1], num_intra_class+num_inter_class, input_size[0], input_size[1]]
        # generate (tf.shape(input)[1]) sets of training samples for each class
        # in each training sample set, only first (num_intra_class) pictures are from current class
        return tf.concat([
            tf.reshape(intra_sample, [-1, self.num_intra_class, self.input_size[0], self.input_size[1]]),
            tf.reshape(inter_sample, [-1, self.num_inter_class, self.input_size[0], self.input_size[1]])], axis=1)

    def build_shift_graph(self, input):
        # input's shape is usually (class_number * picture_num_in_each_class * (10+20), 2, 256)

        # input_size[1] is usually 256
        valid_break_points = tf.constant([
            int(self.input_size[1] - 5),
            int(self.input_size[1] - 4),
            int(self.input_size[1] - 3),
            int(self.input_size[1] - 2),
            int(self.input_size[1] - 1),
            0, 1, 2, 3, 4, 5], dtype=tf.int32)

        def shuffle_block(x):
            break_point = valid_break_points[tf.random_uniform((), 0, tf.shape(valid_break_points)[0], dtype=tf.int32)]
            a = x[:, :, break_point:]
            b = x[:, :, 0:break_point]
            return tf.concat([a, b], axis=2)

        # s is number of pictures in total
        s = tf.shape(input)[0]
        cut = tf.cast(s / 2, tf.int32)

        def shuffle_and_roll_half(x):
            indices = tf.range(s, dtype=tf.int32)
            shuffled_indices = tf.random_shuffle(indices)
            reversed_indices = tf.contrib.framework.argsort(shuffled_indices)

            # shuffle pictures
            x = tf.gather(x, shuffled_indices)

            # shuffle half pic's blocks
            # each pic has 2 blocks
            a = shuffle_block(x[0: cut, ...])
            b = x[cut:, ...]

            # put shuffled pics back into original positions
            x = tf.concat([a, b], axis=0)
            x = tf.gather(x, reversed_indices)
            return x

        X = input
        for i in range(100):
            X = shuffle_and_roll_half(X)
        out = X
        return out

    def generate(self, sess, seed):
        return sess.run(self.samples, feed_dict={self.inputs: seed})


if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    np.set_printoptions(threshold=np.nan)

    data = np.reshape(np.arange(4 * 5 * 2 * 10), [4, 5, 2, 10])
    print(data)
    generator = Sample_generator((2, 10), 1, 2)
    gened = generator.generate(sess, data)
    print(gened)

    shift = tf.random_uniform((), 0, data.shape[3], dtype=tf.int32)
    print(sess.run(shift))

    sess.close()
