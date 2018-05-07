''' Estimating Gait Normality Index based on Point Clouds using Deep Neural Network
    BSD 2-Clause "Simplified" License
    Author: Trong-Nguyen Nguyen'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf

from utils import *
from ops import *

'''============== DATA REGION ==============='''
loaded = np.load('dataset/DIRO_normalized_hists.npz')
data = loaded['data']
n_subject, n_gait, n_frame = data.shape[:3]
separation = loaded['split']
training_subjects = np.where(separation == 'train')[0]
test_subjects = np.where(separation == 'test')[0]

##for leave-one-out ONLY
#test_subjects = [0]
#training_subjects = np.setdiff1d(list(range(n_subject)), test_subjects)

print('training subjects: ' + str(training_subjects))

training_img_normal = data[training_subjects, 0]
test_img_normal = data[test_subjects, 0]
test_img_abnormal = data[test_subjects, 1:]

'''flatten data to 2D matrix'''
training_img_normal = training_img_normal.reshape((-1,256))
test_img_normal = test_img_normal.reshape((-1,256))
test_img_abnormal = test_img_abnormal.reshape((-1,256))

print('data shape:')
print(training_img_normal.shape)
print(test_img_normal.shape)
print(test_img_abnormal.shape)
print('')

'''============= MODEL REGION ================'''
tf.reset_default_graph()

xavier = True
leaky_param = 0.1
p_keep_dropout_const = 0.7
l2_lambda = 0.0001 #different from original work in Mathematica
learning_rate = 0.0001
max_epoch = 800
batch_size = 512

def ae4_sigmoid_nodrop(X):
     with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_s_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(tf.nn.sigmoid(h0), 128, 64, scope='ae4_s_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(tf.nn.sigmoid(h1), 64, 32, scope='ae4_s_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(tf.nn.sigmoid(h2), 32, 16, scope='ae4_s_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(tf.nn.sigmoid(h3), 16, 32, scope='ae4_s_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(tf.nn.sigmoid(h4), 32, 64, scope='ae4_s_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(tf.nn.sigmoid(h5), 64, 128, scope='ae4_s_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(tf.nn.sigmoid(h6), 128, 256, scope='ae4_s_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
         
def ae4_sigmoid_drop(X, p_keep_drop):
    with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_sd_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(tf.nn.dropout(tf.nn.sigmoid(h0), p_keep_drop), 128, 64, scope='ae4_sd_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(tf.nn.dropout(tf.nn.sigmoid(h1), p_keep_drop), 64, 32, scope='ae4_sd_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(tf.nn.dropout(tf.nn.sigmoid(h2), p_keep_drop), 32, 16, scope='ae4_sd_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(tf.nn.dropout(tf.nn.sigmoid(h3), p_keep_drop), 16, 32, scope='ae4_sd_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(tf.nn.dropout(tf.nn.sigmoid(h4), p_keep_drop), 32, 64, scope='ae4_sd_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(tf.nn.dropout(tf.nn.sigmoid(h5), p_keep_drop), 64, 128, scope='ae4_sd_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(tf.nn.dropout(tf.nn.sigmoid(h6), p_keep_drop), 128, 256, scope='ae4_sd_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)

def ae4_tanh_nodrop(X):
     with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_t_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(tf.nn.tanh(h0), 128, 64, scope='ae4_t_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(tf.nn.tanh(h1), 64, 32, scope='ae4_t_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(tf.nn.tanh(h2), 32, 16, scope='ae4_t_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(tf.nn.tanh(h3), 16, 32, scope='ae4_t_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(tf.nn.tanh(h4), 32, 64, scope='ae4_t_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(tf.nn.tanh(h5), 64, 128, scope='ae4_t_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(tf.nn.tanh(h6), 128, 256, scope='ae4_t_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
         
def ae4_tanh_drop(X, p_keep_drop):
    with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_td_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(tf.nn.dropout(tf.nn.tanh(h0), p_keep_drop), 128, 64, scope='ae4_td_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(tf.nn.dropout(tf.nn.tanh(h1), p_keep_drop), 64, 32, scope='ae4_td_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(tf.nn.dropout(tf.nn.tanh(h2), p_keep_drop), 32, 16, scope='ae4_td_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(tf.nn.dropout(tf.nn.tanh(h3), p_keep_drop), 16, 32, scope='ae4_td_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(tf.nn.dropout(tf.nn.tanh(h4), p_keep_drop), 32, 64, scope='ae4_td_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(tf.nn.dropout(tf.nn.tanh(h5), p_keep_drop), 64, 128, scope='ae4_td_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(tf.nn.dropout(tf.nn.tanh(h6), p_keep_drop), 128, 256, scope='ae4_td_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)

def ae4_lrelu_nodrop(X):
     with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_l_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(lrelu(h0), 128, 64, scope='ae4_l_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(lrelu(h1), 64, 32, scope='ae4_l_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(lrelu(h2), 32, 16, scope='ae4_l_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(lrelu(h3), 16, 32, scope='ae4_l_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(lrelu(h4), 32, 64, scope='ae4_l_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(lrelu(h5), 64, 128, scope='ae4_l_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(lrelu(h6), 128, 256, scope='ae4_l_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
         
def ae4_lrelu_drop(X, p_keep_drop):
    with tf.variable_scope(tf.get_variable_scope()) as scope:
         h0, w0, b0 = dense(X, 256, 128, scope='ae4_ld_h0', with_w = True, xavier = xavier) #128
         h1, w1, b1 = dense(tf.nn.dropout(lrelu(h0), p_keep_drop), 128, 64, scope='ae4_ld_h1', with_w = True, xavier = xavier) #64
         h2, w2, b2 = dense(tf.nn.dropout(lrelu(h1), p_keep_drop), 64, 32, scope='ae4_ld_h2', with_w = True, xavier = xavier) #32
         h3, w3, b3 = dense(tf.nn.dropout(lrelu(h2), p_keep_drop), 32, 16, scope='ae4_ld_h3', with_w = True, xavier = xavier) #16
         h4, w4, b4 = dense(tf.nn.dropout(lrelu(h3), p_keep_drop), 16, 32, scope='ae4_ld_h4', with_w = True, xavier = xavier) #32
         h5, w5, b5 = dense(tf.nn.dropout(lrelu(h4), p_keep_drop), 32, 64, scope='ae4_ld_h5', with_w = True, xavier = xavier) #64
         h6, w6, b6 = dense(tf.nn.dropout(lrelu(h5), p_keep_drop), 64, 128, scope='ae4_ld_h6', with_w = True, xavier = xavier) #128
         h7, w7, b7 = dense(tf.nn.dropout(lrelu(h6), p_keep_drop), 128, 256, scope='ae4_ld_h7', with_w = True, xavier = xavier) #256
         return h7, tf.nn.l2_loss(w0) + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3) + tf.nn.l2_loss(w4) + tf.nn.l2_loss(w5) \
                + tf.nn.l2_loss(w6) + tf.nn.l2_loss(w7)
    

X = tf.placeholder(tf.float32, shape=[None, 256])
prob_dropout = tf.placeholder_with_default(1.0, shape=()) #for dropout

X_ae4_s, l2reg_ae4_s = ae4_sigmoid_nodrop(X)
X_ae4_sd, l2reg_ae4_sd = ae4_sigmoid_drop(X, prob_dropout)
X_ae4_t, l2reg_ae4_t = ae4_tanh_nodrop(X)
X_ae4_td, l2reg_ae4_td = ae4_tanh_drop(X, prob_dropout)
X_ae4_l, l2reg_ae4_l = ae4_lrelu_nodrop(X)
X_ae4_ld, l2reg_ae4_ld = ae4_lrelu_drop(X, prob_dropout)

'''loss function'''
diff_ae4_s = X_ae4_s - X
loss_ae4_s = tf.reduce_mean(diff_ae4_s**2) + l2_lambda * l2reg_ae4_s
score_ae4_s = tf.reduce_mean(diff_ae4_s**2, axis = 1)

diff_ae4_sd = X_ae4_sd - X
loss_ae4_sd = tf.reduce_mean(diff_ae4_sd**2) + l2_lambda * l2reg_ae4_sd
score_ae4_sd = tf.reduce_mean(diff_ae4_sd**2, axis = 1)

diff_ae4_t = X_ae4_t - X
loss_ae4_t = tf.reduce_mean(diff_ae4_t**2) + l2_lambda * l2reg_ae4_t
score_ae4_t = tf.reduce_mean(diff_ae4_t**2, axis = 1)

diff_ae4_td = X_ae4_td - X
loss_ae4_td = tf.reduce_mean(diff_ae4_td**2) + l2_lambda * l2reg_ae4_td
score_ae4_td = tf.reduce_mean(diff_ae4_td**2, axis = 1)

diff_ae4_l = X_ae4_l - X
loss_ae4_l = tf.reduce_mean(diff_ae4_l**2) + l2_lambda * l2reg_ae4_l
score_ae4_l = tf.reduce_mean(diff_ae4_l**2, axis = 1)

diff_ae4_ld = X_ae4_ld - X
loss_ae4_ld = tf.reduce_mean(diff_ae4_ld**2) + l2_lambda * l2reg_ae4_ld
score_ae4_ld = tf.reduce_mean(diff_ae4_ld**2, axis = 1)

solver_ae4_s = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_s').minimize(loss_ae4_s)
solver_ae4_sd = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_sd').minimize(loss_ae4_sd)
solver_ae4_t = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_t').minimize(loss_ae4_t)
solver_ae4_td = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_td').minimize(loss_ae4_td)
solver_ae4_l = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_l').minimize(loss_ae4_l)
solver_ae4_ld = tf.train.RMSPropOptimizer(learning_rate, momentum = 0.9, name = 'solver_ae4_ld').minimize(loss_ae4_ld)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    '''training'''
    for i in range(max_epoch):
        indices = shuffle_draw_batch(training_img_normal.shape[0], batch_size)
        for idx in indices:
            _, val_loss_ae4_s = sess.run([solver_ae4_s, loss_ae4_s], feed_dict={X: training_img_normal[idx]})
            _, val_loss_ae4_sd = sess.run([solver_ae4_sd, loss_ae4_sd], feed_dict={X: training_img_normal[idx], prob_dropout: p_keep_dropout_const})
            _, val_loss_ae4_t = sess.run([solver_ae4_t, loss_ae4_t], feed_dict={X: training_img_normal[idx]})
            _, val_loss_ae4_td = sess.run([solver_ae4_td, loss_ae4_td], feed_dict={X: training_img_normal[idx], prob_dropout: p_keep_dropout_const})
            _, val_loss_ae4_l = sess.run([solver_ae4_l, loss_ae4_l], feed_dict={X: training_img_normal[idx]})
            _, val_loss_ae4_ld = sess.run([solver_ae4_ld, loss_ae4_ld], feed_dict={X: training_img_normal[idx], prob_dropout: p_keep_dropout_const})
        if (i+1) % 100 == 0:
            print('epoch %3d: sigmoid = (%.3f, %.3f), tanh = (%.3f, %.3f), lrelu = (%.3f, %.3f)' % \
                  (i+1, val_loss_ae4_s, val_loss_ae4_sd, val_loss_ae4_t, val_loss_ae4_td, val_loss_ae4_l, val_loss_ae4_ld))
    '''test on real data -> P(x = abnormal)'''
    print('')
    prob_abnormal_total = np.zeros(test_img_abnormal.shape[0])
    prob_normal_total = np.zeros(test_img_normal.shape[0])
    # sigmoid
    prob_abnormal = score_ae4_s.eval({X: test_img_abnormal}).reshape(-1)
    prob_normal = score_ae4_s.eval({X: test_img_normal}).reshape(-1)
    results_s4 = assessment_full(prob_abnormal, prob_normal, title = 'sigmoid (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # sigmoid + dropout
    prob_abnormal = score_ae4_sd.eval({X: test_img_abnormal, prob_dropout: 1.0}).reshape(-1)
    prob_normal = score_ae4_sd.eval({X: test_img_normal, prob_dropout: 1.0}).reshape(-1)
    results_sd4 = assessment_full(prob_abnormal, prob_normal, title = 'sigmoid + drop (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # tanh
    prob_abnormal = score_ae4_t.eval({X: test_img_abnormal}).reshape(-1)
    prob_normal = score_ae4_t.eval({X: test_img_normal}).reshape(-1)
    results_t4 = assessment_full(prob_abnormal, prob_normal, title = 'tanh (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # tanh + dropout
    prob_abnormal = score_ae4_td.eval({X: test_img_abnormal, prob_dropout: 1.0}).reshape(-1)
    prob_normal = score_ae4_td.eval({X: test_img_normal, prob_dropout: 1.0}).reshape(-1)
    results_td4 = assessment_full(prob_abnormal, prob_normal, title = 'tanh + drop (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # lrelu
    prob_abnormal = score_ae4_l.eval({X: test_img_abnormal}).reshape(-1)
    prob_normal = score_ae4_l.eval({X: test_img_normal}).reshape(-1)
    results_l4 = assessment_full(prob_abnormal, prob_normal, title = 'lrelu (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # lrelu + dropout
    prob_abnormal = score_ae4_ld.eval({X: test_img_abnormal, prob_dropout: 1.0}).reshape(-1)
    prob_normal = score_ae4_ld.eval({X: test_img_normal, prob_dropout: 1.0}).reshape(-1)
    results_ld4 = assessment_full(prob_abnormal, prob_normal, title = 'lrelu + drop (4)')
    prob_abnormal_total += prob_abnormal
    prob_normal_total += prob_normal
    # combination of 6 models
    results_combine = assessment_full(prob_abnormal_total, prob_normal_total, title = 'combination (4)')
