import sys
import time
import numpy as np
from copy import deepcopy
from utils import calculate_perplexity, get_ptb_dataset, Vocab
from utils import ptb_iterator, sample
from model import LanguageModel
import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss

debug=True
"""Loads starter word-vectors and train/dev/test data."""
vocab = Vocab()
vocab.construct(get_ptb_dataset('train'))
encoded_train = np.array([vocab.encode(word) for word in get_ptb_dataset('train')],dtype=np.int32)
encoded_valid = np.array([vocab.encode(word) for word in get_ptb_dataset('valid')],dtype=np.int32)
encoded_test = np.array([vocab.encode(word) for word in get_ptb_dataset('test')],dtype=np.int32)
if debug:                                            
  num_debug = 1024                                   
  encoded_train = encoded_train[:num_debug]
  encoded_valid = encoded_valid[:num_debug]
  encoded_test  = encoded_test[:num_debug]  

print '****** LOADED DATA'
'''**********************************************************************************************************'''

# Hyper Parameters

lr=0.006
batch_size = 64
embed_size = 50
hidden_size = 100
num_steps = 10
max_epochs = 100
early_stopping = 2
dropout = 0.9
# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.1
# lr = tf.train.exponential_decay(starter_learning_rate, global_step,25, 0.94, staircase=True)



# add placeholders

input_placeholder = tf.placeholder(tf.int32,(None, num_steps))
labels_placeholder = tf.placeholder(tf.int32,(None, num_steps))# Please note that float32 is mentioned but I used int32
dropout_placeholder = tf.placeholder(tf.float32,())


# add Embeddings

with tf.device('/cpu:0'):
  np.random.seed(8)
  embeddings = tf.Variable(tf.random_uniform((len(vocab), embed_size),-1.0,1.0,seed = 8))
  embed =tf.nn.embedding_lookup(embeddings,input_placeholder)
  inputs = [tf.squeeze(i,[1]) for i in tf.split(embed,num_steps,1)]  #can use instead tf.unstack 

# initiallize Variables
# H  = tf.Variable(tf.random_normal((hidden_size, hidden_size)),dtype = tf.float32)
# I  = tf.Variable(tf.random_normal((embed_size, hidden_size)),dtype = tf.float32)
# b_1= tf.Variable(tf.random_normal((1,hidden_size)),dtype = tf.float32)
with tf.variable_scope('lstm1'):
  U  = tf.Variable(tf.random_normal((hidden_size, len(vocab))),dtype = tf.float32)
  b_2= tf.Variable(tf.random_normal((1,len(vocab))),dtype = tf.float32)
  LSTM_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(hidden_size)
  initial_hidden_state = tf.zeros((batch_size,hidden_size))
  initial_current_state = tf.zeros((batch_size,hidden_size))
  initial_state = initial_hidden_state,initial_current_state
  final_state = None

def LSTM_fun(INPUT):
  OUTPUT = []
  with tf.variable_scope('lstm1') as scope:
    U  = tf.Variable(tf.random_normal((hidden_size, len(vocab))),dtype = tf.float32)
    b_2= tf.Variable(tf.random_normal((1,len(vocab))),dtype = tf.float32)
    state = initial_state
    for i in range(num_steps):
      if i>0:
        scope.reuse_variables()
      output, state = LSTM_cell(INPUT[i],state)
      pred = tf.matmul(output,U)+b_2
      OUTPUT.append(pred)
      final_state = state
  return (OUTPUT, final_state)

# def LSTM_fun(params, inp):
#   inp = tf.unstack(inp,num_steps,1)
#   outputs, states = tf.contrib.static_rnn(LSTM_cell, inp, dtype = tf.float32)
#   return tf.matmul(outputs,U)+b_2

logits, final_state = LSTM_fun(inputs)
prediction = [tf.nn.softmax(tf.cast(o, 'float64')) for o in logits]

output = tf.reshape(tf.concat( logits,1 ), [-1, len(vocab)])
output = tf.reshape(output,shape = (batch_size,num_steps,len(vocab)))
output = tf.convert_to_tensor(output)
seq_weight = tf.ones((batch_size,num_steps))
loss_op = sequence_loss(output,labels_placeholder,seq_weight)

train_op = tf.train.AdamOptimizer(lr).minimize(loss_op)

print '****** MODEL DEFINED'
'''**********************************************************************************************************'''

def run_epoch(session, data, train_op=None, verbose=10):
  dp = dropout
  if not train_op:
    train_op = tf.no_op()
    dp = 1
  total_steps = sum(1 for x in ptb_iterator(data, batch_size, num_steps))
  total_loss = []
  state = initial_state[0].eval(),initial_state[1].eval()
  for step, (x, y) in enumerate(ptb_iterator(data, batch_size, num_steps)):
    feed = {input_placeholder: x,labels_placeholder: y,initial_state: state, dropout_placeholder: dp}
    loss, state, _ = session.run([loss_op, final_state, train_op], feed_dict=feed)
    total_loss.append(loss)
    if verbose and step % verbose == 0:
        sys.stdout.write('\r{} / {} : pp = {}'.format(
            step, total_steps, np.exp(np.mean(total_loss))))
        sys.stdout.flush()
  if verbose:
    sys.stdout.write('\r')
  return np.exp(np.mean(total_loss))

def generate_text(session, starting_text='<eos>',stop_length=100, stop_tokens=None, temp=1.0):
  state = initial_state.eval()
  tokens = [[vocab.encode(word)] for word in starting_text.split()]
  for i in xrange(stop_length):
    feed_dict = {input_placeholder : [tokens[-1]], initial_state:state ,dropout_placeholder:1.0}
    y_pred, state = session.run([prediction,final_state],feed_dict)
    y_pred = y_pred[-1]
    next_word_idx = sample(y_pred[0], temperature=temp)
    tokens.append([next_word_idx])
    if stop_tokens and vocab.decode(tokens[-1][0]) in stop_tokens:
      break
  output = [vocab.decode(word_idx[0]) for word_idx in tokens]
  return output

def generate_sentence(session, *args, **kwargs):
  """Convenice to generate a sentence from the model."""
  return generate_text(session, *args, stop_tokens=['<eos>'], **kwargs)

'''**********************************************************************************************************'''

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with tf.Session() as session:
  best_val_pp = float('inf')
  best_val_epoch = 0
  
  Tpp = []
  Vpp = []

  session.run(init)
  for epoch in xrange(max_epochs):
    start = time.time()
    ###
    train_pp = run_epoch(session, encoded_train,train_op=train_op)
    valid_pp = run_epoch(session, encoded_valid)
    print 'Training perplexity: {}'.format(train_pp)
    print 'Validation perplexity: {}'.format(valid_pp)
    Tpp.append(train_pp)
    Vpp.append(valid_pp)
    if valid_pp < best_val_pp:
      best_val_pp = valid_pp
      best_val_epoch = epoch
      saver.save(session, 'ptb_rnnlm.weights')
    if epoch - best_val_epoch > early_stopping:
      break
    print 'Total time: {}'.format(time.time() - start)
  print 'Train PP: ',Tpp
  print 'Validation PP: ',Vpp
  saver.restore(session, 'ptb_rnnlm.weights')
  test_pp = run_epoch(session, encoded_test)
  print '=-=' * 5
  print 'Test perplexity: {}'.format(test_pp)
  print '=-=' * 5











