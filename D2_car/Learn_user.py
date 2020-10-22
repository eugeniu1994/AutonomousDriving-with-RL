import tensorflow as tf
import numpy as np
import os
#from D2_car.car_env import CarEnv
from car_env import CarEnv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import getch

keys = {'a': -1,'d': 0,'w':1}

def record_user(dir='./saved_data'):
    sarsa_pairs = []
    print("press + to save")
    esc = False
    while not esc:
        done = False
        _last_obs = env.reset()
        while not done:
            env.render()
            action = None
            while action is None:
                keys_pressed = getch.getch()
                if keys_pressed is '+':
                    esc = True
                    break

                pressed = [x for x in keys if x in keys_pressed]
                action = keys[pressed[0]] if len(pressed) > 0 else None

            if esc:
                print("ENDING")
                done = True
                break

            obs, reward, done = env.step(action)

            sarsa = (_last_obs, action)
            _last_obs = obs
            sarsa_pairs.append(sarsa)
            print(sarsa)

        if esc:
            break

    print("SAVING")
    sarsa_pairs = np.array(sarsa_pairs)
    with open(os.path.join(dir, 'data.npy'), 'wb') as f:
        np.save(f, sarsa_pairs)

    print('Done')

def process_data(dir='./saved_data'):
    states, actions = [], []
    with open('./saved_data/data.npy', 'rb') as f:
        data = np.load(f,allow_pickle=True)
        shard_states, unprocessed_actions = zip(*data)
        shard_states = [x.flatten() for x in shard_states]

        # Add the shard to the dataset
        states.extend(shard_states)
        actions.extend(unprocessed_actions)

    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions)

    print('states:{}, actions:{}'.format(np.shape(states), np.shape(actions)))

    num_bins = 25
    samples_per_bin = 250# 200
    hist, bins = np.histogram(actions, num_bins)
    center = (bins[:-1] + bins[1:]) * 0.5
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(actions), np.max(actions)), (samples_per_bin, samples_per_bin))
    print('total data:', len(data))
    plt.show()

    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(actions)):
            if actions[i] >= bins[j] and actions[i] <= bins[j + 1]:
                list_.append(i)
        list_ = shuffle(list_)
        list_ = list_[samples_per_bin:]
        remove_list.extend(list_)

    print('removed:', len(remove_list))
    actions = np.delete(actions, remove_list)
    states = np.delete(states, remove_list, axis=0)
    print('actions remaining:{}, states:{}'.format(np.shape(actions),np.shape(states)))

    hist, _ = np.histogram(actions, (num_bins))
    plt.bar(center, hist, width=0.05)
    plt.plot((np.min(actions), np.max(actions)), (samples_per_bin, samples_per_bin))
    plt.show()

    return states, actions

def create_model():
    state_ph = tf.placeholder(tf.float32, shape=[None, 5])

    with tf.variable_scope("layer1"):
        hidden = tf.layers.dense(state_ph, 128, activation=tf.nn.relu)

    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)

    with tf.variable_scope("layer3"):
        logits = tf.layers.dense(hidden, 3)

    with tf.variable_scope("output"):
        action = tf.argmax(input=logits, axis=1)

    return state_ph, action, logits

def create_training(logits):
    label_ph = tf.placeholder(tf.int32, shape=[None])

    with tf.variable_scope("loss"):
        onehot_labels = tf.one_hot(indices=tf.cast(label_ph, tf.int32), depth=3)

        loss = tf.losses.softmax_cross_entropy(
            onehot_labels=onehot_labels, logits=logits)
        loss = tf.reduce_mean(loss)

        tf.summary.scalar('loss', loss)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss=loss)

    return train_op, loss, label_ph

def run_main():
    state_data, action_data = process_data()

    x, model, logits = create_model()
    train, loss, labels = create_training(logits)

    sess = tf.Session()

    # Create summaries
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.global_variables_initializer())

    tick = 0
    while True:
        done = False
        obs = env.reset()
        while not done:
            env.render()

            if tick<1500:#train
                #batch_index = np.random.choice(len(state_data), 64)
                #state_batch, action_batch = state_data[batch_index], action_data[batch_index]

                # Train the model.
                _, cur_loss, cur_summaries = sess.run([train, loss, merged], feed_dict={
                    x:  state_data,
                    labels: action_data
                })
                print("Loss: {}".format(cur_loss))
                train_writer.add_summary(cur_summaries, tick)
            else:
                print('Testing--')
                print()

            action = sess.run(model, feed_dict={x: [obs.flatten()]})[0]

            obs, reward, done = env.step(action)

            tick += 1

if __name__ == '__main__':
    DISCRETE_ACTION = True
    env = CarEnv(discrete_action=DISCRETE_ACTION)

    #record_user()

    run_main()