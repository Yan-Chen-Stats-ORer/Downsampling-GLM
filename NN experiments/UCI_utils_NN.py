import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import numpy as np
import tensorflow as tf

# Define the neural network model
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(10, input_dim=input_dim, activation='relu'))  # Small network
    model.add(Dense(8, input_dim=input_dim, activation='relu'))  # Small network
    model.add(Dense(5, input_dim=input_dim, activation='relu'))  # Small network
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    return model

# Custom callback to store losses
class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        # print(f"Epoch {epoch+1}: Loss = {logs['loss']}")

# Modify the experiment_NN function
def experiment_NN_yeast_me2(alpha_list, feature, target, exp_num=100, eps=1e-5):
    all_custom_losses = []
    all_crossentropy_losses = []

    log_loss_custom = []
    log_loss_crossentropy = []
    log_loss_custom_std = []
    log_loss_crossentropy_std = []

    # Custom loss function based on objective_joint_MLE
    def custom_loss(alpha, eps, X_downsample, Y_downsample, model):
        def loss(y_true, y_pred):
            Fi = model(X_downsample)
            sum_term = tf.reduce_mean(Y_downsample * tf.math.log(1 - Fi) + 
                                    (1 - Y_downsample) * tf.math.log(alpha * Fi))
            integral_val_approx = tf.reduce_mean(1 - (1 - alpha) * Fi)
            return -(sum_term - tf.math.log(integral_val_approx))
        return loss

    for alpha in alpha_list:
        log_loss_custom_current = []
        log_loss_crossentropy_current = []

        custom_losses = []
        crossentropy_losses = []

        for i in range(exp_num):
            np.random.seed(i)
            X, X_test, Y, Y_test = train_test_split(feature, target, test_size=0.2, random_state=i)

            # Create downsample_set for conditional MLE 
            indices_Y_1 = np.where(Y == 1)[0]
            num_sample_Y_0 = int(np.sum(Y == 0) * alpha)
            indices_Y_0_sampled = np.random.choice(np.where(Y == 0)[0], num_sample_Y_0, replace=False)
            downsample_set = np.union1d(indices_Y_1, indices_Y_0_sampled)

            X_downsample = X[downsample_set]
            Y_downsample = Y[downsample_set]

            d = X.shape[1]

            # Save the model every 10 epochs
            checkpoint_custom = ModelCheckpoint(filepath=f'models_2/model_custom_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                save_freq='epoch',
                                                period=10,
                                                verbose=0)

            # Custom callback to store losses
            loss_history_custom = LossHistory()

            # Train using the custom loss function
            model_custom = create_nn_model(input_dim=d)
            model_custom.compile(optimizer=Adam(), 
                     loss=custom_loss(alpha, eps, X_downsample, Y_downsample, model_custom))

            model_custom.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0, 
                             callbacks=[checkpoint_custom, loss_history_custom])

            custom_losses.extend(loss_history_custom.losses)

            prediction_score_custom = model_custom.predict(X_test).flatten()
            loss_custom = log_loss(Y_test, prediction_score_custom)
            log_loss_custom_current.append(loss_custom)

            # Save the model every 10 epochs
            checkpoint_crossentropy = ModelCheckpoint(filepath=f'models_cross/model_crossentropy_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                      save_freq='epoch',
                                                      period=10,
                                                      verbose=0)

            # Custom callback to store losses
            loss_history_crossentropy = LossHistory()

            # Train using the standard cross-entropy loss
            model_crossentropy = create_nn_model(input_dim=d)
            model_crossentropy.compile(optimizer=Adam(), loss='binary_crossentropy')
            model_crossentropy.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0,
                                   callbacks=[checkpoint_crossentropy, loss_history_crossentropy])

            crossentropy_losses.extend(loss_history_crossentropy.losses)

            prediction_score_crossentropy = model_crossentropy.predict(X_test).flatten()
            loss_crossentropy = log_loss(Y_test, prediction_score_crossentropy)
            log_loss_crossentropy_current.append(loss_crossentropy)

        all_custom_losses.append(custom_losses)
        all_crossentropy_losses.append(crossentropy_losses)

        log_loss_custom.append(np.mean(log_loss_custom_current))
        log_loss_crossentropy.append(np.mean(log_loss_crossentropy_current))
        
        log_loss_custom_std.append(np.std(log_loss_custom_current))
        log_loss_crossentropy_std.append(np.std(log_loss_crossentropy_current))

    # Plot the log-loss for each epoch
    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors if needed
    linestyles = ['-', '--', '-.', ':']  # Different line styles for different loss functions

    for idx, alpha in enumerate(alpha_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[0]
        plt.plot(all_custom_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Custom Loss (alpha={alpha})')

        linestyle = linestyles[1]
        plt.plot(all_crossentropy_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Crossentropy Loss (alpha={alpha})')

    plt.xlabel('Epoch')
    plt.ylabel('Log-Loss')
    plt.title('Log-Loss (yeast_me2) for Custom and Crossentropy Loss Functions across Epochs')
    plt.legend()
    plt.show()

    return log_loss_custom, log_loss_crossentropy, log_loss_custom_std, log_loss_crossentropy_std

def experiment_NN_abalone_19(alpha_list, feature, target, exp_num=100, eps=1e-5):
    all_custom_losses = []
    all_crossentropy_losses = []

    log_loss_custom = []
    log_loss_crossentropy = []
    log_loss_custom_std = []
    log_loss_crossentropy_std = []

    # Custom loss function based on objective_joint_MLE
    def custom_loss(alpha, eps, X_downsample, Y_downsample, model):
        def loss(y_true, y_pred):
            Fi = model(X_downsample)
            sum_term = tf.reduce_mean(Y_downsample * tf.math.log(1 - Fi) + 
                                    (1 - Y_downsample) * tf.math.log(alpha * Fi))
            integral_val_approx = tf.reduce_mean(1 - (1 - alpha) * Fi)
            return -(sum_term - tf.math.log(integral_val_approx))
        return loss

    for alpha in alpha_list:
        log_loss_custom_current = []
        log_loss_crossentropy_current = []

        custom_losses = []
        crossentropy_losses = []

        for i in range(exp_num):
            np.random.seed(i)
            X, X_test, Y, Y_test = train_test_split(feature, target, test_size=0.2, random_state=i)

            # Create downsample_set for conditional MLE 
            indices_Y_1 = np.where(Y == 1)[0]
            num_sample_Y_0 = int(np.sum(Y == 0) * alpha)
            indices_Y_0_sampled = np.random.choice(np.where(Y == 0)[0], num_sample_Y_0, replace=False)
            downsample_set = np.union1d(indices_Y_1, indices_Y_0_sampled)

            X_downsample = X[downsample_set]
            Y_downsample = Y[downsample_set]

            d = X.shape[1]

            # Save the model every 10 epochs
            checkpoint_custom = ModelCheckpoint(filepath=f'models_2/model_custom_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                save_freq='epoch',
                                                period=10,
                                                verbose=0)

            # Custom callback to store losses
            loss_history_custom = LossHistory()

            # Train using the custom loss function
            model_custom = create_nn_model(input_dim=d)
            model_custom.compile(optimizer=Adam(), 
                     loss=custom_loss(alpha, eps, X_downsample, Y_downsample, model_custom))

            model_custom.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0, 
                             callbacks=[checkpoint_custom, loss_history_custom])

            custom_losses.extend(loss_history_custom.losses)

            prediction_score_custom = model_custom.predict(X_test).flatten()
            loss_custom = log_loss(Y_test, prediction_score_custom)
            log_loss_custom_current.append(loss_custom)

            # Save the model every 10 epochs
            checkpoint_crossentropy = ModelCheckpoint(filepath=f'models_cross/model_crossentropy_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                      save_freq='epoch',
                                                      period=10,
                                                      verbose=0)

            # Custom callback to store losses
            loss_history_crossentropy = LossHistory()

            # Train using the standard cross-entropy loss
            model_crossentropy = create_nn_model(input_dim=d)
            model_crossentropy.compile(optimizer=Adam(), loss='binary_crossentropy')
            model_crossentropy.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0,
                                   callbacks=[checkpoint_crossentropy, loss_history_crossentropy])

            crossentropy_losses.extend(loss_history_crossentropy.losses)

            prediction_score_crossentropy = model_crossentropy.predict(X_test).flatten()
            loss_crossentropy = log_loss(Y_test, prediction_score_crossentropy)
            log_loss_crossentropy_current.append(loss_crossentropy)

        all_custom_losses.append(custom_losses)
        all_crossentropy_losses.append(crossentropy_losses)

        log_loss_custom.append(np.mean(log_loss_custom_current))
        log_loss_crossentropy.append(np.mean(log_loss_crossentropy_current))
        
        log_loss_custom_std.append(np.std(log_loss_custom_current))
        log_loss_crossentropy_std.append(np.std(log_loss_crossentropy_current))

    # Plot the log-loss for each epoch
    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors if needed
    linestyles = ['-', '--', '-.', ':']  # Different line styles for different loss functions

    for idx, alpha in enumerate(alpha_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[0]
        plt.plot(all_custom_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Custom Loss (alpha={alpha})')

        linestyle = linestyles[1]
        plt.plot(all_crossentropy_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Crossentropy Loss (alpha={alpha})')

    plt.xlabel('Epoch')
    plt.ylabel('Log-Loss')
    plt.title('Log-Loss (abalone_19) for Custom and Crossentropy Loss Functions across Epochs')
    plt.legend()
    plt.show()

    return log_loss_custom, log_loss_crossentropy, log_loss_custom_std, log_loss_crossentropy_std


def experiment_NN_ecoli(alpha_list, feature, target, exp_num=100, eps=1e-5):
    all_custom_losses = []
    all_crossentropy_losses = []

    log_loss_custom = []
    log_loss_crossentropy = []
    log_loss_custom_std = []
    log_loss_crossentropy_std = []

    # Custom loss function based on objective_joint_MLE
    def custom_loss(alpha, eps, X_downsample, Y_downsample, model):
        def loss(y_true, y_pred):
            Fi = model(X_downsample)
            sum_term = tf.reduce_mean(Y_downsample * tf.math.log(1 - Fi) + 
                                    (1 - Y_downsample) * tf.math.log(alpha * Fi))
            integral_val_approx = tf.reduce_mean(1 - (1 - alpha) * Fi)
            return -(sum_term - tf.math.log(integral_val_approx))
        return loss

    for alpha in alpha_list:
        log_loss_custom_current = []
        log_loss_crossentropy_current = []

        custom_losses = []
        crossentropy_losses = []

        for i in range(exp_num):
            np.random.seed(i)
            X, X_test, Y, Y_test = train_test_split(feature, target, test_size=0.2, random_state=i)

            # Create downsample_set for conditional MLE 
            indices_Y_1 = np.where(Y == 1)[0]
            num_sample_Y_0 = int(np.sum(Y == 0) * alpha)
            indices_Y_0_sampled = np.random.choice(np.where(Y == 0)[0], num_sample_Y_0, replace=False)
            downsample_set = np.union1d(indices_Y_1, indices_Y_0_sampled)

            X_downsample = X[downsample_set]
            Y_downsample = Y[downsample_set]

            d = X.shape[1]

            # Save the model every 10 epochs
            checkpoint_custom = ModelCheckpoint(filepath=f'models_2/model_custom_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                save_freq='epoch',
                                                period=10,
                                                verbose=0)

            # Custom callback to store losses
            loss_history_custom = LossHistory()

            # Train using the custom loss function
            model_custom = create_nn_model(input_dim=d)
            model_custom.compile(optimizer=Adam(), 
                     loss=custom_loss(alpha, eps, X_downsample, Y_downsample, model_custom))

            model_custom.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0, 
                             callbacks=[checkpoint_custom, loss_history_custom])

            custom_losses.extend(loss_history_custom.losses)

            prediction_score_custom = model_custom.predict(X_test).flatten()
            loss_custom = log_loss(Y_test, prediction_score_custom)
            log_loss_custom_current.append(loss_custom)

            # Save the model every 10 epochs
            checkpoint_crossentropy = ModelCheckpoint(filepath=f'models_cross/model_crossentropy_alpha_{alpha}_exp_{i}_epoch_{{epoch:02d}}.h5',
                                                      save_freq='epoch',
                                                      period=10,
                                                      verbose=0)

            # Custom callback to store losses
            loss_history_crossentropy = LossHistory()

            # Train using the standard cross-entropy loss
            model_crossentropy = create_nn_model(input_dim=d)
            model_crossentropy.compile(optimizer=Adam(), loss='binary_crossentropy')
            model_crossentropy.fit(X_downsample, Y_downsample, epochs=100, batch_size=128, verbose=0,
                                   callbacks=[checkpoint_crossentropy, loss_history_crossentropy])

            crossentropy_losses.extend(loss_history_crossentropy.losses)

            prediction_score_crossentropy = model_crossentropy.predict(X_test).flatten()
            loss_crossentropy = log_loss(Y_test, prediction_score_crossentropy)
            log_loss_crossentropy_current.append(loss_crossentropy)

        all_custom_losses.append(custom_losses)
        all_crossentropy_losses.append(crossentropy_losses)

        log_loss_custom.append(np.mean(log_loss_custom_current))
        log_loss_crossentropy.append(np.mean(log_loss_crossentropy_current))
        
        log_loss_custom_std.append(np.std(log_loss_custom_current))
        log_loss_crossentropy_std.append(np.std(log_loss_crossentropy_current))

    # Plot the log-loss for each epoch
    plt.figure(figsize=(12, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # You can add more colors if needed
    linestyles = ['-', '--', '-.', ':']  # Different line styles for different loss functions

    for idx, alpha in enumerate(alpha_list):
        color = colors[idx % len(colors)]
        linestyle = linestyles[0]
        plt.plot(all_custom_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Custom Loss (alpha={alpha})')

        linestyle = linestyles[1]
        plt.plot(all_crossentropy_losses[idx], color=color, linestyle=linestyle, 
                 label=f'Crossentropy Loss (alpha={alpha})')

    plt.xlabel('Epoch')
    plt.ylabel('Log-Loss')
    plt.title('Log-Loss (ecoli) for Custom and Crossentropy Loss Functions across Epochs')
    plt.legend()
    plt.show()

    return log_loss_custom, log_loss_crossentropy, log_loss_custom_std, log_loss_crossentropy_std




def plot_log_loss(log_loss_inv,log_loss_inv_std,
                  log_loss_joint,log_loss_joint_std,
                  alpha_list, p1, exp_num=500):
    
    log_loss_inv = np.array(log_loss_inv)
    log_loss_inv_std = np.array(log_loss_inv_std)
    log_loss_joint = np.array(log_loss_joint)
    log_loss_joint_std = np.array(log_loss_joint_std)

    # Calculate confidence intervals
    def calc_conf_intervals(mean_list, std_list):
        upper = mean_list + 1.96 * std_list / np.sqrt(exp_num)
        lower = mean_list - 1.96 * std_list / np.sqrt(exp_num)
        return upper, lower

    joint_upper, joint_lower = calc_conf_intervals(log_loss_joint, log_loss_joint_std)
    inv_upper, inv_lower = calc_conf_intervals(log_loss_inv, log_loss_inv_std)
    
    # Create a figure with 2 subplots
    fig, ax1  = plt.subplots(1, 1, figsize=(10, 6))
    
    # Define colors for consistency
    colors = {'our loss function': 'green','cross-entropy loss': 'purple'}

    # Plot data and confidence intervals on the first subplot
    ax1.plot(alpha_list, log_loss_joint, 's-', label='full-MLE downsample loss', color=colors['our loss function'])
    ax1.plot(alpha_list, log_loss_inv, 'o-', label='CE loss', color=colors['cross-entropy loss'])

    # Shaded areas for confidence intervals
    ax1.fill_between(alpha_list, joint_lower, joint_upper, color='grey', alpha=0.3)
    ax1.fill_between(alpha_list, inv_lower, inv_upper, color='grey', alpha=0.3)

    # Upper confidence boundaries with stronger lines
    ax1.plot(alpha_list, joint_upper, '--', color=colors['our loss function'], linewidth=2)
    # Lower confidence boundaries with '*' markers
    ax1.plot(alpha_list, joint_lower, '--', color=colors['our loss function'], linewidth=2)

    ax1.plot(alpha_list, inv_upper, '--', color=colors['cross-entropy loss'], linewidth=2)
    ax1.plot(alpha_list, inv_lower, '--', color=colors['cross-entropy loss'], linewidth=2)

    ax1.set_title(f'NN Log-loss comparison P1={p1:.6f}')
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('log-loss')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True)
    ax1.set_xticks(alpha_list)
    ax1.set_xticklabels(alpha_list, rotation=45, ha='right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()
