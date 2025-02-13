import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize
from scipy.integrate import quad
import random
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.utils import to_categorical

random.seed(10)
RANDOM_STATE = 10

#P(Y=0|X) = exp(theta_0+theta_1*X)/(1+exp(theta_0+theta_1*X))
def experiment_logit(theta_0,alpha_list,feature,target,exp_num=1000,eps=1e-3):
    def F(theta_0, theta_1, X):
        return 1 / (1 + np.exp(- theta_0 - np.dot(X,theta_1)))

    def inv_function_0(theta_0,theta_1,X,alpha):
        return 1/alpha*np.log(F(theta_0, theta_1, X)+eps)
        
    ### Initialize the empty lists recording the mean squared errors 
    ### or mean difference in squared-errors and standard deviations
    """ MLE of joint distribution of downsample (\tilde{Y},\tilde{X})"""
    log_loss_joint = []
    auc_joint = []
    log_loss_joint_std = []
    auc_joint_std = []
    """Inverse-reweighting objective function"""
    log_loss_inv = []
    auc_inv = []
    log_loss_inv_std = []
    auc_inv_std = []

    #run experiments for each alpha in the list 
    for alpha in alpha_list: 
        #joint MLE from the joint distribution of downsamples
        solution_joint_MLE = [] 
        #inverse reweighting estimator
        solution_inv = []
        #prediction score for inverse-estimator
        # pred_score_inv = []
        # #prediction score for full MLE
        # pred_score_joint =[]
        auc_joint_current = []
        log_loss_joint_current = []
        auc_inv_current = []
        log_loss_inv_current = []

        for i in range(exp_num): #by default run 1000 rounds of experiment
            #set random seed for the current iteration
            np.random.seed(i) 
            # Split the data into train and test sets
            X, X_test, Y, Y_test = train_test_split(feature, target, 
                                                    test_size=0.2, 
                                                    random_state=i)
            d = X.shape[1]
            #probability of P(Y=1|X) at each X
            n = len(Y)
            p = sum(Y)/len(Y)
            # print('Prob(Y=1)',p)
            # Initial guess
            """Random Initialization for each iteration"""
            initial_theta = np.random.uniform(low=-1.0,high=1.0,size=d)
            # Downsample, record indices where Y=1 and sample from Y=0
            indices_Y_1 = np.where(Y == 1)[0]
            # Number of Y=0 elements to sample
            num_sample_Y_0 = int(np.sum(Y == 0) * alpha) 
            # Sample num_sample_Y_0 number of data points without placement from Y=0 
            indices_Y_0_sampled = np.random.choice(np.where(Y == 0)[0], 
                                                   num_sample_Y_0, 
                                                   replace=False)    
            
            # The likelihood function of joint MLE, 
            # including the integral with respect to density of X
            def objective_joint_MLE(theta_1):
                X_downsample = X[downsample_set]
                sum_term = 0
                Fi = F(theta_0, theta_1, X_downsample)
                sum_term = np.mean(Y[downsample_set] * np.log(1 - Fi) + (1 - Y[downsample_set]) * np.log(alpha*Fi))
                integral_val_approx = np.mean((1 - (1 - alpha) * F(theta_0, theta_1, X_downsample)))
                
                return -(sum_term - np.log(integral_val_approx))
        
            # Create downsample_set for conditional MLE 
            downsample_set = np.union1d(indices_Y_1, 
                                        indices_Y_0_sampled)
            # Size of downsample_set
            N = len(downsample_set)

            X_0 = X[indices_Y_0_sampled]
            X_1 = X[indices_Y_1]
            
            # objective function for the inverse-reweigting estimator
            def objective_inv(theta_1):
                pos_part = 1-F(theta_0, theta_1, X_1)
                neg_part = inv_function_0(theta_0,theta_1,X_0,alpha)
                return -(np.sum(np.log(pos_part))+np.sum(neg_part))/N
            
            # Finding the estimator that maximizes the empirical MLE for joint distribution of downsample
            result_joint_MLE = minimize(lambda theta_1: objective_joint_MLE(theta_1), 
                                        initial_theta)
            sol_joint_MLE = result_joint_MLE.x  
            # print('sol_joint_MLE',sol_joint_MLE)
            solution_joint_MLE.append(sol_joint_MLE)
            prediction_score_joint = 1-F(theta_0, sol_joint_MLE, X_test)
            # print('prediction_score_joint',prediction_score_joint)
            # pred_score_joint.append(prediction_score_joint)
            loss_joint = log_loss(Y_test, prediction_score_joint)
            log_loss_joint_current.append(loss_joint)
            # Compute the AUC score
            auc_score_joint = roc_auc_score(Y_test, prediction_score_joint)
            auc_joint_current.append(auc_score_joint)

            # Finding the inverse reweighting estimator: 
            result_inv = minimize(lambda theta_1: objective_inv(theta_1), initial_theta)
            sol_inv = result_inv.x 
            # print('sol_inv',sol_inv)
            solution_inv.append(sol_inv)
            prediction_score_inv = 1-F(theta_0, sol_inv, X_test)
            # print('prediction_score_inv',prediction_score_inv)
            # pred_score_inv.append(prediction_score_inv)
            loss_inv = log_loss(Y_test, prediction_score_inv)
            log_loss_inv_current.append(loss_inv)
            auc_score_inv = roc_auc_score(Y_test, prediction_score_inv)
            auc_inv_current.append(auc_score_inv)

        log_loss_inv.append(np.mean(log_loss_inv_current))
        log_loss_inv_std.append(np.std(log_loss_inv_current))
        auc_inv.append(np.mean(auc_inv_current))
        auc_inv_std.append(np.std(auc_inv_current))

        log_loss_joint.append(np.mean(log_loss_joint_current))
        log_loss_joint_std.append(np.std(log_loss_joint_current))
        auc_joint.append(np.mean(auc_joint_current))
        auc_joint_std.append(np.std(auc_joint_current))

    #compute the difference of the squared errors 
    return log_loss_inv,log_loss_inv_std,auc_inv,auc_inv_std,log_loss_joint,log_loss_joint_std,auc_joint,auc_joint_std


def plot_log_loss(log_loss_inv, log_loss_inv_std,
                  log_loss_joint, log_loss_joint_std,
                  alpha_list, p1, dataset_name, exp_num=500):
    
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
    colors = {'joint': 'green', 'inverse': 'purple'}

    # Plot data and confidence intervals on the first subplot
    ax1.plot(alpha_list, log_loss_joint, 's-', label='log-loss: full-MLE', color=colors['joint'])
    ax1.plot(alpha_list, log_loss_inv, 'o-', label='log-loss: inverse-weighting', color=colors['inverse'])

    # Shaded areas for confidence intervals
    ax1.fill_between(alpha_list, joint_lower, joint_upper, color='grey', alpha=0.3)
    ax1.fill_between(alpha_list, inv_lower, inv_upper, color='grey', alpha=0.3)

    # Upper confidence boundaries with stronger lines
    ax1.plot(alpha_list, joint_upper, '--', color=colors['joint'], linewidth=2)
    # Lower confidence boundaries with '*' markers
    ax1.plot(alpha_list, joint_lower, '--', color=colors['joint'], linewidth=2)

    ax1.plot(alpha_list, inv_upper, '--', color=colors['inverse'], linewidth=2)
    ax1.plot(alpha_list, inv_lower, '--', color=colors['inverse'], linewidth=2)

    ax1.set_title(f'UCI {dataset_name}: Log-loss P(Y=1)={p1:.6f}')
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
