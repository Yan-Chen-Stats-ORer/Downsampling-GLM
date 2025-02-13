import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize
from scipy.integrate import quad
import random

random.seed(10)
RANDOM_STATE = 10

#P(Y=0|X) = exp(theta_0+theta_1*X)/(1+exp(theta_0+theta_1*X))
def experiment_logit(theta_0,alpha_list,theta_1=0.5,
                     exp_num=500,sample_size = pow(10,6)):
    #probability of P(Y=1|X) at each X
    n = sample_size
    X = np.random.uniform(low=0.0,high=1.0,size=n)
    p = 1/(1+np.exp(theta_0+np.dot(theta_1,X)))
    print('Prob(Y=1)',np.mean(p))

    def F(theta_0, theta_1, X):
        return np.exp(theta_0 + np.dot(theta_1, X)) / (1 + np.exp(theta_0 + np.dot(theta_1, X)))

    def inv_function_0(theta_0,theta_1,X,alpha):
        return 1/alpha*np.log(F(theta_0, theta_1, X))
    
    ### Initialize the empty lists recording the mean squared errors 
    ### or mean difference in squared-errors and standard deviations
    """ MLE of joint distribution of downsample (\tilde{Y},\tilde{X})"""
    # mean-squared-error 
    MSE_pseudo_MLE_list = []
    # standard deviation for the squared error 
    std_SE_pseudo_MLE_list = []

    """conditional MLE of downsample (\tilde{Y}|\tilde{X})"""
    # mean-squared-error 
    MSE_cond_MLE_list = []
    # standard deviation for the squared error 
    std_SE_cond_MLE_list = []

    """Inverse-reweighting objective function"""
    # mean-squared-error
    MSE_inverse_list = []
    # standard deviation for the squared error 
    std_SE_inverse_list = []

    """
    difference in the squared error 
    between pseudo MLE and inverse-reweighting estimator
    """
    # mean of the difference in the squared errors 
    mean_diff_SE_inverse_pseudo_list = []
    # standard deviation for the difference for the squared errors
    std_diff_SE_inverse_pseudo_list = []


    """
    difference in the squared error 
    between conditional MLE and pseudo MLE
    """
    # mean of the difference in the squared errors 
    mean_diff_SE_cond_pseudo_list = []
    # standard deviation for the difference for the squared errors
    std_diff_SE_cond_pseudo_list = []

    #run experiments for each alpha in the list 
    for alpha in alpha_list: 
        #joint MLE from the joint distribution of downsamples
        solution_pseudo_MLE = [] 
        #conditional MLE from the joint distribution of downsamples
        solution_cond_MLE = []
        #inverse reweighting estimator
        solution_inv = []
    
        for i in range(exp_num): #by default run 500 rounds of experiment
            #set random seed for the current iteration
            np.random.seed(i) 
            # Initial guess
            """Random Initialization for each iteration"""
            initial_theta = np.random.uniform(low=0.2,high=0.8,size=1)[0]
            # Sample Y_i's according to the conditional distributions P(Y=1|X)
            Y = np.random.binomial(1, p)
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
            def objective_pseudo_MLE(theta_1):
                X_downsample = X[downsample_set]
                sum_term = 0
                Fi = F(theta_0, theta_1, X_downsample)
                sum_term = np.mean(Y[downsample_set] * np.log(1 - Fi) + (1 - Y[downsample_set]) * np.log(alpha*Fi))
                integral_val_approx = np.mean(1 - (1 - alpha) * F(theta_0, theta_1, X_downsample))
                return -(sum_term - np.log(integral_val_approx))
        
            # Create downsample_set for conditional MLE 
            downsample_set = np.union1d(indices_Y_1, 
                                        indices_Y_0_sampled)
            # Size of downsample_set
            N = len(downsample_set)

            # likelihood function / objective for the conditional MLE
            def objective_cond_MLE(theta_1):
                X_downsample = X[downsample_set]
                sum_term = 0
                Fi = F(theta_0, theta_1, X_downsample)
                sum_term = np.mean(Y[downsample_set] * np.log(1 - Fi) + (1 - Y[downsample_set]) * np.log(alpha*Fi)-np.log(1-(1-alpha)*Fi))
                return - sum_term

            X_0 = X[indices_Y_0_sampled]
            X_1 = X[indices_Y_1]
            
            # objective function for the inverse-reweigting estimator
            def objective_inv(theta_1):
                pos_part = 1-F(theta_0, theta_1, X_1)
                neg_part = inv_function_0(theta_0,theta_1,X_0,alpha)
                return -(np.sum(np.log(pos_part))+np.sum(neg_part))/N
            
            # Finding the estimator that maximizes the empirical MLE for joint distribution of downsample
            result_pseudo_MLE = minimize(lambda theta_1: objective_pseudo_MLE(theta_1[0]), [initial_theta])
            sol_pseudo_MLE = result_pseudo_MLE.x  
            solution_pseudo_MLE.append(sol_pseudo_MLE)

            # Finding the inverse reweighting estimator 
            result_cond_MLE = minimize(lambda theta_1: objective_cond_MLE(theta_1[0]), [initial_theta])
            sol_cond_MLE = result_cond_MLE.x 
            solution_cond_MLE.append(sol_cond_MLE)
            
            # Finding the inverse reweighting estimator: 
            result_inv = minimize(lambda theta_1: objective_inv(theta_1[0]), [initial_theta])
            sol_inv = result_inv.x 
            solution_inv.append(sol_inv)

             
        #compute the squared error for each iteration 
        SE_pseudo_MLE = [(a - theta_1)**2 for a in solution_pseudo_MLE]
        SE_cond_MLE = [(a - theta_1)**2 for a in solution_cond_MLE]
        SE_inv = [(a - theta_1)**2 for a in solution_inv]

        
        #compute the difference of the squared errors 
        diff_SE_inv_pseudo = [(SE_inv[i] - SE_pseudo_MLE[i])[0] for i in range(exp_num)]
        diff_SE_cond_pseudo = [(SE_cond_MLE[i] - SE_pseudo_MLE[i])[0] for i in range(exp_num)]

        #compute the mean and standard deviations of the squared errors
        MSE_pseudo_MLE = np.mean(SE_pseudo_MLE)
        std_SE_pseudo_MLE = np.std(SE_pseudo_MLE)
        MSE_pseudo_MLE_list.append(MSE_pseudo_MLE)
        std_SE_pseudo_MLE_list.append(std_SE_pseudo_MLE)

        MSE_cond_MLE = np.mean(SE_cond_MLE)
        std_SE_cond_MLE = np.std(SE_cond_MLE)
        MSE_cond_MLE_list.append(MSE_cond_MLE)
        std_SE_cond_MLE_list.append(std_SE_cond_MLE)

        MSE_inv = np.mean(SE_inv)
        std_SE_inv = np.std(SE_inv)
        MSE_inverse_list.append(MSE_inv)
        std_SE_inverse_list.append(std_SE_inv)

        mean_diff_SE_inv_pseudo = np.mean(diff_SE_inv_pseudo)
        std_diff_SE_inv_pseudo = np.std(diff_SE_inv_pseudo)
        mean_diff_SE_inverse_pseudo_list.append(mean_diff_SE_inv_pseudo)
        std_diff_SE_inverse_pseudo_list.append(std_diff_SE_inv_pseudo)


        mean_diff_SE_cond_joint = np.mean(diff_SE_cond_pseudo)
        std_diff_SE_cond_joint = np.std(diff_SE_cond_pseudo)
        mean_diff_SE_cond_pseudo_list.append(mean_diff_SE_cond_joint)
        std_diff_SE_cond_pseudo_list.append(std_diff_SE_cond_joint)

    return  MSE_pseudo_MLE_list,std_SE_pseudo_MLE_list,\
            MSE_cond_MLE_list,std_SE_cond_MLE_list,\
            MSE_inverse_list,std_SE_inverse_list,\
            mean_diff_SE_inverse_pseudo_list,std_diff_SE_inverse_pseudo_list,\
            mean_diff_SE_cond_pseudo_list,std_diff_SE_cond_pseudo_list

def plot_results_all(MSE_pseudo_MLE_list,std_SE_pseudo_MLE_list,
                    MSE_cond_MLE_list,std_SE_cond_MLE_list,
                    MSE_inverse_list,std_SE_inverse_list,
                    mean_diff_SE_inverse_pseudo_list,std_diff_SE_inverse_pseudo_list,
                    mean_diff_SE_cond_pseudo_list,std_diff_SE_cond_pseudo_list,
                    alpha_list,theta_0,p1,exp_num = 500):
    
    #make the lists into arrays
    MSE_pseudo_MLE_list = np.array(MSE_pseudo_MLE_list)
    std_SE_pseudo_MLE_list = np.array(std_SE_pseudo_MLE_list)

    MSE_cond_MLE_list = np.array(MSE_cond_MLE_list)
    std_SE_cond_MLE_list = np.array(std_SE_cond_MLE_list)

    MSE_inverse_list = np.array(MSE_inverse_list)
    std_SE_inverse_list = np.array(std_SE_inverse_list)

    mean_diff_SE_inverse_pseudo_list = np.array(mean_diff_SE_inverse_pseudo_list)
    std_diff_SE_inverse_pseudo_list = np.array(std_diff_SE_inverse_pseudo_list)
    
    mean_diff_SE_cond_pseudo_list = np.array(mean_diff_SE_cond_pseudo_list)
    std_diff_SE_cond_pseudo_list = np.array(std_diff_SE_cond_pseudo_list)
    

    #the confidence intervals for pseudo MLE at each alpha
    pseudo_upper = MSE_pseudo_MLE_list + 1.96 * std_SE_pseudo_MLE_list/ np.sqrt(exp_num)
    pseudo_lower = MSE_pseudo_MLE_list - 1.96 * std_SE_pseudo_MLE_list / np.sqrt(exp_num)

    #the confidence intervals for conditional MLE at each alpha
    cond_upper = MSE_cond_MLE_list + 1.96 * std_SE_cond_MLE_list/ np.sqrt(exp_num)
    cond_lower = MSE_cond_MLE_list - 1.96 * std_SE_cond_MLE_list/ np.sqrt(exp_num)

    #the confidence intervals for inverse-reweighting estimator at each alpha
    inv_upper = MSE_inverse_list + 1.96 * std_SE_inverse_list/ np.sqrt(exp_num)
    inv_lower = MSE_inverse_list - 1.96 * std_SE_inverse_list/ np.sqrt(exp_num)

    #the confidence intervals for the differences in square errors at each alpha:
    inv_pseudo_upper = mean_diff_SE_inverse_pseudo_list + 1.96 * std_diff_SE_inverse_pseudo_list/ np.sqrt(exp_num)
    inv_pseudo_lower = mean_diff_SE_inverse_pseudo_list - 1.96 * std_diff_SE_inverse_pseudo_list/ np.sqrt(exp_num)

    cond_pseudo_upper = mean_diff_SE_cond_pseudo_list + 1.96 * std_diff_SE_cond_pseudo_list/ np.sqrt(exp_num)
    cond_pseudo_lower = mean_diff_SE_cond_pseudo_list - 1.96 * std_diff_SE_cond_pseudo_list/ np.sqrt(exp_num)

    # Create a figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    
    # Plot data on the first subplot
    ax1.plot(alpha_list, MSE_pseudo_MLE_list, '+-', 
         label='MSE:pseudo',color='green')
    
    ax1.plot(alpha_list, MSE_cond_MLE_list, '+-', 
         label='MSE:conditional',color='red')
    
    ax1.plot(alpha_list, MSE_inverse_list, '+-', 
         label='MSE:inverse',color='blue')
    
    #confidence intervals for SE of joint MLE
    for i in range(len(MSE_pseudo_MLE_list)):
        if i == 0:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [pseudo_lower[i], pseudo_upper[i]], '--',color='green',
                     label='0.95 CI: pseudo')
        else:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [pseudo_lower[i], pseudo_upper[i]], '--',color='green')

    
    #confidence intervals for SE of conditional MLE
    for i in range(len(MSE_cond_MLE_list)):
        if i == 0:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [cond_lower[i], cond_upper[i]], '--',color='red',
                     label='0.95 CI: conditional')
        else:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [cond_lower[i], cond_upper[i]], '--',color='red')

    #confidence intervals for SE of inverse-reweighting estimator
    for i in range(len(MSE_inverse_list)):
        if i == 0:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [inv_lower[i], inv_upper[i]], '--',color='blue',
                     label='0.95 CI: inverse')
        else:
            ax1.plot([alpha_list[i], alpha_list[i]], 
                     [inv_lower[i], inv_upper[i]], '--',color='blue')

    ax1.set_title('MSE (theta0=%.2f,P1=%.6f)'%(theta_0,p1))
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('mean-squared-error')
    ax1.legend(fontsize=8,loc='upper right')
    ax1.grid(True)
    ax1.set_xticks(alpha_list)
    ax1.set_xticklabels(alpha_list, rotation=45, ha='right')

    # Plot data on the second subplot
    ax2.plot(alpha_list, mean_diff_SE_inverse_pseudo_list, 
             '+-', 
         label='inverse - pseudo',color='green')
    
    ax2.plot(alpha_list, mean_diff_SE_cond_pseudo_list, 
             '+-', 
         label='conditional - pseudo',color='purple')
        
    #confidence intervals for difference between SE of inverse vs. joint MLE
    for i in range(len(mean_diff_SE_inverse_pseudo_list)):
        if i == 0:
            ax2.plot([alpha_list[i], alpha_list[i]], 
                     [inv_pseudo_lower[i], inv_pseudo_upper[i]], '--',color='green',
                     label='0.95 CI: inverse - pseudo')
        else:
            ax2.plot([alpha_list[i], alpha_list[i]], 
                     [inv_pseudo_lower[i], inv_pseudo_upper[i]], '--',color='green')


    #confidence intervals for difference of SE: conditional MLE - pseudo MLE
    for i in range(len(mean_diff_SE_cond_pseudo_list)):
        if i == 0:
            ax2.plot([alpha_list[i], alpha_list[i]], 
                     [cond_pseudo_lower[i], cond_pseudo_upper[i]], '--',color='purple',
                     label='0.95 CI: conditional - pseudo')
        else:
            ax2.plot([alpha_list[i], alpha_list[i]], 
                     [cond_pseudo_lower[i], cond_pseudo_upper[i]], '--',color='purple')

    ax2.set_title('Average diff (SE) (theta0=%.2f,P1=%.6f)'%(theta_0,p1))
    ax2.set_xlabel('Alpha')
    ax2.set_ylabel('difference of squared errors')
    ax2.legend(fontsize=8,loc='upper right')
    ax2.grid(True)
    ax2.set_xticks(alpha_list)
    ax2.set_xticklabels(alpha_list, rotation=45, ha='right')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()





            
            
            