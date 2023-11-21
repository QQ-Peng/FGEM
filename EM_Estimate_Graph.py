
import numpy as np
import torch

class random_graph:
    '''
    implementation of paper 'Estimating network structure from unreliable measurements'
    section 'Random graphs and independent measurements'
    '''
    def __init__(self):
          pass

    def E_step(self,Q,E,N):
            
            # estimation of w
            num_node = Q.shape[0]
            w = np.triu(Q,1).sum()/(num_node*(num_node-1)/2)
            
            # estimation of alpha
            numerator_alpha = np.triu(Q*E,1).sum()
            denominator_alpha = np.triu(Q*N,1).sum()
            alpha = numerator_alpha/denominator_alpha

            # estimation of beta
            numerator_beta = np.triu((1-Q)*E,1).sum()
            denominator_beta = np.triu((1-Q)*N,1).sum()
            beta = numerator_beta/denominator_beta


            return alpha, beta, w

    def M_step(self,alpha, beta, w,N,E):
            """Run the Maximization(M) step of the EM algorithm.
            """
            numerator = w*np.power(alpha,E)*np.power(1-alpha,N-E)
            
            denominator = w*np.power(alpha,E)*np.power(1-alpha,N-E) +\
                        (1-w)*np.power(beta,E)*np.power(1-beta,N-E)
            Q = numerator/denominator
            return Q


    def EM(self,E,N,tolerance=.000001):
            
            # Record previous values to confirm convergence
            alpha_p = 0
            beta_p = 0
            
            # Do an initial E-step with random alpha, beta and O
            # Beta must be smaller than alpha
            # beta, alpha = np.sort(np.random.rand(2))
            alpha, beta = 0.4, 0.001
            w = np.random.rand()
            
            # Calculate initial Q
            Q = self.M_step(alpha, beta, w,N,E)
            iterations = 0
            while abs(alpha_p - alpha) > tolerance or abs(beta_p - beta) > tolerance:
                alpha_p = alpha
                beta_p = beta
                alpha, beta, w = self.E_step(Q,E,N)
                Q = self.M_step(alpha, beta, w,N,E)
                iterations += 1
                if iterations % 10 == 0:
                    print(f'*'*30)
                    print(f'iter: {iterations} done.')
                    print(f"alpha: {alpha}, beta: {beta}, w: {w}, Q: {Q}")

            return Q, alpha, beta, w, iterations



class poisson_graph:
    '''
    implementation of paper 'A principled approach for weighted multilayer network aggregation'
    '''
    def __init__(self) -> None:
          pass
    
    def E_step(self,num,Q):
        numerator = num**2
        denominator = Q.sum()
        theta = numerator/denominator
        return theta

    def M_step(self,theta,epsilon,E):
        
        numerator = epsilon*E
        denominator = E + theta
        Q = numerator/denominator
        return Q

    def EM(self,E,num,epsilon=None,tolerance=.000001):

        theta_p = 0
        # theta = np.random.rand()
        theta = 0.05
        if epsilon is None:
            '''
            as we don't know the prior edge weight in every facet PPI graph, we select the number '350' for  Iterative stability.
            '''
            epsilon = np.zeros_like(E)
            epsilon[E!=0] = 350
            epsilon[E==0] = 0
        
        # Calculate initial Q
        Q = self.M_step(theta,epsilon,E)
        iterations = 0
        while abs(theta_p - theta) > tolerance:
            theta_p = theta
            theta = self.E_step(num,Q)
            Q = self.M_step(theta,epsilon,E)
            iterations += 1
            if iterations % 2 == 0:
                print(f'*'*30)
                print(f'iter: {iterations} done.')
                print(f"theta: {theta}, Q: {Q}")

        return theta, Q/Q.max()
# graph_estimator = random_graph()
# E = torch.load("./data/EMOGI/data/Gobs_aggr.pkl",map_location='cpu')
# E = E.numpy()
# N = 5

# Q,alpha,beta,w,iterations = graph_estimator.EM(E,N)
# np.save('F:/Data/EMOGI/Q_alpha0.30732268_beta0.0002341604_w0.0023043599_N5.npy',Q)

#######################################################
# graph_estimator = poisson_graph()
# E = torch.load("./data/EMOGI/data/Gobs_aggr.pkl",map_location='cpu')

# E = E.numpy()
# num = E.shape[0]

# theta, Q = graph_estimator.EM(E,num)
# np.save('F:/Data/EMOGI/Q_poisson.npy',Q)

# Q[Q<0.5] = 0
# Q[Q>=0.5] = 1.0

# np.save('F:/Data/EMOGI/PGE0.5.npy',Q)
