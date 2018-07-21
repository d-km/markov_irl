# -*- coding: utf-8 -*-
"""
"Projection Method": Apprenticeship Learning via IRL

@author: D. Kishikawa
"""

import math
#import numpy as np
import numpy as np
import numba
import sys 
from policy_iteration import PolicyIteration




#二次元なのでγは配列
gamma = 0.9
#meros
x_size = 11
y_size = 11

#πE……4種類
pi_expert = np.array([[0,2,5,7,6,10,1],
                      [0,3,5,9,6,10,1],
                      [0,4,5,2,6,10,1], 
                      [0,8,5,9,6,10,1]])



### L2ノルムの計算 ##################################
@numba.jit
def L2norm(array):
    return np.dot(array, array)
#####################################################
    

### Φ(s)の計算 ######################################
@numba.jit
def phi(state):
    #one-hotベクトル化する
    phi_s = np.zeros(x_size*y_size)
    for i in range(x_size*y_size):
        if i == state:
            phi_s[i] = 1
        else:
            phi_s[i] = 0 
    #行列で返す
    return phi_s 
@numba.jit
def Mu(policy):
    #print("π: {}".format(policy))
    Mu_s = np.zeros(x_size*y_size)
    for s in range(len(policy)):
        gamma_s = math.pow(gamma, s)
        Mu_s = Mu_s + gamma_s * phi(policy[s])        
    #print("μ:")
    #(Mu_s)
    #print("")
    return Mu_s
@numba.jit
def MuE(policyE):
    MuE_m = np.zeros(x_size*y_size)

    #print(policyE.shape[0])
    for m in range(policyE.shape[0]):
        policy_Ex = np.zeros(x_size*y_size)
        MuE_s = np.zeros(x_size*y_size)
        
        policy_Ex = policyE[m,]
        MuE_s = Mu(policy_Ex)
        
        MuE_m = MuE_m + MuE_s
        

    
    MuE_a = MuE_m / policyE.shape[0]
    
    #print("-*- calculated μE -*- ")
    #print(MuE_a)
    #print("")
    
    return MuE_a
@numba.jit
def Rcalc(w):
    R = np.zeros(x_size)
    for s in range(x_size):
        R[s] = np.dot(w, phi(s))
        #if (s%10==0):
            #print("R: {}%".format((s/(x_size*y_size))*100))

    return R
#################################################################################


Pi0 = np.array([0,2,6,10,1])
mu0 = Mu(Pi0)
muE = MuE(pi_expert)


mu_i_1 = np.zeros(x_size*y_size)
mu_i_2 = np.zeros(x_size*y_size)

i = 1
w = np.zeros(x_size*y_size)
t = 0
epsilon = 0.1**10

#P no keisan
P_data = np.loadtxt('p.csv', delimiter=',')

P = np.zeros((11, 11, 11))

P_state1 = P_data[:,0]
P_state2 = P_data[:,1]
P_prob = P_data[:,2]

for sta in range(22):
    P[int(P_state1[sta])][int(P_state2[sta])][int(P_state2[sta])] = P_prob[sta]

while(True): ##########################################################
    
    #print("-----------------iter: {}-------------------\n".format(i))
    
    # μ_ の計算
    if (i-1) == 0:
        mu_i_1 = mu0
    else:
        #Projection Methodの計算式
        N1 = np.dot( (mu_old - mu_i_2).T , (muE - mu_i_2) )
        N2 = np.dot( (mu_old - mu_i_2).T , (mu_old - mu_i_2) )
        N3 = N1 / N2
        N4 = np.dot(N3, (mu_old - mu_i_2))
        mu_i_1 = mu_i_2 + N4
    
    #print("===== μ_ = {} =====\n".format(mu_i_1))
    
    # w(weight)の計算
    w = muE - mu_i_1
    #print("===== w = {} =====\n".format(w))
    
    # tの計算・終了判定
    t = L2norm(muE - mu_i_1)
    
    print("===== t = {} =====".format(t))
    
    if t <= epsilon :
        break
    
    #R = np.zeros(x_size*y_size)
    
    #Rの計算
    R = Rcalc(w)
    
    #print("===== R: =====")
    #print(R)
    #print("")
    np.savetxt('R.csv', R, delimiter=',')
    
    #強化学習により、Rからpi_selectedを求める
    state = np.zeros(11)
    for hoge in range(11):
        state[hoge] = hoge
    pi_selected = PolicyIteration(state, P, R, gamma)
   # print(pi_selected)
    #mu(pi_selected) を計算
    mu_old = Mu(pi_selected)
    mu_i_2 = mu_i_1
    #i を加算、ループ続行
    i = i + 1
print(R)   
##########################################################################
