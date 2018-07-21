# -*- coding: utf-8 -*-
#forked from https://gist.github.com/tuxdna/7e29dd37300e308a80fc1559c343c545

############## import Xpy as ArrM ###################################
"""
try:
    try:
        import cupy as ArrM  #cupyが使えるならcupy
    except ImportError:
        import numpy as ArrM  #使えないならnumpy        
except ImportError:
    print("エラー : numpyかcupyをインストールするのです (from PolicyIteration)")
"""

import numpy as ArrM

#####################################################################
def PolicyIteration(states, P, R, gamma):
    #states = [0,1,2,3,4]
    actions = [0,1,2,3,4,5,6,7,8,9,10]
    N_STATES = len(states)
    N_ACTIONS = len(actions)
    #P = np.zeros((N_STATES1, ACT, N_STATES2))  # transition probability
    #R = np.zeros((N_STATES))  # rewards

    policy = [0 for s in range(N_STATES)]
    V = ArrM.zeros(N_STATES)
    
    #is_value_changed = True
    iterations = 0
#    while is_value_changed:
    for i in range(100):
        #is_value_changed = False
        iterations += 1
        # run value iteration for each state
        for s in range(N_STATES):
            V[s] = sum([P[s,policy[s],s1] * (R[s1] + gamma*V[s1]) for s1 in range(N_STATES)])
    
        for s in range(N_STATES):
            q_best = V[s]
            #print(q_best)
            dice = ArrM.random.uniform(0,1)
            if dice > 0.3:
                for a in range(N_ACTIONS):
                    q_sa = sum([R[s1] + (P[s, a, s1] * gamma * V[s1]) for s1 in range(N_STATES)])
                    if q_sa > q_best:
                        policy[s] = a
                        q_best = q_sa
                    #is_value_changed = True
            else:
                policy[s] = ArrM.random.randint(0,11)

    return policy
