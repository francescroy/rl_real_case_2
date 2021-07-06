

# This is a toy problem modeled as a Markov decision process.
# Author: Francesc Roy Campderrós

import numpy as np
from random import *
import math
import time
import sys
import matplotlib.pyplot as plt


X_SIZE =7 # 5,7,9
Y_SIZE =7 # 5,7,9
NUM_STATES = X_SIZE * Y_SIZE
GAMMA = 0.90
OPTIMAL_X, OPTIMAL_Y = 3,3 # 2,3,4
OPTIMAL_FINAL_STATE = False # Can be false if using TD-learning or DP methods but must be true if some MonteCarlo method...
COST_STEP = 0.10
NUM_ACTIONS = 5

ROUND_COUNTER = 0 # in seconds
TOTAL_ROUND_COUNTER = 0 # in seconds
TOTAL_PRICE = 0 # in euros
ID_USER = 0

CPU_TYPE_1 = 1
CPU_TYPE_2 = 5
MAX_JBS = 2
PRICE_JB_S = 0.000025 # 0.09$ la hora (based on real Amazon EC2...)
PRICE_SLA_S = 0.0025 # 9$ la hora
CPU_LOAD_SLA = 80

CYCLE_IN_SECONDS = int(sys.argv[1]) # in seconds (360, 1800,...)
PHOTO_INTERVAL = 5 # in seconds (5 default)
DURACIO_VID = 120 # in seconds
DURACIO_VID_MIN = 110
DURACIO_VID_MAX = 130

horari = [None] * CYCLE_IN_SECONDS # es regenera a cada cycle

# random numbers [15,30] (per patró 1):
random_numbers = [29, 24, 17, 21, 19, 29, 28, 25, 20, 30, 17, 18, 23, 21, 28, 16, 26, 19, 29, 26, 26, 23, 22, 21, 26, 26, 16, 17, 29, 27, 27, 25, 23, 17, 24, 27, 16, 16, 29, 21, 27, 26, 22, 27, 26, 17, 20, 23, 15, 19, 22, 27, 23, 24, 20, 21, 16, 15, 17, 24, 30, 24, 30, 25, 15, 17, 19, 20, 25, 16, 17, 18, 18, 24, 15, 15, 26, 29, 16, 24, 24, 29, 17, 18, 29, 20, 25, 19, 18, 17, 16, 28, 15, 18, 26, 20, 20, 29, 24, 23, 29, 16, 15, 29, 30, 29, 16, 20, 15, 17, 18, 16, 22, 27, 19, 27, 20, 27, 21, 16, 16, 16, 22, 17, 26, 29, 25, 19, 29, 23, 22, 29, 27, 15, 30, 28, 30, 22, 17, 28, 28, 27, 28, 21, 21, 24, 19, 30, 30, 27, 19, 25, 30, 21, 23, 26, 20, 18, 21, 20, 17, 30, 18, 15, 20, 19, 21, 18, 15, 28, 25, 30, 22, 28, 29, 16, 15, 22, 26, 28, 25, 27, 18, 21, 15, 16, 23, 19, 29, 25, 22, 15, 24, 30, 16, 16, 22, 19, 28, 17, 28, 20, 25, 16, 20, 28, 20, 22, 25, 20, 20, 29, 27, 23, 17, 15, 29, 27, 29, 15, 26, 15, 30, 30, 26, 18, 24, 24, 19, 20, 20, 17, 25, 23, 16, 15, 16, 24, 30, 15, 27, 23, 18, 18, 18, 25, 29, 30, 19, 23, 18, 15, 30, 30, 20, 24, 19, 22, 18, 15, 16, 24, 15, 22, 29, 21, 18, 29, 18, 17, 30, 25, 20, 24, 30, 16, 23, 24, 22, 19, 28, 27, 26, 22, 16, 25, 24, 18, 23, 22, 23, 23, 18, 30, 22, 30, 23, 15, 27, 24, 18, 23, 20, 15, 22, 28, 26, 23, 26, 27, 19, 22, 30, 18, 20, 28, 20, 24, 20, 17, 16, 23, 25, 21, 21, 18, 23, 21, 30, 15, 17, 23, 21, 24, 16, 26, 18, 23, 27, 16, 18, 23, 15, 26, 23, 24, 21, 26, 19, 29, 17, 23, 22, 18, 18, 21, 30, 20, 27, 20, 22, 19, 23, 30, 17, 25, 29, 30, 30, 18, 24, 18, 28, 17, 25, 26, 23, 17, 26, 15, 24, 30, 20, 17, 26, 27, 15, 24, 24, 26, 29, 27, 21, 30, 17, 22, 16, 23, 17, 23]

INICI_CONF=[] # fixe per tota la simulacio, de llargaria numero de videoconf. en un cycle (i amb valors de 0 a CYCLE_IN_SECONDS?)
for index_i in range(15): # 15 videoconf.

    index_pos_conf = random_numbers.pop()
    index_pos_primera = index_pos_conf
    INICI_CONF.append(index_pos_conf)

    while True:
        random_int = random_numbers.pop()
        index_pos_conf = index_pos_conf + DURACIO_VID_MAX + random_int

        if index_pos_conf > CYCLE_IN_SECONDS + index_pos_primera - DURACIO_VID_MAX :
            print (INICI_CONF)
            break

        INICI_CONF.append(index_pos_conf)




QUANTIZATION_TIME = 10 # default 10
QUANTIZATION_CPU = 10 # default 10

NUM_EXP = 10
TOTAL_EXP = 0


class User:
    def __init__(self,type_user,duration,id):
        self.id = id
        self.type = type_user # 1 o 2, els 1's fan augmentar la cpu un 1% i els 2 un 5%
        self.duration = duration # same for all participants of same conference, in seconds

class JVB:
    def __init__(self):
        self.users_connected = []
        self.cpu_load = None
        self.up = False

    def start(self):
        self.cpu_load = 0
        self.up = True

    def close(self):
        self.users_connected = []
        self.cpu_load = None
        self.up = False

    def is_up(self):
        return self.up

    def add_user(self,user):

        if self.up:
            self.users_connected.append(user)
            if user.type == 1:
                self.cpu_load = self.cpu_load + CPU_TYPE_1
            else:
                self.cpu_load = self.cpu_load + CPU_TYPE_2
            return 1
        return 0

    def advance_round(self):
        for user in reversed(self.users_connected):
            user.duration = user.duration - 1
            if user.duration == 0:
                self.users_connected.remove(user)
                if user.type==1:
                    self.cpu_load = self.cpu_load - CPU_TYPE_1
                else:
                    self.cpu_load = self.cpu_load - CPU_TYPE_2

class Jitsi:
    def __init__(self):

        self.video_bridges = []
        for i in range(MAX_JBS):
            self.video_bridges.append(JVB())

        self.video_bridges[0].start() # always 1 up...
        self.video_bridges_up = 1

    def get_some_jvb_down(self):

        jvb_down = None

        for jvb in self.video_bridges:
            if not jvb.is_up():
                jvb_down = jvb
                break

        return jvb_down

    def start_jvb(self):

        if self.video_bridges_up < MAX_JBS:
            self.get_some_jvb_down().start()
            self.video_bridges_up += 1
            return 1
        return 0

    def add_user(self,user):
        jvb_selected = self.get_least_loaded_jvb()
        jvb_selected.add_user(user)

    def stop_jvb(self):
        # No imagino cap cas on tingui sentit apagar algun jvb que no sigui el que esta menys carregat... així que de moment...

        if self.video_bridges_up > 1:

            jvb_selected = self.get_least_loaded_jvb()
            users_to_reallocate = jvb_selected.users_connected
            jvb_selected.close()
            self.video_bridges_up -= 1

            for user in users_to_reallocate:
                self.add_user(user)

            return 1
        return 0

    def advance_round(self):
        for jvb in self.video_bridges:
            if jvb.is_up():
                jvb.advance_round()

    def get_users_connected(self):

        num_users=0
        for jvb in self.video_bridges:
            num_users= num_users + len(jvb.users_connected)

        return num_users

    def get_state(self):

        state=None

        state = [self.video_bridges_up, ROUND_COUNTER]

        for jvb in self.video_bridges:
            state.append(jvb.cpu_load)

        return state

    def get_least_loaded_jvb(self):

        jvb_selected = None
        cpu_min = 1000

        for jvb in self.video_bridges:

            if jvb.is_up():

                if jvb.cpu_load <= cpu_min:
                    cpu_min = jvb.cpu_load
                    jvb_selected = jvb

        return jvb_selected

    def get_most_loaded_jvb(self):

        jvb_selected = None
        cpu_max = 0

        for jvb in self.video_bridges:

            if jvb.is_up():

                if jvb.cpu_load >= cpu_max:
                    cpu_max = jvb.cpu_load
                    jvb_selected = jvb

        return jvb_selected

class Autoscaler:

    def __init__(self,jitsi,threshold):
        self.jitsi = jitsi
        self.threshold = threshold

    def perform_action(self,jitsi_state):

        # [jvb,part,cpu1,cpu2] jitsi_state

        if jitsi_state[2] is not None:
            if jitsi_state[2] + self.threshold > CPU_LOAD_SLA:
                self.jitsi.start_jvb()

        if jitsi_state[3] is not None:
            if jitsi_state[3] + self.threshold > CPU_LOAD_SLA:
                self.jitsi.start_jvb()

        if jitsi_state[2] is not None and jitsi_state[3] is not None:
            if jitsi_state[2] + jitsi_state[3] <= CPU_LOAD_SLA - self.threshold:
                self.jitsi.stop_jvb()

class AutoscalerRL:

    def __init__(self,jitsi,policy):
        self.jitsi = jitsi
        self.policy = policy

    def perform_action(self,jitsi_state):
        # It returns the action as well
        coordinates = get_coordinates_state(jitsi_state)

        if self.policy[coordinates[0]][coordinates[1]][coordinates[2]][coordinates[3]] == 1:
            self.jitsi.start_jvb()
            return 1
        elif self.policy[coordinates[0]][coordinates[1]][coordinates[2]][coordinates[3]] == -1:
            self.jitsi.stop_jvb()
            return -1
        return 0

    def set_policy(self,policy):
        self.policy= policy

class NoAutoscaler:
    def __init__(self,jitsi):
        self.jitsi = jitsi

    def perform_action(self,jitsi_state):
        pass



def compute_price(jitsi):
    global TOTAL_PRICE

    TOTAL_PRICE = TOTAL_PRICE + PRICE_JB_S * jitsi.video_bridges_up

    for jvb in jitsi.video_bridges:
        if jvb.is_up():
            if jvb.cpu_load>=CPU_LOAD_SLA:
                TOTAL_PRICE = TOTAL_PRICE + PRICE_SLA_S * (1 + (jvb.cpu_load - CPU_LOAD_SLA)/CPU_LOAD_SLA)

def advance_rounds(jitsi,rounds):
    global ROUND_COUNTER
    global TOTAL_ROUND_COUNTER

    for i in range(rounds):

        # Start of the round:
        new_users(jitsi)
        compute_price(jitsi)

        # End of the round:
        jitsi.advance_round()

        TOTAL_ROUND_COUNTER = TOTAL_ROUND_COUNTER + 1
        ROUND_COUNTER = ROUND_COUNTER + 1
        if ROUND_COUNTER == CYCLE_IN_SECONDS:
            ROUND_COUNTER=0
            populate_timetable() # new!
            #print("Cycle number "+str(TOTAL_ROUND_COUNTER/CYCLE_IN_SECONDS)+" finished")

def new_users(jitsi):
    global ID_USER

    list_of_users_to_connect = horari[ROUND_COUNTER]
    if list_of_users_to_connect is not None:
        for con_user in list_of_users_to_connect:
            jitsi.add_user(User(con_user[0], con_user[1], ID_USER))
            ID_USER = ID_USER + 1

def test_autoscaler(type_autoscaler,policy):

    global ROUND_COUNTER, TOTAL_ROUND_COUNTER, TOTAL_PRICE, ID_USER, TOTAL_EXP

    ROUND_COUNTER = 0  # in seconds
    TOTAL_ROUND_COUNTER = 0  # in seconds
    TOTAL_PRICE = 0  # in euros
    ID_USER = 0

    populate_timetable()

    jitsi = Jitsi()
    autoscaler = None

    if type_autoscaler==1:
        autoscaler = Autoscaler(jitsi,45) 
    elif type_autoscaler==2:
        autoscaler = Autoscaler(jitsi,50)
    elif type_autoscaler==3:
        autoscaler = Autoscaler(jitsi,55)
    elif type_autoscaler==4:
        autoscaler = Autoscaler(jitsi,60)
    elif type_autoscaler==5:
        autoscaler = Autoscaler(jitsi,65)
    elif type_autoscaler==6:
        autoscaler = Autoscaler(jitsi,70)
    elif type_autoscaler==7:
        autoscaler = Autoscaler(jitsi,75)
    elif type_autoscaler==8:
        autoscaler = NoAutoscaler(jitsi)
    elif type_autoscaler==9:
        autoscaler = AutoscalerRL(jitsi,policy)

    a_week = 24*7*3600/PHOTO_INTERVAL
    for i in range(int(a_week)): # 1 WEEK...

        # each PHOTO_INTERVAL seconds we take a "photo":
        advance_rounds(jitsi, PHOTO_INTERVAL)

        state = jitsi.get_state()
        autoscaler.perform_action(state)

    print()
    print(str(round(TOTAL_PRICE, 2)) + " Euros")
    if type_autoscaler==9:
        TOTAL_EXP = TOTAL_EXP + round(TOTAL_PRICE, 2)
    print()





def get_closest(number,to):

    if number is None:
        return None

    number_decimal = float(number) / to
    number_rounded = round(number_decimal)
    number_final = int(number_rounded * to)

    return number_final

    """
    while 1:
        if number % 10 == 0:
            break
        else:
            number = number-1

    return number
    """

def compute_reward_state(state):

    cost = (state[0]*PRICE_JB_S) * PHOTO_INTERVAL

    for i in range(2,len(state)):
        if state[i] is not None:
            if state[i]>=CPU_LOAD_SLA:
                cost = cost + (1 + (state[i]-CPU_LOAD_SLA)/CPU_LOAD_SLA)*PRICE_SLA_S * PHOTO_INTERVAL

    return - cost

def get_coordinates_state(jitsi_state):
    # Un estat seria algo com [1,171,93,None]

    coordernada_x = jitsi_state[0] - 1
    coordernada_y = int(get_closest(jitsi_state[1],QUANTIZATION_TIME) / QUANTIZATION_TIME)
    coordernada_z = None
    coordernada_a = None

    if jitsi_state[2] == None:
        coordernada_z = 0
    else:
        coordernada_z = int(get_closest(jitsi_state[2],QUANTIZATION_CPU) / QUANTIZATION_CPU + 1)

    if jitsi_state[3] == None:
        coordernada_a = 0
    else:
        coordernada_a = int(get_closest(jitsi_state[3],QUANTIZATION_CPU) / QUANTIZATION_CPU + 1)

    return [coordernada_x,coordernada_y,coordernada_z,coordernada_a]




def get_num_action_2(action):
    if(action==-1):
        return 0
    if(action==0):
        return 1
    if(action==1):
        return 2

def get_action_2(action_num):
    if(action_num==0):
        return -1
    if(action_num==1):
        return 0
    if(action_num==2):
        return 1

def argmax_a_2(Q,state):
    coor = get_coordinates_state(state)

    best_action = get_action_2(0)
    best_q = Q[0][coor[0]][coor[1]][coor[2]][coor[3]]

    if Q[1][coor[0]][coor[1]][coor[2]][coor[3]] > best_q:
        best_action = get_action_2(1)
        best_q = Q[1][coor[0]][coor[1]][coor[2]][coor[3]]

    if Q[2][coor[0]][coor[1]][coor[2]][coor[3]] > best_q:
        best_action = get_action_2(2)
        best_q = Q[2][coor[0]][coor[1]][coor[2]][coor[3]]

    return best_action

def max_a_2(Q,state):
    coor = get_coordinates_state(state)

    best_q = Q[0][coor[0]][coor[1]][coor[2]][coor[3]]

    if Q[1][coor[0]][coor[1]][coor[2]][coor[3]] > best_q:
        best_q = Q[1][coor[0]][coor[1]][coor[2]][coor[3]]

    if Q[2][coor[0]][coor[1]][coor[2]][coor[3]] > best_q:
        best_q = Q[2][coor[0]][coor[1]][coor[2]][coor[3]]

    return best_q






def add_conference(temps_x,pos_x,tipus_x):
    # Conf 1.
    for i in range(tipus_x[0]):

        plus = round(np.random.normal(0, 3, 1)[0])

        if plus > 10 or plus < -10:
            plus = 0

        pos = int(pos_x + plus)
        if horari[pos] == None:
            horari[pos] = [[1, temps_x]]
        else:
            horari[pos].append([1, temps_x])

    for i in range(tipus_x[1]):

        plus = round(np.random.normal(0, 3, 1)[0])

        if plus > 10 or plus < -10:
            plus = 0

        pos = int(pos_x + plus)
        if horari[pos] == None:
            horari[pos] = [[2, temps_x]]
        else:
            horari[pos].append([2, temps_x])

def populate_timetable():
    global horari

    horari = [None] * CYCLE_IN_SECONDS
    # MAXIM QUE PUGUI CARREGAR 150! 15*6 = 90 users
    # Enregistrar aquestes dades de Jitsi realment...
    for pos_conf in INICI_CONF:
        # temps_x = randint(60, 120)
        temps_x = round(np.random.normal(DURACIO_VID, 10, 1)[0])
        if temps_x < DURACIO_VID_MIN or temps_x > DURACIO_VID_MAX:
            temps_x = DURACIO_VID
        #pos_x = randint(15, CYCLE_IN_SECONDS - 15 - 1)
        pos_x = pos_conf
        tipus_x = [5, 1] # 5 de tipus 1 i 1 de tipus 2 = 6

        add_conference(temps_x, pos_x, tipus_x)





def main2():

    global ROUND_COUNTER, TOTAL_ROUND_COUNTER, TOTAL_PRICE, ID_USER

    print("What do you want to do?:")
    option_selected = "1"

    for num_experiments in range(NUM_EXP):

        print("##########################################")
        print("##########################################")
        print("EXPERIMENT NUMBER " + str(num_experiments))
        print("##########################################")
        print("##########################################")

        if option_selected == "1":

            test_autoscaler(1, None)

        if option_selected == "1":

            test_autoscaler(2, None)

        if option_selected == "1":

            test_autoscaler(3, None)

        if option_selected == "1":

            test_autoscaler(4, None)

        if option_selected == "1":

            test_autoscaler(5, None)

        if option_selected == "1":

            test_autoscaler(6, None)

        if option_selected == "1":

            test_autoscaler(7, None)

        if option_selected == "1":

            test_autoscaler(8, None)

        if option_selected == "1":

            ROUND_COUNTER = 0  # in seconds
            TOTAL_ROUND_COUNTER = 0  # in seconds
            TOTAL_PRICE = 0  # in euros
            ID_USER = 0

            populate_timetable()

            #"""
            volum_dimensio_number_jitsi = MAX_JBS
            volum_dimensio_temps = int(CYCLE_IN_SECONDS / QUANTIZATION_TIME + 1)
            volum_dimensio_cpu = int(150 / QUANTIZATION_CPU + 2)

            print("Q.time: " + str(QUANTIZATION_TIME))
            print("Q.cpu: " + str(QUANTIZATION_CPU))

            policy_actual = np.zeros((volum_dimensio_number_jitsi, volum_dimensio_temps, volum_dimensio_cpu, volum_dimensio_cpu))
            #"""

            jitsi = Jitsi()
            autoscaler = AutoscalerRL(jitsi,policy_actual)

            # Un estat seria algo com [1,15,43,None]
            # V_policy_actual(s) = un valor
            # V_policy_actual = dimensio de policy? Si
            # Q_policy_actual = mes gran que la dimensio de la policy

            #"""
            ALPHA = 0.1  # Which is the right value? After 50% of iteration decay... after 80% decay... LEARNING RATE...
            print("ALPHA: "+str(ALPHA))
            current_state = jitsi.get_state()
            Q = np.zeros((3, volum_dimensio_number_jitsi, volum_dimensio_temps, volum_dimensio_cpu, volum_dimensio_cpu))

            EPSILON = 1.00
            number_of_iterations = int(sys.argv[2])
            print("ITERATIONS: "+str(number_of_iterations))
            if number_of_iterations != 0:
                DECAYING_EPSILON = 1.0/number_of_iterations

            print("CYCLE: "+ str(CYCLE_IN_SECONDS))
            #"""

            for t in range(number_of_iterations):

                print_wait_info(t, number_of_iterations)

                # (s,a,r,s')
                action = autoscaler.perform_action(current_state)
                advance_rounds(jitsi, PHOTO_INTERVAL)
                next_state = jitsi.get_state()
                reward = compute_reward_state(current_state) # afegir cost si obres o tanques? R(s,a,s')

                coor = get_coordinates_state(current_state)
                Q_t_minus_1 = Q[get_num_action_2(action)][coor[0]][coor[1]][coor[2]][coor[3]]

                #if t%(number_of_iterations/10)== 0 and t!=0:
                #    ALPHA= ALPHA/2

                Q[get_num_action_2(action)][coor[0]][coor[1]][coor[2]][coor[3]] = Q_t_minus_1 + ALPHA * (reward + GAMMA * max_a_2(Q, next_state) - Q_t_minus_1)

                policy_actual[coor[0]][coor[1]][coor[2]][coor[3]] = argmax_a_2(Q, current_state)
                random_int = randint(0, 99)
                if random_int < int(EPSILON * 100):
                    policy_actual[coor[0]][coor[1]][coor[2]][coor[3]] = get_action_2(randint(0, 2))

                autoscaler.set_policy(policy_actual)

                EPSILON = EPSILON - DECAYING_EPSILON

                current_state = next_state






            if True:

                test_autoscaler(9,policy_actual)




    print(TOTAL_EXP / NUM_EXP)







    # Vale pero la foto del sistema la vull fer cada 10 segons... FOTO + ACCIO (*QUE SUPOSARE QUE TE IMPACTE IMMEDIAT*)















class ChanceNode:
    def __init__(self, x,y,action):
        self.action = action
        self.x = x
        self.y = y
        self.trans_probabilities = np.zeros((NUM_STATES,), dtype=float)

        if action=='N':

            self.set_trans_probability(x, y + 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='S':

            self.set_trans_probability(x, y - 1, 0.7)
            self.set_trans_probability(x + 1, y, 0.15)
            self.set_trans_probability(x - 1, y, 0.15)
        elif action=='W':

            self.set_trans_probability(x - 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='E':

            self.set_trans_probability(x + 1, y, 0.7)
            self.set_trans_probability(x, y + 1, 0.15)
            self.set_trans_probability(x, y - 1, 0.15)
        elif action=='·':
            self.set_trans_probability(x ,y, 0.8)
            self.set_trans_probability(x, y + 1, 0.05)
            self.set_trans_probability(x, y - 1, 0.05)
            self.set_trans_probability(x + 1, y, 0.05)
            self.set_trans_probability(x - 1, y, 0.05)

    def set_trans_probability(self,x,y,prob):

        if(0 <= x and x <= X_SIZE-1 and 0 <= y and y <= Y_SIZE-1):
            self.trans_probabilities[x + y*Y_SIZE] = prob
        else:
            self.trans_probabilities[self.x + self.y*Y_SIZE] += prob

    def possible_next_sates(self,states):

        possible_sates = []

        for x in range(X_SIZE):
            for y in range(Y_SIZE):
                prob_to_that_state = self.trans_probabilities[x + y * Y_SIZE]

                if prob_to_that_state != 0:
                    possible_sates.append([find_state(x, y, states), prob_to_that_state])
                    # print (str(x) + " - " +str(y) + " with prob: "+ str(prob_to_that_state))

        return possible_sates

    def next_state(self, states):

        possible_sates = self.possible_next_sates(states)

        # Com a minim hi haura dos possible_state's no?
        random_int = randint(0, 99)
        definitive_next_state = None

        acumulative=0

        for s in possible_sates:

            acumulative = acumulative + s[1]

            if random_int < acumulative * 100:
                definitive_next_state = s[0]
                break


        return definitive_next_state

class State:
    def __init__(self, x, y, end,cost):
        self.x = x
        self.y = y
        self.end = end
        self.chance_nodes = None
        self.cost = cost

        if end==False:
            self.chance_nodes = [ChanceNode(x,y,'N'),ChanceNode(x,y,'S'),ChanceNode(x,y,'W'),ChanceNode(x,y,'E'),ChanceNode(x,y,'·')]

    def next_state(self, action, states):

        if self.end==True:
            return self # o None?
        if action=='N':
            return self.chance_nodes[0].next_state(states)
        if action=='S':
            return self.chance_nodes[1].next_state(states)
        if action=='W':
            return self.chance_nodes[2].next_state(states)
        if action=='E':
            return self.chance_nodes[3].next_state(states)
        if action=='·':
            return self.chance_nodes[4].next_state(states)

    def get_chance_node(self, action):
        if self.end ==True:
            return None
        if action == 'N':
            return self.chance_nodes[0]
        if action == 'S':
            return self.chance_nodes[1]
        if action == 'W':
            return self.chance_nodes[2]
        if action == 'E':
            return self.chance_nodes[3]
        if action == '·':
            return self.chance_nodes[4]


def find_state(x,y,states):

    result = None

    for s in states:
        if s.x==x and s.y==y:
            result = s

    return result

def compute_cost(x,y):
    return (pow(x - OPTIMAL_X, 2) + pow(y - OPTIMAL_Y, 2))

def compute_reward(x,y,action):

    desired_x =x
    desired_y =y

    if action=='N':
        desired_y = desired_y + 1
    elif action=='S':
        desired_y = desired_y - 1
    elif action == 'W':
        desired_x = desired_x - 1
    elif action == 'E':
        desired_x = desired_x + 1
    elif action == '·':
        pass

    reward = 0.0

    if (0 <= desired_x and desired_x <= X_SIZE - 1 and 0 <= desired_y and desired_y <= Y_SIZE - 1):
        if action!='·':
            reward =  (compute_cost(x,y) - compute_cost(desired_x,desired_y)) - COST_STEP
    else:
        reward = -COST_STEP


    return reward

def get_nice_policy(states):
    policy = [None]*NUM_STATES

    for x in range(0,int((X_SIZE/2))):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'S'

    for x in range(X_SIZE):
        for y in range(0,int((Y_SIZE/2))):
            policy[x + y * Y_SIZE] = 'E'

    for x in range(int((X_SIZE/2))+1,X_SIZE):
        for y in range(Y_SIZE):
            policy[x + y*Y_SIZE] = 'N'

    for x in range(int((X_SIZE/2)),X_SIZE):
        for y in range(int((Y_SIZE/2)+1),Y_SIZE):
            policy[x + y * Y_SIZE] = 'W'

    for x in range(X_SIZE):
        for y in range(Y_SIZE):
            if policy[x + y * Y_SIZE] == None:
                if states[x + y * Y_SIZE].end==False:
                    policy[x + y * Y_SIZE] = '·'



    return policy

def get_random_policy(states):

    policy= []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if(find_state(x,y,states).end==False):


                rand_dir = randint(0, 4)
                if rand_dir==0:
                    policy.append('N')
                if rand_dir==1:
                    policy.append('S')
                if rand_dir==2:
                    policy.append('W')
                if rand_dir==3:
                    policy.append('E')
                if rand_dir==4:
                    policy.append('·')




            else:
                policy.append(None)

    return policy

def get_fixed_random_policy(states):
    return ['S', 'N', 'S', '·', '·', 'N', 'S', '·', 'W', 'E', '·', 'S', '·', 'W', 'W', 'S', 'S', 'S', 'E', '·', 'E', 'E', 'W', '·', 'S', 'S', 'W', 'N', 'E', '·', 'N', '·', 'E', 'S', 'W', 'S', '·', 'N', 'W', '·', 'N', 'S', 'S', 'W', '·', 'S', '·', 'S', 'E']

def print_V(V):
    for i in range(X_SIZE):
        for j in range(Y_SIZE):
            print(str(i) + " " + str(j) + ": " + str(V[i + j * Y_SIZE]))

def print_policy(policy):
    print()
    for y in range(Y_SIZE - 1, -1, -1):
        for x in range(X_SIZE):

            if policy[x + y * Y_SIZE] != None:
                print(policy[x + y * Y_SIZE], end=" ")
            else:
                print(" ", end=" ")
        print()
    print()

def get_random_state(states):
    random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    while random_state.end == True:
        random_state = find_state(randint(0, X_SIZE - 1), randint(0, Y_SIZE - 1), states)
    return random_state

def print_wait_info(loop,number_of_iterations):
    if (loop % (number_of_iterations / 10) == 0 and loop != 0):
        print("|", end=' ')
        sys.stdout.flush()

def get_num_action(action):
    if(action=='N'):
        return 0
    if(action=='S'):
        return 1
    if(action=='W'):
        return 2
    if(action=='E'):
        return 3
    if(action=='·'):
        return 4

def get_action(action_num):
    if(action_num==0):
        return 'N'
    if(action_num==1):
        return 'S'
    if(action_num==2):
        return 'W'
    if(action_num==3):
        return 'E'
    if(action_num==4):
        return '·'

def argmax_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_action

def max_a(Q,state):

    best_action = get_action(0)
    best_q = Q[0][state.x + state.y * Y_SIZE]

    if Q[1][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(1)
        best_q = Q[1][state.x + state.y * Y_SIZE]

    if Q[2][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(2)
        best_q = Q[2][state.x + state.y * Y_SIZE]

    if Q[3][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(3)
        best_q = Q[3][state.x + state.y * Y_SIZE]

    if Q[4][state.x + state.y * Y_SIZE] > best_q:
        best_action = get_action(4)
        best_q = Q[4][state.x + state.y * Y_SIZE]

    return best_q




def main():

    states = []

    for x in range(X_SIZE):
        for y in range(Y_SIZE):

            if (x==OPTIMAL_X and y == OPTIMAL_Y):
                states.append(State(x,y,OPTIMAL_FINAL_STATE,compute_cost(x,y)))
            else:
                states.append(State(x, y, False,compute_cost(x,y)))


    #policy_example = get_nice_policy(states) # is simply a list of strings...
    policy_example = get_fixed_random_policy(states)  # is simply a list of strings...

    print_policy(policy_example)

    print("What do you want to do?:")

    option_selected = input()




    # MODEL-FREE MDP ALGO: Q-LEARNING, FIND [ESTIMATED] OPTIMAL POLICY
    if option_selected == "5":

        policy_actual = policy_example

        ALPHA = 0.001  # Which is the right value? After 50% of iteration decay... after 80% decay... LEARNING RATE...
        current_state = get_random_state(states)
        Q = []
        for i in range(NUM_ACTIONS):
            Q.append([0.0] * NUM_STATES)

        EPSILON = 1.00
        number_of_iterations = 5000000
        DECAYING_EPSILON = 1.0/number_of_iterations

        for t in range(number_of_iterations):

            print_wait_info(t, number_of_iterations)

            action = policy_actual[current_state.x + current_state.y * Y_SIZE]
            next_state = current_state.next_state(action, states)
            reward = current_state.cost - next_state.cost

            Q_t_minus_1 = Q[get_num_action(action)][current_state.x + current_state.y * Y_SIZE]

            if t%(number_of_iterations/10)== 0 and t!=0:
                ALPHA= ALPHA/2

            Q[get_num_action(action)][current_state.x + current_state.y *Y_SIZE] = Q_t_minus_1 + ALPHA * (reward + GAMMA * max_a(Q, next_state) - Q_t_minus_1)

            policy_actual[current_state.x + current_state.y * Y_SIZE] = argmax_a(Q, current_state)
            random_int = randint(0, 99)
            if random_int < int(EPSILON * 100):
                policy_actual[current_state.x + current_state.y * Y_SIZE] = get_action(randint(0, 4))

            EPSILON= EPSILON - DECAYING_EPSILON

            #current_state = next_state
            current_state= get_random_state(states)

        print_policy(policy_actual)









    # Suposant que no ens donen les probabilitats, com trobes V de una policy pi?: TD learning... Osigui model free es com RL ja no...? CLAU...

    # After TD learning, Q learning...

    # potser podria usar threads per usar diferents CPU's...
    # mes endavant -> imagina que les transition probabilities cambiessin through time... que en el fons es el que passa al autoscaling problem...

    # l'ultim montecarlo esta bé perque aconsegueix estimator of V que es unbiased!! tot i que es veu que la variance es bastant gran...

    ## THEORY ##
    # It would be nice to undestand why DP policy evaluation algo. works... is because you are using bootstrapping...
    # It's much easier to understand MC policy evaluation algo. works... it relies on sampling, not on bootstrapping...

if __name__ == '__main__':
    main2()