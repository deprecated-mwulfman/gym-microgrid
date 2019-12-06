import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class MicrogridEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.max_u=100

        self.n = 3
        self.dt = .05
        self.dt_2 = 0.5
        self.x2_star = 50*2*np.pi
        self.P_N = np.array([0.505+0.028, 0.261+0.179, 0.168+0.012])
        self.P_star = np.array([0.202+0.008, 0.078+0.054, 0.067+0.004])

        self.load={"1H":0.15000,"1I":0.05000,

                   "3H":0.00276,"3I":0.00224,
                   "4H":0.00432,"4I":0.0000,
                   "5H":0.00725,"5I":0.0000,
                   "6H":0.00550,"6I":0.0000,
                   "7H":0.00000,"7I":0.00077,
                   "8H":0.00588,"8I":0.00000,
                   "9H":0.00000,"9I":0.00574,
                   "10H":0.00477,"10I":0.00068,
                   "11H":0.00331,"11I":0.00000,
                   "12H":0.15000,"12I":0.05000,
                   "13H":0.00000,"13I":0.00032,
                   "14H":0.00207,"14I":0.00330}
        self.y_dict = {
        '5 9': [0.336, 0.126, 5488, 1.54],
        '9 10': [0.399, 0.133, 4832, 0.77],
        '10 5': [0.367, 0.133, 4560, 0.33]
        }

        self.gb_dict = self.rx_to_gb(self.y_dict)

        self.k_P = np.array([0.396+7.143, 0.766+1.117, 1.191+16.667])
        self.K_P = np.diag(self.k_P)
        self.T_1 = self.dt*np.eye(self.n)
        self.T_2 = (self.dt/self.dt_2)*np.eye(self.n)

        self.K_err = np.diag([0.1,0.1,0.1])
        self.c_err = 0.2

        self.viewer = None

        high = 100*np.array([1., 1., 1.,1., 1., 1.])
        self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,f2_hat):
        x = self.state # th := theta
        x_1=x[:self.n]
        x_2=x[self.n:]

        f_1 = x_1 + self.T_1 @ x_2
        P = self.power_flow(self.gb_dict ,x_1) ## TO DO
        f_2 = x_2 - self.T_2 @(x_2 + self.K_P @(P-self.P_star))
        # print("power P: ", P)
        # print("f2: ", f_2)
        # print("f1: ", f_1)
        g_2 = self.T_2

        # print("err(k): ",   np.linalg.norm(x_2-self.x2_star))
        # print("f2-Ke: ", np.linalg.norm(f_2-f2_hat-self.K_err@(self.x2_star-x_2)))
        # print("f2: ", f_2)
        # print("f2_hat: ", f2_hat)
        # noise = np.random.random(3)
        u = np.linalg.inv(g_2) @ (self.x2_star - f2_hat + self.K_err@(self.x2_star-x_2))
        # print("u: ",u)
        # print("g2_inv: ", np.linalg.inv(g_2))
        # print("f2_hat: ", f2_hat)
        # print("self.x2_star: ", self.x2_star)
        # print("self.K_err@(self.x2_star-x_2): ",self.K_err@(self.x2_star-x_2))

        # costs = np.sum(x_1**2) + .1*np.sum(x_2**2) + .001*np.sum((u**2))
        costs, done = self.get_reward(self.state, f2_hat)
        # costs = np.mean((x_2 - self.x2_star)**2)
        # dist = np.linalg.norm(x_2 - self.x2_star)#, axis=1)
        # print("np norm: ",dist)
        # print("np mean: ", costs)
        # costs = np.sum(np.abs(x_2-self.x2_star) < self.c_err)

        self.last_u = f2_hat # for rendering

        newx_1 = f_1
        newx_2 = f_2 + np.dot(g_2,u)

        self.state = np.concatenate((newx_1,newx_2),axis=0)

        return self._get_obs(), costs, done, {}

    def reset(self):
        # high = np.array([np.pi,np.pi,np.pi,np.pi,np.pi,np.pi, 55/(2*np.pi),55/(2*np.pi),55/(2*np.pi),55/(2*np.pi),55/(2*np.pi),55/(2*np.pi)])
        # low = np.array([-np.pi,-np.pi,-np.pi,-np.pi,-np.pi,-np.pi, 45/(2*np.pi),45/(2*np.pi),45/(2*np.pi),45/(2*np.pi),45/(2*np.pi),45/(2*np.pi)])
        # high = np.array([np.pi,np.pi,np.pi, 55*(2*np.pi),55*(2*np.pi),55*(2*np.pi)])
        # low = np.array([-np.pi,-np.pi,-np.pi, 45*(2*np.pi),45*(2*np.pi),45*(2*np.pi)])
        high = np.array([np.pi,np.pi,np.pi, 5*(2*np.pi),5*(2*np.pi),5*(2*np.pi)])
        low = np.array([-np.pi,-np.pi,-np.pi, -5*(2*np.pi),-5*(2*np.pi),-5*(2*np.pi)])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        # x2 = np.expand_dims(self.state, axis = 0)
        #
        # x2 = x2[:, self.n:]
        # target_x2 = np.tile(self.x2_star,np.shape(x2))
        # done_error = 5*2*np.pi
        # dones = np.array(np.sum(np.abs(x2-target_x2)>done_error,axis=1)>0)
        # print(np.abs(x2-target_x2))
        # print(np.abs(x2-target_x2)>done_error)
        # print(np.sum(np.abs(x2-target_x2)>done_error,axis=1)>0)
        # print(dones)

        return self._get_obs()

    def _get_obs(self):
        return self.state

    def render(self, mode='human'):
        pass


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



    def power_flow(self, gb_dict,delta):
        p5 = self.load['5H']+self.load['5I']+gb_dict['5 9'][0]+gb_dict['10 5'][0]-gb_dict['5 9'][0]*np.cos(delta[0]-delta[1])-gb_dict['5 9'][1]*np.sin(delta[0]-delta[1])-gb_dict['10 5'][0]*np.cos(delta[0]-delta[2])-gb_dict['10 5'][1]*np.sin(delta[0]-delta[2])
        # print("load power: ", self.load['5H']+self.load['5I'])
        # print("diagonal network power: ", gb_dict['5 9'][0])
        # print("off- diagonal power: ", )
        p9 = self.load['9H']+self.load['9I']+gb_dict['5 9'][0]+gb_dict['9 10'][0]-gb_dict['5 9'][0]*np.cos(delta[1]-delta[0])-gb_dict['5 9'][1]*np.sin(delta[1]-delta[0])-gb_dict['9 10'][0]*np.cos(delta[1]-delta[2])-gb_dict['9 10'][1]*np.sin(delta[1]-delta[2])
        p10 = self.load['10H']+self.load['10I']+gb_dict['10 5'][0]+gb_dict['9 10'][0]-gb_dict['10 5'][0]*np.cos(delta[2]-delta[0])-gb_dict['10 5'][1]*np.sin(delta[2]-delta[0])-gb_dict['9 10'][0]*np.cos(delta[2]-delta[1])-gb_dict['9 10'][1]*np.sin(delta[2]-delta[1])
        return np.array([p5,p9,p10])

    def rx_to_gb(self, y_dict):
        gb_dict = {}
        for key in y_dict:
            r = y_dict[key][0]*y_dict[key][3]
            x = y_dict[key][1]*y_dict[key][3]
            z = r+1j*x
            g = np.real(1/z)
            b = np.imag(1/z)
            gb_dict[key] = [g, b]
        # gb_dict['5 5'] = gb_dict['5 9'][0] + gb_dict['10 5'][0]
        # gb_dict['9 9'] = gb_dict['5 9'][0] + gb_dict['9 10'][0]
        # gb_dict['10 10'] = gb_dict['9 10'][0] + gb_dict['10 5'][0]
        return gb_dict

    def get_reward(self, observations, actions):

        """get reward/s of given (observations, actions) datapoint or datapoints

        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)

        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """

        #initialize and reshape as needed, for batch mode
        self.reward_dict = {}
        if(len(observations.shape)==1):
            observations = np.expand_dims(observations, axis = 0)
            actions = np.expand_dims(actions, axis = 0)
            batch_mode = False
        else:
            batch_mode = True

        #get vars
        # print('obs shape: ',np.shape(observations))
        x2 = observations[:, self.n:]
        target_x2 = np.tile(self.x2_star,np.shape(x2))

        #calc rew
        # print('x2 shape: ',np.shape(x2))
        # print('target shape: ',np.shape(target_x2))
        # print('x2-target: ',x2-target_x2)
        dist = np.linalg.norm(x2 - target_x2, axis=1)#+np.linalg.norm(f2 - f2_hat, axis=1)
        # print('dist: ',dist)
        self.reward_dict['r_total'] = -dist

        #done is always false for this env
        # dones = np.zeros((observations.shape[0],))
        done_error = 5*2*np.pi
        dones = np.array(np.sum(np.abs(x2-target_x2)>done_error,axis=1)>0)
        # print(np.abs(x2-target_x2))
        # print(np.abs(x2-target_x2)>done_error)
        # print(np.sum(np.abs(x2-target_x2)>done_error,axis=1)>0)
        # print(dones)
        # print("dones shape: ",np.shape(dones))
        # print("dones: ",dones)

        #return
        if(not batch_mode):
            return self.reward_dict['r_total'][0], dones[0]
        return self.reward_dict['r_total'], dones
