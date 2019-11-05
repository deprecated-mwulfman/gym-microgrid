import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

class MicrogridEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    
    self.max_u=100
    
    self.n = 6
    self.dt = .05
    self.dt_2 = 0.5
    
    self.P_N = np.array([[0.505, 0.028, 0.261, 0.179, 0.168, 0.012]])
    self.P_star = np.array([[0.202, 0.008, 0.078, 0.054, 0.067, 0.004]])
    
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
    '10 5': [0.367, 0.133, 4560, 0.33],
    }

               
    
    self.k_P = np.array([[0.396, 7.143, 0.766, 1.117, 1.191, 16.667]])
    self.K_P = np.diag(self.k_P)
    self.T_1 = self.dt=.05*np.eye(self.n)
    self.T_2 = np.diag((self.dt/self.dt_2)*np.ones((self.n,1)))
    
    self.viewer = None

    high = 100*np.array([1., 1., 1.,1., 1., 1.,1., 1., 1.,1., 1., 1.])
    self.action_space = spaces.Box(low=-self.max_u, high=self.max_u, shape=(6,), dtype=np.float32)
    self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    self.seed()
    
      
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  
  def step(self,u):
    x = self.state # th := theta
    x_1=x[:self.n]
    x_2=x[self.n:]
    
    f_1 = x_1 + self.T_1 @ x_2
    P = power_flow(self, rx_to_gb(self.y_dict),x_1) ## TO DO
    f_2 = x_2 - self.T_2 @(x_2+self.K_P @ ( P-self.P_star))
    g_2 = self.T_2
    
  
    self.last_u = u # for rendering
    costs = np.sum(x_1**2) + .1*np.sum(x_2**2) + .001*np.sum((u**2))

    newx_1 = f_1 
    newx_2 = f_2 + np.dot(g_2,u)

    self.state = np.concatenate((newx_1,newx_2),axis=0)
    return self._get_obs(), -costs, False, {}

  def reset(self):
    high = np.array([np.pi,np.pi,np.pi,np.pi,np.pi,np.pi, 100,100,100,100,100,100])
    self.state = self.np_random.uniform(low=-high, high=high)
    self.last_u = None
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
  p1 = self.load['5H']+self.load['5I']+gb_dict['5 9'][0]-gb_dict['5 9'][0]*np.cos(delta[0]-delta[2])-gb_dict['5 9'][1]*np.sin(delta[0]-delta[2])+gb_dict['10 5'][0]-gb_dict['10 5'][0]*np.cos(delta[0]-delta[2])-gb_dict['10 5'][1]*np.sin(delta[0]-delta[2])
  p2 = self.load['5H']+self.load['5I']+gb_dict['5 9'][0]-gb_dict['5 9'][0]*np.cos(delta[1]-delta[3])-gb_dict['5 9'][1]*np.sin(delta[1]-delta[3])+gb_dict['10 5'][0]-gb_dict['10 5'][0]*np.cos(delta[1]-delta[3])-gb_dict['10 5'][1]*np.sin(delta[1]-delta[3])
  p3 = self.load['9H']+self.load['9I']+gb_dict['5 9'][0]-gb_dict['5 9'][0]*np.cos(delta[2]-delta[0])-gb_dict['5 9'][1]*np.sin(delta[2]-delta[0])+gb_dict['9 10'][0]-gb_dict['9 10'][0]*np.cos(delta[2]-delta[4])-gb_dict['9 10'][1]*np.sin(delta[2]-delta[4])
  p4 = self.load['9H']+self.load['9I']+gb_dict['5 9'][0]-gb_dict['5 9'][0]*np.cos(delta[3]-delta[1])-gb_dict['5 9'][1]*np.sin(delta[3]-delta[1])+gb_dict['9 10'][0]-gb_dict['9 10'][0]*np.cos(delta[3]-delta[5])-gb_dict['9 10'][1]*np.sin(delta[3]-delta[5])
  p5 = self.load['10H']+self.load['10I']+gb_dict['10 5'][0]-gb_dict['10 5'][0]*np.cos(delta[4]-delta[0])-gb_dict['10 5'][1]*np.sin(delta[4]-delta[0])+gb_dict['9 10'][0]-gb_dict['9 10'][0]*np.cos(delta[4]-delta[2])-gb_dict['9 10'][1]*np.sin(delta[4]-delta[2])
  p6 = self.load['10H']+self.load['10I']+gb_dict['10 5'][0]-gb_dict['10 5'][0]*np.cos(delta[5]-delta[1])-gb_dict['10 5'][1]*np.sin(delta[5]-delta[1])+gb_dict['9 10'][0]-gb_dict['9 10'][0]*np.cos(delta[5]-delta[3])-gb_dict['9 10'][1]*np.sin(delta[5]-delta[3])
  return np.array([p1,p2,p3,p4,p5,p6])
      
def rx_to_gb(self, y_dict):
    gb_dict = {}
    for key in y_dict:
      r = y_dict[key][0]*y_dict[key][3]
      x = y_dict[key][1]*y_dict[key][3]
      y = r+1j*x
      g = np.real(1/y)
      b = np.imag(1/y)
      gb_dict[key] = [g, b]
    return gb_dict
