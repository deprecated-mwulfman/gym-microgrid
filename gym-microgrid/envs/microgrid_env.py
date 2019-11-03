import gym
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
    
    self.k_P = np.array([[0.396, 7.143, 0.766, 1.117, 1.191, 16.667]])
    self.K_P = np.diag(self.k_p)
    self.T_1 = self.dt=.05*np.eye(n)
    self.T_2 = np.diag(self.dt/self.dt_*np.ones((n,1)))
    self.viewer = None

    high = np.array([1., 1., self.max_speed])
    self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
    self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    self.seed()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]
  def step(self,u):
    x_1, x_2 = self.state # th := theta
    
    f_1 = x_1 + self.T_1 @ x_2
    P = 0
    f_2 = x_2 - self.T_2 @(x_2+self.K_P @ ( P-self.P_star)-u)
    g_2 = self.T_2
    
    u = np.clip(u, -self.max_torque, self.max_torque)[0]
    self.last_u = u # for rendering
    costs = angle_normalize(np.sum(x_1))**2 + .1*x_2**2 + .001*(u**2)

    newx_1 = f_1 
    newx_2 = f_2 + g_2@u

    self.state = np.array([newx_1, newtx_2])
    return self._get_obs(), -costs, False, {}

  def reset(self):
    high = np.array([np.pi, 1])
    self.state = self.np_random.uniform(low=-high, high=high)
    self.last_u = None
    return self._get_obs()

  def _get_obs(self):
    theta, thetadot = self.state
    return np.array([np.cos(theta), np.sin(theta), thetadot])

  def render(self, mode='human'):
    pass

   
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

  def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
