from gym-microgrid.env.microgrid_env import all

def main():
    env = MicrogridEnv()
    env.reset()
    obs = []
    costs = []
    dones = []

    for i in range(100):
        ob, cost, done, _ = env.step(0)
        obs.append(ob)
        costs.append(cost)
        dones.append(done)

    import matplotlib.pyplot as plt
    # print(np.shape(obs))
    plt.plot(np.array(obs))
    plt.show()

if __name__ == "__main__":
    main()
