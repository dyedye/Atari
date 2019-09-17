import gym
from DQN import DQN_Agent
NUM_EPISODES = 12000
NO_OP_STEPS = 30
ENV_NAME = 'MsPacman-v0'

env = gym.make(ENV_NAME)
agent = DQN_Agent(num_actions=env.action_space.n)

for _  in xrange(NUM_EPISODES):
    terminal = False
    observation = env.reset()
    for _ in xrange(random.randint(1, NO_OP_STEPS)):
        last_observation = observation
        observation, _, _, _ = env.step(0)
    state = agent.get_initial_state(observation, last_observation)
    while not terminal:
        last_observation = observation
        action = agent.get_action(state)
        observation, reward, terminal, _ = env.step(action)
        env.render()
        processed_observation = preprocess(observation, last_observation)
        state = agent.run(state, action, reward, terminal, processed_observation)