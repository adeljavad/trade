import numpy as np
import pandas as pd
import random
from collections import deque
import gym
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

# تعریف محیط
class StockTradingEnv(gym.Env):
    def __init__(self, data):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.action_space = gym.spaces.Discrete(3)  # 0: فروش، 1: نگه‌داری، 2: خرید
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(len(data.columns) - 1,), dtype=np.float32)  # یکی کمتر
        self.current_step = 0
        self.balance = 1000  # موجودی اولیه
        self.stock_owned = 0
        self.total_profit = 0

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.stock_owned = 0
        self.total_profit = 0
        return self.data.iloc[self.current_step, 1:].values.astype(np.float32)  # حذف ستون تاریخ

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']

        if action == 0:  # فروش
            if self.stock_owned > 0:
                self.balance += current_price
                self.stock_owned = 0
        elif action == 2:  # خرید
            if self.balance >= current_price:
                self.balance -= current_price
                self.stock_owned += 1

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        if done:
            self.total_profit = self.balance + self.stock_owned * current_price - 1000  # محاسبه سود

        numeric_state = self.data.iloc[self.current_step, 1:].values.astype(np.float32)  # حذف ستون تاریخ
        return numeric_state, self.total_profit, done, {}

# تعریف مدل DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # تخفیف
        self.epsilon = 1.0  # اکتشاف
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))  # استفاده از Input
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# بارگذاری داده‌ها
data = pd.read_csv('data/hourly_btc_data.csv')  # داده‌های سهام خود را بارگذاری کنید
env = StockTradingEnv(data)
agent = DQNAgent(state_size=data.shape[1] - 1, action_size=env.action_space.n)  # یکی کمتر


# آموزش مدل
episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, env.observation_space.shape[0]])
    for time in range(len(data)-1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"Episode: {e}/{episodes}, Profit: {env.total_profit}")
            break
        if len(agent.memory) > 32:
            agent.replay(32)
