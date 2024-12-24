import numpy as np
import pandas as pd
import gym

class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size=5):
        super(StockTradingEnv, self).__init__()
        self.data = data
        self.window_size = window_size
        self.action_space = gym.spaces.Discrete(3)  # 0: فروش، 1: نگه‌داری، 2: خرید
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(window_size + 1,), dtype=np.float32)
        self.current_step = 0
        self.balance = 10000  # موجودی اولیه
        self.stock_owned = 0

    def reset(self):
        self.current_step = self.window_size
        self.balance = 10000
        self.stock_owned = 0
        return self._get_observation()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0

        # اجرای اقدام
        if action == 0:  # فروش
            if self.stock_owned > 0:
                self.balance += current_price
                reward = current_price  # پاداش برابر با قیمت فروش
                self.stock_owned = 0
        elif action == 2:  # خرید
            if self.balance >= current_price:
                self.balance -= current_price
                self.stock_owned += 1
                reward = -current_price  # پاداش منفی برای خرید

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # محاسبه پاداش
        total_value = self.balance + self.stock_owned * current_price
        reward += total_value - 1000  # پاداش بر اساس تغییر در ارزش کل

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        # ایجاد مشاهده شامل قیمت‌های گذشته و قیمت فعلی
        prices = self.data['close'].values[self.current_step - self.window_size:self.current_step]
        return np.append(prices, self.balance)

class MovingAverageAgent:
    def __init__(self, window_size):
        self.window_size = window_size

    def choose_action(self, state):
        # محاسبه میانگین متحرک
        moving_average = np.mean(state[:-1])  # قیمت‌ها بدون موجودی
        current_price = state[-1]  # موجودی

        if current_price > moving_average:
            return 2  # خرید
        elif current_price < moving_average:
            return 0  # فروش
        return 1  # نگه‌داری

# استفاده از محیط و عامل
data = pd.DataFrame({
    'close': [100, 102, 101, 105, 103, 107,115,120,115,110,109,108,109,109,110,115,120,121]  # قیمت‌های فرضی
})

env = StockTradingEnv(data)
agent = MovingAverageAgent(window_size=3)

# حلقه یادگیری
for episode in range(1000):  # تعداد اپیزودها
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.choose_action(state)  # انتخاب اقدام بر اساس میانگین متحرک
        next_state, reward, done, _ = env.step(action)  # اجرای اقدام
        state = next_state  # به روزرسانی حالت
        total_reward += reward

    print(f"Episode: {episode}, Total Reward: {total_reward}")

