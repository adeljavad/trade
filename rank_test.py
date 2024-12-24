import pandas as pd
import numpy as np

def backtesting_new_rank(data, initial_balance=100000, broker_fee_percent=0.1):
    """Backtest a trading strategy based on new_rank."""
    balance = initial_balance
    stock_owned = 0.0  # Allow fractional ownership
    
    # Generate signals based on new_rank
    signals = np.where(data['new_rank'] >1, 1, np.where(data['new_rank'] <-1, -1, 0))  # 1 for buy, -1 for sell
    print(signals)
    for index in range(len(data)):
        signal = signals[index]
        price = data['close'].iloc[index]
        print(f"signal : ,{signal:d}, Price : ,{price:.2f}")
        if signal == 1:  # Buy signal
            if balance >= 0:  # Ensure there is enough balance to buy
                # Calculate how much can be bought
                amount_to_invest = balance * (1 - broker_fee_percent / 100)  # Deduct broker fee
                shares_to_buy = amount_to_invest / price  # Fractional shares
                stock_owned += shares_to_buy
                balance -= shares_to_buy * price * (1 + broker_fee_percent / 100)  # Deduct balance with fee
                print(f"amount_to_invest : ,{amount_to_invest:.2f}, stock_owned : ,{stock_owned:.2f}, balance : {balance:.2f} ")
        elif signal == -1:  # Sell signal
            if stock_owned > 0:  # Check if there are stocks to sell
                balance += stock_owned * price * (1 - broker_fee_percent / 100)  # Add to balance after selling
                stock_owned = 0  # Reset stock owned after selling
                print(f"amount_to_invest : ,{amount_to_invest:.2f}, stock_owned : ,{stock_owned:.2f}, balance : {balance:.2f} ")        
        final_balance =stock_owned * data['close'].iloc[-1]  # Sell remaining stocks at last price
        profit_percentage = ((final_balance - initial_balance) / initial_balance) * 100
        results.append({'final_balance': final_balance, 'profit_percentagef': profit_percentage,'close : ':price,'stock_owned :':stock_owned,'new_rank':data['new_rank'].iloc[index]})        
    
    return final_balance, profit_percentage

# مثال استفاده
file_path = 'output/updated_stock_data3.csv'
data = pd.read_csv(file_path)

# چک کردن وجود ستون‌های مورد نیاز
if 'new_rank' not in data.columns or 'close' not in data.columns:
    raise ValueError("Data must contain 'new_rank' and 'close' columns.")

results = []

final_balance, profit_percentage = backtesting_new_rank(data)
results.append({'final_balance': final_balance, 'profit_percentage': profit_percentage})
print(f"Final Balance: {final_balance:.2f}, Profit Percentage: {profit_percentage:.2f}%")

# تبدیل نتایج به DataFrame و ذخیره در فایل CSV
df = pd.DataFrame(results)
df.to_csv('output/rank_results.csv', index=False)
