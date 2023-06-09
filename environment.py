import numpy as np

from gym_anytrading.envs.trading_env import TradingEnv, Actions, Positions

from utils import inverse_transform_values


class CryptoStockEnv(TradingEnv):

    def __init__(self, df, window_size, frame_bound, scaler: None, initial_balance: float = 1.0):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size)

        self.trade_fee_bid_percent = 0.001  # unit
        # self.trade_fee_bid_percent = 0.0  # unit
        self.trade_fee_ask_percent = 0.0005  # unit
        # self.trade_fee_ask_percent = 0.0  # unit

        self.initial_balance = initial_balance
        self.scaler = scaler

    def reset(self):
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Short
        self._position_history = (self.window_size * [None]) + [self._position]
        self._total_reward = 0.
        self._total_profit = self.initial_balance
        self._first_rendering = True
        self.history = {}
        return self._get_observation()

    def _process_data(self):
        prices = self.df.loc[:, 'close'].to_numpy()

        prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        prices = prices[self.frame_bound[0] - self.window_size:self.frame_bound[1]]

        diff = np.insert(np.diff(prices), 0, 0)
        signal_features = np.column_stack((prices, diff))

        return prices, signal_features

    def _calculate_reward(self, action):
        step_reward = 0

        trade = False
        if ((action == Actions.Buy.value and self._position == Positions.Short) or
            (action == Actions.Sell.value and self._position == Positions.Long)):
            trade = True

        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            price_diff = current_price - last_trade_price

            if self._position == Positions.Long:
                step_reward += price_diff
            elif self._position == Positions.Short:
                step_reward += -price_diff

        return step_reward

    def _update_profit(self, action):
        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or \
           (action == Actions.Sell.value and self._position == Positions.Long):
            trade = True

        if trade or self._done:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
            elif self._position == Positions.Short:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / current_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * last_trade_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            current_price = self.prices[current_tick - 1]
            last_trade_price = self.prices[last_trade_tick]
            if position == Positions.Long:
                shares = (profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price
            else:
                shares = (profit * (1 - self.trade_fee_ask_percent)) / current_price
                profit = (shares * (1 - self.trade_fee_bid_percent)) * last_trade_price
            last_trade_tick = current_tick - 1

        return profit

    def render_all(self, mode='human'):
        if self.scaler:
            prices = self.prices.copy()
            self.prices = inverse_transform_values(self.prices, self.scaler)
            super().render_all(mode)
            self.prices = prices
