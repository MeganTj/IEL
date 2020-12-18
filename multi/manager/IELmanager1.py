"""
This is a IEL manager bot for method 1. Sets initial holdings of each bot randomly at
the start of every period by computing a random base endowment for a specific period
and adding the base to a randomly generated endowment for each bot.

Initializes cash holdings to the same value for every bot at the start of round 1.
For subsequent rounds, cash holdings are carried over from the end of the previous
round.
"""
import logging
import random
import datetime
import pandas as pd
import time
from typing import List
from fmadmin.generic_manager import GenericManagerAgent
from fmclient import Holding, Order, Market
import os
import shutil
import argparse

# Periods are called rounds in the agent and they start from 1 onwards (instead of 0)
NUM_PERIODS = 5  # 10

# Timing of periods in SECONDS
ROUNDS_DURATION = [60] * NUM_PERIODS
# ROUNDS_DURATION = [180, 180, 180, 180, 180, 180, 180, 180, 180, 180]

# ROUNDS_DELAY = [15 for i in range(NUM_PERIODS)]
ROUNDS_DELAY = [15 for i in range(NUM_PERIODS)]

# Dividends in CENTS

NAME_COL = "account_name"
MARKET_ID_COL = "market_id"
MARKET_ITEMS_COL = "market_items"
EMAIL_COL = "account_email"
PASSWORD_COL = "account_password"

# INITIAL HOLDINGS. Set this variable to True will cause the bot to set the initial holdings before the start of an
# experiment, every time the bot is run.
DO_SET_INITIAL_HOLDINGS = True
# Set the path to save the results of the simulations run
SAVE_TO = "/Users/MeganT/Documents/IEL/"


class IELManager(GenericManagerAgent):

    def __init__(self, config_file, total_rounds=10, name="IELManager1",
                  upper_endow = 0, base_cash = 0):
        self.init_from_config(config_file)
        # super().__init__(account, email, password, marketplace_id,
        #                  total_rounds=total_rounds, name=name)
        super().__init__(self.account_name, self.account_email, self.account_password,
                         self.marketplace_id, total_rounds=total_rounds, name=name)
        self.upper_endow = upper_endow
        self.base_cash = base_cash
        self.dir_path = SAVE_TO + "sim-" + f"{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')}"
        # This deletes the simulation directory if it already exists
        if os.path.isdir(self.dir_path):
            shutil.rmtree(self.dir_path)
        os.mkdir(self.dir_path)

    def has_experiment_ended(self, round_num) -> bool:
        """
        Specify the conditions for experiment end.
        :param round_num:
        :return:
        """
        if round_num > NUM_PERIODS:
            return True
        else:
            return False

    def get_break_duration(self, round_num) -> int:
        return ROUNDS_DELAY[round_num - 1]

    def get_round_duration(self, round_num) -> int:
        return ROUNDS_DURATION[round_num - 1]

    def holdings_updated_callback(self):
        self.inform(f"Holdings updated for round {self.current_round}")

        # Mark intervention as completed, so that the first period can start.
        self.action_completed(self.perform_on_after_close)

    def set_holdings_after_round(self, user_holdings):
        """
        This method is called, after each period finishes, to update the users' holdings.

        :param user_holdings: Holdings at the end of the trading period
        :return: updated holdings
        """

        # Calculate the base endowment for the next round
        round_endow = random.randint(0, self.upper_endow)
        self.inform("Base endowment is: {}".format(round_endow))
        for email, holdings in user_holdings.items():
            for market, h in holdings.assets.items():
                # For each agent, add a randomly generated value in range [0, upper_endow]
                holdings.assets[market].units_initial = round_endow + random.randint(0, self.upper_endow)
                # Carry over held cash from previous round
                holdings.cash_initial = holdings.cash

        try:
            self.inform("Sending new holdings to the server.")
            self.update_holdings(user_holdings, self.holdings_updated_callback)
        except Exception as e:
            self.error(e)

    def initial_holdings_updated_callback(self):
        self.inform(f"Holdings updated for round {self.current_round}")
        # Mark intervention as completed, so that the first period can start.
        self.action_completed(self.perform_on_before_open)

    def set_initial_holdings(self, user_holdings):
        """
        Initialize each bot with the same base endowment + additional endowment in U[0, 5].
        All bots have the same cash holding.
        :param user_holdings:
        :return:
        """
        self.inform("I have received old holdings. I will now calculate the initial holdings.")
        markets = []
        for item in self.market_items:
            markets.append(Market.get_by_item(item))
        # Calculate the base endowment for the next round
        round_endow = random.randint(0, self.upper_endow)
        self.inform("Base endowment for round 1 is: {}".format(round_endow))
        for email, holdings in user_holdings.items():
            for market in markets:
                holdings.assets[market].units_initial = round_endow + random.randint(0, self.upper_endow)
                holdings.cash_initial = self.base_cash

        try:
            self.update_holdings(user_holdings, self.initial_holdings_updated_callback)
        except Exception as e:
            self.error(e)

    def perform_on_after_close(self):
        file = self.dir_path + '/per' + str(self.current_round)
        if self.current_round > 0:
            self.get_holdings(self.set_holdings_after_round, save_to_file_path=file)
        else:
            self.get_holdings(callback=None, save_to_file_path=file)
            self.action_completed(self.perform_on_after_close)

    def perform_on_before_open(self):
        if self.current_round == 1 and DO_SET_INITIAL_HOLDINGS:
            self.get_holdings(self.set_initial_holdings)
        else:
            self.action_completed(self.perform_on_before_open)

    def init_from_config(self, fname):
        df = pd.read_excel(fname, nrows=1, keep_default_na=False)
        self.marketplace_id = df[MARKET_ID_COL].item()
        self.account_name = df[NAME_COL].item()
        self.account_email = df[EMAIL_COL].item()
        self.account_password = df[PASSWORD_COL].item()
        self.market_items = df[MARKET_ITEMS_COL].item().split(', ')

    """
    Implement the methods below if the manager is required to trade along with participants, 
    or if she is required to monitor the orders..
    """
    def order_accepted(self, order: Order):
        pass

    def order_rejected(self, info: dict, order: Order):
        pass

    def received_orders(self, orders: List[Order]):
        pass

    def received_holdings(self, holdings: Holding):
        pass


if __name__ == "__main__":
    # Define the upper bound on randomly generated endowments and the starting cash
    # for all bots
    endow = 5
    cash = 1000
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('config_file', type=str,
                        help='The path of the configuration file')
    args = parser.parse_args()
    manager = IELManager(args.config_file, total_rounds=NUM_PERIODS, upper_endow = endow,
                         base_cash = cash)
    manager.run()
