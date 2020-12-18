import copy
import sys
from collections import OrderedDict
from typing import List
from fmclient import Agent
from fmclient import Order, OrderSide, OrderType, Holding
from fmclient import Market, Asset
from fmclient import Session, SessionState
import random
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import argparse
import traceback as tb

NAME_COL = "account_name"
MARKET_ID_COL = "market_id"
MARKET_ITEMS_COL = "market_items"
BOT_NAME_COL = "bot_name"
BOT_PASSWORD_COL = "bot_password"
HOLDINGS_COL = "holdings"
MIN_HOLDINGS_COL = "min_holdings"
MAX_HOLDINGS_COL = "max_holdings"
VAL_COL = "val"
J_COL = "J"
K_COL = "K"
MUV_COL= "muv"
MUL_COL = "mul"
T_COL = "T"

# Cut down the name of the item to 3 for order references
ITEM_PREF = 3

class IELAgent(Agent):
    """
    The IEL agent makes bids and offers depending on the foregone utility
    of each bid/offer. Foregone utility is currently calculated by looking
    only at the present book.

     Attributes:
        J	      The number of items in "considered set" S
        muv       The rate of mutation of value
        sigmav    The variance on the mutation of value. float((su-sl))/10
        mul       The rate of mutation of length
        K         The max length of a "considered strategy"
        T         A measure of how far back to start keeping track of past prices
        su        Upper bound on the strategy set [sl,su]
        sl        Lower bound on strategy set [sl,su]
        BUY       Always set to 0 such that utilities and strategies for buying
                  are stored in index 0
        SELL      Always set to 1 such that utilities and strategies for buying
                  are stored in index 1
    """
    K = 1
    J = 100
    muv = .033
    mul = .0033
    T = 5

    BUY = 0
    SELL = 1

    def __init__(self, config_file, bot_num):

        """

        The constructor takes arguments for the flexemarkets account details and the I D/S parameters.

        :param account: Flexemarkets account name

        :param email: Flexemarkets account email

        :param password: Flexemarkets account password

        :param marketplace_id: Flexemarkets marketplace id

        :param M: The maximum number of commodities

        :param r: The number of commodities that the agent starts with

        :param valuations: An array containing marginal utilities of the i'th commodity,
        where i corresponds with indices in the array.

        """
        name = "IEL" + str(bot_num)
        self.bot_name = "test" + str(bot_num) + "@test"
        self.bot_password = "pass" + str(bot_num)
        # Parameters for 2+ item utility functions
        self.val_dict = {}
        self.init_from_config(config_file, bot_num)
        # Calls the __init__  method of the base Agent class so that this
        # subclass does not repeat code to set up account,  email, password,
        # marketplace_id, and the name of the type of agent.
        super().__init__(self.account_name, self.bot_name, self.bot_password, self.marketplace_id, name=name)

        # Updated after bot has been initialised and can retrieve these
        # values from the market
        self.su = {}
        self.sl = {}
        self.curr_best_bid = {}
        self.curr_best_offer = {}

        # Updated in received_holdings. This is an OrderedDict because the order in which
        # item holdings appear matters when computing utility
        self.curr_units = OrderedDict()
        self.utilities = {}
        self.strategies = {}

        self.curr_strat = {}

        # Fetch the market ids of each item in initialised
        self.market_ids = {}

        # IMPORTANT! Always send orders for 1 unit of item.
        self._units = 1
        self._max_market_making_orders = 1

        # Orders waiting to be processed by the server
        self._orders_waiting_ackn = {}

        # Keep track of prices at which past transactions has taken place
        self.past_prices = {}

        # Prices at which past trades were made, with the most recent
        # trading prices at the beginning of the list.
        self.past_trades = {}

        self._mm_buy_prefix = "b"

        self._mm_sell_prefix = "s"

        self._mm_cancel_prefix = "c"

        # Description will be shown on the hosting platform.
        self.description = "IEL bot"  # self.get_description()

        # Simple counter to track of orders sent
        self._counter = 0
        self._timer = 0

        # Set to a float when we don't want buy/sell orders to update on a timer.
        self._rank_frequency = 20

    def init_from_config(self, fname, bot_num):
        '''
        Read in general info shared across all bots. Then read in
        bot-specific information

        :param fname: the name of the excel spreadsheet

        :param bot_num: the number of this bot
        '''
        gen_info = pd.read_excel(fname, nrows=1)
        # Get the id of the marketplace, which can contain multiple item markets
        self.marketplace_id = gen_info[MARKET_ID_COL].item()
        self.account_name = gen_info[NAME_COL].item()
        # Get the list of all market items
        items = gen_info[MARKET_ITEMS_COL].item().split(', ')

        # Read in bot specific parameters
        bot_info = pd.read_excel(fname, header=3)
        row_index = bot_num - 1
        val_sheet = bot_info[VAL_COL].iloc[row_index]

        # Now read in valuation function parameters from the second sheet (index 1)
        val_set = pd.read_excel(fname, sheet_name=str(val_sheet))
        for col, data in val_set.items():
            self.val_dict[col] = data[~np.isnan(data)].to_numpy()

        # optional parameters we could configure. If not set in the
        # config file, just use the default class values
        min_holdings = [0] * len(items)
        max_holdings = [sys.maxsize] * len(items)
        if HOLDINGS_COL in bot_info.columns:
            holdings_sheet = bot_info[HOLDINGS_COL].iloc[row_index]
            if not pd.isnull(holdings_sheet):
                holdings = pd.read_excel(fname, sheet_name=str(holdings_sheet))
                if MIN_HOLDINGS_COL in holdings.columns:
                    min_holdings = holdings[MIN_HOLDINGS_COL]
                if MAX_HOLDINGS_COL in holdings.columns:
                    max_holdings = holdings[MAX_HOLDINGS_COL]

        # The number of entries should match the number of items
        self.min_holdings = dict(zip(items, min_holdings))
        self.max_holdings = dict(zip(items, max_holdings))

        if J_COL in bot_info.columns:
            val = bot_info[J_COL].iloc[row_index]
            if not np.isnan(val):
                self.J = int(val)
        if K_COL in bot_info.columns:
            val = bot_info[K_COL].iloc[row_index]
            if not np.isnan(val):
                self.K = int(val)
        if T_COL in bot_info.columns:
            val = bot_info[T_COL].iloc[row_index]
            if not np.isnan(val):
                self.T = int(val)
        if MUV_COL in bot_info.columns:
            val = bot_info[MUV_COL].iloc[row_index]
            if not np.isnan(val):
                self.muv = val
        if MUL_COL in bot_info.columns:
            val = bot_info[MUL_COL].iloc[row_index]
            if not np.isnan(val):
                self.mul = val

    def initialised(self):
        """
        Called after initialization of the robot is complete.
        :return:
        """
        try:
            for market_id, market in self.markets.items():
                item = market.item
                self.market_ids[item] = market_id
                self.su[item] = self.markets[market_id].max_price
                self.sl[item] = self.markets[market_id].min_price
                self.utilities[item] = [[1] * self.J for _ in range(2)]
        except Exception:
            tb.print_exc()

    def add_one(self, item):
        """
        Add one to the unit count of the passed in item and return
        the unit count for all items

        :param item: the item for which we add one to its unit count
        """
        copy_dict = copy.deepcopy(self.curr_units)
        copy_dict[item] += 1
        return copy_dict

    def form1(self, units):
        '''
        Computes a valuation using the functional form v = mr' - (a/2) r Sr’.  
        If there are K goods, r is the vector of holdings,  r= (r_1, …,  r_K).
        m is a vector of parameters m=(m_1,…,m_K), a is a parameter, a real number.  
        And S is a KxK matrix.

        :param units: the unit counts of all items
        '''
        S = np.asarray(self.val_dict['S'])
        # Since S is passed in as a list, we need to reshape it to be a n by n array
        m = self.val_dict['m']
        S = S.reshape((len(m), len(m)))
        m = np.asarray(m)
        a = self.val_dict['a']
        r = np.asarray(list(units.values()))
        r = np.reshape(r, (1, r.size))
        ans = np.dot(m, np.transpose(r)) - (float(a) / 2.0) * np.matmul(r, np.matmul(S, np.transpose(r)))
        return max(0, ans)

    def get_val(self, units):
        '''
        Returns the valuation of the passed in holdings (the counts of units
        we have for each item)

        :param units: the unit counts of all items
        '''
        form = int(self.val_dict['form'][0])
        if form == 1:
            return self.form1(units)

    def choiceprobabilities(self, item, action):
        """
        Calculates the probability of choosing a strategy for all strategies of
        the same action. Probability is proportional to the foregone utility of
        one strategy divided by the sum of foregone utilities over all
        strategies.

        :param item: the item that we're trading

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        choicep = []
        sumw = sum(self.utilities[item][action])
        if sumw == 0:
            return np.zeros(self.J)
        for j in range(self.J):
            choicep.append(self.utilities[item][action][j] / float(sumw))
        return choicep

    def strat_selection(self, item, action):
        """
        Chooses a strategy out of all strategies of the same action.

        :param item: the item that we're trading

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """

        if action == self.BUY:
            self.strategies[item][action] = list(filter(lambda x: x < self.get_val(self.add_one(item)),
                                                    self.strategies[item][action]))
        else:
            self.strategies[item][action] = list(filter(lambda x: x > self.get_val(self.curr_units),
                                                  self.strategies[item][action]))
        if action == self.BUY:
            if len(self.strategies[item][action]) < self.J:
                if len(self.strategies[item][action]) == 0:
                    self.curr_strat[item][action] = self.sl[item]
                x = self.J - len(self.strategies[item][action])
                for i in range(x):
                    self.strategies[item][action].append(int(random.uniform(self.sl[item],
                                                                            self.get_val(self.add_one(item)))))
        else:
            if len(self.strategies[item][action]) < self.J:
                if len(self.strategies[item][action]) == 0:
                    self.curr_strat[action] = self.su[item]
                x = self.J - len(self.strategies[item][action])
                for i in range(x):
                    self.strategies[item][action].append(int(random.uniform(self.get_val(self.curr_units),
                                                                            self.su[item])))

            self.updateW(action)
        choicep = self.choiceprobabilities(item, action)

        if sum(choicep) == 0:
            choicep = [1 / len(self.strategies[item][action]) for _ in self.strategies[item][action]]
        self.curr_strat[item][action] = int(self.rand_choice(self.strategies[item][action], choicep))

    def rand_choice(self, strats, distr):
        '''
        Randomly chooses a strategy from the passed in strats array, and using
        a probability distribution for picking a given strategy

        :param strats: length J array of strategies we're considering

        :param distr: the probability distribution associated with the
        strategies
        '''
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        saved_strat = 0
        for strat, strat_probability in zip(strats, distr):
            cumulative_probability += strat_probability
            if x < cumulative_probability:
                saved_strat = strat
                break
        return saved_strat

    def Vexperimentation(self, item, action):
        """
        Value experimentation for strategies of the same action. With a
        probability determined by muv, takes a strategy as a center of a
        distribution and generates a new strategy around the center.

        :param item: the item that we're trading

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for j in range(self.J):
            if action == self.BUY:
                sigmav = max(1, 0.1 * (self.get_val(self.add_one(item)) - self.sl[item]))
                if random.uniform(0, 1) < self.muv:
                    centers = self.strategies[item][action][j]
                    r = (truncnorm.rvs((self.sl[item] - centers) / float(sigmav),
                                       (self.su[item] - centers) / float(sigmav),
                                       loc=centers, scale=sigmav, size=1))
                    self.strategies[item][action][j] = int(np.array(r).tolist()[0])
            else:
                sigmav = max(1, 0.1 * (self.su[item] - self.get_val(self.curr_units)))
                if random.uniform(0, 1) < self.muv:
                    centers = self.strategies[item][action][j]
                    r = (truncnorm.rvs((self.sl[item] - centers) / float(sigmav),
                                       (self.su[item] - centers) / float(sigmav),
                                       loc=centers, scale=sigmav, size=1))
                    self.strategies[item][action][j] = int(np.array(r).tolist()[0])

    def replicate(self, item, action):
        """
        Replicates strategies of the same action by comparing two randomly
        chosen strategies and replacing that with the lower utility with
        the other strategy.

        :param item: the item that we're trading

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for j in range(self.J):
            j1 = random.randrange(self.J)
            j2 = random.randrange(self.J)
            self.strategies[item][action][j] = self.strategies[item][action][j2]
            self.utilities[item][action][j] = self.utilities[item][action][j2]
            if self.utilities[item][action][j1] > self.utilities[item][action][j2]:
                self.strategies[item][action][j] = self.strategies[item][action][j1]
                self.utilities[item][action][j] = self.utilities[item][action][j1]

    def foregone_utility(self, item, j, action):
        """
        Calculates the foregone utility of a strategy with index j and
        corresponding to a certain action. Currently, we only consider the
        current book and look at the current highest bid and current lowest
        offer.

        :param item: the item that we're trading

        :param j: The index of the strategy for which we want to update
        the foregone utility.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """

        if action == 0:
            bid = self.strategies[item][action][j]
            # Return 0 for case where bid exceeds the valuation of item to be bought
            if bid <= self.curr_best_offer[item] or self.curr_units[item] >= self.max_holdings[item] or \
                    bid > self.get_val(self.add_one(item)):
                return 0
            else:
                return self.get_val(self.add_one(item)) - self.curr_best_offer[item]
        else:
            offer = self.strategies[item][action][j]
            # Return 0 for case where the offer is lower than the valuation of item to be sold
            if offer >= self.curr_best_bid[item] or self.curr_units[item] <= self.min_holdings[item] or \
                    offer < self.get_val(self.curr_units):
                return 0
            else:
                return self.curr_best_bid[item] - self.get_val(self.curr_units)

    def foregone_utility_past(self, item, j, action):
        """
        Calculates the foregone utility of a strategy with index j and
        corresponding to a certain action. We consider the
        current book--looking at the current highest bid/current lowest
        offer--as well as past transactions.

        :param item: the item that we're trading

        :param j: The index of the strategy for which we want to update
        the foregone utility.

        :param action: 0 corresponds to buying, 1 corresponds to selling

        """
        # print("Now we are foregone_utility_past")
        if action == 0:
            bid = self.strategies[item][action][j]
            # get minimum of past prices
            p_min = min(self.past_trades[item])
            z = min(p_min, self.curr_best_offer[item])
            if bid <= z or self.curr_units[item] >= self.max_holdings[item] or \
                    bid > self.get_val(self.add_one(item)):
                return 0
            elif bid > self.curr_best_offer[item]:
                return self.get_val(self.add_one(item)) - self.curr_best_offer[item]
            elif z < bid <= self.curr_best_offer[item]:
                return self.get_val(self.add_one(item)) - bid
        else:
            offer = self.strategies[item][action][j]
            # get maximum of past prices
            p_max = max(self.past_trades[item])
            z = max(p_max, self.curr_best_bid[item])
            if offer >= z or self.curr_units[item] <= self.min_holdings[item] or \
                    offer < self.get_val(self.curr_units):
                return 0
            elif offer < self.curr_best_bid[item]:
                return self.curr_best_offer[item] - self.get_val(self.curr_units)
            elif z > offer >= self.curr_best_bid[item]:
                return offer - self.get_val(self.curr_units)

    def updateW(self, action):
        """
        Updates the foregone utilities for strategies of the same action.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for item in self.curr_units.keys():
            for j in range(self.J):
                if len(self.past_trades[item]) > 0:
                    self.utilities[item][action][j] = self.foregone_utility_past(item, j, action)
                else:
                    self.utilities[item][action][j] = self.foregone_utility(item, j, action)

    def received_orders(self, orders: List[Order]):
        """
        Decide what to do when an order book update arrives.

        We categorize orders based on {ours, others} and {buy, sell}, then depending on the type of robot we either

        react or market make.

        :param orders: List of Orders received from FM via WS
        """
        try:
            if len(self.strategies) == 0 or not self.is_session_active():
                return
            is_mine = {}
            for item in self.curr_units.keys():
                is_mine[item] = True
            for order in orders:
                order_item = order.market.item
                # Updates the list of past trades for computing foregone past
                # utility
                if order.has_traded and order.mine:
                    self.past_trades[order_item].insert(0, order.price)
                    if len(self.past_trades) > self.T:
                        self.past_trades[order_item] = self.past_trades[order_item][0:self.T]

                if order.mine and order.order_type == OrderType.LIMIT and \
                        order.has_traded:
                    order_item = order.market.item
                    if order.order_side == OrderSide.BUY:
                        self.inform('Added a unit of ' + order_item)
                        self.curr_units[order_item] += 1
                    else:
                        self.inform('Subtracted a unit of ' + order_item)
                        self.curr_units[order_item] -= 1
                    self.updateW(self.BUY)
                    self.updateW(self.SELL)
                if not order.mine:
                    is_mine[order_item] = False
            for item, val in is_mine.items():
                if not val:
                    order_book = self._categorize_orders(item)
                    self._react_to_book(item, order_book)
        except Exception as e:
            tb.print_exc()

    def received_holdings(self, holdings: Holding):
        """
        Called when holdings information is received from the exchange. If the
        strategy set empty, this means a new session has just started and the robot
        has new holdings. In this case, it must initialize a new strategy set to random
        values within the proper ranges for bids and offers based on the marginal utility
        of the next item. Updates the foregone utilities when holdings have changed.
        :param holdings:
        :return
        """
        pass

    def initialize_strat_set(self):
        """
        Only initializes the strategy set if they are currently uninitialized
        """
        for item in self.curr_units.keys():
            if item not in self.strategies:
                self.strategies[item] = [[], []]
                for j in range(self.J):
                    if self.su[item] < self.get_val(self.curr_units) or \
                            self.curr_units[item] >= self.max_holdings[item]:
                        self.strategies[item][self.BUY].append(self.sl[item])
                    else:
                        self.strategies[item][self.BUY].append(
                                random.randint(self.sl[item], int(self.get_val(self.add_one(item)))))
                    if self.sl[item] > self.get_val(self.add_one(item)) or \
                            self.curr_units[item] <= self.min_holdings[item]:
                        self.strategies[item][self.SELL].append(self.su[item])
                    else:
                        self.strategies[item][self.SELL].append(
                            random.randint(int(self.get_val(self.curr_units)), self.su[item]))
        self.updateW(self.BUY)
        self.updateW(self.SELL)

    def received_session_info(self, session: Session):
        """
        Called when marketplace information is received from the exchange.
        :param session:
        :return:
        """
        try:
            if session.is_open:
                for item, market_id in self.market_ids.items():
                    self.curr_best_bid[item] = self.sl[item]
                    self.curr_best_offer[item] = self.su[item]
                    market = self.markets[market_id]
                    self.curr_units[item] = self.holdings.assets[market].units
                    self.curr_strat[item] = [self.sl[item], self.su[item]]
                    self._orders_waiting_ackn[item] = {}
                    self.past_trades[item] = []
                self.initialize_strat_set()
                self.updateW(self.BUY)
                self.updateW(self.SELL)
            elif session.is_closed:
                # The purpose of this is to reset the strategy set after a period ends
                self.past_trades = {}
                self.strategies = {}
                self.curr_units = {}
                self._orders_waiting_ackn = {}
        except Exception as e:
            tb.print_exc()

    def pre_start_tasks(self):
        """
        The sub-classed trading agent should override this method to perform any task or schedule tasks
        before Agent starts interacting in the marketplace
        :return:
        """
        super().execute_periodically(self.check_book, 2)

    def check_book(self):
        try:
            if self.current_session is not None and self.is_session_active():
                for item in self.curr_units.keys():
                    if len(self.strategies[item]) != 0 and self.is_session_active():
                        order_book = self._categorize_orders(item)
                        if len(order_book["mine"]["buy"]) > 0:
                            self.cancel_ord(self.BUY, order_book)
                        if len(order_book["mine"]["sell"]) > 0:
                            self.cancel_ord(self.SELL, order_book)
                        self.send_ord(item, self.BUY, order_book)
                        self.send_ord(item, self.SELL, order_book)
        except Exception as e:
            tb.print_exc()

    def received_trades(self, orders: List[Order], market: Market = None):
        """
        Called when completed orders are sent by the exchange.
        :param orders: List of traded orders
        :param market: Market for which trades were received
        :return:
        """
        pass

    def respond_to_user(self, message: str):
        """
        Called when marketplace information is received from the exchange.
        :param message: Incoming message from user
        :return:
        """
        pass

    def _get_order(self, item, order_side):
        """
        Takes a request for a either a buy or sell order, the type of which is
        specified by order_side. Returns an order.

        :param order_side: either OrderSide.BUY or OrderSide.SELL
        """
        action = self.BUY
        if order_side == OrderSide.SELL:
            action = self.SELL
        self.Vexperimentation(item, action)
        self.updateW(action)
        self.replicate(item, action)
        self.strat_selection(item, action)
        if order_side == OrderSide.SELL and (
                self.curr_strat[item][self.SELL] < self.sl[item] or self.curr_strat[item][self.SELL] > self.su[item]):
            raise Exception('Price should not be less than the minimum or greater than the maximum when selling.')
        if order_side == OrderSide.BUY and (
                self.curr_strat[item][self.BUY] < self.sl[item] or self.curr_strat[item][self.BUY] > self.su[item]):
            raise Exception('Price should not be less than the minimum or greater than the maximum when selling.')
        price = max(self.sl[item], self.curr_strat[item][action])
        order = Order.create_new(self.markets[self.market_ids[item]])
        order.price = price
        order.units = self._units
        order.order_type = OrderType.LIMIT
        order.order_side = order_side
        return order

    def get_description(self):

        """

        The description to be displayed on the hosting platform.

        :return: Concatenation of I D/S parameters.
        """
        return self.description

    def _categorize_orders(self, item):

        """

        Categorizes orders in 4 groups: {Mine, Others} and {Buy, Sell}.

        :param orders: List of orders

        :return: a dictionary with top level key for the owner, and second level key for the order side.

        The values are sorted list of orders.

        """

        # Uses a dictionary mapping from  a string to a list within a dicitonary
        # mapping  a string to a string to retrieve orders in one of the four
        # categories.
        orders_dict = {"mine": {"buy": [], "sell": []}, "others": {"buy": [], "sell": []}}
        for order in Order.all().values():
            # Make sure to exclude cancel orders
            order_item = order.market.item
            if order.order_type == OrderType.LIMIT and order.is_pending and \
                order_item == item:
                if order.mine and order.order_side == OrderSide.BUY:
                    orders_dict["mine"]["buy"].append(order)
                elif order.mine and order.order_side == OrderSide.SELL:
                    orders_dict["mine"]["sell"].append(order)
                elif not order.mine and order.order_side == OrderSide.SELL:
                    orders_dict["others"]["sell"].append(order)
                elif not order.mine and order.order_side == OrderSide.BUY:
                    orders_dict["others"]["buy"].append(order)

        # IMPORTANT! Sort the orders to make it easier to reason on what to do.
        for owner_key in orders_dict.keys():
            for type_key in orders_dict[owner_key]:
                # multiplier ensures the sorting is correct for the buy ands sell side
                if type_key == "buy":
                    multiplier = -1
                else:
                    multiplier = 1
                orders_dict[owner_key][type_key].sort(key=lambda o: multiplier * o.price)
        return orders_dict

    def cancel_ord(self, action, order_book):
        order_str = "buy"
        if action == self.SELL:
            order_str = "sell"
        if len(order_book["mine"][order_str]) != 0:
            order = order_book["mine"][order_str].pop(0)
            cancel_order = copy.copy(order)
            cancel_order.order_type = OrderType.CANCEL
            cancel_order.ref = self._increment_counter(order.market.item, self._mm_cancel_prefix)
            self.send_order(cancel_order)

    def send_ord(self, item, action, order_book):
        my_orders = order_book["mine"]
        if action == self.BUY:
            if self.sl[item] <= self.get_val(self.add_one(item)):
                waiting = self._waiting_for(item, self._mm_buy_prefix)
                if not waiting and len(my_orders["buy"]) == 0 and self.curr_units[item] < self.max_holdings[item]:
                    order = self._get_order(item, OrderSide.BUY)
                    cash_i_have = self.holdings.cash_available
                    if len(my_orders["buy"]) < self._max_market_making_orders \
                            and cash_i_have >= order.price:
                        order.ref = self._increment_counter(item, self._mm_buy_prefix)
                        self.send_order(order)
        else:
            if self.su[item] >= self.get_val(self.curr_units):
                waiting = self._waiting_for(item, self._mm_sell_prefix)
                if not waiting and len(my_orders["sell"]) == 0:
                    order = self._get_order(item, OrderSide.SELL)
                    market = self.markets[self.market_ids[item]]
                    units_i_have = self.holdings.assets[market].units_available
                    if len(my_orders["sell"]) < self._max_market_making_orders \
                            and units_i_have >= order.units and self.curr_units[item] > self.min_holdings[item]:
                        order.ref = self._increment_counter(item, self._mm_sell_prefix)
                        self.send_order(order)

    def _react_to_book(self, item, order_book):

        """
        The agent checks if it can react to any order.

        :param orders_dict: Dictionary of orders

        :return: Does not return anything, but may send an order.

        """
        # others_orders is a dictionary with keys "buy" and "sell", which
        # map to lists of orders
        others_orders = order_book["others"]

        # If the best bid/offer has changed, update the utilities
        curr_bids = others_orders["buy"]
        curr_offers = others_orders["sell"]
        submit_bid = False
        submit_offer = False
        if len(curr_bids) != 0:
            h_bid = curr_bids[0].price
            if h_bid != self.curr_best_bid:
                submit_offer = True
                # Updates the current best bid
                self.curr_best_bid[item] = h_bid
                self.updateW(self.SELL)

        if len(curr_offers) != 0:
            l_offer = curr_offers[0].price
            if l_offer != self.curr_best_offer:
                submit_bid = True
                # Updates the current best offer
                self.curr_best_offer[item] = l_offer
                self.updateW(self.BUY)

        if submit_bid:
            self.cancel_ord(self.BUY, order_book)
        if submit_offer:
            self.cancel_ord(self.SELL, order_book)

    def _increment_counter(self, item, prefix):
        """
        Increments the counter and returns a reference with a prefix
        corresponding to the type of an order.

        :return: Reference of an order
        """
        ref = str(item[0:ITEM_PREF]) + '_' + str(prefix) + str(self._counter)
        self._counter += 1
        return ref

    def _waiting_for(self, item, prefix):
        """
        Check if there exists an order that we are waiting for the server to process

        such that the order's ref starts with a given prefix.

        :param prefix: prefix to check in order's reference

        :return: True if such an order exists, False otherwise

        """

        for key in self._orders_waiting_ackn[item].keys():
            if key.startswith(str(item[0:ITEM_PREF]) + '_' + prefix):
                return True
        return False

    def send_order(self, order):

        """

        We override the parent's send_order method to allow tracking of orders sent but not yet acknowledged.

        :param order: order object

        :return:

        """

        # Takes the additional action of adding an order that is about to be sent
        # to a dictionary that keeps track of objects using their reference.

        if order.ref is None:
            order.ref = self._increment_counter(order.market.item, "n")
        self._orders_waiting_ackn[order.market.item][order.ref] = order
        super().send_order(order)

    def received_marketplace_info(self, marketplace_info):

        _extra_info = "Waiting for marketplace to open." if not marketplace_info["status"] else ""

    def order_accepted(self, order):
        ''' Now takes in the order book as a parameter. '''
        if order.mine:
            order_item = order.market.item
            if order.ref in self._orders_waiting_ackn[order_item]:
                del self._orders_waiting_ackn[order_item][order.ref]

    def order_rejected(self, info, order):
        if order.mine:
            order_item = order.market.item
            if order.ref in self._orders_waiting_ackn:
                del self._orders_waiting_ackn[order_item][order.ref]


    def run(self):
        super().run()


if __name__ == "__main__":
    # initialise and run the bot.
    parser = argparse.ArgumentParser(description='Run a simulation.')
    parser.add_argument('config_file', type=str,
                        help='The path of the configuration file')
    parser.add_argument('bot_num', type=int,
                        help='The bot\'s number')
    args = parser.parse_args()

    fm_bot = IELAgent(args.config_file, args.bot_num)
    fm_bot.run()
