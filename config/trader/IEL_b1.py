import copy
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
BOT_NAME_COL = "bot_name"
BOT_PASSWORD_COL = "bot_password"
VAL_SET_COL = "val_set"
MAX_HOLDINGS_COL = "max_holdings"
MIN_HOLDINGS_COL = "min_holdings"
J_COL = "J"
K_COL = "K"
MUV_COL= "muv"
MUL_COL = "mul"
T_COL = "T"

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
        self.init_from_config(config_file, bot_num)
        # Calls the __init__  method of the base Agent class so that this
        # subclass does not repeat code to set up account,  email, password,
        # marketplace_id, and the name of the type of agent.
        super().__init__(self.account_name, self.bot_name, self.bot_password, self.marketplace_id, name=name)

        # Updated after bot has been initialised and can retrieve these
        # values from the market
        self.su = 0
        self.sl = 0
        self.curr_best_bid = 0
        self.curr_best_offer = 0

        # Updated in received_holdings
        self.utilities = [[1] * self.J for i in range(2)]
        self.strategies = []

        self.curr_strat = [0, 0]
        self.onstrat = [0, 0]

        # Keep track of prices at which past transactions has taken place
        self.past_prices = []

        # IMPORTANT! We assume there is only 1 market in the marketplace.
        # The market id is fetched automatically on initialisation.
        self._market_id = -1

        # IMPORTANT! Always send orders for 1 unit of item.
        self._units = 1
        self._max_market_making_orders = 1

        # Orders waiting to be processed by the server
        self._orders_waiting_ackn = {}

        # Orders that are still waiting to be traded
        self.orders_not_traded = []
        # Prices at which past trades were made, with the most recent
        # trading prices at the beginning of the list.
        self.past_trades = []

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
        '''
        gen_info = pd.read_excel(fname, nrows=1)
        self.marketplace_id = gen_info[MARKET_ID_COL].item()
        self.account_name = gen_info[NAME_COL].item()

        bot_info = pd.read_excel(fname, header=3)
        row_index = bot_num - 1
        val_set = bot_info[VAL_SET_COL].iloc[row_index]

        # Now read in valuations from the second sheet (index 1)
        val_sets = pd.read_excel(fname, sheet_name=1)
        self.valuations = list(val_sets[val_set].values)
        # optional parameters we could configure. If not set in the
        # config file, just use the default class values
        self.min_holdings = 0
        self.max_holdings = len(self.valuations) - 1
        # additional 0 is added to the valuation array fix the index error
        self.valuations.append(0)
        if MIN_HOLDINGS_COL in bot_info.columns:
            val = bot_info[MIN_HOLDINGS_COL].iloc[row_index]
            if not np.isnan(val):
                self.min_holdings = max(self.min_holdings, int(val))
        if MAX_HOLDINGS_COL in bot_info.columns:
            val = bot_info[MAX_HOLDINGS_COL].iloc[row_index]
            if not np.isnan(val):
                self.max_holdings = min(self.max_holdings, int(val))
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

        ##self.inform("Who am I? {}".format(self.get_description()))
        self._market_id = list(self.markets.keys())[0]
        self.su = self.markets[self._market_id].max_price
        self.sl = self.markets[self._market_id].min_price
        self.curr_best_bid = self.sl
        self.curr_best_offer = self.su

    def get_units(self):
        """ Returns the number of units that the agent holds """
        if self.holdings is not None:
            market = self.markets[self._market_id]
            return self.holdings.assets[market].units
        return None

    def choiceprobabilities(self, action):
        """
        Calculates the probability of choosing a strategy for all strategies of
        the same action. Probability is proportional to the foregone utility of
        one strategy divided by the sum of foregone utilities over all
        strategies.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        choicep = []
        sumw = sum(self.utilities[action])
        if sumw == 0:
            return np.zeros(self.J)
        for j in range(self.J):
            choicep.append(self.utilities[action][j] / float(sumw))
        return choicep

    def strat_selection(self, action):
        """
        Chooses a strategy out of all strategies of the same action.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """

        if action == self.BUY:
            self.strategies[action] = list(filter(lambda x: x < self.valuations[self.get_units() + 1], self.strategies[action]))
        else:
            self.strategies[action] = list(filter(lambda x: x > self.valuations[self.get_units()], self.strategies[action]))
        if action == self.BUY:
            if len(self.strategies[action]) < self.J:
                if len(self.strategies[action]) == 0:
                    self.curr_strat[action] = self.sl
                x = self.J - len(self.strategies[action])
                for i in range(x):
                    self.strategies[action].append(int(random.uniform(self.sl, self.valuations[self.get_units() + 1])))
        else:
            if len(self.strategies[action]) < self.J:
                if len(self.strategies[action]) == 0:
                    self.curr_strat[action] = self.su
                x = self.J - len(self.strategies[action])
                for i in range(x):
                    self.strategies[action].append(int(random.uniform(self.valuations[self.get_units()], self.su)))

            self.updateW(action)
        choicep = self.choiceprobabilities(action)

        if sum(choicep) == 0:
            choicep = [1 / len(self.strategies[action]) for x in self.strategies[action]]
        self.curr_strat[action] = int(self.rand_choice(self.strategies[action], choicep))

    def rand_choice(self, items, distr):
        x = random.uniform(0, 1)
        cumulative_probability = 0.0
        saved_item = 0
        for item, item_probability in zip(items, distr):
            cumulative_probability += item_probability
            if x < cumulative_probability:
                saved_item = item
                break
        return saved_item

    def Vexperimentation(self, action):
        """
        Value experimentation for strategies of the same action. With a
        probability determined by muv, takes a strategy as a center of a
        distribution and generates a new strategy around the center.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for j in range(self.J):
            if action == self.BUY:
                sigmav = max(1, 0.1 * (self.valuations[self.get_units() + 1] - self.sl))
                if random.uniform(0, 1) < self.muv:
                    centers = self.strategies[action][j]
                    r = (truncnorm.rvs((self.sl - centers) / float(sigmav),
                                       (self.su - centers) / float(sigmav),
                                       loc=centers, scale=sigmav, size=1))
                    self.strategies[action][j] = int(np.array(r).tolist()[0])
            else:
                sigmav = max(1, 0.1 * (self.su - self.valuations[self.get_units()]))
                if random.uniform(0, 1) < self.muv:
                    centers = self.strategies[action][j]
                    r = (truncnorm.rvs((self.sl - centers) / float(sigmav),
                                       (self.su - centers) / float(sigmav),
                                       loc=centers, scale=sigmav, size=1))
                    self.strategies[action][j] = int(np.array(r).tolist()[0])

    def replicate(self, action):
        """
        Replicates strategies of the same action by comparing two randomly
        chosen strategies and replacing that with the lower utility with
        the other strategy.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for j in range(self.J):
            j1 = random.randrange(self.J)
            j2 = random.randrange(self.J)
            self.strategies[action][j] = self.strategies[action][j2]
            self.utilities[action][j] = self.utilities[action][j2]
            if self.utilities[action][j1] > self.utilities[action][j2]:
                self.strategies[action][j] = self.strategies[action][j1]
                self.utilities[action][j] = self.utilities[action][j1]

    def foregone_utility(self, j, action):
        """
        Calculates the foregone utility of a strategy with index j and
        corresponding to a certain action. Currently, we only consider the
        current book and look at the current highest bid and current lowest
        offer.

        :param j: The index of the strategy for which we want to update
        the foregone utility.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """

        if action == 0:
            bid = self.strategies[action][j]
            # Return 0 for case where bid exceeds the valuation of item to be bought
            if bid <= self.curr_best_offer or self.get_units() >= self.max_holdings or \
                    bid > self.valuations[self.get_units() + 1]:
                return 0
            else:
                return self.valuations[self.get_units() + 1] - self.curr_best_offer
        else:
            offer = self.strategies[action][j]
            # Return 0 for case where the offer is lower than the valuation of item to be sold
            if offer >= self.curr_best_bid or self.get_units() <= self.min_holdings or \
                    offer < self.valuations[self.get_units()]:
                return 0
            else:
                return self.curr_best_bid - self.valuations[self.get_units()]

    def foregone_utility_past(self, j, action):
        """
        Calculates the foregone utility of a strategy with index j and
        corresponding to a certain action. We consider the
        current book--looking at the current highest bid/current lowest
        offer--as well as past transactions.

        :param j: The index of the strategy for which we want to update
        the foregone utility.

        :param action: 0 corresponds to buying, 1 corresponds to selling

        """
        # #print("Now we are foregone_utility_past")
        if action == 0:
            bid = self.strategies[action][j]
            # get minimum of past prices
            p_min = min(self.past_trades)
            z = min(p_min, self.curr_best_offer)
            if bid <= z or self.get_units() >= self.max_holdings or \
                    bid > self.valuations[self.get_units() + 1]:
                return 0
            elif bid > self.curr_best_offer:
                return self.valuations[self.get_units() + 1] - self.curr_best_offer
            elif z < bid <= self.curr_best_offer:
                return self.valuations[self.get_units() + 1] - bid
        else:
            offer = self.strategies[action][j]
            # get maximum of past prices
            p_max = max(self.past_trades)
            z = max(p_max, self.curr_best_bid)
            if offer >= z or self.get_units() <= self.min_holdings or \
                    offer < self.valuations[self.get_units()]:
                return 0
            elif offer < self.curr_best_bid:
                return self.curr_best_offer - self.valuations[self.get_units()]
            elif z > offer >= self.curr_best_bid:
                return offer - self.valuations[self.get_units()]

    def updateW(self, action):
        """
        Updates the foregone utilities for strategies of the same action.

        :param action: 0 corresponds to buying, 1 corresponds to selling
        """
        for j in range(self.J):
            # if len(self.past_trades) > 0:
            #    self.utilities[action][j] = self.foregone_utility_past(j, action)
            # else:
            self.utilities[action][j] = self.foregone_utility(j, action)

    def update_list(self, order, add):
        for i in range(len(self.orders_not_traded)):
            if self.orders_not_traded[i] == order:
                if not add:
                    self.orders_not_traded.pop(i)
                return
        if add:
            self.orders_not_traded.append(order)

    def received_orders(self, orders: List[Order]):
        """
        Decide what to do when an order book update arrives.

        We categorize orders based on {ours, others} and {buy, sell}, then depending on the type of robot we either

        react or market make.

        :param orders: List of Orders received from FM via WS
        :return:
        """
        try:
            if len(self.strategies) == 0 or not self.is_session_active():
                return
            is_mine = True
            for order in orders:
                if order.has_traded and order.mine:
                    self.update_list(order, False)
                    self.update_list(order.traded_order, False)
                    self.past_trades.insert(0, order.price)
                    if len(self.past_trades) > self.T:
                        self.past_trades = self.past_trades[0:self.T]
                # else:
                #     if order.is_consumed:
                #         self.update_list(order, False)
                #     else:
                #         self.update_list(order, True)
                if not order.mine:
                    is_mine = False
            if not is_mine:
                order_book = self._categorize_orders()
                self._react_to_book(order_book)
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
        :return:
        """
        try:
            if len(self.strategies) != 0:
                self.updateW(self.BUY)
                self.updateW(self.SELL)
        except Exception as e:
            tb.print_exc()

    def initialize_strat_set(self):
        """
        Only initializes the strategy set if they are currently uninitialized
        """
        if len(self.strategies) == 0:
            self.strategies = [[], []]
            for j in range(self.J):
                if self.sl > self.valuations[self.get_units() + 1] and self.su < self.valuations[self.get_units()]:
                    break
                elif self.su < self.valuations[self.get_units()]:
                    if self.get_units() >= self.max_holdings:
                        self.strategies[self.BUY].append(self.sl)
                    else:
                        self.strategies[self.BUY].append(
                            random.randint(self.sl, int(self.valuations[self.get_units() + 1])))
                elif self.sl > self.valuations[self.get_units() + 1]:
                    if self.get_units() <= self.min_holdings:
                        self.strategies[self.SELL].append(self.su)
                    else:
                        self.strategies[self.SELL].append(
                            random.randint(int(self.valuations[self.get_units()]), self.su))
                else:
                    if self.get_units() >= self.max_holdings:
                        self.strategies[self.BUY].append(self.sl)
                    else:
                        self.strategies[self.BUY].append(
                            random.randint(self.sl, int(self.valuations[self.get_units() + 1])))
                    if self.get_units() <= self.min_holdings:
                        self.strategies[self.SELL].append(self.su)
                    else:
                        self.strategies[self.SELL].append(
                            random.randint(int(self.valuations[self.get_units()]), self.su))
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
                self.curr_best_bid = self.sl
                self.curr_best_offer = self.su
                self.initialize_strat_set()
            elif session.is_closed:
                # The purpose of this is to reset the strategy set after a period ends
                self.past_trades = []
                self.strategies = []
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
            if len(self.strategies) != 0 and self.is_session_active():
                order_book = self._categorize_orders()
                if len(order_book["mine"]["buy"]) > 0:
                    self.cancel_ord(self.BUY, order_book)
                if len(order_book["mine"]["sell"]) > 0:
                    self.cancel_ord(self.SELL, order_book)
                self.send_ord(self.BUY, order_book)
                self.send_ord(self.SELL, order_book)
        except Exception as e:
            tb.print_exc(e)

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

    def _get_order(self, order_side):
        """
        Takes a request for a either a buy or sell order, the type of which is
        specified by order_side. Returns an order.

        :param order_side: either OrderSide.BUY or OrderSide.SELL
        """
        action = self.BUY
        if order_side == OrderSide.SELL:
            action = self.SELL
        self.Vexperimentation(action)
        self.updateW(action)
        self.replicate(action)
        self.strat_selection(action)
        if order_side == OrderSide.SELL and (
                self.curr_strat[self.SELL] < self.sl or self.curr_strat[self.SELL] > self.su):
            raise Exception('Price should not be less than the minimum or greater than the maximum when selling.')
        if order_side == OrderSide.BUY and (
                self.curr_strat[self.BUY] < self.sl or self.curr_strat[self.BUY] > self.su):
            raise Exception('Price should not be less than the minimum or greater than the maximum when selling.')
        price = max(self.sl, self.curr_strat[action])
        order = Order.create_new(self.markets[self._market_id])
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

    def _categorize_orders(self):

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
            if order.order_type == OrderType.LIMIT and order.is_pending:
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
            cancel_order.ref = self._increment_counter(self._mm_cancel_prefix)
            self.send_order(cancel_order)

    def send_ord(self, action, order_book):
        my_orders = order_book["mine"]
        if action == self.BUY:
            if self.sl <= self.valuations[self.get_units() + 1]:
                waiting = self._waiting_for(self._mm_buy_prefix)
                if not waiting and len(my_orders["buy"]) == 0 and self.get_units() < self.max_holdings:
                    order = self._get_order(OrderSide.BUY)
                    cash_i_have = self.holdings.cash_available
                    if len(my_orders["buy"]) < self._max_market_making_orders \
                            and cash_i_have >= order.price:
                        order.ref = self._increment_counter(self._mm_buy_prefix)
                        self.send_order(order)
        else:
            if self.su >= self.valuations[self.get_units()]:
                waiting = self._waiting_for(self._mm_sell_prefix)
                if not waiting and len(my_orders["sell"]) == 0:
                    order = self._get_order(OrderSide.SELL)
                    market = self.markets[self._market_id]
                    units_i_have = self.holdings.assets[market].units_available
                    if len(my_orders["sell"]) < self._max_market_making_orders \
                            and units_i_have >= order.units and self.get_units() > self.min_holdings:
                        order.ref = self._increment_counter(self._mm_sell_prefix)
                        self.send_order(order)

    def _react_to_book(self, order_book):

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
                self.curr_best_bid = h_bid
                self.updateW(self.SELL)

        if len(curr_offers) != 0:
            l_offer = curr_offers[0].price
            if l_offer != self.curr_best_offer:
                submit_bid = True
                # Updates the current best offer
                self.curr_best_offer = l_offer
                self.updateW(self.BUY)

        if submit_bid:
            self.cancel_ord(self.BUY, order_book)
        if submit_offer:
            self.cancel_ord(self.SELL, order_book)

    def _increment_counter(self, prefix):
        """
        Increments the counter and returns a reference with a prefix
        corresponding to the type of an order.

        :return: Reference of an order
        """
        ref = str(prefix) + str(self._counter)
        self._counter += 1
        return ref

    def _waiting_for(self, prefix):

        """

        Check if there exists an order that we are waiting for the server to process

        such that the order's ref starts with a given prefix.

        :param prefix: prefix to check in order's reference

        :return: True if such an order exists, False otherwise

        """

        for key in self._orders_waiting_ackn.keys():
            if key.startswith(prefix):
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
            order.ref = self._increment_counter("n")
        self._orders_waiting_ackn[order.ref] = order
        super().send_order(order)

    def received_marketplace_info(self, marketplace_info):

        _extra_info = "Waiting for marketplace to open." if not marketplace_info["status"] else ""

    def order_accepted(self, order):
        ''' Now takes in the order book as a parameter. '''
        if order.mine:
            if order.ref in self._orders_waiting_ackn:
                del self._orders_waiting_ackn[order.ref]

    def order_rejected(self, info, order):
        if order.mine:
            if order.ref in self._orders_waiting_ackn:
                del self._orders_waiting_ackn[order.ref]


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
