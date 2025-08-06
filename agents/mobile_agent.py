from agents.base_agent import BaseAgent
import numpy as np

class MobileAgent(BaseAgent):
    def __init__(
            self,
            agent_class,
            agent_id,
            config,
            build_payout,
            risk_aversion,
            env,
            labour,
            llm
        ):
        super().__init__(agent_class, agent_id, config, env)
        self.agent_class = agent_class
        self.agent_id = agent_id
        self.config = config
        self.env = env
        self.build_payout = build_payout
        self.risk_aversion = risk_aversion
        self.inventory = {'wood': 0, 'stone': 0, 'coins': 0}
        self.escrow = {'wood': 0, 'stone': 0, 'coins': 0}
        self.utility = [0]
        self.labour = labour
        self.llm = llm

    def observe(self):
        """
        Agents observation space:
        1. Neighbourhood (everything in the view_size radius)
        2. Inventory
        3. Build payout
        TODO:
        4. Taxation 
            i. Planner tax schedule
            ii. Agents current tax bracket
            iii. Anonymized and sorted income distribution of all agents
        5. Time to next tax period
        TODO:
        6. Monetary policy
            i. Interest rate
            ii. Inflation rate
            iii. Money supply
            iv. Target inflation rate
        7. Position
        """
        # get the location
        location = self.env.current_agent_positions[self.agent_id]
        # get the neighbourhood
        neighbourhood = {}
        env_map = self.env.map
        for level in env_map.keys():
            x_upper = max(0, location[0] - self.view_size)
            x_lower = min(self.env.map_size[0], location[0] + self.view_size)
            y_upper = max(0, location[1] - self.view_size)
            y_lower = min(self.env.map_size[1], location[1] + self.view_size)
            neighbourhood[level] = env_map[level][x_upper:x_lower, y_upper:y_lower]

        # utility
        utility = self.utility[-1]
        for i in range(len(self.utility) - 1):
            utility += self.env.discount_factor ** (i + 1) * self.utility[i]

        # make the prompt:
        prompt = f"""
        You are a mobile agent in a 2D grid world.
        Your goal is to maximize your utility function.
        The utility function is the sum of the discounted utility of each turn.
        You are given the following information:
        - Your inventory
        - Your location
        - The neighbourhood
        - The build payout
        - Your current utility

        Here are your observation for this turn:
        - Your location: {location}
        - Your inventory: {self.inventory}
        - The neighbourhood: {neighbourhood}
        - The build payout: {self.build_payout}
        - Your current utility: {utility}
        """

        return prompt

    def step(self):
        # this function will step. so we need to observe, then get the action, then execute the action
        prompt = self.observe()
        action = self.get_action(prompt)
        self.execute_action(action)




        # action is in the form of: {'action_type': 'move/trade/build', 'action_args': {}}
        # if action is move, then args is {'direction': 'up/down/left/right'}
        # if action is trade, then the args is {'action': 'buy/sell', 'item': 'wood/stone', 'amount': int}
        # if action is build, args is empty

    def get_action(self, observation_prompt):
        prompt = f"""
        You can take one of the following actions:
        - Move
        - Trade
        - Build

        If you decide to move, you can move in the following directions:
        - Up
        - Down
        - Left
        - Right
        You will move one tile, if possible. If you cannot move, you will stay in the same tile.

        If you decide to trade, you can buy or sell the following items:
        - Wood
        - Stone
        If you place a sell order and you do not have the item, you will not be able to sell it.
        If you place a buy order and you do not have the coins, you will not be able to buy it.
        If you can place the order, your resources will be placed into escrow until the order is filled.
        Once the order is filled, your inventory will be updated accordingly.

        You may also decide to do nothing.

        Please respond with the following format:
        Action, Arguments

        For example:
        Move, Up
        Move, Down
        Buy, Wood 10
        Sell, Stone 5
        Nothing, Nothing

        If your response is of an invalid format, you will do nothing.
        """

        prompt_to_llm = observation_prompt + "\n" + prompt
        action = self.llm.generate_response(prompt_to_llm)
        return action

    def reset(self):
        pass

    def get_utility(self):
        if self.risk_aversion == 1:
            self.utility.append(np.log(self.inventory["coins"]))
        else:
            utility = (self.inventory["coins"] ** (1 - self.risk_aversion) - 1) / (1 - self.risk_aversion)
            self.utility.append(utility - self.labour)