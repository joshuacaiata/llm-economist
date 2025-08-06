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

    def execute_action(self, action):
        action_type = action['action_type']
        action_args = action['action_args']

        if action_type == "Move":
            direction = action_args.split(" ")[1]
            if direction in ["Up", "Down", "Left", "Right"]:
                self.move(direction)
        elif action_type == "Trade":
            action = action_args.split(" ")
            if action[0] in ["Buy", "Sell"]:
                pass # TODO: implement trade
        elif action_type == "Build":
            self.build()
        else:
            pass


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
        result = self.llm.generate_response(prompt_to_llm)
        action = result.split(",")
        return {'action_type': action[0], 'action_args': action[1]}

    def reset(self):
        pass

    def get_utility(self):
        if self.risk_aversion == 1:
            self.utility.append(np.log(self.inventory["coins"]))
        else:
            utility = (self.inventory["coins"] ** (1 - self.risk_aversion) - 1) / (1 - self.risk_aversion)
            self.utility.append(utility - self.labour)

    def move(self, direction):
        location = self.env.current_agent_positions[self.agent_id]
        if direction == "Up":
            new_location = (location[0] - 1, location[1])
        elif direction == "Down":
            new_location = (location[0] + 1, location[1])
        elif direction == "Left":
            new_location = (location[0], location[1] - 1)
        elif direction == "Right":
            new_location = (location[0], location[1] + 1)

        # check if its valid. if so, update the env. if not, do nothing.
        # its valid if the location is in the map, and is not water
        if new_location[0] < 0 or new_location[0] >= self.env.map_size[0] or new_location[1] < 0 or new_location[1] >= self.env.map_size[1]:
            return
        if self.env.map["Water"][new_location[0], new_location[1]] == 1:
            return

        self.env.current_agent_positions[self.agent_id] = new_location

        # if the location has a stone or wood, we collect it with probability 

    def build(self):
        # build if we have at least one wood and one stone
        # and if the location is buildable

        location = self.env.current_agent_positions[self.agent_id]
        if self.env.map["Buildable"][location[0], location[1]] == 0:
            return

        if self.inventory["wood"] < 1 or self.inventory["stone"] < 1:
            return

        self.env.map["Houses"][location[0], location[1]] = 1
        self.env.map["Buildable"][location[0], location[1]] = 0
        self.inventory["wood"] -= 1
        self.inventory["stone"] -= 1
        self.inventory["coins"] += self.build_payout