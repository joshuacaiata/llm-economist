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
            llm,
            action_mechanism,
            gather_skill
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
        self.labour = [0]
        self.llm = llm
        self.action_mechanism = action_mechanism
        self.gather_skill = gather_skill

        self.metrics_history = {
            'coins': [0],
            'wood': [0], 
            'stone': [0],
            'utility': [0],
            'houses_built': [0]
        }
        self.total_houses_built = 0

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
        if self.action_mechanism == "llm":
            prompt = self.observe()
            action = self.get_action(prompt)
        elif self.action_mechanism == "random":
            action = self.get_action()
        else:
            raise ValueError(f"Invalid action mechanism: {self.action_mechanism}")
        self.execute_action(action)
        self.record_metrics()

    def execute_action(self, action):
        action_type = action['action_type']
        action_args = action['action_args']

        if action_type == "Move":
            # For random actions, direction is directly in action_args
            # For LLM actions, direction is after "Move " in action_args
            if " " in action_args:
                direction = action_args.split(" ")[1]
            else:
                direction = action_args
            if direction in ["Up", "Down", "Left", "Right"]:
                self.move(direction)
        elif action_type == "Trade":
            action_parts = action_args.split(" ")
            if action_parts[0] in ["Buy", "Sell"]:
                pass # TODO: implement trade
        elif action_type == "Build":
            self.build()
        elif action_type == "Nothing":
            pass  # Do nothing
        else:
            pass


    def get_action(self, observation_prompt=None):
        if observation_prompt is None:
            # Get all valid actions based on current state
            valid_actions = self._get_valid_actions()
            
            # If no valid actions, do nothing
            if not valid_actions:
                return {'action_type': "Nothing", 'action_args': "Nothing"}
            
            # Randomly select from valid actions
            action = np.random.choice(valid_actions)
            return action

        


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

    def _get_valid_actions(self):
        """Get all valid actions based on current state and environment."""
        valid_actions = []
        location = self.env.current_agent_positions[self.agent_id]
        
        # Check valid moves
        directions = ["Up", "Down", "Left", "Right"]
        for direction in directions:
            if self._is_valid_move(location, direction):
                valid_actions.append({'action_type': "Move", 'action_args': direction})
        
        # Check if build is valid
        if self._is_valid_build(location):
            valid_actions.append({'action_type': "Build", 'action_args': ""})
        
        # Check valid trades
        valid_trades = self._get_valid_trades()
        valid_actions.extend(valid_trades)
        
        # Always allow doing nothing
        valid_actions.append({'action_type': "Nothing", 'action_args': "Nothing"})
        
        return valid_actions
    
    def _is_valid_move(self, current_location, direction):
        """Check if a move in the given direction is valid."""
        if direction == "Up":
            new_location = (current_location[0] - 1, current_location[1])
        elif direction == "Down":
            new_location = (current_location[0] + 1, current_location[1])
        elif direction == "Left":
            new_location = (current_location[0], current_location[1] - 1)
        elif direction == "Right":
            new_location = (current_location[0], current_location[1] + 1)
        else:
            return False
        
        # Check if location is within map bounds
        if (new_location[0] < 0 or new_location[0] >= self.env.map_size[0] or 
            new_location[1] < 0 or new_location[1] >= self.env.map_size[1]):
            return False
        
        # Check if location is not water
        if self.env.map["Water"][new_location[0], new_location[1]] == 1:
            return False
        
        return True
    
    def _is_valid_build(self, location):
        """Check if building at current location is valid."""
        # Need at least 1 wood and 1 stone
        if self.inventory["wood"] < 1 or self.inventory["stone"] < 1:
            return False
        
        # Location must be buildable
        if self.env.map["Buildable"][location[0], location[1]] == 0:
            return False
        
        return True
    
    def _get_valid_trades(self):
        """Get all valid trade actions."""
        valid_trades = []
        
        # Valid sell orders - can only sell what we have (1 item per order)
        items = ["Wood", "Stone"]
        for item in items:
            item_key = item.lower()
            if self.inventory[item_key] > 0:
                # Generate a random price for selling 1 item
                price = np.random.randint(1, 1000)  # Random price between 1-1000 coins
                valid_trades.append({
                    'action_type': "Trade", 
                    'action_args': f"Sell {item} {price}"
                })
        
        # Valid buy orders - can only bid what we can afford (1 item per order)
        if self.inventory["coins"] > 0:
            for item in items:
                # Generate a random bid price up to our available coins
                max_bid = self.inventory["coins"]
                price = np.random.randint(1, max_bid + 1)
                valid_trades.append({
                    'action_type': "Trade", 
                    'action_args': f"Buy {item} {price}"
                })
        
        return valid_trades

    def record_metrics(self):
        """Record current metrics for tracking over time."""
        self.get_utility()
        
        self.metrics_history['coins'].append(self.inventory['coins'])
        self.metrics_history['wood'].append(self.inventory['wood'])
        self.metrics_history['stone'].append(self.inventory['stone'])
        self.metrics_history['utility'].append(self.utility[-1])
        self.metrics_history['houses_built'].append(self.total_houses_built)

    def reset(self):
        pass

    def get_utility(self):
        if self.risk_aversion == 1:
            base_utility = np.log(self.inventory["coins"] + 1e-8)
        else:
            base_utility = (self.inventory["coins"] ** (1 - self.risk_aversion) - 1) / (1 - self.risk_aversion)

        labour_cost = sum(self.labour)
        self.utility.append(base_utility - labour_cost)


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

        action_labour = self.env.move_labour

        # if the location has a stone or wood, we collect it. with probability gather_skill we collect a second one
        # we then set it to 0 and allow the map to regenerate it
        if self.env.map["Wood"][new_location[0], new_location[1]] == 1:
            action_labour += self.env.gather_labour
            self.inventory["wood"] += 1
            if np.random.random() < self.gather_skill:
                self.inventory["wood"] += 1

            self.env.map["Wood"][new_location[0], new_location[1]] = 0

        if self.env.map["Stone"][new_location[0], new_location[1]] == 1:
            action_labour += self.env.gather_labour
            self.inventory["stone"] += 1
            if np.random.random() < self.gather_skill:
                self.inventory["stone"] += 1

            self.env.map["Stone"][new_location[0], new_location[1]] = 0
        
        self.labour.append(action_labour)

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
        self.total_houses_built += 1
        self.labour.append(self.env.build_labour)