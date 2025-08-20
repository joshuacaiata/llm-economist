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
        self.active_orders = 0

        self.last_year_coins = 0

        self.metrics_history = {
            'coins': [0],
            'wood': [0], 
            'stone': [0],
            'utility': [0],
            'houses_built': [0],
            'active_orders': [0],
            'escrow_coins': [0],
            'escrow_wood': [0],
            'escrow_stone': [0]
        }
        self.total_houses_built = 0
        self.movement_history = []
        self.decision_history = []
        self.memory_turns = config['llm'].get('memory_turns', 10)

        self.view_size = config['agent_view_size']

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
        # Get decision history

        decision_history = self.format_decision_history()

        # get sorted income distribution
        income_distribution = self.get_income_distribution()
        
        prompt = f"""
        You are a mobile agent in a 2D grid world.
        Your goal is to maximize your utility function.
        The utility function is the sum of the discounted utility of each turn.

        CURRENT STATE:
        - Location: {location}
        - Inventory: {self.inventory}
        - Neighbourhood: {neighbourhood}
        - Build payout: {self.build_payout}
        - Current utility: {utility}
        - Total houses built: {self.total_houses_built}
        - Active market orders: {self.active_orders}
        - Resources in escrow: {self.escrow}
        - Income: {self.get_income()}
        - Tax bracket: {self.get_tax_bracket()}
        - Tax rates: {self.env.planner.tax_rates}
        - Sorted anonymized income distribution: {income_distribution}

        {decision_history}

        Based on your history:
        1. Look for patterns in which actions led to the highest utility gains
        2. Consider your current needs (wood/stone for building, coins for trading)
        3. Think about market timing - when to buy/sell resources
        4. Plan moves to reach valuable resources in your neighborhood
        """

        return prompt

    def step(self):
        # this function will step. so we need to observe, then get the action, then execute the action
        if self.action_mechanism not in ["llm", "random"]:
            raise ValueError(f"Invalid agent action mechanism: {self.action_mechanism}")
            
        if self.action_mechanism == "llm":
            prompt = self.observe()
            action = self.get_action(prompt)
        else:  # random
            action = self.get_action()
        self.execute_action(action)
        self.record_metrics()

    def execute_action(self, action):
        action_type = action['action_type']
        action_args = action['action_args']
        
        # Record the decision before executing it
        self.record_decision(action_type, action_args)

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
                # Convert Buy/Sell to buy/sell for trading system
                transaction_type = action_parts[0].lower()
                self.env.trading_system.make_order(self.agent_id, action_parts[1].lower(), int(action_parts[2]), transaction_type)
        elif action_type == "Build":
            self.build()
        elif action_type == "Nothing":
            pass  # Do nothing
        else:
            pass

    def get_income_distribution(self):
        # get sorted income distribution
        income_distribution = sorted(self.env.mobile_agents, key=lambda x: x.get_income())
        return income_distribution

    def _get_action_descriptions(self):
        """Generate dynamic descriptions of valid actions."""
        valid_actions = self._get_valid_actions()
        
        # Group actions by type
        moves = [a['action_args'] for a in valid_actions if a['action_type'] == "Move"]
        trades = [a['action_args'] for a in valid_actions if a['action_type'] == "Trade"]
        can_build = any(a['action_type'] == "Build" for a in valid_actions)
        
        descriptions = []
        
        # Movement description
        if moves:
            descriptions.append("You can move in the following directions:")
            for direction in moves:
                descriptions.append(f"- {direction}")
            descriptions.append(f"Moving costs {self.env.move_labour} labour. If you move to a tile with wood or stone, " +
                             f"you will collect it (costs {self.env.gather_labour} additional labour). " +
                             f"With probability {self.gather_skill}, you will collect a second unit.")
        
        # Trading description
        if trades:
            descriptions.append("\nYou can make the following trades:")
            for trade in trades:
                parts = trade.split()
                if parts[0] == "Buy":
                    max_possible_bid = min(self.inventory['coins'], self.env.max_order_price)
                    descriptions.append(f"- Buy {parts[1]} (you have {self.inventory['coins']} coins, can bid between 1 and {max_possible_bid} coins)")
                else:
                    item = parts[1].lower()
                    descriptions.append(f"- Sell {parts[1]} (you have {self.inventory[item]} {item}, can ask between 1 and {self.env.max_order_price} coins)")
            descriptions.append(f"Trading costs {self.env.trade_labour} labour. " +
                             f"When you place a buy/sell order, your resources are placed in escrow. " +
                             f"If your order is matched, you'll receive the traded resource/coins. " +
                             f"If your order expires after {self.env.trading_system.max_order_lifetime} steps, " +
                             f"your escrowed resources will be returned to you.")
        
        # Building description
        if can_build:
            descriptions.append("\nYou can build a house:")
            descriptions.append("- Requires 1 wood and 1 stone")
            descriptions.append(f"- Rewards {self.build_payout} coins")
            descriptions.append(f"- Costs {self.env.build_labour} labour")
        
        # Always show nothing option
        descriptions.append("\nYou can always choose to do nothing (costs no labour).")
        
        return "\n".join(descriptions)

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


        # Get dynamic action descriptions based on current state
        action_descriptions = self._get_action_descriptions()
        
        prompt = f"""
        Here are the actions available to you this turn:
        {action_descriptions}

        Please respond with your chosen action in the following format:
        Action, Arguments

        Examples:
        - To move: "Move, Up"
        - To trade: "Buy, Wood 10" or "Sell, Stone 5" (where the number is your price in coins)
        - To build: "Build, "
        - To do nothing: "Nothing, Nothing"

        Your response must match one of the available actions exactly.
        Return nothing else in your response except the action and correct arguments.
        If your response is invalid, you will do nothing, or conduct undefined behaviour.
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
        
        if self.env.config.get('trading_system', False):
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
                price = np.random.randint(1, self.env.max_order_price + 1)
                valid_trades.append({
                    'action_type': "Trade", 
                    'action_args': f"Sell {item} {price}"
                })
        
        # Valid buy orders - can only bid what we can afford (1 item per order)
        if self.inventory["coins"] > 0:
            for item in items:

                max_bid = min(self.inventory["coins"], self.env.max_order_price)
                if max_bid > 1:
                    price = np.random.randint(1, max_bid + 1)
                    valid_trades.append({
                        'action_type': "Trade", 
                        'action_args': f"Buy {item} {price}"
                    })
                elif max_bid == 1:
                    valid_trades.append({
                        'action_type': "Trade", 
                        'action_args': f"Buy {item} 1"
                    })
        
        return valid_trades

    def record_decision(self, action_type, action_args):
        """Record a decision and its outcome."""
        # Get state before action
        prev_state = {
            'location': self.env.current_agent_positions[self.agent_id],
            'wood': self.inventory['wood'],
            'stone': self.inventory['stone'],
            'coins': self.inventory['coins'],
            'utility': self.utility[-1] if self.utility else 0
        }
        
        # Store the decision
        self.decision_history.append({
            'state': prev_state,
            'action': f"{action_type}, {action_args}",
            'outcome': None  # Will be updated after we see the results
        })
        
        # Keep only last N turns
        if len(self.decision_history) > self.memory_turns:
            self.decision_history.pop(0)

    def update_last_decision_outcome(self):
        """Update the outcome of the last decision after seeing its results."""
        if not self.decision_history:
            return
            
        last_decision = self.decision_history[-1]
        prev_state = last_decision['state']
        
        # Calculate changes
        changes = {
            'wood': self.inventory['wood'] - prev_state['wood'],
            'stone': self.inventory['stone'] - prev_state['stone'],
            'coins': self.inventory['coins'] - prev_state['coins'],
            'utility': self.utility[-1] - prev_state['utility']
        }
        
        # Format outcome string
        outcome_parts = []
        if changes['wood'] != 0:
            outcome_parts.append(f"wood {changes['wood']:+.0f}")
        if changes['stone'] != 0:
            outcome_parts.append(f"stone {changes['stone']:+.0f}")
        if changes['coins'] != 0:
            outcome_parts.append(f"coins {changes['coins']:+.0f}")
        outcome_parts.append(f"utility {changes['utility']:+.2f}")
        
        last_decision['outcome'] = ", ".join(outcome_parts)

    def format_decision_history(self):
        """Format decision history for the prompt."""
        if not self.decision_history:
            return "No previous decisions.\n"
            
        history = ["RECENT DECISIONS:"]
        for i, decision in enumerate(self.decision_history):
            state = decision['state']
            turn_str = f"Turn {i+1}: At {state['location']} with {state['wood']} wood, {state['stone']} stone, {state['coins']} coins\n"
            turn_str += f"Action: {decision['action']}"
            if decision['outcome']:
                turn_str += f" -> {decision['outcome']}"
            history.append(turn_str)
            
        return "\n".join(history) + "\n"

    def record_metrics(self):
        """Record current metrics for tracking over time."""
        self.get_utility()
        self.update_last_decision_outcome()  # Update the outcome of the last decision
        
        self.metrics_history['coins'].append(self.inventory['coins'])
        self.metrics_history['wood'].append(self.inventory['wood'])
        self.metrics_history['stone'].append(self.inventory['stone'])
        self.metrics_history['utility'].append(self.utility[-1])
        self.metrics_history['houses_built'].append(self.total_houses_built)
        self.metrics_history['active_orders'].append(self.active_orders)
        self.metrics_history['escrow_coins'].append(self.escrow['coins'])
        self.metrics_history['escrow_wood'].append(self.escrow['wood'])
        self.metrics_history['escrow_stone'].append(self.escrow['stone'])

    def reset(self):
        self.last_year_coins = 0
        self.inventory = {'wood': 0, 'stone': 0, 'coins': 0}
        self.escrow = {'wood': 0, 'stone': 0, 'coins': 0}
        self.utility = [0]
        self.labour = [0]
        self.movement_history = []
        self.decision_history = []

    def reset_year(self):
        self.last_year_coins = self.inventory['coins']

    def get_income(self):
        return self.inventory['coins'] - self.last_year_coins

    def get_utility(self):
        if self.risk_aversion == 1:
            base_utility = np.log(self.inventory["coins"] + 1e-8)
        else:
            base_utility = (self.inventory["coins"] ** (1 - self.risk_aversion) - 1) / (1 - self.risk_aversion)

        labour_cost = sum(self.labour)
        utility_value = base_utility - labour_cost
        self.utility.append(utility_value)
        return utility_value


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
        self.movement_history.append(new_location)

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