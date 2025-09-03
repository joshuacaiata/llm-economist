from agents.base_agent import BaseAgent
import numpy as np

class PlannerAgent(BaseAgent):
    def __init__(
            self,
            agent_class,
            agent_id,
            config,
            env,
            llm,
            action_mechanism,
    ):
        super().__init__(agent_class, agent_id, config, env)
        self.agent_class = agent_class
        self.agent_id = agent_id
        self.config = config
        self.env = env
        self.llm = llm
        self.action_mechanism = action_mechanism
        self.original_tax_rates = config['default_tax_rates']
        self.tax_brackets = config['tax_brackets']
        self.num_tax_brackets = len(self.tax_brackets)
        self.tax_rates = config['default_tax_rates']
        self.tax_collection_history = []
        self.tax_bracket_history = [self.tax_rates]
        self.utility_type = config['planner_utility_type']
        self.fixed_tax_rates = config['fixed_tax_rates']
        
        assert self.num_tax_brackets + 1 == len(self.tax_rates), "Number of tax brackets and tax rates must match"
        assert self.utility_type in ["utilitarian", "nash_welfare"], "Invalid utility type"

        if self.utility_type == "utilitarian":
            self.utility_function = self.get_utilitarian_utility
        elif self.utility_type == "nash_welfare":
            self.utility_function = self.get_nash_welfare_utility

        self.utility_history = []
        
        self.decision_history = []
        self.memory_turns = config.get('planner_memory_turns', 5)


    def observe(self):
        # TODO: Add the ability for the planner to also observe the central bank
        """
        1. Bids, asks, traded prices, number of trades per resource (wood and stone)
        2. Agent inventories (coins, wood, stone)
        3. Current tax rates, previous year's income, tax brackets
        4. Current time step and time to next tax year

        Return the prompt to the LLM
        """

        prompt = f"""
You are a planner agent in a multi-agent economy.
You are responsible for setting the tax rates for the next year.
Mobile agents can move, build, trade, and do nothing. By building and trading, agents earn coins.
Your job is to tax their income to increase your overall utility.
When you collect taxes, you redistribute them equally to all agents.
Your utility function is {self.utility_type}.
        """

        if self.utility_type == "utilitarian":
            prompt += """
            Your utility function is utilitarian.
            You want to maximize the sum of the utilities of all agents.
            """
        elif self.utility_type == "nash_welfare":
            prompt += """
            Your utility function is nash welfare.
            You want to maximize the product of the utilities of all agents.
            """

        num_bids = len(self.env.trading_system.buy["wood"]) + len(self.env.trading_system.buy["stone"])
        num_asks = len(self.env.trading_system.sell["wood"]) + len(self.env.trading_system.sell["stone"])
        num_trades = num_bids + num_asks
        # traded prices should be the average over the last tax period
        traded_prices = 0 # TODO: get the average traded price over the last tax period

        agent_inventories = {
            agent.agent_id: agent.inventory
            for agent in self.env.mobile_agents
        }

        current_tax_rates = self.tax_rates
        previous_year_incomes = {
            agent.agent_id: agent.get_income()
            for agent in self.env.mobile_agents
        }

        time_to_next_tax_year = self.env.time_to_next_tax_year
        current_time_step = self.env.time

        prompt += f"""
You will be given the following information:
1. Bids, asks, traded prices, number of trades per resource (wood and stone)
2. Agent inventories (coins, wood, stone)
3. Current tax rates, previous year's income, tax brackets
4. Current time step and time to next tax year

{self.format_decision_history()}

Provided this information, you must set the new tax rates. 
Bids: {num_bids}
Asks: {num_asks}
Trades: {num_trades}
Traded prices: {traded_prices}
Agent inventories: {agent_inventories}
Current tax rates: {current_tax_rates}
Tax brackets: {self.tax_brackets}
Previous year's incomes: {previous_year_incomes}
Time to next tax year: {time_to_next_tax_year}
Current time step: {current_time_step}

Based on your history and the current economic situation:
1. Look for patterns in which tax policies led to better utility outcomes
2. Consider the trade-off between tax collection and economic growth
3. Think about how tax rates affect inequality and overall welfare
4. Plan tax rates that will improve your utility function over time

You must respond with a set of numbers between 0 and 1 and of length {self.num_tax_brackets + 1}.
For example: 0.1 0.2 0.3 0.4 0.5 0.6 means that the first marginal rate is 10%, then 20%, etc.
Return ONLY the set of numbers between 0 and 1 and of length {self.num_tax_brackets + 1}. Do not return anything else.
If you return anything else, you will be penalized heavily, and your plan will be rejected. The tax rates will then be set at random.
        """
        return prompt

    def step(self):
        if self.action_mechanism not in ["llm", "random"]:
            raise ValueError(f"Invalid planner action mechanism: {self.action_mechanism}")
        
        prev_state = self.get_current_state()
            
        if self.action_mechanism == "llm":
            prompt = self.observe()
            tax_rates, valid_response = self.get_tax_rates(prompt)
        else:  # random
            tax_rates, valid_response = self.get_tax_rates()
        
        self.store_decision(prev_state, tax_rates, valid_response)
        
        self.execute_action(tax_rates)
        utility = self.utility_function()
        if not valid_response:
            utility += -100
        self.utility_history.append(utility)
        
        self.update_last_decision_outcome()

    def reset(self):
        self.tax_rates = self.original_tax_rates
        self.utility_history = []
        self.decision_history = []

    def get_tax_rates(self, prompt=None):
        if self.fixed_tax_rates:
            return self.tax_rates, True
        
        if prompt is None:
            tax_rates = np.random.uniform(0, 1, self.num_tax_brackets + 1)
            return tax_rates, True
        else:
            result = self.llm.generate_response(prompt)
            tax_rate_strings = result.split()
            valid_response = True
            tax_rates = []
            
            try:
                for tax_rate_str in tax_rate_strings:
                    tax_rate = float(tax_rate_str)
                    if tax_rate < 0 or tax_rate > 1:
                        valid_response = False
                        break
                    tax_rates.append(tax_rate)
                    
                if len(tax_rates) != self.num_tax_brackets + 1:
                    valid_response = False
                    
            except (ValueError, TypeError):
                valid_response = False
            
            if not valid_response:
                tax_rates = np.random.uniform(0, 1, self.num_tax_brackets + 1)
                return tax_rates, False
            
            return np.array(tax_rates), True
    
    def execute_action(self, tax_rates):
        # collect taxes from the agent based on previous tax rates
        # then update the tax rates
        collected_taxes = 0
        
        # Store pre-redistribution coins for next year's baseline
        pre_redistribution_coins = {}
        
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            
            # get the agents income
            income = agent.get_income()

            # Store pre-redistribution coins (after tax collection)
            pre_redistribution_coins[agent.agent_id] = agent.inventory['coins']

            # calculate the taxes owed using tax brackets 
            taxes_owed = self.calculate_taxes_owed(income, self.tax_rates)

            # collect the taxes from the agent
            to_collect = min(taxes_owed, agent.inventory['coins'])
            agent.inventory['coins'] -= to_collect
            collected_taxes += to_collect
            
            # Update pre-redistribution coins after tax collection
            pre_redistribution_coins[agent.agent_id] = agent.inventory['coins']
        

        # update the tax rates
        self.tax_rates = tax_rates
        self.tax_collection_history.append(collected_taxes)
        self.tax_bracket_history.append(self.tax_rates)
        
        # disburse the collected taxes to the agents equally
        # TODO: not sure if there is other ways to redistribute the income
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            agent.inventory['coins'] += collected_taxes / len(self.env.mobile_agents) 
        
        # Set next year's baseline to pre-redistribution amounts
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            agent.last_year_coins = pre_redistribution_coins[agent.agent_id]

    
    def calculate_taxes_owed(self, income, tax_rates):
        total_tax = 0
        remaining_income = income

        if income <= self.tax_brackets[0]:
            return income * tax_rates[0]
        
        for i in range(len(self.tax_brackets) - 1):
            bracket_size = self.tax_brackets[i + 1] - self.tax_brackets[i]
            if remaining_income <= 0:
                break

            taxable_amount = min(bracket_size, remaining_income)
            total_tax += taxable_amount * tax_rates[i + 1]
            remaining_income -= taxable_amount
        
        if remaining_income > 0:
            total_tax += remaining_income * tax_rates[-1]
        
        return total_tax

    def calculate_gini(self, values):
        """Calculate the Gini coefficient of a list of values."""
        if not values or len(values) <= 1:
            return 0
        
        sorted_values = sorted(values)
        n = len(values)
        cumsum = 0
        for i, value in enumerate(sorted_values):
            cumsum += (n - i) * value
        
        return (2 * cumsum) / (n * sum(values)) - (n + 1) / n

    def get_utilitarian_utility(self):
        """Calculate utility with weights inversely proportional to agent wealth."""
        agent_coins = {
            agent.agent_id: max(1e-8, agent.inventory["coins"])
            for agent in self.env.mobile_agents
            if agent.agent_id != self.agent_id
        }
        
        agent_utilities = {}
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            utility = agent.get_utility()
            if utility is not None:
                agent_utilities[agent.agent_id] = utility

        # If no valid utilities, return 0
        if not agent_utilities:
            return 0

        inverse_weights = {
            agent_id: 1.0 / coins
            for agent_id, coins in agent_coins.items()
            if agent_id in agent_utilities 
        }
        
        total_inverse_weight = sum(inverse_weights.values())
        if total_inverse_weight > 0:
            normalized_weights = {
                agent_id: weight / total_inverse_weight
                for agent_id, weight in inverse_weights.items()
            }
        else:
            normalized_weights = {
                agent_id: 1.0 / len(agent_utilities)
                for agent_id in agent_utilities.keys()
            }
        
        return sum(
            normalized_weights[agent_id] * agent_utilities[agent_id]
            for agent_id in agent_utilities.keys()
        )
        
    def get_nash_welfare_utility(self):
        """Calculate Nash welfare utility based on coins and equality."""
        coins = [
            agent.inventory["coins"] 
            for agent in self.env.mobile_agents 
            if agent.agent_id != self.agent_id
        ]
        n_agents = len(coins)

        if n_agents <= 1:
            return sum(coins)
        
        gini = self.calculate_gini(coins)
        equality = 1 - (n_agents / (n_agents - 1)) * gini
        productivity = sum(coins)

        return equality * productivity
    
    def get_current_state(self):
        """Capture current economic state before making a decision."""
        agent_incomes = [agent.get_income() for agent in self.env.mobile_agents]
        
        return {
            'tax_rates': self.tax_rates.copy(),
            'agent_incomes': agent_incomes,
            'utility': self.utility_function()
        }
    
    def store_decision(self, prev_state, tax_rates, valid_response):
        """Store the decision made by the planner."""
        action_str = f"Tax rates: {[f'{rate:.3f}' for rate in tax_rates]}"
        if not valid_response:
            action_str += " (INVALID RESPONSE - using random rates)"
            
        self.decision_history.append({
            'state': prev_state,
            'action': action_str,
            'outcome': None 
        })
        
        if len(self.decision_history) > self.memory_turns:
            self.decision_history.pop(0)
    
    def update_last_decision_outcome(self):
        """Update the outcome of the last decision after seeing its results."""
        if not self.decision_history:
            return
            
        last_decision = self.decision_history[-1]
        prev_state = last_decision['state']
        current_state = self.get_current_state()
        
        # Calculate utility change (the only thing the planner cares about)
        utility_change = current_state['utility'] - prev_state['utility']
        
        last_decision['outcome'] = f"utility {utility_change:+.2f}"
    
    def format_decision_history(self):
        """Format decision history for the prompt."""
        if not self.decision_history:
            return "No previous fiscal policy decisions.\n"
            
        history = ["RECENT FISCAL POLICY DECISIONS:"]
        for i, decision in enumerate(self.decision_history):
            state = decision['state']
            turn_str = f"Year {i+1}: Tax rates {[f'{rate:.3f}' for rate in state['tax_rates']]}, Agent incomes {[f'{inc:.1f}' for inc in state['agent_incomes']]}\n"
            turn_str += f"Policy: {decision['action']}"
            if decision['outcome']:
                turn_str += f" -> {decision['outcome']}"
            history.append(turn_str)
            
        return "\n".join(history) + "\n"
        