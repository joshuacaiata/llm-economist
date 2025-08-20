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


    def observe(self):
        """
        1. Bids, asks, traded prices, number of trades per resource (wood and stone)
        2. Agent inventories (coins, wood, stone)
        3. Current tax rates, previous year's income, tax brackets
        4. Current time step and time to next tax year
        """
        pass

    def step(self):
        if self.action_mechanism not in ["llm", "random"]:
            raise ValueError(f"Invalid planner action mechanism: {self.action_mechanism}")
            
        if self.action_mechanism == "llm":
            prompt = self.observe()
            tax_rates = self.get_tax_rates(prompt)
        else:  # random
            tax_rates = self.get_tax_rates()
        self.execute_action(tax_rates)
        self.utility_history.append(self.utility_function())

    def reset(self):
        self.tax_rates = self.original_tax_rates
        self.utility_history = []

    def get_tax_rates(self, prompt=None):
        if self.fixed_tax_rates:
            return self.tax_rates
        
        if prompt is None:
            # set tax rates at random, each between 0 and 1
            tax_rates = np.random.uniform(0, 1, self.num_tax_brackets)
            return tax_rates
        else:
            # TODO: after getting LLM prompt and response
            pass
    
    def execute_action(self, tax_rates):
        # collect taxes from the agent based on previous tax rates
        # then update the tax rates
        collected_taxes = 0
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            
            # get the agents income
            income = agent.get_income()


            # calculate the taxes owed using tax brackets 
            taxes_owed = self.calculate_taxes_owed(income, self.tax_rates)

            # collect the taxes from the agent
            to_collect = min(taxes_owed, agent.inventory['coins'])
            agent.inventory['coins'] -= to_collect
            collected_taxes += to_collect

        # update the tax rates
        self.tax_rates = tax_rates
        self.tax_collection_history.append(collected_taxes)
        self.tax_bracket_history.append(self.tax_rates)
        
        # disburse the collected taxes to the agents equally
        # TODO: not sure if there is other ways to redistribute the income
        for agent in self.env.mobile_agents:
            if agent.agent_id == self.agent_id:
                continue
            agent.inventory['coins'] += collected_taxes / (len(self.env.mobile_agents) - 1)

    
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
        