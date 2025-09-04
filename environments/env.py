import numpy as np
import random
from agents.mobile_agent import MobileAgent
from agents.planner_agent import PlannerAgent
from services.plotting_service import PlottingService
from services.trading_system import TradingSystem
from services.llm_services import create_llm

class EconomyEnv:
    def __init__(self, config: dict):
        self.config = config
        
        self.map_path = config['map_path']
        self.map_size = config['map_size']
        self.wood_regen_prob = config['wood_regen_prob']
        self.stone_regen_prob = config['stone_regen_prob']

        self.build_payout_min = config['build_payout_min']
        self.build_payout_max = config['build_payout_max']
        self.build_payout_multiplier = config['build_payout_multiplier']
        self.risk_aversion = config['risk_aversion']
        self.discount_factor = config['discount_factor']
        self.agent_action_mechanism = config['agent_action_mechanism']
        self.planner_action_mechanism = config['planner_action_mechanism']
        self.gather_skill_range = config['gather_skill_range']
    
        self.move_labour = config['move_labour']
        self.build_labour = config['build_labour']
        self.trade_labour = config['trade_labour']
        self.gather_labour = config['gather_labour']
        self.max_order_price = config['max_order_price']

        self.year_length = config['year_length']
        self.planner_enabled = config.get('planner', True)  
        self.map = self.generate_map()

        self.n_agents = config['n_agents']
        self.mobile_agents = []

        for i in range(self.n_agents):
            llm = None
            if self.agent_action_mechanism == "llm":
                llm = create_llm(config['llm'], agent_type="mobile", agent_id=i, experiment_name=config.get('experiment_name'))

            build_payout = self.build_payout_multiplier * (
                random.randint(
                    self.build_payout_min, 
                    self.build_payout_max)
            )

            self.mobile_agents.append(
                MobileAgent(
                    "MobileAgent",
                    i,
                    config,
                    build_payout,
                    self.risk_aversion,
                    self,
                    llm,
                    self.agent_action_mechanism,
                    random.uniform(self.gather_skill_range[0], self.gather_skill_range[1])
                )
            )
        
        if self.planner_enabled:
            planner_llm = None
            if self.planner_action_mechanism == "llm":
                planner_llm = create_llm(config['llm'], agent_type="planner", experiment_name=config.get('experiment_name'))
            
            self.planner = PlannerAgent(
                "PlannerAgent",
                self.n_agents,
                config,
                self,
                planner_llm,
                self.planner_action_mechanism
            )
        else:
            self.planner = None

        self.initial_agent_positions = self.initialize_agents()
        self.current_agent_positions = self.initial_agent_positions.copy()

        self.episode_length = config['episode_length']

        self.trading_system = TradingSystem(config, self) if config.get('trading_system', False) else None
        
        self.plotting_service = PlottingService()

        self.time = 0

    @property
    def time_to_next_tax_year(self):
        """Calculate steps until next tax collection."""
        if not self.planner_enabled:
            return None
        return self.year_length - (self.time % self.year_length)

    def generate_map(self):
        map_dict = {
            "Wood": np.zeros(self.map_size, dtype=int),
            "Stone": np.zeros(self.map_size, dtype=int),
            "Water": np.zeros(self.map_size, dtype=int),
            "Buildable": np.ones(self.map_size, dtype=int),
            "Houses": np.zeros(self.map_size, dtype=int),
        }

        # Read the map file
        with open(self.map_path, 'r') as f:
            content = f.read()
            map_lines = content.split(';')[:-1]

        self.wood_tiles = []
        self.stone_tiles = []
        self.water_tiles = []
        
        # Process each line of the map
        for row, line in enumerate(map_lines):
            for col, char in enumerate(line):
                if char == 'W':
                    # Wood with probability
                    if random.random() < self.wood_regen_prob:
                        map_dict["Wood"][row, col] = 1
                    
                    self.wood_tiles.append((row, col))
                    map_dict["Buildable"][row, col] = 0
                
                elif char == 'S':
                    # Stone with probability
                    if random.random() < self.stone_regen_prob:
                        map_dict["Stone"][row, col] = 1
                    
                    self.stone_tiles.append((row, col))
                    map_dict["Buildable"][row, col] = 0
                
                elif char == '@':
                    # Water is always blocking
                    map_dict["Water"][row, col] = 1
                    map_dict["Buildable"][row, col] = 0
                    self.water_tiles.append((row, col))
        
        return map_dict
    
    def initialize_agents(self):
        valid_positions = []
        agent_initial_positions = {}
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.map["Water"][i, j] == 0:
                    valid_positions.append((i, j))
        
        if len(valid_positions) < self.n_agents:
            raise ValueError(f"Not enough valid positions for {self.n_agents} agents")
        
        selected_positions = random.sample(valid_positions, self.n_agents)
        
        for i, agent in enumerate(self.mobile_agents):
            agent.position = selected_positions[i]
            agent_initial_positions[agent.agent_id] = agent.position
            assert self.map["Water"][agent.position[0], agent.position[1]] == 0

        return agent_initial_positions

    def regenerate_tiles(self):
        for i, (row, col) in enumerate(self.wood_tiles):
            if self.map["Wood"][row, col] == 0 and random.random() < self.wood_regen_prob:
                self.map["Wood"][row, col] = 1

        for i, (row, col) in enumerate(self.stone_tiles):
            if self.map["Stone"][row, col] == 0 and random.random() < self.stone_regen_prob:
                self.map["Stone"][row, col] = 1

    def step(self):
        for agent in self.mobile_agents:
            agent.step()
        
        if self.trading_system:
            self.trading_system.step()
            
        self.regenerate_tiles()

        if self.time > 0 and self.time % self.year_length == 0 and self.planner_enabled:
            self.planner.step()

        self.time += 1


    def reset_env(self, randomize_agent_positions=False):
        self.map["Houses"] = np.zeros(self.map_size, dtype=int)

        # Clear all agents inventory
        for agent in self.all_agents:
            agent.reset_episode()
    
        # if randomize_agent_positions, randomize the agent positions
        if randomize_agent_positions:
            self.initialize_agents()

        # otherwise, put them back to the initial positions
        else:
            for agent in self.mobile_agents:
                agent.position = self.agent_initial_positions[agent.agent_id]

        # clear all orders
        if self.trading_system:
            self.trading_system.reset_episode()
        self.time = 0

    def run_economy(self):
        from tqdm import tqdm
        for i in tqdm(range(self.episode_length), desc="Running simulation"):
            self.step()
            if self.time % self.year_length == 0 and self.time > 0:
                self.plot_agent_metrics()
            
        return self.mobile_agents

    def plot_agent_metrics(self):
        """Plot and save agent metrics over time using the plotting service."""
        self.plotting_service.plot_agent_metrics(
            self.mobile_agents, 
            self.config, 
            self.trading_system if self.config.get('trading_system', False) else None, 
            self
        )
        self.plotting_service.plot_map_state(self, self.mobile_agents, self.config)
    
    def plot_specific_metric(self, metric, filename=None):
        """Plot a specific metric using the plotting service."""
        self.plotting_service.plot_specific_metric(self.mobile_agents, metric, self.config, filename)