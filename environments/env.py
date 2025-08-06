import numpy as np
import random
from agents.mobile_agent import MobileAgent

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

        self.map = self.generate_map()

        self.n_agents = config['n_agents']
        self.mobile_agents = []
        self.labour_disutilities = config['labour_disutilities']
        for i in range(self.n_agents):
            if isinstance(self.labour_disutilities, list):
                labour_disutility = self.labour_disutilities[i]
            else:
                labour_disutility = self.labour_disutilities
            self.mobile_agents.append(
                MobileAgent(
                    "MobileAgent",
                    i,
                    config,
                    self.build_payout_multiplier * (
                        random.randint(
                            self.build_payout_min, 
                            self.build_payout_max)
                    ),
                    self.risk_aversion,
                    self,
                    labour_disutility,
                    None # TODO: pass in an instance of the llm
                )
            )
        
        self.initial_agent_positions = self.initialize_agents()
        self.current_agent_positions = self.initial_agent_positions.copy()

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

    def reset_year(self):
        # For each agent, call reset year
        shuffled_agents = random.sample(self.mobile_agents, len(self.mobile_agents))

        for agent in shuffled_agents:
            agent.reset_year()

    def reset_env(self, randomize_agent_positions=False):
        # Undo all houses (set to 0)
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
        self.trading_system.reset_episode()