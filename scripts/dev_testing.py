from environments.env import EconomyEnv
import numpy as np
import os

# Set numpy to print full arrays without truncation
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

config = {
    'experiment_name': 'dev_testing',
    'map_path': 'maps/env-pure_and_mixed-15x15.txt',
    'map_size': (15, 15),
    'wood_regen_prob': 1.0,
    'stone_regen_prob': 1.0,
    'build_payout_min': 0,
    'build_payout_max': 100,
    'build_payout_multiplier': 1.5,
    'risk_aversion': 0.5,
    'n_agents': 2, 
    'discount_factor': 0.95,
    'agent_action_mechanism': "llm",
    'planner_action_mechanism': "llm",
    'episode_length': 100,
    'plot_path': 'plots',
    'gather_skill_range': (0.0, 1.0),
    'move_labour': 0.21,
    'build_labour': 2.1,
    'trade_labour': 0.05,
    'gather_labour': 0.21,
    'nothing_labour': 10,
    'max_order_lifetime': 50,
    'max_order_price': 1000,
    'llm': {
        'type': 'ollama',
        'model': 'qwen2.5:1.5b',
        'temperature': 1.0,
        'base_url': 'http://localhost:11434',
        'memory_turns': 100
    },
    'agent_view_size': 10,

    
    "planner": False,
    'tax_brackets': [11, 47, 100, 191, 243, 609], #[11601, 47151, 100526, 191951, 243726, 609351],
    'default_tax_rates': [0.0, 0.1, 0.48, 0.52, 0.88, 0.92, 0.99],
    'fixed_tax_rates': False, 
    'year_length': 10,
    'planner_utility_type': "utilitarian",
    'planner_memory_turns': 10,
    'trading_system': True
}

env = EconomyEnv(config)

print("Initial agent positions:", env.current_agent_positions)

# Run the economy
print(f"Running economy for {config['episode_length']} steps...")
agents = env.run_economy()

# Plot and save metrics
print("Generating plots...")
env.plot_agent_metrics()

# Print final stats
print("\nFinal agent stats:")
for agent in agents:
    print(f"Agent {agent.agent_id}:")
    print(f"  Inventory: {agent.inventory}")
    print(f"  Houses built: {agent.total_houses_built}")
    print(f"  Final utility: {agent.utility[-1]:.2f}")
    print()