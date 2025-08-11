from environments.env import EconomyEnv
import numpy as np
import os

# Set numpy to print full arrays without truncation
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

config = {
    'map_path': 'maps/env-pure_and_mixed-15x15.txt',
    'map_size': (15, 15),
    'wood_regen_prob': 0.5,
    'stone_regen_prob': 0.5,
    'build_payout_min': 10,
    'build_payout_max': 20,
    'build_payout_multiplier': 1.5,
    'risk_aversion': 0.5,
    'n_agents': 10,
    'discount_factor': 0.95,
    'action_mechanism': "random",
    'episode_length': 1000,
    'plot_path': 'plots',
    'gather_skill_range': (0.0, 1.0),
    'move_labour': 0.21,
    'build_labour': 2.1,
    'trade_labour': 0.05,
    'gather_labour': 0.21,
    'max_order_lifetime': 50,
    'max_order_price': 1000,
    'llm': {
        'type': 'openai',
        'model': 'gpt-4o-mini',
        'temperature': 0.5,
        'api_key': os.getenv('OPENAI_API_KEY'),
        'memory_turns': 10,
        'log_dir': 'logs',
        'log_file': 'llm_conversation.txt' 
    },
    'agent_view_size': 10
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