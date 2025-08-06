from environments.env import EconomyEnv
import numpy as np

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
    'n_agents': 1,
    'discount_factor': 0.95,
    'labour_disutilities': 0
}

env = EconomyEnv(config)

print(env.current_agent_positions)