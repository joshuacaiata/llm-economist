def get_llm_config(llm_type='ollama'):
    """
    Get LLM configuration for different providers
    """
    configs = {
        'ollama': {
            'type': 'ollama',
            'model': 'qwen2.5:1.5b',
            'temperature': 1.0,  # High for exploration during data collection
            'base_url': 'http://localhost:11434',
            'memory_turns': 100,
            'agent_action_mechanism': 'llm'
        },
        'openai': {
            'type': 'openai',
            'model': 'gpt-4o-mini',
            'temperature': 1.0,  # High for exploration during data collection
            'api_key': None,  # Will use OPENAI_API_KEY env var
            'memory_turns': 100,
            'agent_action_mechanism': 'llm'
        },
        'anthropic': {
            'type': 'anthropic',
            'model': 'claude-3-haiku',
            'temperature': 1.0,
            'api_key': None,  # Will use ANTHROPIC_API_KEY env var
            'memory_turns': 100,
            'agent_action_mechanism': 'llm'
        }
    }
    
    if llm_type not in configs:
        raise ValueError(f"Unknown LLM type: {llm_type}. Available: {list(configs.keys())}")
    
    return configs[llm_type]

def get_level_1_training_config():
    """
    Level 1: Single Agent Training Configuration
    All parameters configurable here - no hardcoded values elsewhere
    """
    return {
        # Training metadata
        'level': 1,
        'level_name': 'single_agent',
        'description': 'Single agent learning resource collection and building',
        
        # Data collection settings
        'data_collection': {
            'num_episodes': 2,
            'start_episode': 0,
            'output_dir': 'training_data/level_1',
            'base_experiment_name': 'level_1_single_agent',
            'episode_length': 30,
            'save_summary': True
        },
        
        # Environment configuration
        'environment': {
            'n_agents': 1,
            'planner': False,
            'trading_system': False,
            'map_path': 'maps/env-pure_and_mixed-15x15.txt',
            'map_size': (15, 15),
            'wood_regen_prob': 1.0,
            'stone_regen_prob': 1.0,
            'build_payout_min': 0,
            'build_payout_max': 100,
            'build_payout_multiplier': 1.5,
            'discount_factor': 0.95,
            'risk_aversion': 0.5,
            'gather_skill_range': (0.0, 1.0),
            'agent_view_size': 10,
            'plot_path': 'plots'
        },
        
        # Labour costs - encourage active behavior
        'labour_costs': {
            'move_labour': 0.21,
            'build_labour': 2.1,
            'trade_labour': 0.05,
            'gather_labour': 0.21,
            'nothing_labour': 10  # High penalty for inaction
        },
        
        # LLM settings - change get_llm_config() to switch types
        'llm': get_llm_config('ollama'),
        
        # RL training parameters
        'rl_training': {
            'discount_factor': 0.95,
            'balance_dataset': True,
            'oversample_builds': 3,  # Multiply build actions by this factor
            'filter_negative_returns': True,
            'min_episodes_for_training': 10
        },
        
        # Fixed environment settings (rarely changed)
        'fixed_settings': {
            'max_order_lifetime': 50,
            'max_order_price': 1000,
            'year_length': 10,
            'tax_brackets': [11, 47, 100, 191, 243, 609],
            'default_tax_rates': [0.0, 0.1, 0.48, 0.52, 0.88, 0.92, 0.99],
            'fixed_tax_rates': True,
            'planner_utility_type': "utilitarian",
            'planner_memory_turns': 5,
            'collect_training_data': False
        }
    }

def get_level_2_training_config():
    """
    Level 2: Multi-Agent Training Configuration
    Inherits from Level 1 and modifies for multi-agent trading
    """
    config = get_level_1_training_config()
    
    # Update for multi-agent
    config.update({
        'level': 2,
        'level_name': 'multi_agent_trading',
        'description': 'Multiple agents learning to trade and compete',
    })
    
    # Modify specific settings
    config['data_collection'].update({
        'base_experiment_name': 'level_2_multi_agent',
        'output_dir': 'training_data/level_2',
        'num_episodes': 15  # Fewer episodes due to complexity
    })
    
    config['environment'].update({
        'n_agents': 4,  # Multiple agents
        'trading_system': True  # Enable trading
    })
    
    return config

def get_level_3_training_config():
    """
    Level 3: Government Training Configuration  
    Multi-agent + Planner
    """
    config = get_level_2_training_config()
    
    config.update({
        'level': 3,
        'level_name': 'government_economy',
        'description': 'Agents learning to adapt to government taxation',
    })
    
    config['data_collection'].update({
        'base_experiment_name': 'level_3_government',
        'output_dir': 'training_data/level_3',
        'num_episodes': 10  # Even fewer due to complexity
    })
    
    config['environment'].update({
        'planner': True  # Enable government
    })
    
    return config


def create_episode_config(training_config, episode_id):
    """
    Create a single episode configuration from training config
    """
    tc = training_config
    
    # Build the episode config
    episode_config = {
        'experiment_name': f"{tc['data_collection']['base_experiment_name']}_ep_{episode_id}",
        'episode_length': tc['data_collection']['episode_length'],
        
        # Environment settings
        **tc['environment'],
        
        # Labour costs
        **tc['labour_costs'],
        
        # LLM settings
        'llm': tc['llm'].copy(),
        'agent_action_mechanism': tc['llm']['agent_action_mechanism'],
        'planner_action_mechanism': tc['llm']['agent_action_mechanism'],
        
        # Fixed settings
        **tc['fixed_settings']
    }
    
    return episode_config
