#!/usr/bin/env python3

import argparse
import sys
import os
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.env import EconomyEnv
from scripts.rl_training_config import (
    get_level_1_training_config,
    get_level_2_training_config, 
    get_level_3_training_config,
    create_episode_config
)


def calculate_discounted_returns(training_examples: List[Dict[str, Any]], gamma: float = 0.95) -> List[Dict[str, Any]]:
    """
    Calculate discounted future returns for training examples.
    Works backwards from the end of the episode.
    
    Args:
        training_examples: List of training examples in chronological order
        gamma: Discount factor (0 < gamma <= 1)
        
    Returns:
        List of training examples with added 'discounted_return' field
    """
    if not training_examples:
        return training_examples
    
    # Sort by step to ensure chronological order
    sorted_examples = sorted(training_examples, key=lambda x: x.get('step', 0))
    
    # Calculate discounted returns working backwards
    returns = []
    running_return = 0.0
    
    # Work backwards through the episode
    for example in reversed(sorted_examples):
        immediate_reward = example.get('reward', 0.0)
        running_return = immediate_reward + gamma * running_return
        returns.append(running_return)
    
    # Reverse to get chronological order
    returns = list(reversed(returns))
    
    # Add discounted returns to examples
    enhanced_examples = []
    for i, example in enumerate(sorted_examples):
        enhanced_example = example.copy()
        enhanced_example['discounted_return'] = returns[i]
        enhanced_examples.append(enhanced_example)
    
    return enhanced_examples

class TrainingDataCollector:
    """Manages collection of training data for any level"""
    
    def __init__(self, training_config: Dict[str, Any]):
        self.config = training_config
        self.output_dir = training_config['data_collection']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.episode_summary = []
        self.summary_file = os.path.join(self.output_dir, "episode_summary.json")
        
        # Load existing summary if it exists
        if os.path.exists(self.summary_file):
            with open(self.summary_file, 'r') as f:
                self.episode_summary = json.load(f)
        
        print(f"Initialized Level {self.config['level']} Data Collector")
        print(f"Output: {self.output_dir}\n")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met based on LLM type"""
        llm_type = self.config['llm']['type'].lower()
        
        if llm_type == 'ollama':
            return self._check_ollama()
        elif llm_type == 'openai':
            return self._check_openai()
        else:
            print(f"No prerequisite checks needed for LLM type: {llm_type}")
            return True
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            import requests
            base_url = self.config['llm'].get('base_url', 'http://localhost:11434')
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama is ready\n")
                return True
        except Exception as e:
            pass
        
        base_url = self.config['llm'].get('base_url', 'http://localhost:11434')
        print(f"Ollama not running at {base_url}")
        print("Please start it with: ollama serve\n")
        return False
    
    def _check_openai(self) -> bool:
        """Check if OpenAI API key is available"""
        import os
        
        # Check for API key in config or environment
        api_key = self.config['llm'].get('api_key') or os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("OpenAI API key not found")
            print("Set OPENAI_API_KEY environment variable or add 'api_key' to config\n")
            return False
        
        print("OpenAI API key found\n")
        return True
    
    def run_episode(self, episode_id: int) -> bool:
        """Run a single episode and collect data"""
        try:
            # Create episode configuration
            episode_config = create_episode_config(self.config, episode_id)
            
            # Record episode start
            episode_info = {
                'episode_id': episode_id,
                'experiment_name': episode_config['experiment_name'],
                'start_time': datetime.now().isoformat(),
                'level': self.config['level'],
                'status': 'running',
                'log_file': f"logs/{episode_config['experiment_name']}/mobile_agent_0.txt",
                'plot_dir': f"plots/{episode_config['experiment_name']}/"
            }
            
            self.episode_summary.append(episode_info)
            self._save_summary()
            
            print(f"\n\nStarting Episode {episode_id}: {episode_config['experiment_name']}")
            
            # Create and run environment
            env = EconomyEnv(episode_config)
            agents = env.run_economy()
            
            # Record successful completion
            self._record_completion(episode_info, agents, success=True)
            
            return True
            
        except Exception as e:
            print(f"\n\nEpisode {episode_id} failed: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            
            # Record failure
            self._record_completion(episode_info, [], success=False, error=str(e))
            
            return False
    
    def _record_completion(self, episode_info: Dict, agents: List, success: bool, error: str = None):
        """Record episode completion"""
        episode_info['end_time'] = datetime.now().isoformat()
        episode_info['status'] = 'completed' if success else 'failed'
        
        if error:
            episode_info['error'] = error
        
        if success and agents:
            episode_info['agent_metrics'] = {}
            episode_info['summary_metrics'] = {}
            
            total_houses = 0
            total_utility = 0
            total_coins = 0
            
            for i, agent in enumerate(agents):
                agent_metrics = {
                    'agent_id': i,
                    'total_houses_built': getattr(agent, 'total_houses_built', 0),
                    'final_utility': agent.utility[-1] if hasattr(agent, 'utility') and agent.utility else 0,
                    'final_inventory': getattr(agent, 'inventory', {}),
                    'final_coins': getattr(agent, 'inventory', {}).get('coins', 0),
                    'episode_length': len(agent.utility) if hasattr(agent, 'utility') and agent.utility else 0,
                    'training_examples': getattr(agent, 'training_examples', [])
                }
                
                episode_info['agent_metrics'][f'agent_{i}'] = agent_metrics
                
                total_houses += agent_metrics['total_houses_built']
                total_utility += agent_metrics['final_utility']
                total_coins += agent_metrics['final_coins']
            
            gamma = self.config.get('rl_training', {}).get('discount_factor', 0.95)
            for agent_key, agent_data in episode_info['agent_metrics'].items():
                training_examples = agent_data.get('training_examples', [])
                if training_examples:
                    enhanced_examples = calculate_discounted_returns(training_examples, gamma)
                    agent_data['training_examples'] = enhanced_examples
            
            num_agents = len(agents)
            episode_info['summary_metrics'] = {
                'num_agents': num_agents,
                'total_houses_all_agents': total_houses,
                'avg_houses_per_agent': total_houses / num_agents,
                'avg_utility_per_agent': total_utility / num_agents,
                'avg_coins_per_agent': total_coins / num_agents
            }
        
        self._save_summary()
        
        print(f"\nEpisode {episode_info['episode_id']} completed")
        if success and agents:
            summary = episode_info.get('summary_metrics', {})
            if summary:
                print(f"   {summary['num_agents']} agents")
                print(f"   Total houses: {summary['total_houses_all_agents']}")
                print(f"   Avg houses/agent: {summary['avg_houses_per_agent']:.1f}")
                print(f"   Avg utility/agent: {summary['avg_utility_per_agent']:.1f}")
                
                total_training_examples = sum(
                    len(agent_data.get('training_examples', [])) 
                    for agent_data in episode_info.get('agent_metrics', {}).values()
                )
                
                if total_training_examples > 0:
                    all_rewards = []
                    all_returns = []
                    for agent_data in episode_info.get('agent_metrics', {}).values():
                        for example in agent_data.get('training_examples', []):
                            all_rewards.append(example.get('reward', 0))
                            all_returns.append(example.get('discounted_return', 0))
                    
                    if all_rewards:
                        avg_reward = sum(all_rewards) / len(all_rewards)
                        avg_return = sum(all_returns) / len(all_returns)
                        print(f"   Training examples: {total_training_examples}")
                        print(f"   Avg reward: {avg_reward:.3f}, Avg discounted return: {avg_return:.3f}")
                else:
                    print(f"   Training examples: {total_training_examples}")
    
    def collect_data(self, num_episodes: int, start_from: int = 0) -> Dict[str, Any]:
        """Collect training data by running multiple episodes"""
        
        print(f"Starting Level {self.config['level']} Data Collection")
        print(f"Episodes to run: {num_episodes}")
        print(f"Starting from episode: {start_from}")
        
        if not self.check_prerequisites():
            return {'success': False, 'error': 'Prerequisites not met'}
        
        # Run episodes
        successful_episodes = 0
        failed_episodes = 0
        
        for i in range(start_from, start_from + num_episodes):
            print(f"\nRunning Episode {i+1}/{start_from + num_episodes}")
            
            success = self.run_episode(i)
            
            if success:
                successful_episodes += 1
            else:
                failed_episodes += 1
            
            # Print progress every 5 episodes
            if (i + 1) % 5 == 0:
                self._print_summary()
        
        # Final results
        results = {
            'success': True,
            'total_episodes': num_episodes,
            'successful_episodes': successful_episodes,
            'failed_episodes': failed_episodes,
            'success_rate': successful_episodes / num_episodes if num_episodes > 0 else 0,
            'output_dir': self.output_dir,
            'episode_files': self.get_completed_episode_files()
        }
        
        print("\nData collection complete.")
        self._print_summary()
        
        return results
    
    def get_completed_episode_files(self) -> List[Dict[str, Any]]:
        """Get list of completed episode log files"""
        completed_episodes = []
        for ep in self.episode_summary:
            if ep['status'] == 'completed' and os.path.exists(ep['log_file']):
                completed_episodes.append(ep)
        return completed_episodes
    
    def _save_summary(self):
        """Save episode summary to file"""
        with open(self.summary_file, 'w') as f:
            json.dump(self.episode_summary, f, indent=2)
    
    def _print_summary(self):
        """Print collection progress summary"""
        total = len(self.episode_summary)
        completed = len([ep for ep in self.episode_summary if ep['status'] == 'completed'])
        failed = len([ep for ep in self.episode_summary if ep['status'] == 'failed'])
        
        completed_episodes = [ep for ep in self.episode_summary if ep['status'] == 'completed']
        
        if completed_episodes:
            # Use new summary_metrics structure
            total_houses_across_episodes = 0
            total_avg_utility = 0
            
            for ep in completed_episodes:
                summary_metrics = ep.get('summary_metrics', {})
                if summary_metrics:
                    total_houses_across_episodes += summary_metrics.get('total_houses_all_agents', 0)
                    total_avg_utility += summary_metrics.get('avg_utility_per_agent', 0)
                else:
                    # Fallback to old structure if it exists
                    old_metrics = ep.get('final_metrics', {})
                    total_houses_across_episodes += old_metrics.get('total_houses_built', 0)
                    total_avg_utility += old_metrics.get('final_utility', 0)
            
            avg_houses = total_houses_across_episodes / len(completed_episodes)
            avg_utility = total_avg_utility / len(completed_episodes)
        else:
            avg_houses = 0
            avg_utility = 0
        
        print("\n" + "="*50)
        print(f"Level {self.config['level']} collection summary")
        print("="*50)
        print(f"Total Episodes: {total}")
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {completed/total:.1%}" if total > 0 else "Success Rate: 0%")
        print(f"Avg Houses/Episode: {avg_houses:.1f}")
        print(f"Avg Final Utility: {avg_utility:.1f}")
        print("="*50)

def get_training_config(level: int) -> Dict[str, Any]:
    """Get training configuration for specified level"""
    config_map = {
        1: get_level_1_training_config,
        2: get_level_2_training_config,
        3: get_level_3_training_config
    }
    
    if level not in config_map:
        raise ValueError(f"Invalid level: {level}. Must be 1-3 (Level 4 not implemented yet).")
    
    return config_map[level]()

def main():
    parser = argparse.ArgumentParser(description='Collect RL training data')
    parser.add_argument('--level', type=int, required=True, choices=[1, 2, 3],
                       help='Training level (1=single agent, 2=multi-agent, 3=government)')
    parser.add_argument('--start-from', type=int, default=0,
                       help='Episode number to start from (default: 0)')
    
    args = parser.parse_args()
    
    # Get training configuration
    training_config = get_training_config(args.level)
    
    # All settings come from config - no command line overrides
    
    # Create collector and run
    collector = TrainingDataCollector(training_config)
    results = collector.collect_data(
        training_config['data_collection']['num_episodes'], 
        args.start_from
    )
    
    if not results['success']:
        sys.exit(1)

if __name__ == "__main__":
    main()
