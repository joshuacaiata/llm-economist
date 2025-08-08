import matplotlib.pyplot as plt
import os


class PlottingService:
    """Service class to handle all plotting functionality for agent metrics."""
    
    def __init__(self, plot_path=None):
        """
        Initialize the plotting service.
        
        Args:
            plot_path (str, optional): Directory to save plots. If None, plots won't be saved.
        """
        self.plot_path = plot_path
        self.metrics_to_plot = ['coins', 'wood', 'stone', 'utility', 'houses_built']
    
    def plot_agent_metrics(self, agents, config=None):
        """
        Plot and save agent metrics over time.
        
        Args:
            agents (list): List of agent objects with metrics_history
            config (dict, optional): Configuration dictionary containing plot_path
        """
        # Determine plot path
        plot_path = self._get_plot_path(config)
        if not plot_path:
            print("Warning: No plot path specified, skipping plotting")
            return
            
        # Create directory if it doesn't exist
        os.makedirs(plot_path, exist_ok=True)
        
        # Create individual metric plots
        self._create_individual_plots(agents, plot_path)
        
        # Create summary dashboard
        self._create_summary_plot(agents, plot_path)
        
        print(f"All agent metrics plots saved successfully to {plot_path}!")
    
    def _get_plot_path(self, config):
        """Determine the plot path from various sources."""
        if self.plot_path:
            return self.plot_path
        elif config and 'plot_path' in config:
            return config['plot_path']
        else:
            return None
    
    def _create_individual_plots(self, agents, plot_path):
        """Create individual plots for each metric."""
        for metric in self.metrics_to_plot:
            plt.figure(figsize=(10, 6))
            
            for agent in agents:
                if hasattr(agent, 'metrics_history') and metric in agent.metrics_history:
                    time_steps = range(len(agent.metrics_history[metric]))
                    plt.plot(time_steps, agent.metrics_history[metric], 
                            label=f'Agent {agent.agent_id}', linewidth=2)
            
            plt.xlabel('Time Step')
            plt.ylabel(self._format_metric_name(metric))
            plt.title(f'Agent {self._format_metric_name(metric)} Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            filename = f'agent_{metric}.png'
            filepath = os.path.join(plot_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved {filename}")
    
    def _create_summary_plot(self, agents, plot_path):
        """Create a summary plot with all metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(self.metrics_to_plot):
            ax = axes[i]
            
            for agent in agents:
                if hasattr(agent, 'metrics_history') and metric in agent.metrics_history:
                    time_steps = range(len(agent.metrics_history[metric]))
                    ax.plot(time_steps, agent.metrics_history[metric], 
                           label=f'Agent {agent.agent_id}', linewidth=2)
            
            ax.set_xlabel('Time Step')
            ax.set_ylabel(self._format_metric_name(metric))
            ax.set_title(f'Agent {self._format_metric_name(metric)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot if we have an odd number of metrics
        if len(self.metrics_to_plot) < len(axes):
            axes[-1].set_visible(False)
            
        plt.tight_layout()
        summary_filepath = os.path.join(plot_path, 'agent_metrics_summary.png')
        plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved agent_metrics_summary.png")
    
    def _format_metric_name(self, metric):
        """Format metric name for display."""
        return metric.replace('_', ' ').title()
    
    def plot_specific_metric(self, agents, metric, config=None, filename=None):
        """
        Plot a specific metric for all agents.
        
        Args:
            agents (list): List of agent objects
            metric (str): Metric name to plot
            config (dict, optional): Configuration dictionary
            filename (str, optional): Custom filename for the plot
        """
        plot_path = self._get_plot_path(config)
        if not plot_path:
            print("Warning: No plot path specified, skipping plotting")
            return
            
        os.makedirs(plot_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        for agent in agents:
            if hasattr(agent, 'metrics_history') and metric in agent.metrics_history:
                time_steps = range(len(agent.metrics_history[metric]))
                plt.plot(time_steps, agent.metrics_history[metric], 
                        label=f'Agent {agent.agent_id}', linewidth=2)
        
        plt.xlabel('Time Step')
        plt.ylabel(self._format_metric_name(metric))
        plt.title(f'Agent {self._format_metric_name(metric)} Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if filename is None:
            filename = f'agent_{metric}.png'
        filepath = os.path.join(plot_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {filename}")
    
    def set_plot_path(self, plot_path):
        """Set the plot path for this service."""
        self.plot_path = plot_path
    
    def add_custom_metric(self, metric_name):
        """Add a custom metric to be plotted."""
        if metric_name not in self.metrics_to_plot:
            self.metrics_to_plot.append(metric_name)