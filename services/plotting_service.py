import matplotlib.pyplot as plt
import os
import numpy as np


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
        self.env = None
    
    def plot_agent_metrics(self, agents, config=None, trading_system=None, env=None):
        """
        Plot and save agent metrics over time.
        
        Args:
            agents (list): List of agent objects with metrics_history
            config (dict, optional): Configuration dictionary containing plot_path
            trading_system (TradingSystem, optional): Trading system instance for market metrics
            env (EconomyEnv, optional): Environment instance for map visualization
        """
        self.env = env  # Store environment reference for map plotting
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
        self._create_summary_plot(agents, plot_path, trading_system)
        
        # Plot market metrics if trading system is provided
        if trading_system:
            self.plot_market_metrics(trading_system, config)
        
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
    
    def _create_summary_plot(self, agents, plot_path, trading_system=None):
        """Create a summary plot with all metrics including market data if available."""
        num_rows = 2
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
        axes = axes.flatten()
        
        # Plot agent metrics
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
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Plot market metrics if trading system is provided
        if trading_system:
            # Plot wood orders
            ax_wood = axes[5]
            time_steps = range(len(trading_system.order_history["wood"]["buy"]))
            buy_counts = trading_system.order_history["wood"]["buy"]
            sell_counts = trading_system.order_history["wood"]["sell"]
            
            ax_wood.plot(time_steps, buy_counts, label='Buy Orders', linewidth=2)
            ax_wood.plot(time_steps, sell_counts, label='Sell Orders', linewidth=2)
            ax_wood.set_xlabel('Time Step')
            ax_wood.set_ylabel('Number of Orders')
            ax_wood.set_title('Wood Market Orders')
            ax_wood.legend(loc='upper left')
            ax_wood.grid(True, alpha=0.3)
            
            # Plot stone orders
            ax_stone = axes[6]
            time_steps = range(len(trading_system.order_history["stone"]["buy"]))
            buy_counts = trading_system.order_history["stone"]["buy"]
            sell_counts = trading_system.order_history["stone"]["sell"]
            
            ax_stone.plot(time_steps, buy_counts, label='Buy Orders', linewidth=2)
            ax_stone.plot(time_steps, sell_counts, label='Sell Orders', linewidth=2)
            ax_stone.set_xlabel('Time Step')
            ax_stone.set_ylabel('Number of Orders')
            ax_stone.set_title('Stone Market Orders')
            ax_stone.legend(loc='upper left')
            ax_stone.grid(True, alpha=0.3)
            
            # Plot price history
            ax_prices = axes[7]
            for resource in ["wood", "stone"]:
                times = [trade[0] for trade in trading_system.trades[resource]]
                prices = [trade[1] for trade in trading_system.trades[resource]]
                
                if times and prices:  # Only plot if we have data
                    ax_prices.plot(times, prices, label=f'{resource.title()} Price', linewidth=2)
            
            ax_prices.set_xlabel('Time Step')
            ax_prices.set_ylabel('Price')
            ax_prices.set_title('Resource Prices')
            ax_prices.legend(loc='upper left')
            ax_prices.grid(True, alpha=0.3)
        
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
            
    def plot_map_state(self, env, agents, config=None):
        """Create a standalone map visualization."""
        plot_path = self._get_plot_path(config)
        if not plot_path:
            print("Warning: No plot path specified, skipping plotting")
            return
            
        os.makedirs(plot_path, exist_ok=True)
        
        # Create figure with space for legend below
        fig = plt.figure(figsize=(12, 10))
        ax = plt.gca()
        self._plot_map_state(env, agents, ax)
        
        # Adjust layout to make room for legend
        plt.subplots_adjust(bottom=0.2)  # Make space for legend at bottom
        
        # Save the plot
        filepath = os.path.join(plot_path, 'map_state.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved map_state.png")
        
    def _plot_map_state(self, env, agents, ax):
        """Plot the map state with agent positions and paths."""
        # Create a colormap for the map
        cmap = plt.cm.colors.ListedColormap(['white', 'blue'])  # white for normal, blue for water
        
        # Plot the base map (water)
        ax.imshow(env.map["Water"], cmap=cmap)
        
        # Plot houses with X markers
        house_positions = zip(*np.where(env.map["Houses"] == 1))
        for pos in house_positions:
            ax.plot(pos[1], pos[0], 'kX', markersize=10)  # Black X for houses
        
        # Plot agent paths with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(agents)))
        for agent, color in zip(agents, colors):
            # Plot starting position (triangle)
            start_pos = env.initial_agent_positions[agent.agent_id]
            ax.plot(start_pos[1], start_pos[0], marker='^', color=color, markersize=10, 
                   label=f'Agent {agent.agent_id} Start')
            
            # Plot movement path
            if hasattr(agent, 'movement_history') and agent.movement_history:
                path = np.array(agent.movement_history)
                ax.plot(path[:, 1], path[:, 0], '-', color=color, alpha=0.5, linewidth=2)
            
            # Plot current position (circle)
            end_pos = env.current_agent_positions[agent.agent_id]
            ax.plot(end_pos[1], end_pos[0], marker='o', color=color, markersize=10,
                   label=f'Agent {agent.agent_id} End')
        
        # Customize the plot
        ax.grid(True, which='both', color='gray', linewidth=0.5)
        ax.set_xticks(np.arange(-0.5, env.map_size[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env.map_size[0], 1), minor=True)
        ax.set_xticks(np.arange(0, env.map_size[1], 1))
        ax.set_yticks(np.arange(0, env.map_size[0], 1))
        
        ax.set_title('Map State')
        # Place legend below the chart, centered and auto-wrapped
        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='center', ncol=4, bbox_transform=ax.transAxes)
            
    def plot_market_metrics(self, trading_system, config=None):
        """
        Plot market metrics including number of buy/sell orders and last prices.
        
        Args:
            trading_system: The trading system instance containing market data
            config (dict, optional): Configuration dictionary containing plot_path
        """
        plot_path = self._get_plot_path(config)
        if not plot_path:
            print("Warning: No plot path specified, skipping plotting")
            return
            
        os.makedirs(plot_path, exist_ok=True)
        
        # Plot number of orders
        self._plot_order_counts(trading_system, plot_path)
        
        # Plot last prices
        self._plot_price_history(trading_system, plot_path)
        
    def _plot_order_counts(self, trading_system, plot_path):
        """Plot the number of buy and sell orders for each resource."""
        plt.figure(figsize=(10, 6))
        
        resources = ["wood", "stone"]
        for resource in resources:
            time_steps = range(len(trading_system.order_history[resource]["buy"]))
            buy_counts = trading_system.order_history[resource]["buy"]
            sell_counts = trading_system.order_history[resource]["sell"]
            
            plt.plot(time_steps, buy_counts, label=f'{resource.title()} Buy Orders', linewidth=2)
            plt.plot(time_steps, sell_counts, label=f'{resource.title()} Sell Orders', linewidth=2)
        
        plt.xlabel('Time Step')
        plt.ylabel('Number of Orders')
        plt.title('Market Order Counts Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(plot_path, 'market_orders.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved market_orders.png")
        
    def _plot_price_history(self, trading_system, plot_path):
        """Plot the last trade price for each resource over time."""
        plt.figure(figsize=(10, 6))
        
        resources = ["wood", "stone"]
        
        for resource in resources:
            # Extract time steps and prices from trades
            times = [trade[0] for trade in trading_system.trades[resource]]
            prices = [trade[1] for trade in trading_system.trades[resource]]
            
            if times and prices:  # Only plot if we have data
                plt.plot(times, prices, label=f'{resource.title()} Price', linewidth=2)
        
        plt.xlabel('Time Step')
        plt.ylabel('Price')
        plt.title('Resource Prices Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        filepath = os.path.join(plot_path, 'market_prices.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved market_prices.png")