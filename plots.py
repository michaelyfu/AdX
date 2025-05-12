#!/usr/bin/env python3
# using the eisenberg_agent, modulate various hyperparameters

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Sequence
from simplified.eisenberg_agent import EisenbergAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent

def run_greed_experiment(greed_values, num_simulations=50):
    """
    Run simulations with different greed values and return the average profits.
    
    Args:
        greed_values: List of greed values to test
        num_simulations: Number of simulations to run for each greed value
    
    Returns:
        List of average profits for each greed value
    """
    results = []
    
    for greed in greed_values:
        print(f"Testing greed factor: {greed:.2f}")
        
        # Create our agent with the current greed value
        eisenberg_agent = EisenbergAgent(name=f"Eisenberg-{greed:.2f}", greed_factor=greed)
        
        # Create baseline agents
        baseline_agents = [Tier1NDaysNCampaignsAgent(name=f"Tier1-{i}") for i in range(9)]
        
        # All agents for the simulation - using a list but not explicitly typing it
        all_agents = [eisenberg_agent] + baseline_agents
        
        # Run the simulation
        simulator = AdXGameSimulator()
        simulation_results = simulator.run_simulation(agents=all_agents, num_simulations=num_simulations)
        
        # Extract our agent's average profit
        if simulation_results and eisenberg_agent.name in simulation_results:
            # The simulation_results contains total profits across all simulations
            # Divide by num_simulations to get the average profit
            eisenberg_profit = simulation_results[eisenberg_agent.name] / num_simulations
            results.append(eisenberg_profit)
            # print(f"  Average profit: {eisenberg_profit:.2f}")
        else:
            print(f"  No results for {eisenberg_agent.name}")
            results.append(0.0)  # Append a default value if no results
    
    return results

def plot_results(greed_values, profits):
    """Plot the results of the greed experiment."""
    plt.figure(figsize=(10, 6))
    plt.plot(greed_values, profits, 'o-', linewidth=2)
    plt.xlabel('Greed Factor')
    plt.ylabel('Average Profit')
    plt.title('EisenbergAgent Performance vs Greed Factor')
    plt.grid(True)
    
    # Find and mark the optimal greed value
    optimal_idx = np.argmax(profits)
    optimal_greed = greed_values[optimal_idx]
    optimal_profit = profits[optimal_idx]
    
    plt.scatter([optimal_greed], [optimal_profit], color='red', s=100, zorder=5)
    plt.annotate(f'Optimal: {optimal_greed:.2f}\nProfit: {optimal_profit:.2f}',
                 xy=(optimal_greed, optimal_profit),
                 xytext=(10, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.savefig('greed_experiment_results.png')
    plt.show()

if __name__ == "__main__":
    # Define the range of greed values to test
    greed_values = np.linspace(1.0, 2.0, 21)  # Test from 1.0 to 2.0 in steps of 0.05 #21 
    
    # Run multiple experiments and average results
    num_experiments = 5
    all_profits = []
    
    for _ in range(num_experiments):
        profits = run_greed_experiment(greed_values)
        all_profits.append(profits)
    
    # Average the profits across experiments
    avg_profits = np.mean(all_profits, axis=0)
    
    # Plot the averaged results
    plot_results(greed_values, avg_profits)
    
    # Print the optimal greed value based on averaged results
    optimal_idx = np.argmax(avg_profits)
    print(f"\nOptimal greed factor: {greed_values[optimal_idx]:.2f}")
    print(f"Optimal average profit: {avg_profits[optimal_idx]:.2f}")