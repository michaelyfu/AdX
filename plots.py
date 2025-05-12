#!/usr/bin/env python3
# using the eisenberg_agent, modulate various hyperparameters

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Union, Sequence
from model_equilibrium.beta import BetaAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent

def run_experiment(beta_values, num_simulations=50):
    """
    Run simulations with different values and return the average profits.
    
    Args:
        values: List of values to test
        num_simulations: Number of simulations to run for each value
    
    Returns:
        List of average profits for each value
    """
    results = []
    
    for beta in beta_values:
        print(f"Testing beta factor: {beta:.2f}")
        
        # Create our agent with the current value
        eisenberg_agent = BetaAgent(name=f"Eisenberg-{beta:.2f}", beta=beta)
        
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

def plot_results(values, profits):
    """Plot the results of the experiment."""
    plt.figure(figsize=(10, 6))
    plt.plot(values, profits, 'o-', linewidth=2)
    plt.xlabel('Beta Value')
    plt.ylabel('Average Profit')
    plt.title('EisenbergAgent Performance vs Beta Value')
    plt.grid(True)
    
    # Find and mark the optimal beta value
    optimal_idx = np.argmax(profits)
    optimal_beta = beta_values[optimal_idx]
    optimal_profit = profits[optimal_idx]
    
    plt.scatter([optimal_beta], [optimal_profit], color='red', s=100, zorder=5)
    plt.annotate(f'Optimal: {optimal_beta:.2f}\nProfit: {optimal_profit:.2f}',
                 xy=(optimal_beta, optimal_profit),
                 xytext=(10, -30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle='->'))
    
    plt.savefig('beta_experiment_results-2.png')
    plt.show()

if __name__ == "__main__":
    # Define the range of values to test
    beta_values = np.arange(0.35, 0.45 + 0.02, 0.02)  # Test from 0.35 to 0.45 in steps of 0.02
    
    # Run multiple experiments and average results
    num_experiments = 5
    all_profits = []
    
    for i in range(num_experiments):
        print('experiment num:', i + 1)
        profits = run_experiment(beta_values)
        all_profits.append(profits)
    
    # Average the profits across experiments
    avg_profits = np.mean(all_profits, axis=0)
    
    # Plot the averaged results
    plot_results(beta_values, avg_profits)
    
    # Print the optimal value based on averaged results
    optimal_idx = np.argmax(avg_profits)
    print(f"\nOptimal beta value: {beta_values[optimal_idx]:.2f}")
    print(f"Optimal average profit: {avg_profits[optimal_idx]:.2f}")