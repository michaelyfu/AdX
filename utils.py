from simplified.eisenberg_agent import EisenbergAgent 
from john_agent import JohnAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 

def main():
    bots = [EisenbergAgent()] + [JohnAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(9)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)


if __name__ == "__main__":
    main()