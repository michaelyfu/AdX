from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent 
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict, List
from math import isfinite, atan
from collections import Counter
import matplotlib.pyplot as plt
import random, copy

# ONE DAY GAME

REACH_FACTORS = [0.3, 0.5, 0.7]
NUM_AGENTS = 19
TOTAL_USERS = 10000
# MARKET_SEGMENT = MarketSegment(("Male", "Young"))
EPSILON = 1e-9

agents = []
prev_output = []

class Agent:

    def __init__(self, id):

        market_segment = random.choice(MarketSegment.all_segments())
        # print(market_segment.name)
        # print(self.get_segment_size(market_segment.name))

        reach_factor = random.choice(REACH_FACTORS) # if id != "John" else 0.3
        reach = reach_factor * self.get_segment_size(market_segment.name)
        budget = reach
        # BID_SHADING_FACTOR = 0.8 + 0.3 * random.random()
        BID_SHADING_FACTOR = 0.80000002 + 0.2 * random.random()
        
        # bid_per_item = BID_SHADING_FACTOR * budget / reach 
        bid_per_item = random.choice(prev_output)

        self.id = id
        self.campaign = Campaign(reach, market_segment, 1, 1)
        self.bid = Bid(id, market_segment, bid_per_item, budget)

        # print(f"Initializing agent {id} with reach {reach} and bid {bid_per_item} and segment {market_segment.name}")
    
    def get_id(self):
        return self.id
    
    def get_segment_size(self, segment):
        market_segment_map = {
            "MALE_YOUNG_LOWINCOME":   1836,
            "MALE_YOUNG_HIGHINCOME":   517,
            "FEMALE_YOUNG_LOWINCOME":  1980,
            "FEMALE_YOUNG_HIGHINCOME":  256,
            "MALE_OLD_LOWINCOME":      1795,
            "MALE_OLD_HIGHINCOME":      808,
            "FEMALE_OLD_LOWINCOME":    2401,
            "FEMALE_OLD_HIGHINCOME":    407,

            "FEMALE_LOWINCOME":        4381,
            "FEMALE_HIGHINCOME":        663,
            "YOUNG_LOWINCOME":         3816,
            "YOUNG_HIGHINCOME":         773,
            "OLD_LOWINCOME":           4196,
            "OLD_HIGHINCOME":          1215,

            "MALE_YOUNG":               2353,
            "MALE_OLD":                 2603,
            "FEMALE_YOUNG":             2236,
            "FEMALE_OLD":               2808,
            "MALE_LOWINCOME":          3631,
            "MALE_HIGHINCOME":         1325,
        }

        return market_segment_map[segment.upper()]

# Run auction

class AuctionSimulation:

    def __init__(self):
        # self.MARKET_SEGMENT = MARKET_SEGMENT
        self.reset()

    def reset(self):
        agents = [Agent(i) for i in range(NUM_AGENTS)]
        
        self.agents = agents
        self.agents.sort(key = lambda agent: agent.bid.bid_per_item, reverse = True)

        self.distribution = {
            MarketSegment(("Male", "Young", "LowIncome")): 1836,
            MarketSegment(("Male", "Young", "HighIncome")): 517,
            MarketSegment(("Male", "Old", "LowIncome")): 1795,
            MarketSegment(("Male", "Old", "HighIncome")): 808,
            MarketSegment(("Female", "Young", "LowIncome")): 1980,
            MarketSegment(("Female", "Young", "HighIncome")): 256,
            MarketSegment(("Female", "Old", "LowIncome")): 2401,
            MarketSegment(("Female", "Old", "HighIncome")): 407
        }

        # print(f"Agents in descending order: {[agent.get_id() for agent in self.agents]}")
        
    def sample_random_users(self):
    
        segments = list(self.distribution.keys())
        weights = list(self.distribution.values())

        population = [
        segment
        for segment, count in self.distribution.items()
        for _ in range(count)
        ]

        users = random.sample(population, TOTAL_USERS)

        return users



    def get_allocation_and_payment(self, my_agent: Agent, k: int,
                                   agent_rankings: Dict[MarketSegment, List[Agent]],
                                   users: List[MarketSegment]):
        
        # Insert my agent at kth position in each list of agent_rankings

        def add_my_agent_to_rankings(rankings):
            for user_segment in rankings:
                segment_agent_rankings = rankings[user_segment]

                if not my_agent.bid.item.issubset(user_segment):
                    # if not applicable to my_agent, skip
                    # print(f"Agents in descending order: {[agent.get_id() for agent in segment_agent_rankings]} in {user_segment.name}")
                    continue

                # add my_agent
                # POSSIBLE SOURCE OF ERROR: kth place is different across multiple segments, so bids might differ
                my_agent.bid.bid_per_item = segment_agent_rankings[k].bid.bid_per_item + EPSILON if k < len(segment_agent_rankings) else EPSILON
                segment_agent_rankings.insert(min(k, len(segment_agent_rankings)), my_agent)

                # print(f"Agents in descending order: {[agent.get_id() for agent in segment_agent_rankings]} in {user_segment}")
        
        new_rankings = {key: value.copy() for key, value in agent_rankings.items()}

        add_my_agent_to_rankings(new_rankings)

        # Create agent list with my_agent in it

        agents_list = self.agents.copy()
        agents_list.append(my_agent)
        
        # agents = {agent.id: agent for agent in agents_list}
        bid = {agent: agent.bid.bid_per_item for agent in agents_list}
        x = {agent: 0 for agent in agents_list}
        p = {agent: 0 for agent in agents_list}
        eff_reach = {agent: 0 for agent in agents_list}
        utility = {agent: 0 for agent in agents_list}

        # Loop through all users
        for user in users:

            # Go through bidders in order; check to make sure budget hasn't been exceeded first
            for i in range(len(new_rankings[user])):

                agent_i = new_rankings[user][i]

                if (i + 1 >= len(new_rankings[user])):
                    # no more agents who bid below, so it's free
                    x[agent_i] += 1
                    break

                agent_iplus1 = new_rankings[user][i + 1]

                if (p[agent_i] + agent_iplus1.bid.bid_per_item > agent_i.bid.bid_limit):
                    # we are over budget
                    continue

                else:
                    x[agent_i] += 1
                    p[agent_i] += agent_iplus1.bid.bid_per_item
                    break
            
        for agent in eff_reach.keys():
            eff_reach[agent] = self.effective_reach(x[agent], agent.campaign.reach)
            utility[agent] = eff_reach[agent] * agent.bid.bid_limit - p[agent]
            # print(f"""Agent {agent.get_id()} bids {agent.bid.bid_per_item}, gets allocation {x[agent]}, payment {p[agent]}, effective reach {eff_reach[agent]}, utility {utility[agent]}""")

        return bid, x, p, eff_reach, utility
        
    def effective_reach(self, x: int, R: int) -> float:
        return (2.0 / 4.08577) * (atan(4.08577 * ((x + 0.0) / R) - 3.08577) - atan(-3.08577))

    def simulate(self):

        self.reset()
        
        my_agent = Agent("John")
        effective_reaches = []
        my_agent_utilities = []
        bids = []

        # Generate TOTAL_USERS random users

        users = self.sample_random_users()

        # For each market segment, get bid rankings (already sorted because agent list is sorted)

        agent_rankings = {k: [] for k in self.distribution.keys()}

        for user_segment in self.distribution.keys():
            for agent in self.agents:
                if agent.bid.item.issubset(user_segment):
                    agent_rankings[user_segment].append(agent)
                    

        for k in range(4):
            # print("====================================")
            # print(f"Running simulation for position {k}")

            # SIMULATION AT A HIGH LEVEL:

            # 1. Fix the bid of each bot agent for each individual market segment, then get rankings.

            # 2. Loop through values of k and fix my agent's bid for each individual market segment (in get_allocation_and_payment)
            #    To simplify, we use the same value of k for every market segment.

            # 3. Simulate a round of the auction; simulate n users coming in. For each user go through the bidders in order and
            #    check which one has the highest bid and that isn't yet over budget. 

            bid, x, p, eff_reach, utility = self.get_allocation_and_payment(my_agent, k, agent_rankings, users)

            bids.append(bid[my_agent])
            effective_reaches.append(eff_reach[my_agent])
            my_agent_utilities.append(utility[my_agent])

            # print("------------------------------------")
            # print(f"Final stats for position {k}: bid {bid[my_agent]}, allocation {x[my_agent]}, payment {p[my_agent]}, effective reach {eff_reach[my_agent]}, utility {utility[my_agent]}")
        
        max_eff_reach = max(effective_reaches)
        max_index = effective_reaches.index(max_eff_reach)

        max_utility = my_agent_utilities[max_index]
        best_bid = bids[max_index]

        # print("====================================")
        # print(f"Best position for {my_agent.get_id()} is position {max_index} with bid {best_bid}, with effective reach {max_eff_reach} and utility {max_utility}")

        return best_bid

prev_output = [0.8 + 0.2 * random.random() for _ in range(50)]

simulation = AuctionSimulation()

NUM_SIMULATIONS = 50
best_bids = []
averages = []


for j in range(500):

    for i in range(NUM_SIMULATIONS):
        best_bids.append(simulation.simulate())

    # print(best_bids)
    print(f"Average best bid: {sum(best_bids) / len(best_bids)}")
    averages.append(sum(best_bids) / len(best_bids))

    prev_output = best_bids


x = [i for i in range(500)]

plt.plot(x, averages)
plt.title("Average best bid")
plt.xlabel("trial")
plt.ylabel("avg best bid")
plt.grid(True)
plt.show()


# agent_i = agents_list[i]
            # bid_segment = agent_i.bid.item
            # subsets = []

            # for segment in MarketSegment.all_segments():
            #     if bid_segment.issubset(segment):
            #         subsets.append(segment)
            
            # # print(subsets)

            # # Randomly sample users up to budget

            # num_desired_users_left = sum([freq_map[segment] for segment in subsets if segment in freq_map])

            # num_desired_users = num_desired_users_left if i == NUM_AGENTS else agent_i.bid.bid_limit // agents_list[i + 1].bid.bid_per_item

            # # Now allocate bidders
            
            # # if last place bidder or not enough left, allocate all

            # if i == NUM_AGENTS or num_desired_users_left < num_desired_users:
            #     for segment in subsets:
            #         if segment in freq_map: freq_map[segment] = 0
                
            #     x_agent = current_supply

            # else:

            #     x_agent = num_desired_users

            #     # print(x_agent)

            #     filtered_counts = {segment: freq_map[segment] for segment in subsets if segment in freq_map}

            #     # print(filtered_counts)

            #     # randomly sample this many agents with max equal to number in freq_map
    
            #     population = [seg for seg, count in filtered_counts.items() for _ in range(count)]

            #     # print(population)

            #     sampled_users = random.sample(population, int(num_desired_users))

            #     print(sampled_users)

            #     for user in sampled_users:
            #         freq_map[user] -= 1


            # p_agent = 0 if i == NUM_AGENTS else x_agent * agents_list[i + 1].bid.bid_per_item
            # # else, 

            # x_agent = current_supply if i == NUM_AGENTS else min(agent_i.bid.bid_limit // agents_list[i + 1].bid.bid_per_item, current_supply)
            
            # id = agents_list[i].get_id()