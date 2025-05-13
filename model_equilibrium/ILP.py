from itertools import chain, combinations
from gurobipy import Model, GRB
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
'''
SIMPLIFIED VERSION:
'''
CONFIG = {
        'num_agents': 10,
        'num_days': 1,
        'quality_score_alpha': 0.5,
        'campaigns_per_day': 1,
        'campaign_reach_dist': [0.3],
        'campaign_length_dist': [1, 2, 3],
        'market_segment_dist': [
            MarketSegment(("Male", "Young")),
        ],
        'market_segment_pop': {
            MarketSegment(("Male", "Young")): 10,
                },
        'user_segment_pmf': {
            MarketSegment(("Male", "Young", "LowIncome")): 1,
                }
    }

class CP:
    def __init__(self, agents: list, goods: list, valuations: dict):
        self.model = Model("walrasian_ilp")
        self.agents = agents
        self.goods = goods
        self.valuations = valuations  # dict of the form {(i, frozenset(S)): v_i(S)}
        self.x = {}  # decision variables

        all_bundles = list(self.powerset(goods))
        for i in agents:
            for S in all_bundles:
                S_frozen = frozenset(S)
                self.x[i, S_frozen] = self.model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{list(S)}")

        self.add_constraints()
        self.set_objective()

    def powerset(self, iterable):
        "powerset([1,2,3]) --> (), (1,), (2,), (3,), (1,2), (1,3), ..."
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def add_constraints(self):
        # Each agent gets at most one bundle
        for i in self.agents:
            expr = sum(self.x[i, frozenset(S)] for S in self.powerset(self.goods))
            self.model.addConstr(expr <= 1) # type: ignore

        # Each good is assigned at most once
        for g in self.goods:
            expr = sum(self.x[i, frozenset(S)] for i in self.agents for S in self.powerset(self.goods) if g in S)
            self.model.addConstr(expr <= 1) # type: ignore

    def set_objective(self):
        self.model.setObjective(
            sum(self.valuations[i, frozenset(S)] * self.x[i, frozenset(S)]
                for i in self.agents for S in self.powerset(self.goods)),
            GRB.MAXIMIZE
        )

    def run_model(self):
        self.model.optimize()
        return {
            (i, S): var.X for (i, S), var in self.x.items() if var.X > 0.5
        }

def estimate_valuations(agents, goods, campaigns):
    valuations = {}
    for i, campaign in enumerate(campaigns):
        for S in powerset(goods):
            S_frozen = frozenset(S)
            match_count = len(S_frozen)
            # Value is proportional to how much of the reach requirement is satisfied
            value = campaign.budget * min(match_count, campaign.reach) / campaign.reach
            valuations[i, S_frozen] = value
    return valuations

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def dual_lp(allocation, valuations, agents, goods):
    model = Model("dual_prices")
    u = {i: model.addVar(lb=0, name=f"u_{i}") for i in agents}
    p = {g: model.addVar(lb=0, name=f"p_{g}") for g in goods}

    # Add constraints
    for i in agents:
        for S in powerset(goods):
            S_frozen = frozenset(S)
            if (i, S_frozen) in valuations:
                model.addConstr(u[i] + sum(p[g] for g in S_frozen) >= valuations[i, S_frozen])

    # Add allocation constraints
    for (i, S) in allocation:
        model.addConstr(u[i] + sum(p[g] for g in S) == valuations[i, S])

    # Set objective
    model.setObjective(sum(u.values()) + sum(p.values()), GRB.MINIMIZE)
    model.optimize()

    return {g: p[g].X for g in goods}

def create_sample_campaigns(num_agents):
    campaigns = []
    segment = list(CONFIG['market_segment_dist'])[0]  
    
    for i in range(num_agents):
        # Create a campaign for each agent
        reach = CONFIG['campaign_reach_dist'][i % len(CONFIG['campaign_reach_dist'])]  # Cycle through reach options
        # Scale reach to the population size
        actual_reach = int(reach * CONFIG['market_segment_pop'][segment])
        
        campaign = Campaign(
            reach=actual_reach,  # Number of users to reach
            target=segment,
            start_day=0,
            end_day=min(CONFIG['campaign_length_dist']) - 1  # Use minimum campaign length from CONFIG
        )
        # Set the budget
        campaign.budget = 100
        campaigns.append(campaign)
    
    return campaigns

def main():
    # Use values from CONFIG
    num_agents = CONFIG['num_agents']
    num_goods = CONFIG['market_segment_pop'][list(CONFIG['market_segment_pop'].keys())[0]]  # Use population as number of goods
    agents = list(range(num_agents))
    goods = list(range(num_goods))
    
    print("Creating sample campaigns...")
    campaigns = create_sample_campaigns(num_agents)
    
    print("Estimating valuations...")
    valuations = estimate_valuations(agents, goods, campaigns)
    
    # Print valuations for debugging
    print("Valuations:")
    for (i, S), v in valuations.items():
        if len(S) <= 2:  # Only print smaller bundles to avoid cluttering output
            print(f"Agent {i}, Bundle {S}: Value = {v}")
    
    print("\nRunning ILP to find optimal allocation...")
    ilp = CP(agents, goods, valuations)
    allocation = ilp.run_model()
    
    # Convert allocation from {(i, S): 1.0} to {(i, S)} format for dual_lp
    allocation_set = {(i, S) for (i, S), val in allocation.items()}
    
    # Print allocation results
    print("\nOptimal Allocation:")
    for (i, S) in allocation_set:
        print(f"Agent {i} gets bundle {S}")
    
    print("\nCalculating prices using dual LP...")
    prices = dual_lp(allocation_set, valuations, agents, goods)
    
    print("\nComputed Prices:")
    for g, price in prices.items():
        print(f"Good {g}: Price = {price:.2f}")
    
    # Verify welfare maximization
    social_welfare = sum(valuations[i, S] for (i, S) in allocation_set)
    print(f"\nTotal Social Welfare: {social_welfare:.2f}")
    
    # Verify that each agent gets a utility-maximizing bundle at these prices
    print("\nVerifying rationality of allocation:")
    for i in agents:
        agent_bundle = None
        agent_value = 0
        for (agent, bundle) in allocation_set:
            if agent == i:
                agent_bundle = bundle
                agent_value = valuations[i, bundle]
                break
        
        if agent_bundle is not None:
            bundle_price = sum(prices[g] for g in agent_bundle)
            utility = agent_value - bundle_price
            print(f"Agent {i} gets utility {utility:.2f} from bundle {agent_bundle}")
            
            # Check a few alternative bundles to verify optimality
            for S in list(powerset(goods))[:10]:  # Check only first few bundles to avoid too much output
                S_frozen = frozenset(S)
                if S_frozen != agent_bundle and (i, S_frozen) in valuations:
                    other_price = sum(prices[g] for g in S_frozen)
                    other_utility = valuations[i, S_frozen] - other_price
                    # print(f"  Alternative bundle {S_frozen} would give utility {other_utility:.2f}")
                    if other_utility > utility + 1e-6:
                        print(f"  WARNING: Agent {i} would prefer bundle {S_frozen}!")
        else:
            # print(f"Agent {i} got no bundle")
            pass

if __name__ == "__main__":
    main()
