from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent 
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import math

from game_theoretic_equilibrium import Agent, AuctionSimulation

class MSegment():

    def __init__(self, market_str):

        parsed_market_str = self.parse(market_str.upper())

        gender = parsed_market_str // 9
        income = (parsed_market_str % 9) // 3
        age = parsed_market_str % 3

        self.ms_int = parsed_market_str
        self.is_male = (gender == 0) or (gender == 2)
        self.is_female = (gender == 1) or (gender == 2)
        self.is_low_income = (income == 0) or (income == 2)
        self.is_high_income = (income == 1) or (income == 2)
        self.is_young = (age == 0) or (age == 2)
        self.is_old = (age == 0) or (age == 2)
        

    def parse(self, market_str):
        base3_map = {
            "MALE_YOUNG_LOWINCOME":    0,   # [gender=0, income=0, age=0] → 0*9 + 0*3 + 0
            "MALE_YOUNG_HIGHINCOME":   3,   # [0,1,0] → 0*9 + 1*3 + 0
            "FEMALE_YOUNG_LOWINCOME":  9,   # [1,0,0] → 1*9 + 0*3 + 0
            "FEMALE_YOUNG_HIGHINCOME":12,   # [1,1,0] → 1*9 + 1*3 + 0

            "MALE_OLD_LOWINCOME":      1,   # [0,0,1] → 0*9 + 0*3 + 1
            "MALE_OLD_HIGHINCOME":     4,   # [0,1,1] → 0*9 + 1*3 + 1
            "FEMALE_OLD_LOWINCOME":   10,   # [1,0,1] → 1*9 + 0*3 + 1
            "FEMALE_OLD_HIGHINCOME":  13,   # [1,1,1] → 1*9 + 1*3 + 1

            "FEMALE_LOWINCOME":       11,   # [1,0,2] → 1*9 + 0*3 + 2
            "FEMALE_HIGHINCOME":      14,   # [1,1,2] → 1*9 + 1*3 + 2
            "YOUNG_LOWINCOME":        18,   # [2,0,0] → 2*9 + 0*3 + 0
            "YOUNG_HIGHINCOME":       21,   # [2,1,0] → 2*9 + 1*3 + 0
            "OLD_LOWINCOME":          19,   # [2,0,1] → 2*9 + 0*3 + 1
            "OLD_HIGHINCOME":         22,   # [2,1,1] → 2*9 + 1*3 + 1

            "MALE_YOUNG":               6,   # [0,2,0] → 0*9 + 2*3 + 0
            "MALE_OLD":                 7,   # [0,2,1] → 0*9 + 2*3 + 1
            "FEMALE_YOUNG":            15,   # [1,2,0] → 1*9 + 2*3 + 0
            "FEMALE_OLD":              16,   # [1,2,1] → 1*9 + 2*3 + 1

            "MALE_LOWINCOME":          2,   # [0,0,2] → 0*9 + 0*3 + 2
            "MALE_HIGHINCOME":         5,   # [0,1,2] → 0*9 + 1*3 + 2
        }

        if market_str not in base3_map:
            print(market_str)
            raise Exception("market_str not valid")
        
        return base3_map[market_str]
    
    def get_map(self):
        binary_indicators = {
            "male": self.is_male,
            "female": self.is_female,
            "low_income": self.is_low_income,
            "high_income": self.is_high_income,
            "young": self.is_young,
            "old": self.is_old
        }
        return binary_indicators


class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "game-theory"  # TODO: enter a name.

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

        self.market_segment_map = market_segment_map

        self.price_index = {}
        self.env_index = 1.0

        

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        
        self.price_index = {}
        self.env_index = 1.0

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in

        quality_score = self.get_quality_score()

        bundles = set()

        active_campaigns = self.get_active_campaigns()

        for campaign in active_campaigns:
            
            MARKET_SEGMENT = campaign.target_segment
            simulation = AuctionSimulation(MARKET_SEGMENT)
            
            NUM_SIMULATIONS = 100 
            best_bids = []
            for i in range(NUM_SIMULATIONS):
                best_bids.append(simulation.simulate())
            avg_best_bid = sum(best_bids) / len(best_bids)
            
            
            
            bid:Bid = Bid(bidder = self, auction_item = MARKET_SEGMENT, bid_per_item = avg_best_bid, bid_limit = campaign.reach)

            bid_set = set()
            bid_set.add(bid)
            bundle = BidBundle(campaign_id = campaign.uid, limit = campaign.reach, bid_entries = bid_set)
            bundles.add(bundle)

        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 

        # make enum that's like 0 if no overlap
        # 1 if general segment overlap (like male old and male young)
        # 2 if actual overlap (male old, male old high income)

        quality_score = self.get_quality_score()

        active_campaigns = self.get_active_campaigns()

        total_indicators = {
            "male": 0,
            "female": 0,
            "low_income": 0,
            "high_income": 0,
            "young": 0,
            "old": 0
        }

        campaign_overlap_map = self.market_segment_map.copy()  # 0 = no overlap, 1 = partial overlap, 2 = full overlap

        for campaign in campaign_overlap_map:
            campaign_overlap_map[campaign] = 0

        for campaign in active_campaigns:
            
            market_segment_obj = MSegment(campaign.target_segment.name)
            # active_campaign_market_segments.add(market_segment_obj)
            for k, v in market_segment_obj.get_map().items():
                total_indicators[k] += v

        # find partial overlap

        for campaign in campaigns_for_auction:

            market_segment_obj = MSegment(campaign.target_segment.name)

            market_segment_map = market_segment_obj.get_map()

            for indicator in total_indicators:
                if total_indicators[indicator] and market_segment_map[indicator]:
                    campaign_overlap_map[campaign.target_segment.name.upper()] = 1
    
        
        # find total overlap

        for my_campaign in active_campaigns:

            my_campaign_int = MSegment(my_campaign.target_segment.name).ms_int

            for auction_campaign in campaigns_for_auction:

                auction_campaign_int = MSegment(auction_campaign.target_segment.name).ms_int

                is_gender_overlap = (my_campaign_int // 9 == 2) or (auction_campaign_int // 9 == 2) or (my_campaign_int // 9 == auction_campaign_int // 9)
                is_income_overlap = ((my_campaign_int % 9) // 3 == 2) or ((auction_campaign_int % 9) // 3 == 2) or ((my_campaign_int % 9) // 3 == (auction_campaign_int % 9) // 3)
                is_age_overlap = (my_campaign_int % 3 == 2) or (auction_campaign_int % 3 == 2) or (my_campaign_int % 3 == auction_campaign_int % 3)

                is_overlap = is_gender_overlap and is_income_overlap and is_age_overlap

                if is_overlap: campaign_overlap_map[campaign.target_segment.name.upper()] = 2

        res_dict = {}

        for campaign in campaigns_for_auction:
            if campaign_overlap_map[campaign.target_segment.name.upper()] == 1: # partial
                res_dict[campaign] = self.clip_campaign_bid(campaign, campaign.reach * 0.9 * self.env_index)
            
            if campaign_overlap_map[campaign.target_segment.name.upper()] == 0: # no overlap
                res_dict[campaign] = self.clip_campaign_bid(campaign, campaign.reach * 0.75 * self.env_index)

        return res_dict


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=50)

my_agent_submission = MyNDaysNCampaignsAgent()