from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent 
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
from typing import Set, Dict
import numpy as np
from utils import MSegment

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self):
        # TODO: fill this in (if necessary)
        super().__init__()
        self.name = "john"  # TODO: enter a name.

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

        

    def on_new_game(self) -> None:
        # TODO: fill this in (if necessary)
        
        pass

    def get_ad_bids(self) -> Set[BidBundle]:
        # TODO: fill this in
        bundles = set()

        active_campaigns = self.get_active_campaigns()

        for campaign in active_campaigns:

            reach_proportion = campaign.reach / (self.market_segment_map[campaign.target_segment.name.upper()] ) 

            item_bid = 0.9
            budget = campaign.reach

            if reach_proportion < 0.4:
                item_bid = 0.88001
            elif reach_proportion < 0.6:
                item_bid = 0.90001
            else:
                item_bid = 0.92001
            
            if campaign.reach < 800:
                budget *= 1.1
            elif campaign.reach < 2000:
                budget *= 1.05
            
            if campaign.start_day < 4:
                item_bid += 0.02
                budget *= 1.05
            elif campaign.start_day < 7:
                item_bid += 0.01
                budget *= 1.03

            # print(item_bid)


            bid:Bid = Bid(bidder = self, auction_item = campaign.target_segment, bid_per_item = item_bid, bid_limit = budget)

            bid_set = set()
            bid_set.add(bid)

            bundle = BidBundle(campaign_id = campaign.uid, limit = budget, bid_entries = bid_set)

            bundles.add(bundle)

        return bundles

    def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
        # TODO: fill this in 

        # make enum that's like 0 if no overlap
        # 1 if general segment overlap (like male old and male young)
        # 2 if actual overlap (male old, male old high income)

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
                res_dict[campaign] = campaign.reach * 0.99
            
            if campaign_overlap_map[campaign.target_segment.name.upper()] == 0: # no overlap
                res_dict[campaign] = campaign.reach * 0.9

        return res_dict


if __name__ == "__main__":
    # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
    test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

    # Don't change this. Adapt initialization to your environment
    simulator = AdXGameSimulator()
    simulator.run_simulation(agents=test_agents, num_simulations=500)

my_agent_submission = MyNDaysNCampaignsAgent()