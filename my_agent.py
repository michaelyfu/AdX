# ─────────────────── imports ─────────────────────────────────────────────
import cvxpy as cp
import numpy as np
from typing import Set, Dict

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
# ─────────────────────────────────────────────────────────────────────────

# ─────────────────── helper to inspect market segments ───────────────────
class MSegment:
    """
    Convenience wrapper that turns a market‑segment string into useful booleans
    and a unique int code (0‑26).  Only needed by get_campaign_bids().
    """
    base3_map = {
        "MALE_YOUNG_LOWINCOME":    0,  "MALE_YOUNG_HIGHINCOME":   3,
        "FEMALE_YOUNG_LOWINCOME":  9,  "FEMALE_YOUNG_HIGHINCOME":12,

        "MALE_OLD_LOWINCOME":      1,  "MALE_OLD_HIGHINCOME":     4,
        "FEMALE_OLD_LOWINCOME":   10,  "FEMALE_OLD_HIGHINCOME":  13,

        "FEMALE_LOWINCOME":       11,  "FEMALE_HIGHINCOME":      14,
        "YOUNG_LOWINCOME":        18,  "YOUNG_HIGHINCOME":       21,
        "OLD_LOWINCOME":          19,  "OLD_HIGHINCOME":         22,

        "MALE_YOUNG":               6,  "MALE_OLD":                 7,
        "FEMALE_YOUNG":            15,  "FEMALE_OLD":              16,

        "MALE_LOWINCOME":          2,  "MALE_HIGHINCOME":         5,
    }

    def __init__(self, market_str: str):
        self.ms_int = self.parse(market_str.upper())

        gender = self.ms_int // 9
        income = (self.ms_int % 9) // 3
        age    = self.ms_int % 3

        self.is_male        = (gender == 0) or (gender == 2)
        self.is_female      = (gender == 1) or (gender == 2)
        self.is_low_income  = (income == 0) or (income == 2)
        self.is_high_income = (income == 1) or (income == 2)
        self.is_young       = (age    == 0) or (age    == 2)
        self.is_old         = (age    == 1) or (age    == 2)

    @classmethod
    def parse(cls, s):
        if s not in cls.base3_map:
            raise ValueError(f"Unknown market segment {s}")
        return cls.base3_map[s]

    def get_map(self):
        return {
            "male":        self.is_male,
            "female":      self.is_female,
            "low_income":  self.is_low_income,
            "high_income": self.is_high_income,
            "young":       self.is_young,
            "old":         self.is_old,
        }

class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

    def __init__(self, name: str = "LINSANITY", greed_factor=1.15):
        super().__init__()
        self.name = name

        # Average number of *daily* users in each primitive segment (hand‑out table).
        self.market_segment_map: Dict[str, int] = {
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
            "MALE_YOUNG":              2353,
            "MALE_OLD":                2603,
            "FEMALE_YOUNG":            2236,
            "FEMALE_OLD":              2808,
            "MALE_LOWINCOME":          3631,
            "MALE_HIGHINCOME":         1325,
        }


        # Only the 8 atomic segments — per‑day counts from the spec
        self.atomic_segment_map: Dict[str, int] = {
            "MALE_YOUNG_LOWINCOME":   1836,
            "MALE_YOUNG_HIGHINCOME":   517,
            "FEMALE_YOUNG_LOWINCOME":  1980,
            "FEMALE_YOUNG_HIGHINCOME":  256,
            "MALE_OLD_LOWINCOME":      1795,
            "MALE_OLD_HIGHINCOME":      808,
            "FEMALE_OLD_LOWINCOME":    2401,
            "FEMALE_OLD_HIGHINCOME":    407,
        }

        self.greed_factor = greed_factor # it's set to 1.2 in the paper
        # --- Greed / CI parameters --------------------------------------
        self.GREED = max(1.0, greed_factor)           
        self._ci   = 1.0                               # Competing‑Index (CI)
        self._prev_day_bids: Dict[int, float] = {}     # uid → bid we sent
        # ----------------------------------------------------------------


    # ───────────────── EG solver ──────────────────────────
    def _solve_eg_market(self, campaigns):
        if not campaigns:
            return {}, None, {}
        segments = sorted({c.target_segment.name.upper() for c in campaigns})
        m, n = len(segments), len(campaigns)

        seg_to_idx = {s: j for j, s in enumerate(segments)}
        supplies   = np.array([self.market_segment_map[s] for s in segments], dtype=float)

        X  = cp.Variable((n, m), nonneg=True)
        ε  = 1e-6
        u  = cp.sum(X, axis=1) + ε
        B  = np.array([c.budget for c in campaigns], dtype=float)

        constraints = [cp.sum(X, axis=0) <= supplies]
        objective   = cp.Maximize(cp.sum(cp.multiply(B, cp.log(u))))
        prob        = cp.Problem(objective, constraints).solve(solver=cp.SCS, verbose=False)

        if prob is None:
            return {s: 1.0 for s in segments}, None, seg_to_idx

        p = constraints[0].dual_value
        prices = {s: float(max(v, 0)) for s, v in zip(segments, p)}
        return prices, X.value, seg_to_idx

    # ─────────────────  daily ad bids  ────────────────────────────────
    def get_ad_bids(self) -> Set[BidBundle]:
        bundles = set()
        active_campaigns = list(self.get_active_campaigns())
        
        try:
            prices, alloc, idx = self._solve_eg_market(active_campaigns)
            
            for i, camp in enumerate(active_campaigns):
                try:
                    seg = camp.target_segment.name.upper()
                    p_m = prices.get(seg, 1.0)
                    
                    # Check if seg is in idx before accessing
                    if alloc is not None and seg in idx:
                        qty = alloc[i, idx[seg]]
                    else:
                        qty = camp.reach / max(1, camp.end_day - camp.start_day + 1)
                        
                    spend = p_m * qty
                    bid = Bid(self, camp.target_segment, p_m, spend)
                    bundles.add(BidBundle(camp.uid, spend, {bid}))
                except Exception as e:
                    # print('error', e)
                    continue
        except Exception as e:
            # Return empty set if the whole process fails
            print('shouldnt happen', e)
            return set()
        
        return bundles

    def estimate_segment_size(self, target_segment: MarketSegment) -> int:
        """
        Return the *daily* number of users matching target_segment,
        by summing over all atomic (3‑attribute) segments whose attributes
        contain target_segment (i.e. atomic ⊇ target).
        """
        total = 0
        for seg_str, count in self.atomic_segment_map.items():
            atomic_attrs = set(seg_str.split("_"))  # e.g. {"FEMALE","OLD","LOWINCOME"}
            # if every attr in target_segment is in this atomic_attrs
            if set(target_segment).issubset(atomic_attrs):
                total += count
        return total
    
    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:

        # --- incorporate previous‑day auction result --------------------
        self._roll_ci_after_contract_auction(campaigns_for_auction)
        # ----------------------------------------------------------------
        
        campaign_bids = {}
        current_day = self.get_current_day()
        quality_score = self.get_quality_score()

        # Step 1: Build the set of segments we're already targeting
        active_campaigns = self.get_active_campaigns()
        committed_segments = set()
        for camp in active_campaigns:
            committed_segments.add(frozenset(camp.target_segment))  # store as sets for subset checks

        # Step 2: Decide bids
        for campaign in campaigns_for_auction:
            R = campaign.reach
            start_day, end_day = campaign.start_day, campaign.end_day
            duration = end_day - start_day + 1

            # Step 3: Skip campaigns with overlap
            overlap = any(frozenset(campaign.target_segment).issubset(seg) or seg.issubset(campaign.target_segment)
                        for seg in committed_segments)
            if overlap:
                continue  # too much overlap with currently active segments

            # Step 4: Estimate user supply and value
            # **normalize**: get per‑day users, then total expected over the campaign window
            daily_users    = self.estimate_segment_size(campaign.target_segment)
            expected_users = daily_users * duration
            expected_value = min(R, expected_users)

            # Step 5: Determine bid based on quality and urgency
            is_short = duration <= 2
            base_bid = 0.25 * R if quality_score < 0.9 else 0.5 * R
            # raw_bid = base_bid * (0.9 if current_day >= start_day else 1.0) * (0.9 if is_short else 1.0)
            raw_bid  = base_bid * self._ci   # ← ①  CI here

            clipped_bid = self.clip_campaign_bid(campaign, raw_bid) 

            if self.is_valid_campaign_bid(campaign, clipped_bid):
                campaign_bids[campaign] = clipped_bid
                self._prev_day_bids[campaign.uid] = clipped_bid  # track
        return campaign_bids

    # ----- CI book‑keeping  ---------------------------------------------
    def _roll_ci_after_contract_auction(self,
                                        todays_contracts: Set[Campaign]) -> None:
        """Update CI using yesterday’s outcomes (paper Eq. (5))."""
        # ❶ campaigns we still see today ⇒ we *lost* yesterday
        lost_uids = {
            c.uid
            for c in todays_contracts
            if c.uid in self._prev_day_bids
        }

        # ❷ campaigns that moved to our active list ⇒ we *won*
        won_campaigns = {
            c for c in self.get_active_campaigns()
            if c.uid in self._prev_day_bids
        }

        # -- CI update ---------------------------------------------------
        if lost_uids:
            # we failed → CI ← G * CI
            self._ci *= self.GREED
        for camp in won_campaigns:
            bid     = self._prev_day_bids[camp.uid]
            # budget is only exposed to the winner
            if np.isclose(camp.budget, bid):
                # random assignment → CI unchanged
                pass
            else:
                # standard 2nd‑price win → CI ← CI / G
                self._ci /= self.GREED

        # clear history for next round
        self._prev_day_bids.clear()


    # optional: clear per‑game state
    def on_new_game(self):
        pass

# ─────────────────── quick offline test harness ─────────────────────────
if __name__ == "__main__":
    bots = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(9)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)

my_agent_submission = MyNDaysNCampaignsAgent()

# from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent 
# from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
# from agt_server.local_games.adx_arena import AdXGameSimulator 
# from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment
# from typing import Set, Dict
# import math

# from game_theoretic_equilibrium import Agent, AuctionSimulation

# class MSegment():

#     def __init__(self, market_str):

#         parsed_market_str = self.parse(market_str.upper())

#         gender = parsed_market_str // 9
#         income = (parsed_market_str % 9) // 3
#         age = parsed_market_str % 3

#         self.ms_int = parsed_market_str
#         self.is_male = (gender == 0) or (gender == 2)
#         self.is_female = (gender == 1) or (gender == 2)
#         self.is_low_income = (income == 0) or (income == 2)
#         self.is_high_income = (income == 1) or (income == 2)
#         self.is_young = (age == 0) or (age == 2)
#         self.is_old = (age == 0) or (age == 2)
        

#     def parse(self, market_str):
#         base3_map = {
#             "MALE_YOUNG_LOWINCOME":    0,   # [gender=0, income=0, age=0] → 0*9 + 0*3 + 0
#             "MALE_YOUNG_HIGHINCOME":   3,   # [0,1,0] → 0*9 + 1*3 + 0
#             "FEMALE_YOUNG_LOWINCOME":  9,   # [1,0,0] → 1*9 + 0*3 + 0
#             "FEMALE_YOUNG_HIGHINCOME":12,   # [1,1,0] → 1*9 + 1*3 + 0

#             "MALE_OLD_LOWINCOME":      1,   # [0,0,1] → 0*9 + 0*3 + 1
#             "MALE_OLD_HIGHINCOME":     4,   # [0,1,1] → 0*9 + 1*3 + 1
#             "FEMALE_OLD_LOWINCOME":   10,   # [1,0,1] → 1*9 + 0*3 + 1
#             "FEMALE_OLD_HIGHINCOME":  13,   # [1,1,1] → 1*9 + 1*3 + 1

#             "FEMALE_LOWINCOME":       11,   # [1,0,2] → 1*9 + 0*3 + 2
#             "FEMALE_HIGHINCOME":      14,   # [1,1,2] → 1*9 + 1*3 + 2
#             "YOUNG_LOWINCOME":        18,   # [2,0,0] → 2*9 + 0*3 + 0
#             "YOUNG_HIGHINCOME":       21,   # [2,1,0] → 2*9 + 1*3 + 0
#             "OLD_LOWINCOME":          19,   # [2,0,1] → 2*9 + 0*3 + 1
#             "OLD_HIGHINCOME":         22,   # [2,1,1] → 2*9 + 1*3 + 1

#             "MALE_YOUNG":               6,   # [0,2,0] → 0*9 + 2*3 + 0
#             "MALE_OLD":                 7,   # [0,2,1] → 0*9 + 2*3 + 1
#             "FEMALE_YOUNG":            15,   # [1,2,0] → 1*9 + 2*3 + 0
#             "FEMALE_OLD":              16,   # [1,2,1] → 1*9 + 2*3 + 1

#             "MALE_LOWINCOME":          2,   # [0,0,2] → 0*9 + 0*3 + 2
#             "MALE_HIGHINCOME":         5,   # [0,1,2] → 0*9 + 1*3 + 2
#         }

#         if market_str not in base3_map:
#             print(market_str)
#             raise Exception("market_str not valid")
        
#         return base3_map[market_str]
    
#     def get_map(self):
#         binary_indicators = {
#             "male": self.is_male,
#             "female": self.is_female,
#             "low_income": self.is_low_income,
#             "high_income": self.is_high_income,
#             "young": self.is_young,
#             "old": self.is_old
#         }
#         return binary_indicators


# class MyNDaysNCampaignsAgent(NDaysNCampaignsAgent):

#     def __init__(self):
#         # TODO: fill this in (if necessary)
#         super().__init__()
#         self.name = "game-theory"  # TODO: enter a name.

#         market_segment_map = {
#             "MALE_YOUNG_LOWINCOME":   1836,
#             "MALE_YOUNG_HIGHINCOME":   517,
#             "FEMALE_YOUNG_LOWINCOME":  1980,
#             "FEMALE_YOUNG_HIGHINCOME":  256,
#             "MALE_OLD_LOWINCOME":      1795,
#             "MALE_OLD_HIGHINCOME":      808,
#             "FEMALE_OLD_LOWINCOME":    2401,
#             "FEMALE_OLD_HIGHINCOME":    407,

#             "FEMALE_LOWINCOME":        4381,
#             "FEMALE_HIGHINCOME":        663,
#             "YOUNG_LOWINCOME":         3816,
#             "YOUNG_HIGHINCOME":         773,
#             "OLD_LOWINCOME":           4196,
#             "OLD_HIGHINCOME":          1215,

#             "MALE_YOUNG":               2353,
#             "MALE_OLD":                 2603,
#             "FEMALE_YOUNG":             2236,
#             "FEMALE_OLD":               2808,
#             "MALE_LOWINCOME":          3631,
#             "MALE_HIGHINCOME":         1325,
#         }

#         self.market_segment_map = market_segment_map

#         self.price_index = {}
#         self.env_index = 1.0

        

#     def on_new_game(self) -> None:
#         # TODO: fill this in (if necessary)
        
#         self.price_index = {}
#         self.env_index = 1.0

#     def get_ad_bids(self) -> Set[BidBundle]:
#         # TODO: fill this in

#         quality_score = self.get_quality_score()

#         bundles = set()

#         active_campaigns = self.get_active_campaigns()

#         for campaign in active_campaigns:
            
#             MARKET_SEGMENT = campaign.target_segment
#             simulation = AuctionSimulation(MARKET_SEGMENT)
            
#             NUM_SIMULATIONS = 100 
#             best_bids = []
#             for i in range(NUM_SIMULATIONS):
#                 best_bids.append(simulation.simulate())
#             avg_best_bid = sum(best_bids) / len(best_bids)
            
            
            
#             bid:Bid = Bid(bidder = self, auction_item = MARKET_SEGMENT, bid_per_item = avg_best_bid, bid_limit = campaign.reach)

#             bid_set = set()
#             bid_set.add(bid)
#             bundle = BidBundle(campaign_id = campaign.uid, limit = campaign.reach, bid_entries = bid_set)
#             bundles.add(bundle)

#         return bundles

#     def get_campaign_bids(self, campaigns_for_auction:  Set[Campaign]) -> Dict[Campaign, float]:
#         # TODO: fill this in 

#         # make enum that's like 0 if no overlap
#         # 1 if general segment overlap (like male old and male young)
#         # 2 if actual overlap (male old, male old high income)

#         quality_score = self.get_quality_score()

#         active_campaigns = self.get_active_campaigns()

#         total_indicators = {
#             "male": 0,
#             "female": 0,
#             "low_income": 0,
#             "high_income": 0,
#             "young": 0,
#             "old": 0
#         }

#         campaign_overlap_map = self.market_segment_map.copy()  # 0 = no overlap, 1 = partial overlap, 2 = full overlap

#         for campaign in campaign_overlap_map:
#             campaign_overlap_map[campaign] = 0

#         for campaign in active_campaigns:
            
#             market_segment_obj = MSegment(campaign.target_segment.name)
#             # active_campaign_market_segments.add(market_segment_obj)
#             for k, v in market_segment_obj.get_map().items():
#                 total_indicators[k] += v

#         # find partial overlap

#         for campaign in campaigns_for_auction:

#             market_segment_obj = MSegment(campaign.target_segment.name)

#             market_segment_map = market_segment_obj.get_map()

#             for indicator in total_indicators:
#                 if total_indicators[indicator] and market_segment_map[indicator]:
#                     campaign_overlap_map[campaign.target_segment.name.upper()] = 1
    
        
#         # find total overlap

#         for my_campaign in active_campaigns:

#             my_campaign_int = MSegment(my_campaign.target_segment.name).ms_int

#             for auction_campaign in campaigns_for_auction:

#                 auction_campaign_int = MSegment(auction_campaign.target_segment.name).ms_int

#                 is_gender_overlap = (my_campaign_int // 9 == 2) or (auction_campaign_int // 9 == 2) or (my_campaign_int // 9 == auction_campaign_int // 9)
#                 is_income_overlap = ((my_campaign_int % 9) // 3 == 2) or ((auction_campaign_int % 9) // 3 == 2) or ((my_campaign_int % 9) // 3 == (auction_campaign_int % 9) // 3)
#                 is_age_overlap = (my_campaign_int % 3 == 2) or (auction_campaign_int % 3 == 2) or (my_campaign_int % 3 == auction_campaign_int % 3)

#                 is_overlap = is_gender_overlap and is_income_overlap and is_age_overlap

#                 if is_overlap: campaign_overlap_map[campaign.target_segment.name.upper()] = 2

#         res_dict = {}

#         for campaign in campaigns_for_auction:
#             if campaign_overlap_map[campaign.target_segment.name.upper()] == 1: # partial
#                 res_dict[campaign] = self.clip_campaign_bid(campaign, campaign.reach * 0.9 * self.env_index)
            
#             if campaign_overlap_map[campaign.target_segment.name.upper()] == 0: # no overlap
#                 res_dict[campaign] = self.clip_campaign_bid(campaign, campaign.reach * 0.75 * self.env_index)

#         return res_dict


# if __name__ == "__main__":
#     # Here's an opportunity to test offline against some TA agents. Just run this file to do so.
#     test_agents = [MyNDaysNCampaignsAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Agent {i + 1}") for i in range(9)]

#     # Don't change this. Adapt initialization to your environment
#     simulator = AdXGameSimulator()
#     simulator.run_simulation(agents=test_agents, num_simulations=50)

# my_agent_submission = MyNDaysNCampaignsAgent()