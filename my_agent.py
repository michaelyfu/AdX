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

class MyAgent(NDaysNCampaignsAgent):

    def __init__(self, name: str = "B99"):
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
        bundles          = set()
        active_campaigns = list(self.get_active_campaigns())
        prices, alloc, idx = self._solve_eg_market(active_campaigns)
        
        for i, camp in enumerate(active_campaigns):
            seg   = camp.target_segment.name.upper()
            p_m   = prices.get(seg, 1.0)
            qty   = alloc[i, idx[seg]] if alloc is not None else camp.reach / max(1, camp.end_day - camp.start_day + 1)
            spend = p_m * qty

            bid = Bid(self, camp.target_segment, p_m, spend)
            bundles.add(BidBundle(camp.uid, spend, {bid}))

        return bundles

    # ─────────────────  unchanged heuristic for campaign auctions  ────
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        # your original overlap heuristic is kept here for brevity
        # (or drop in your own EG‑based bidding rule later)
        bids = {}
        for c in auctions:
            bids[c] = max(0.1 * c.reach, 0.9 * c.reach)  # very simple placeholder
        return bids

    # optional: clear per‑game state
    def on_new_game(self):
        pass

# ─────────────────── quick offline test harness ─────────────────────────
if __name__ == "__main__":
    bots = [MyAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(9)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)

    
my_agent_submission = MyAgent()