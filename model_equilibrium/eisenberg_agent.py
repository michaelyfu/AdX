import cvxpy as cp
import numpy as np
from typing import Set, Dict

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment

class EisenbergAgent(NDaysNCampaignsAgent):

    def __init__(self, name: str = "EisenbergAgent"):
        super().__init__()
        self.name = name

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

    # ───────────────── daily campaign bids ────────────────────────── 
    def get_campaign_bids(self, auctions: Set[Campaign]) -> Dict[Campaign, float]:
        bids = {}
        for c in auctions:
            bids[c] = max(0.1 * c.reach, 0.9 * c.reach)
        return bids

    # optional: clear per‑game state
    def on_new_game(self):
        pass

if __name__ == "__main__":
    bots = [EisenbergAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(9)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)