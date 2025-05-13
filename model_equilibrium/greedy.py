import cvxpy as cp
import numpy as np
from typing import Set, Dict

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator
from agt_server.agents.utils.adx.structures import Bid, Campaign, BidBundle, MarketSegment

class Greedy(NDaysNCampaignsAgent):

    def __init__(self, name: str = "Greedy", greed_factor=1.2):
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
        epislon  = 1e-6
        u  = cp.sum(X, axis=1) + epislon
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
    
    # ───────────────── daily campaign bids ──────────────────────────
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
        # campaigns we still see today ⇒ we *lost* yesterday
        lost_uids = {
            c.uid
            for c in todays_contracts
            if c.uid in self._prev_day_bids
        }

        # campaigns that moved to our active list ⇒ we *won*
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
    bots = [Greedy()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(9)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)