from typing import Dict, Set, List, Tuple
import random
import math
from collections import defaultdict, deque

from agt_server.agents.base_agents.adx_agent import NDaysNCampaignsAgent
from agt_server.agents.utils.adx.structures import Bid, BidBundle, Campaign, MarketSegment
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent
from agt_server.local_games.adx_arena import AdXGameSimulator

############################################################
#  Advanced-NL (ANL)‑style agent adapted for the CSCI 1440 #
#  10‑day AdX project.                                     #
############################################################
#  Key ideas ported from the TAC AdX’14 champion “ANL”:    #
#   • Contract bidding based on a private‑value model that #
#     combines a dynamic competing‑index (CI) with a price #
#     index (PI) predicted for each segment.               #
#   • Impression bidding that tracks remaining reach &     #
#     budget, then allocates a daily spend per campaign.   #
#   • A lightweight responsive PI estimator using an       #
#     exponential moving average of realised prices.       #
############################################################

class TACAgent(NDaysNCampaignsAgent):
    """Advanced agent that implements the core ANL/AdvANL ideas in the
    simplified CSCI‑1440 AdX environment (no UCS auction).
    """

    # ----- Tunable hyper‑parameters ----- #
    ALPHA_PI = 0.2          # EMA smoothing for realised price‑index updates
    INIT_PI = 0.001         # \$ default cost per impression when no data

    GREED     = 1.20        # G_greed from the paper (controls CI update)
    MAX_DAILY_BURN = 0.25   # fraction of remaining budget spent each day
    BEHIND_MULT     = 2.0   # bid multiplier when campaign is behind schedule

    def __init__(self, name: str = "ANL‑Lite"):
        super().__init__()
        self.name = name
        # segment‑level running price index (rough cost per impression)
        self.segment_pi: Dict[MarketSegment, float] = {
            s: self.INIT_PI for s in MarketSegment.all_segments()
        }
        # (popularity, price, day) history queue per segment (bounded)
        self._hist: Dict[MarketSegment, deque] = {
            s: deque(maxlen=30) for s in MarketSegment.all_segments()
        }
        # dynamic competing index for contract auctions
        self.ci = 1.0
        # day counter (reset each game)
        self._current_sim_day = 0

    # ----------------------------------------------------- #
    #  Lifecycle hooks                                     #
    # ----------------------------------------------------- #

    def on_new_game(self):
        """Reset all state at the start of each simulation run."""
        self._current_sim_day = 0
        self.ci = 1.0
        for s in MarketSegment.all_segments():
            self.segment_pi[s] = self.INIT_PI
            self._hist[s].clear()

    # ----------------------------------------------------- #
    #  Helper methods                                      #
    # ----------------------------------------------------- #

    def _days_left(self, c: Campaign) -> int:
        return max(1, c.end_day - self.get_current_day() + 1)

    def _remaining_reach(self, c: Campaign) -> int:
        return c.reach - self.get_cumulative_reach(c)

    def _remaining_budget(self, c: Campaign) -> float:
        return c.budget - self.get_cumulative_cost(c)

    def _average_pi(self, segs: Set[MarketSegment]) -> float:
        return sum(self.segment_pi[s] for s in segs) / len(segs)

    # ----------------------------------------------------- #
    #  Contract‑auction bidding                             #
    # ----------------------------------------------------- #

    def get_campaign_bids(self, campaigns_for_auction: Set[Campaign]) -> Dict[Campaign, float]:
        bids: Dict[Campaign, float] = {}
        for camp in campaigns_for_auction:
            est_cpi   = self._average_pi({camp.target_segment})
            est_cost  = est_cpi * camp.reach
            private_v = self.ci * est_cost

            # Clip into allowed range [0.1R, R]
            bid_raw = max(0.1 * camp.reach, min(camp.reach, private_v))
            bid     = self.clip_campaign_bid(camp, bid_raw)
            bids[camp] = bid
        return bids

    # ----------------------------------------------------- #
    #  Impression‑auction bidding                           #
    # ----------------------------------------------------- #

    def get_ad_bids(self) -> Set[BidBundle]:
        bundles: Set[BidBundle] = set()
        today = self.get_current_day()
        self._current_sim_day = today

        for camp in self.get_active_campaigns():
            days_left  = self._days_left(camp)
            reach_left = self._remaining_reach(camp)
            budget_left= self._remaining_budget(camp)
            if reach_left <= 0:
                continue  # already satisfied

            # Daily allowance: conservative burn + catch‑up if behind
            daily_budget = min(budget_left, max(1.0, self.MAX_DAILY_BURN * budget_left))
            completed_ratio = (camp.reach - reach_left) / camp.reach
            expected_ratio  = 1.0 - days_left / max(1, camp.end_day - camp.start_day + 1)
            if completed_ratio + 0.05 < expected_ratio:
                daily_budget *= self.BEHIND_MULT  # spend harder to catch up

            # Bid per impression: PI + small premium
            pi = self.segment_pi[camp.target_segment]
            bid_price = pi * self.BEHIND_MULT if daily_budget > budget_left * self.MAX_DAILY_BURN else pi * 1.1

            # Build Bid + Bundle
            bid_obj = Bid(
                bidder=self,
                auction_item=camp.target_segment,
                bid_per_item=bid_price,
                bid_limit=daily_budget,
            )
            bundle = BidBundle(
                campaign_id=camp.uid,
                limit=daily_budget,
                bid_entries={bid_obj},
            )
            bundles.add(bundle)
        return bundles

    # ----------------------------------------------------- #
    #  Post‑day callbacks (price‑index learning)            #
    # ----------------------------------------------------- #

    def update_after_impression_report(self, reports: List[Tuple[Campaign, int, float]]):
        """Called externally by the simulator **once per day** with a list of
        tuples (campaign, impressions_won, total_cost). This is not part of
        the base API but we expose it so that the local launcher can forward
        the daily ad‑auction outcomes.  Adjust PI estimates using EMA.
        """
        for camp, imps, cost in reports:
            if imps == 0:
                continue
            seg = camp.target_segment
            cpi_realised = cost / imps
            self.segment_pi[seg] = (
                (1 - self.ALPHA_PI) * self.segment_pi[seg] + self.ALPHA_PI * cpi_realised
            )

    # ----------------------------------------------------- #
    #  Competing‑index adjustment (based on auction wins)   #
    # ----------------------------------------------------- #

    def notify_campaign_result(self, campaign: Campaign, won: bool, random_assignment: bool):
        """Hook called by the simulator after campaign auctions. We update CI
        per ANL logic: shrink on win (unless random), grow on loss."""
        if won:
            if random_assignment:
                pass  # leave CI unchanged
            else:
                self.ci /= self.GREED
        else:
            self.ci *= self.GREED

# NOTE -------------------------------------------------------------------- #
# The local simulator bundled with the project does not yet invoke         #
# `update_after_impression_report` or `notify_campaign_result`.            #
# To use those hooks, extend the simulator or call them manually from      #
# your test harness (AdXGameSimulator) after each simulated day.           #
# ----------------------------------------------------------------------- #
# ─────────────────── quick offline test harness ─────────────────────────
if __name__ == "__main__":
    bots = [TACAgent()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(10)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=50)