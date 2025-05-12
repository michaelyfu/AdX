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

from john_agent import JohnAgent
from agt_server.agents.test_agents.adx.tier1.my_agent import Tier1NDaysNCampaignsAgent 
from agt_server.local_games.adx_arena import AdXGameSimulator 
from model_equilibrium.B99 import B99
from model_equilibrium.beta import BetaAgent
from model_equilibrium.eisenberg_agent import EisenbergAgent
from model_equilibrium.greedy import Greedy

def main():
    bots = [JohnAgent()] + [B99()] + [BetaAgent()] + [EisenbergAgent()] + [Greedy()] + [Tier1NDaysNCampaignsAgent(name=f"Tier1 {i}") for i in range(5)]
    AdXGameSimulator().run_simulation(agents=bots, num_simulations=100)


if __name__ == "__main__":
    main()