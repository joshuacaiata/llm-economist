from agents.base_agent import BaseAgent

class MobileAgent(BaseAgent):
    def __init__(
            self,
            agent_class,
            agent_id,
            config,
            build_payout,
            risk_aversion,
            env
        ):
        super().__init__(agent_class, agent_id, config, env)
        self.agent_class = agent_class
        self.agent_id = agent_id
        self.config = config
        self.env = env
        self.build_payout = build_payout
        self.risk_aversion = risk_aversion

    def observe(self, env):
        """
        Agents observation space:
        1. Neighbourhood (everything in the view_size radius)
        2. Inventory
        3. Build payout
        TODO:
        4. Taxation 
            i. Planner tax schedule
            ii. Agents current tax bracket
            iii. Anonymized and sorted income distribution of all agents
        5. Time to next tax period
        TODO:
        6. Monetary policy
            i. Interest rate
            ii. Inflation rate
            iii. Money supply
            iv. Target inflation rate
        7. Position
        """
        pass

    def step(self, action):
        pass

    def reset(self):
        pass