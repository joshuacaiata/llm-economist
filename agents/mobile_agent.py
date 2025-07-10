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


    def observe(self, env):
        pass

    def step(self, action):
        pass