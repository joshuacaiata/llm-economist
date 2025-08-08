import numpy as np

class TradingSystem:
    def __init__(self, config: dict, env):
        self.config = config
        self.buy = {
            "wood": [],
            "stone": []
        }
        self.sell = {
            "wood": [],
            "stone": []
        }
        self.time = 0 

        self.env = env

        self.max_order_lifetime = config["max_order_lifetime"]

        self.num_trades = {
            "wood": 0,
            "stone": 0
        }

        self.last_price = {
            "wood": 0,
            "stone": 0
        }

        self.trades = {
            "wood": [],
            "stone": []
        }
        
        self.order_history = {
            "wood": {"buy": [], "sell": []},
            "stone": {"buy": [], "sell": []}
        }
    
    def make_order(
            self,
            agent_id: int,
            resource: str,
            price: int,
            transaction: str,
    ):
        if transaction == "buy":
            buy = {
                "agent_id": agent_id,
                "price": price,
                "lifetime": 0
            }
            self.buy[resource].append(buy)
        elif transaction == "sell":
            sell = {
                "agent_id": agent_id,
                "price": price,
                "lifetime": 0
            }
            self.sell[resource].append(sell)
    
    def satisfy_sell(self, resource, ask):
        agent_id = ask["agent_id"]
        price = ask["price"]
        
        agent = next((a for a in self.env.mobile_agents if a.agent_id == agent_id), None)
        if agent:
            agent.inventory["coins"] += price
            agent.escrow[resource] -= 1
            agent.active_orders -= 1
        
        self.sell[resource].remove(ask)

    def satisfy_buy(self, resource, bid, ask_price):
        agent_id = bid["agent_id"]
        
        agent = next((a for a in self.env.mobile_agents if a.agent_id == agent_id), None)
        if agent:
            agent.inventory[resource] += 1
            agent.escrow["coins"] -= ask_price
            agent.active_orders -= 1

            difference = bid["price"] - ask_price
            agent.escrow["coins"] -= difference
            agent.inventory["coins"] += difference
        
        self.buy[resource].remove(bid)
    
    def step(self):
        resources = ["wood", "stone"]
        self.time += 1 

        for resource in resources:
            self.order_history[resource]["buy"].append(len(self.buy[resource]))
            self.order_history[resource]["sell"].append(len(self.sell[resource]))

        for resource in resources:
            sorted_buys = sorted(self.buy[resource], key=lambda x: x["price"], reverse=True)
            sorted_sells = sorted(self.sell[resource], key=lambda x: x["price"])

            matched_buys = []
            matched_sells = []

            for buy in sorted_buys:
                buy_agent_id = buy["agent_id"]

                if buy in matched_buys:
                    continue
                
                for sell in sorted_sells:
                    sell_agent_id = sell["agent_id"]

                    if sell in matched_sells or buy_agent_id == sell_agent_id:
                        continue
                    
                    if buy["price"] >= sell["price"]:
                        matched_buys.append(buy)
                        matched_sells.append(sell)

                        self.satisfy_sell(resource, sell)
                        self.satisfy_buy(resource, buy, sell["price"])

                        self.num_trades[resource] += 1

                        self.trades[resource].append((self.env.time, sell["price"]))

                        break
            
            if len(matched_sells) > 0:
                matched_prices = [sell["price"] for sell in matched_sells]
                self.last_price[resource] = np.mean(matched_prices)
        
        for resource in resources:
            expired_sells = []

            for sell in self.sell[resource]:
                sell["lifetime"] += 1

                if sell["lifetime"] > self.max_order_lifetime:
                    expired_sells.append(sell)
                    
            for sell in expired_sells:
                agent = next((a for a in self.env.mobile_agents if a.agent_id == sell["agent_id"]), None)
                if agent:
                    agent.inventory[resource] += 1
                    agent.escrow[resource] -= 1
                    agent.active_orders -= 1

                self.sell[resource].remove(sell)
            
            expired_buys = []

            for buy in self.buy[resource]:
                buy["lifetime"] += 1

                if buy["lifetime"] > self.max_order_lifetime:
                    expired_buys.append(buy)
                    
            for buy in expired_buys:
                agent = next((a for a in self.env.mobile_agents if a.agent_id == buy["agent_id"]), None)
                if agent:
                    agent.inventory["coins"] += buy["price"]
                    agent.escrow["coins"] -= buy["price"]
                    agent.active_orders -= 1 

                self.buy[resource].remove(buy)

    def reset_episode(self):
        self.buy = {
            "wood": [],
            "stone": []
        }
        self.sell = {
            "wood": [],
            "stone": []
        }

        self.num_trades = {
            "wood": 0,
            "stone": 0
        }

        self.last_price = {
            "wood": 0,
            "stone": 0
        }

        self.trades = {
            "wood": [],
            "stone": []
        }

        self.order_history = {
            "wood": {"buy": [], "sell": []},
            "stone": {"buy": [], "sell": []}
        }
        
        self.time = 0