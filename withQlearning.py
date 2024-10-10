import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(0)

# Parameters
T = 100  # Time steps
n_agents = 90  # Number of agents
income_mean = 100  # Mean income for agents
income_std = 10  # Income variability
beta = 1  # Impatience factor
initial_wealth = 0
marg_consumption = 0.5

investment_options = {
    "cash": {"mean": 0.05, "std": 0.05},  # Safe but low return
    "etf": {"mean": 0.2, "std": 0.1},    # Medium risk, medium return
    "stocks": {"mean": 0.30, "std": 0.15}    # High risk, high return
}

agents = {
    "risk_averse": {"alpha": 0.5, "risk_tolerance": 0.05},  # Concave utility, low tolerance for risk
    "risk_neutral": {"alpha": 1.0, "risk_tolerance": 0.07}, # Linear utility, moderate tolerance for risk
    "risk_seeking": {"alpha": 1.5, "risk_tolerance": 0.1}  # Convex utility, high tolerance for risk
}

class Agent:
    def __init__(self, risk_type):
        self.risk_type = risk_type
        self.wealth = initial_wealth
        self.option = None # Will be updated later
        self.q_table = {option: 0 for option in investment_options.keys()}
        self.counter = 0  # initialize q-values for each option

    def simulate_income(self):
        return np.random.normal(income_mean, income_std) # draw income for the agent
        
    def decide(self, income, cumwealth, exploration_rate):
        consumption = marg_consumption * income
        saving = income - consumption

        # Exploration/Exploitation
        if np.random.rand() < exploration_rate:  # Explore or exploit? 
            self.option = np.random.choice(list(investment_options.keys()))
        else: 
            if self.counter == 0: # Exploit - For the first period, choose randomly
                self.option = np.random.choice(list(investment_options.keys()))
            else:  # Exploit - for t > 0, choose the option with the highest Q-value
                self.option = max(self.q_table, key=self.q_table.get)

        chosen_return = np.real(np.random.normal(investment_options[self.option]['mean'], investment_options[self.option]['std']))

        # Update wealth
        self.wealth = (cumwealth + saving) * (1 + chosen_return)

        # period_wealth
        period_wealth = saving * (1 + chosen_return)

        self.counter +=1

        return self.wealth, period_wealth, chosen_return

    def utility(self, wealth, chosen_return, option):
        consumption = marg_consumption * income
        alpha = agents[self.risk_type]["alpha"]
        risk_tolerance = agents[self.risk_type]["risk_tolerance"]
        
        #Sanity check, making sure all args are real numbers
        wealth = np.real(wealth)
        chosen_return = np.real(chosen_return)
        consumption = np.real(consumption)
    
        if consumption <= 0:
            return -np.inf  # No negative consumption allowed

        base_utility = (consumption ** alpha) + beta * (wealth ** alpha)
        option_risk = investment_options[option]["std"]
        
        penalty = 0
        if option_risk > risk_tolerance:
            penalty = abs(chosen_return - risk_tolerance) * (consumption ** alpha)

        return base_utility - penalty

    def update_q_table(self, current_utility, t):
        # Update Q-value using the an update rule where I give 0.5 as the historical weight and 0.5 to the current utility value
        self.q_table[self.option] = (self.q_table[self.option] + current_utility)/(t+1)









# Run simulation
exploration_factors = [0.01, 0.5, 0.9]

results_risk_averse = [[[np.nan for _ in range(T)] for _ in range(len(exploration_factors))] for _ in range(n_agents)]
results_risk_neutral = [[[np.nan for _ in range(T)] for _ in range(len(exploration_factors))] for _ in range(n_agents)]
results_risk_seeking = [[[np.nan for _ in range(T)] for _ in range(len(exploration_factors))] for _ in range(n_agents)]

len_averse, len_neutral, len_seeking = 0, 0, 0

# Track investment choices by risk profile and exploration rate for the second plot
investment_counts = {
    "risk_averse": {rate: {option: 0 for option in investment_options.keys()} for rate in exploration_factors},
    "risk_neutral": {rate: {option: 0 for option in investment_options.keys()} for rate in exploration_factors},
    "risk_seeking": {rate: {option: 0 for option in investment_options.keys()} for rate in exploration_factors},
}

# Track investment choices per period by risk profile and exploration rate
selection_counts = {
    risk_type: {
        exploration_rate: {option: np.zeros(T) for option in investment_options.keys()}
        for exploration_rate in exploration_factors
    }
    for risk_type in agents.keys()
}

total_agents_per_type_exploration = {
    risk_type: {exploration_rate: 0 for exploration_rate in exploration_factors}
    for risk_type in agents.keys()
}

### Start Simulation
for id in range(n_agents):
    risk_type = np.random.choice(list(agents.keys())) 

    len_averse += (risk_type == 'risk_averse')
    len_neutral += (risk_type == 'risk_neutral')
    len_seeking += (risk_type == 'risk_seeking')

    agent = Agent(risk_type)
    income = agent.simulate_income()

    for exploration_index, exploration_rate in enumerate(exploration_factors):
        cumwealth = 0

        total_agents_per_type_exploration[risk_type][exploration_rate] += 1

        for t in range(T):
            wealth, period_wealth, chosen_return = agent.decide(income, cumwealth, exploration_rate)
            print(id, t, exploration_rate, wealth, period_wealth, income, chosen_return, risk_type)

            cumwealth += period_wealth
            option = agent.option
            utility = agent.utility(cumwealth, chosen_return, option)
            period_utility = agent.utility(period_wealth,chosen_return, option)

            # Update Q-table with the learned utility
            agent.update_q_table(period_utility, t)

            # Count investment selections by risk profile and exploration rate
            investment_counts[risk_type][exploration_rate][option] += 1
            # Count selections of each investment option over time
            selection_counts[risk_type][exploration_rate][option][t] += 1

            # Update results based on risk type
            if risk_type == 'risk_averse':
                results_risk_averse[id][exploration_index][t] = utility
            elif risk_type == 'risk_neutral':
                results_risk_neutral[id][exploration_index][t] = utility
            elif risk_type == 'risk_seeking':
                results_risk_seeking[id][exploration_index][t] = utility

# Average the results over the number of agents
mean_results_risk_averse = np.nanmean(results_risk_averse, axis=0)
mean_results_risk_neutral = np.nanmean(results_risk_neutral, axis=0)
mean_results_risk_seeking = np.nanmean(results_risk_seeking, axis=0)


################# FIRST PLOT #######################################
# Plot the avg.  wealth per exploration rate per risk profile type

fig, axs = plt.subplots(1, 3, figsize=(20, 10))

# Plot for risk-averse agents
for i, exploration_rate in enumerate(exploration_factors):
    axs[0].plot(mean_results_risk_averse[i], label=f'Exploration Rate {exploration_rate}')

axs[0].set_title('Utility by Period for Risk-Averse Agents')
axs[0].set_xlabel('Time')
axs[0].set_ylabel(f'Avg. Utility (N={len_averse})')
axs[0].legend()

# Plot for risk-neutral agents
for i, exploration_rate in enumerate(exploration_factors):
    axs[1].plot(mean_results_risk_neutral[i], label=f'Exploration Rate {exploration_rate}')

axs[1].set_title('Utility by Period for Risk-Neutral Agents')
axs[1].set_xlabel('Time')
axs[1].set_ylabel(f'Avg. Utility (N={len_neutral})')
axs[1].legend()

# Plot for risk-seeking agents
for i, exploration_rate in enumerate(exploration_factors):
    axs[2].plot(mean_results_risk_seeking[i], label=f'Exploration Rate {exploration_rate}')

axs[2].set_title('Utility by Period for Risk-Seeking Agents')
axs[2].set_xlabel('Time')
axs[2].set_ylabel(f'Avg. Utility (N={len_seeking})')
axs[2].legend()
axs[2].ticklabel_format(axis = 'y',style = 'plain')

plt.tight_layout()
plt.savefig('/Users/agustinafarias/Documents/3. MBR/Simulations/multi_arm_bandit/figure_1.png')
plt.show()


################# SECOND PLOT #######################################
# Plot the counts of each investment type chosen by agents

fig, axs = plt.subplots(1, 3, figsize=(20, 10))

for i, risk_type in enumerate(agents.keys()):
    for exploration_rate in exploration_factors:
        investment_selections = investment_counts[risk_type][exploration_rate]
        axs[i].bar(investment_selections.keys(), investment_selections.values(), label=f'Exploration Rate {exploration_rate}')
    
    axs[i].set_title(f'Investment Selections by {risk_type.replace("_", " ").title()} Agents')
    axs[i].set_xlabel('Investment Type')
    axs[i].set_ylabel('Count of Selections')
    axs[i].legend()

plt.tight_layout()
plt.show()

################# THIRD PLOT #######################################
# Plot the % of each investment type chosen by agents per exploration rate and period

# Calculate percentages
selection_percentages = {
    risk_type: {
        exploration_rate: {
            option: (selection_counts[risk_type][exploration_rate][option] / total_agents_per_type_exploration[risk_type][exploration_rate]) * 100
            for option in investment_options.keys()
        }
        for exploration_rate in exploration_factors
    }
    for risk_type in agents.keys()
}

# Plot the percentage of selections over time
fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
agent_types = list(agents.keys())


for i, risk_type in enumerate(agent_types):
    for j, exploration_rate in enumerate(exploration_factors):
        for option in investment_options.keys():
            axs[i, j].plot(range(T), selection_percentages[risk_type][exploration_rate][option], label=f'{option.title()}')

        axs[i, j].set_title(f'{risk_type.replace("_", " ").title()} Agents - Exp. Rate {exploration_rate}')
        if i == 2:
            axs[i, j].set_xlabel('Time Period')
        if j == 0:
            axs[i, j].set_ylabel('Percentage of Selections (%)')
        axs[i, j].legend()

plt.tight_layout()
plt.savefig('/Users/agustinafarias/Documents/3. MBR/Simulations/multi_arm_bandit/figure_3.png')
plt.show()


