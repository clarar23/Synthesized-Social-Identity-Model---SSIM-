import numpy as np 

n = 200
num_steps = 200000
record_every = 100

class Agent:
    pass

def run_simulation(seed, 
                   prop_A=0.1, 
                   X_A_min=0.75, 
                   X_A_max=1.0, 
                   X_B_min=0.0, 
                   X_B_max=0.75,         
                   threshold_base=0.2, 
                   e_min=0.8, 
                   e_max=1.0, 
                   alpha=0.2,      
                   gamma=0.5,      
                   delta_charge=0.05,   
                   delta_drain=0.05, 
                   beta=0.01,       
                   use_SIA=True,          
                   homophily=0.0,         
                   hardcore_activists=False): 
    
    np.random.seed(seed)
    agents = []
    
    # Agent Initialization
    for i in range(n):
        ag = Agent()
        ag.id = i
        
        if i < prop_A * n:
            ag.group = "A"
            ag.opinion = np.random.uniform(X_A_min, X_A_max) 
        else: 
            ag.group = "B"
            ag.opinion = np.random.uniform(X_B_min, X_B_max)
            
        ag.threshold_base = threshold_base
        ag.selfesteem = np.random.uniform(e_min, e_max)
        
        ag.alpha = alpha
        ag.gamma = gamma
        ag.beta = beta
        ag.delta_charge = delta_charge
        ag.delta_drain = delta_drain
        
        agents.append(ag)
        
    if hardcore_activists:
        for ag in agents[:4]:
            ag.beta = 0.0
            ag.gamma = 0.9

    if use_SIA:
        opinionsA = [ag.opinion for ag in agents if ag.group == "A"]
        opinionsB = [ag.opinion for ag in agents if ag.group == "B"]
        identityA = sum(opinionsA)/len(opinionsA) if len(opinionsA) > 0 else 0
        identityB = sum(opinionsB)/len(opinionsB) if len(opinionsB) > 0 else 0
        
        for ag in agents: 
            ag.identity = identityA if ag.group == "A" else identityB
    else:
        for ag in agents:
            ag.identity = ag.opinion

    # Universal Dictionary: Tracks BOTH Macro-level stats and Micro-level agents
    history = {
        'op_A': [], 'op_B': [], 'id_A': [], 'id_B': [], 'se_A': [], 'se_B': [], 
        'sd_op_A': [], 'sd_op_B': [],
        'ind_opinions': [], 'ind_identities': [], 'ind_selfesteem': [], 
        'agent_groups': [ag.group for ag in agents]
    }

    def record_state():
        # Macro (Group Means & Spread)
        history['op_A'].append(np.mean([ag.opinion for ag in agents if ag.group == "A"])) 
        history['op_B'].append(np.mean([ag.opinion for ag in agents if ag.group == "B"])) 
        history['id_A'].append(np.mean([ag.identity for ag in agents if ag.group == "A"]))  
        history['id_B'].append(np.mean([ag.identity for ag in agents if ag.group == "B"])) 
        history['se_A'].append(np.mean([ag.selfesteem for ag in agents if ag.group == "A"])) 
        history['se_B'].append(np.mean([ag.selfesteem for ag in agents if ag.group == "B"]))
        history['sd_op_A'].append(np.std([ag.opinion for ag in agents if ag.group == "A"]))
        history['sd_op_B'].append(np.std([ag.opinion for ag in agents if ag.group == "B"]))
        
        # Micro (Individual Agent Tracking for the Representative Seed Plot)
        history['ind_opinions'].append([ag.opinion for ag in agents])
        history['ind_identities'].append([ag.identity for ag in agents])
        history['ind_selfesteem'].append([ag.selfesteem for ag in agents])
        
    record_state() 

    agents_A = [ag for ag in agents if ag.group == "A"]
    agents_B = [ag for ag in agents if ag.group == "B"]

    # Social Identity Model Mechanisms
    for t in range(num_steps):
        ag_x = agents[np.random.randint(0, n)]
        
        if np.random.rand() < homophily:
            pool = agents_A if ag_x.group == "A" else agents_B
            ag_y = pool[np.random.randint(0, len(pool))]
        else:
            ag_y = agents[np.random.randint(0, n)]

        if ag_y.id != ag_x.id: 
            threshold = ag_x.threshold_base * ag_x.selfesteem
            error = ag_y.opinion - ag_x.identity
            
            if abs(error) <= threshold:
                ag_x.opinion = ag_x.opinion + (ag_x.alpha * (ag_y.opinion - ag_x.opinion)) 
                ag_x.selfesteem = ag_x.selfesteem + ag_x.delta_charge
            else:
                ag_x.opinion = ag_x.opinion + (ag_x.gamma * (ag_x.identity - ag_x.opinion))
                ag_x.selfesteem = ag_x.selfesteem - ag_x.delta_drain
                
        if ag_x.selfesteem <= 0.0001: 
            ag_x.identity = ag_x.identity + (ag_x.beta * error) 
            ag_x.selfesteem = 1.0
            
        ag_x.opinion = max(0.0, min(1.0, ag_x.opinion))
        ag_x.selfesteem = max(0.0, min(1.0, ag_x.selfesteem))

        if (t + 1) % record_every == 0:
            record_state()
            
    return history