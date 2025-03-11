import streamlit as st
import networkx as nx
from pyvis.network import Network
import numpy as np
import streamlit.components.v1 as components
import pandas as pd


# ====== DATA PROCESSING AND PAGE SETUP ======

events = pd.read_csv("data/events_subset.csv")
ginf = pd.read_csv("data/ginf.csv")

st.set_page_config(layout="wide")
st.markdown(
    '''
    <style>
    body {
        background-color: black;
        color: white;
    }
    .block-container {
        background-color: black;
    }
    </style>
    '''
    , unsafe_allow_html=True
)


# ====== USER INPUTS INFORMATION ABOUT THE CURRENT GAME ======

st.sidebar.title("Current game time")
time = st.sidebar.slider("(minutes)", min_value=0, max_value=95, value=0, step=1)

st.sidebar.title("Parameters")
home_away = st.sidebar.number_input("Home (0) or Away (1)", min_value=0, max_value=1, value=0, step=1)
team1_substitutions = st.sidebar.number_input("Team 1 Substitutions", min_value=0, step=1)
team2_substitutions = st.sidebar.number_input("Team 2 Substitutions", min_value=0, step=1)
team1_yellow_cards = st.sidebar.number_input("Team 1 Yellow Cards", min_value=0, step=1)
team2_yellow_cards = st.sidebar.number_input("Team 2 Yellow Cards", min_value=0, step=1)
team1_total_shots = st.sidebar.number_input("Team 1 Total Shots", min_value=0, step=1)
team2_total_shots = st.sidebar.number_input("Team 2 Total Shots", min_value=0, step=1)
team1_goals = st.sidebar.number_input("Team 1 Current Goals", min_value=0, step=1)
team2_goals = st.sidebar.number_input("Team 2 Current Goals", min_value=0, step=1)


# ====== BAYESIAN ANALYSIS ====== #

def calculate_historical_probability(home_away, time, team1_goals, team2_goals):
    team_wins = 0
    total_matches = 0
    for match_id in events["id_odsp"].unique():
        match_events = events[(events["id_odsp"] == match_id) & (events["time"] <= time)]
        homegoals = match_events[(match_events["side"] == 1) & (match_events["is_goal"] == 1)].shape[0]
        awaygoals = match_events[(match_events["side"] == 2) & (match_events["is_goal"] == 1)].shape[0]
        match_result = ginf[ginf["id_odsp"] == match_id]
        if not match_result.empty and (home_away == 0 and awaygoals == team2_goals and homegoals == team1_goals or home_away == 1 and awaygoals == team1_goals and homegoals == team2_goals):
            fthg = match_result["fthg"].values[0]
            ftag = match_result["ftag"].values[0]
            if fthg > ftag:
                if home_away == 0:
                    team_wins += 1
            elif fthg < ftag:
                if home_away == 1:
                    team_wins += 1
            total_matches += 1
    
    return team_wins / total_matches if total_matches > 0 else 0.5

def p_morale(substitutions, home_away):
    return max(min(1.0, 0.5 + 0.1 * substitutions - 0.1 * home_away), 0)

def p_aggression(total_shots, yellow_cards):
    return max(min(1.0, 0.4 + 0.05 * total_shots - 0.2 * yellow_cards), 0)

def p_team1_win(morale1, aggression1, morale2, aggression2, home_away, time, team1_goals, team2_goals):
    morale_factor = 0.3 * (morale1 - morale2)  
    aggression_factor = 0.2 * (aggression1 - aggression2)  
    goal_factor = 0.4 * (team1_goals - team2_goals)  

    home_advantage = 0.1 if home_away == 0 else -0.1  
    
    base_prob = 0.5 + morale_factor + aggression_factor + goal_factor + home_advantage
    
    return max(0, min(1, (base_prob + calculate_historical_probability(home_away, time, team1_goals, team2_goals)) / 2))

def bernoulli(p):
    return 1 if np.random.uniform() < p else 0

def sample_particle():
    morale1 = p_morale(team1_substitutions, home_away)
    aggression1 = p_aggression(team1_total_shots, team1_yellow_cards)
    morale2 = p_morale(team1_substitutions, home_away)
    aggression2 = p_aggression(team1_total_shots, team1_yellow_cards)
    team1_predicted_goals = np.random.poisson((1.5 * morale1 + 1.2 * aggression1)*time/95)
    team2_predicted_goals = np.random.poisson((1.5 * morale2 + 1.2 * aggression2)*time/95)
    win1 = p_team1_win(bernoulli(morale1), bernoulli(aggression1), bernoulli(morale2), bernoulli(aggression2), home_away, time, team1_predicted_goals, team2_predicted_goals)
    return team1_predicted_goals, team2_predicted_goals, win1

N_SAMPLES = 1000
def rejection_sampling():
    samples = [sample_particle() for _ in range(N_SAMPLES)]
    filtered_samples = [s for s in samples if s[0] == team1_goals and s[1] == team2_goals]
    if len(filtered_samples) == 0:
        return 0  
    win_count = sum(s[2] for s in filtered_samples)
    return win_count / len(filtered_samples)

p_win = rejection_sampling()


# ====== DISPLAY RESULTS IN GRAPH ======

edges = [
    ("Home/Away", "Team 1 Morale"),
    ("Home/Away", "Team 2 Morale"),
    ("Home/Away", "Team 1 Win"),
    ("Team 1 Current Goals", "Team 1 Win"),
    ("Team 2 Current Goals", "Team 1 Win"),
    ("Team 1 Total Shots", "Team 1 Aggression"),
    ("Team 2 Total Shots", "Team 2 Aggression"),
    ("Team 1 Yellow Cards", "Team 1 Aggression"),
    ("Team 2 Yellow Cards", "Team 2 Aggression"),
    ("Team 1 Substitutions", "Team 1 Morale"),
    ("Team 2 Substitutions", "Team 2 Morale"),
    ("Team 1 Aggression", "Team 1 Win"),
    ("Team 2 Aggression", "Team 1 Win"),
    ("Team 1 Morale", "Team 1 Win"),
    ("Team 2 Morale", "Team 1 Win"),
    ("Team 1 Aggression", "Team 1 Current Goals"),  
    ("Team 1 Morale", "Team 1 Current Goals"),     
    ("Team 2 Aggression", "Team 2 Current Goals"),  
    ("Team 2 Morale", "Team 2 Current Goals")      
]

G = nx.DiGraph()
G.add_edges_from(edges)

net = Network(height="810px", width="110%", directed=True, bgcolor="#000000", font_color="white")

def get_node_color(node):
    if "Team 1" in node:
        return "#00AA00"
    elif "Team 2" in node:
        return "#AA0000"
    elif "Home/Away" in node:
        return "#FFD700"
    else:
        return "#FFFFFF"

def get_label(node):
    if "win" in node.lower():
        return f"{node.lower()}\n({round(p_win, 2)})"
    elif "morale" in node.lower():
        morale_value = p_morale(team1_substitutions, home_away) if "team 1" in node.lower() else p_morale(team2_substitutions, home_away)
        return f"{node.lower()}\n({round(morale_value, 2)})"
    elif "aggression" in node.lower():
        aggression_value = p_aggression(team1_total_shots, team1_yellow_cards) if "team 1" in node.lower() else p_aggression(team2_total_shots, team2_yellow_cards)
        return f"{node.lower()}\n({round(aggression_value, 2)})"
    elif "goals" in node.lower():
        goals_value = team1_goals if "team 1" in node.lower() else team2_goals
        return f"{node.lower()}\n({goals_value})"
    elif "shots" in node.lower():
        shots_value = team1_total_shots if "team 1" in node.lower() else team2_total_shots
        return f"{node.lower()}\n({shots_value})"
    elif "yellow cards" in node.lower():
        cards_value = team1_yellow_cards if "team 1" in node.lower() else team2_yellow_cards
        return f"{node.lower()}\n({cards_value})"
    elif "substitutions" in node.lower():
        subs_value = team1_substitutions if "team 1" in node.lower() else team2_substitutions
        return f"{node.lower()}\n({subs_value})"
    elif "home/away" in node.lower():
        return f"{node.lower()}\n({home_away})"
    else:
        return node.lower()

for node in G.nodes():
    label = get_label(node)
    net.add_node(node, label=label, color=get_node_color(node), size=10)

for edge in edges:
    net.add_edge(edge[0], edge[1])

net.save_graph("bayesian_network.html")

st.title("Bayesian Rejection Sampling for Football Outcomes")

with open("bayesian_network.html", "r", encoding="utf-8") as f:
    html_string = f.read()

html_string = html_string.replace(
    "#mynetwork {", 
    "#mynetwork { top: -15px; left: -5px; margin: 0; width: 100%; height: 100%;"
)
components.html(html_string, height=800, scrolling=False)
