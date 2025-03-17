# Import required libraries
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import random
from mesa import Agent, Model
from mesa.datacollection import DataCollector

# --- Define Agents and Model ---
class IBDAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.session_duration = 0
        self.achievement = 0

    def step(self):
        self.achievement += self.model.calculate_achievement_change(self.session_duration)

class IBDModel(Model):
    def __init__(self, N, durations):
        super().__init__()
        self.num_agents = N
        self.durations = durations
        self.current_timepoint = 1
        
        # Create agents
        self.agents = [IBDAgent(i, self) for i in range(self.num_agents)]
        
        # Data collector for tracking agent achievements
        self.datacollector = DataCollector(
            agent_reporters={"Achievement": lambda a: a.achievement}
        )

    def step(self):
        if self.current_timepoint <= len(self.durations):
            for agent in self.agents:
                agent.session_duration = self.durations[self.current_timepoint - 1]
        
        random.shuffle(self.agents)
        for agent in self.agents:
            agent.step()
        
        self.datacollector.collect(self)
        self.current_timepoint += 1

    def calculate_achievement_change(self, session_duration):
        if self.current_timepoint < 3:
            b_x = 0.1294
        elif self.current_timepoint < 5:
            b_x = 0.2885
        else:
            b_x = 0.4476

        z_strength = self.current_timepoint * 0.07955
        return session_duration * b_x * z_strength

# --- Generate Plots ---
def generate_plot(durations):
    model = IBDModel(50, durations)
    for _ in range(len(durations)):
        model.step()
    
    agent_data = model.datacollector.get_agent_vars_dataframe()
    timepoints = list(range(1, len(durations) + 1))
    avg_achievements = [float(agent_data.xs(t, level="Step")["Achievement"].mean()) for t in timepoints]

    fig = go.Figure(data=go.Scatter(x=timepoints, y=avg_achievements, mode='lines+markers'))
    fig.update_layout(title="Average Achievement Over Time",
                      xaxis_title="Timepoint",
                      yaxis_title="Average Achievement",
                      xaxis=dict(tickmode='linear'),
                      yaxis=dict(range=[0, max(avg_achievements) + 50]))
    return fig

# --- Dash App Setup ---
app = dash.Dash(__name__)
server = app.server  # For deployment on Render

app.layout = html.Div([
    html.H1("Agent-Based Model: Achievement Over Time"),
    
    html.Div([
        html.Label(f"Interval {i+1} Duration"),
        dcc.Slider(id=f'interval-{i}', min=70, max=800, step=10, value=400)
    ] for i in range(5)),

    dcc.Graph(id='achievement-plot')
])

@app.callback(
    Output('achievement-plot', 'figure'),
    [Input(f'interval-{i}', 'value') for i in range(5)]
)
def update_plot(*durations):
    return generate_plot(list(durations))

if __name__ == '__main__':
    app.run_server(debug=True)
