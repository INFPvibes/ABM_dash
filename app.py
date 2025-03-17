# Import required libraries
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import random
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

# --- Define Agents and Models ---
class MyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = random.randint(13, 18)
        self.T1Depression = random.randint(40, 90)
        self.CSQ8 = random.randint(18, 32)
        self.ChngDepression = random.randint(-17, 35)

    def step(self):
        pass

class IBDAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.session_duration = 0
        self.achievement = 0

    def step(self):
        self.achievement += self.model.calculate_achievement_change(self.session_duration)

class IBDModel(Model):
    def __init__(self, N, durations):
        self.num_agents = N
        self.schedule = []
        self.current_timepoint = 1
        self.durations = durations
        
        for i in range(self.num_agents):
            self.schedule.append(IBDAgent(i, self))
        
        self.datacollector = DataCollector(
            agent_reporters={"Achievement": lambda a: a.achievement}
        )

    def step(self):
        if self.current_timepoint <= 5:
            for agent in self.schedule:
                agent.session_duration = self.durations[self.current_timepoint - 1]
        for agent in self.schedule:
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
def generate_plots(csq8_influence, baseline_depression_influence):
    num_agents = 50
    csq8_values_clustered = [max(18, min(32, int(np.random.normal(csq8_influence, 2)))) for _ in range(num_agents)]
    t1_depression_values_clustered = [max(40, min(90, int(np.random.normal(baseline_depression_influence, 5)))) for _ in range(num_agents)]

    chng_depression_csq8_values = [-1.2437 * (csq8 - 25) + np.random.normal(0, 2) for csq8 in csq8_values_clustered]
    chng_depression_t1_values = [-0.2412 * (t1_depression - 65) + np.random.normal(0, 2) for t1_depression in t1_depression_values_clustered]

    chng_depression_moderated_values = [
        calculate_depression_change(csq8, bd) for csq8, bd in zip(csq8_values_clustered, t1_depression_values_clustered)
    ]

    fig1 = go.Figure(data=[
        go.Scatter(
            x=csq8_values_clustered,
            y=chng_depression_csq8_values,
            mode='markers',
            marker=dict(opacity=0.6)
        )
    ])
    
    fig2 = go.Figure(data=[
        go.Scatter(
            x=t1_depression_values_clustered,
            y=chng_depression_t1_values,
            mode='markers',
            marker=dict(opacity=0.6)
        )
    ])
    
    fig3 = go.Figure(data=[
        go.Scatter(
            x=csq8_values_clustered,
            y=chng_depression_moderated_values,
            mode='markers',
            marker=dict(
                color=t1_depression_values_clustered,
                colorscale='Plasma',
                opacity=0.7,
                size=10,
                colorbar=dict(title='Baseline Depression')
            )
        )
    ])

    return fig1, fig2, fig3

# Add this function definition
def calculate_depression_change(csq8, baseline_depression):
    normalized_baseline = (baseline_depression - 44) / (60 - 44)
    if baseline_depression <= 44:
        effect = -1.8489
    elif baseline_depression >= 60:
        effect = -0.9117
    else:
        effect = -1.8489 + normalized_baseline * (-0.9117 + 1.8489)
    change = effect * (csq8 - 25)
    return change + np.random.normal(0, 2)

# --- Dash App ---
app = dash.Dash(__name__)
server = app.server # For deployment

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Depression Model', children=[
            html.H1("Agent-Based Model for Depression"),
            html.Label("CSQ8 Influence"),
            dcc.Slider(id='csq8-slider', min=18, max=32, step=1, value=25),
            html.Label("Baseline Depression"),
            dcc.Slider(id='baseline-slider', min=40, max=90, step=1, value=65),
            html.Div(className='row', children=[
                dcc.Graph(id='plot1', style={'display': 'inline-block', 'width': '33%'}),
                dcc.Graph(id='plot2', style={'display': 'inline-block', 'width': '33%'}),
                dcc.Graph(id='plot3', style={'display': 'inline-block', 'width': '33%'})
            ])
        ]),
        dcc.Tab(label='IBD Model', children=[
            html.H1("Agent-Based Model for IBD"),
            html.Div([
                html.Label(f'Interval {i+1}'),
                dcc.Slider(id=f'interval-slider-{i+1}', min=70, max=800, step=10, value=400)
                for i in range(5)
            ]),
            dcc.Graph(id='ibd-plot')
        ])
    ])
])

@app.callback(
    [Output('plot1', 'figure'),
     Output('plot2', 'figure'),
     Output('plot3', 'figure')],
    Input('csq8-slider', 'value'),
    Input('baseline-slider', 'value')
)
def update_graph(csq8_influence, baseline_depression_influence):
    fig1, fig2, fig3 = generate_plots(csq8_influence, baseline_depression_influence)
    return fig1, fig2, fig3

@app.callback(
    Output('ibd-plot', 'figure'),
    [Input(f'interval-slider-{i+1}', 'value') for i in range(5)]
)
def update_ibd_graph(*intervals):
    durations = list(intervals)
    
    model = IBDModel(50, durations)
    
    for _ in range(6):
        model.step()
    
    agent_data = model.datacollector.get_agent_vars_dataframe()
    
    timepoints = list(range(1, 7))
    
    average_achievements = [
        agent_data.xs(timestep_idx, level="Step")["Achievement"].mean()
        for timestep_idx in timepoints
    ]
    
    fig = go.Figure(data=[
        go.Scatter(
            x=timepoints,
            y=average_achievements,
            mode='lines+markers'
        )
    ])
    
    fig.update_layout(
        title='Average Achievement Over Time',
        xaxis_title='Timepoint',
        yaxis_title='Average Achievement Score',
        yaxis_range=[0, 350]
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
