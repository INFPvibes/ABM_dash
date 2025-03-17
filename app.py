# Import required libraries
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import numpy as np
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid

# --- Define Original Depression Model ---
class DepressionAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.age = random.randint(13, 18)
        self.T1Depression = random.randint(40, 90)
        self.CSQ8 = random.randint(18, 32)
        self.ChngDepression = random.randint(-17, 35)

    def step(self):
        pass

class DepressionModel(Model):
    def __init__(self, N, width, height):
        super().__init__()
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        for i in range(self.num_agents):
            a = DepressionAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        self.schedule.step()

# --- Define New IBD Achievement Model ---
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
        self.schedule = RandomActivation(self)
        self.current_timepoint = 1
        self.durations = durations
        
        for i in range(self.num_agents):
            self.schedule.add(IBDAgent(i, self))
        
        self.datacollector = DataCollector(
            agent_reporters={"Achievement": lambda a: a.achievement}
        )

    def step(self):
        if self.current_timepoint <= 5:
            for agent in self.schedule.agents:
                agent.session_duration = self.durations[self.current_timepoint - 1]
        self.schedule.step()
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

# --- Combined Dashboard Layout ---
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Mental Health Intervention ABM Dashboard"),
    
    # Depression Model Section
    html.Div([
        html.H2("Depression Model"),
        html.Label("CSQ8 Influence"),
        dcc.Slider(id='csq8-slider', min=18, max=32, step=1, value=25),
        html.Label("Baseline Depression"),
        dcc.Slider(id='baseline-slider', min=40, max=90, step=1, value=65),
    ], style={'padding': '20px', 'border': '1px solid #ddd'}),
    
    # IBD Achievement Model Section
    html.Div([
        html.H2("IBD Achievement Model"),
        html.Label("Session Duration Between Weeks 1-2"),
        dcc.Slider(id='interval1', min=70, max=800, value=400),
        html.Label("Session Duration Between Weeks 2-3"),
        dcc.Slider(id='interval2', min=70, max=800, value=400),
        html.Label("Session Duration Between Weeks 3-4"),
        dcc.Slider(id='interval3', min=70, max=800, value=400),
        html.Label("Session Duration Between Weeks 4-5"),
        dcc.Slider(id='interval4', min=70, max=800, value=400),
        html.Label("Session Duration Between Weeks 5-6"),
        dcc.Slider(id='interval5', min=70, max=800, value=400),
    ], style={'padding': '20px', 'border': '1px solid #ddd'}),
    
    # Visualizations
    html.Div([
        dcc.Graph(id='depression-plot1'),
        dcc.Graph(id='depression-plot2'),
        dcc.Graph(id='depression-plot3'),
        dcc.Graph(id='achievement-plot')
    ], style={'columnCount': 2})
])

# --- Combined Callbacks ---
@app.callback(
    [Output('depression-plot1', 'figure'),
     Output('depression-plot2', 'figure'),
     Output('depression-plot3', 'figure'),
     Output('achievement-plot', 'figure')],
    [Input('csq8-slider', 'value'),
     Input('baseline-slider', 'value'),
     Input('interval1', 'value'),
     Input('interval2', 'value'),
     Input('interval3', 'value'),
     Input('interval4', 'value'),
     Input('interval5', 'value')]
)
def update_all_plots(csq8_influence, baseline_influence, i1, i2, i3, i4, i5):
    try:
        # Update depression plots
        fig1, fig2, fig3 = generate_depression_plots(csq8_influence, baseline_influence)
        
        # Update achievement plot
        achievement_fig = generate_achievement_plot([i1, i2, i3, i4, i5])
        
        return fig1, fig2, fig3, achievement_fig
    except Exception as e:
        print(f"Error in update_all_plots: {str(e)}")
        # Return empty figures in case of error
        return [go.Figure() for _ in range(4)]

def generate_depression_plots(csq8_influence, baseline_influence):
    num_agents = 50
    csq8_values = [max(18, min(32, int(np.random.normal(csq8_influence, 2)))) for _ in range(num_agents)]
    t1_values = [max(40, min(90, int(np.random.normal(baseline_influence, 5)))) for _ in range(num_agents)]
    
    chng_depression_csq8_values = [-1.2437 * (csq8 - 25) + np.random.normal(0, 2) for csq8 in csq8_values]
    chng_depression_t1_values = [-0.2412 * (t1_depression - 65) + np.random.normal(0, 2) for t1_depression in t1_values]
    
    chng_depression_moderated_values = [
        calculate_depression_change(csq8, bd) for csq8, bd in zip(csq8_values, t1_values)
    ]

    fig1 = go.Figure(data=go.Scatter(x=csq8_values,
                                     y=chng_depression_csq8_values,
                                     mode='markers',
                                     marker=dict(opacity=0.6)))
    fig1.update_layout(title='CSQ8 vs Change in Depression',
                       xaxis_title='CSQ8 Score',
                       yaxis_title='Change in Depression',
                       xaxis_range=[17, 33],
                       yaxis_range=[-40, 10])

    fig2 = go.Figure(data=go.Scatter(x=t1_values,
                                     y=chng_depression_t1_values,
                                     mode='markers',
                                     marker=dict(opacity=0.6)))
    fig2.update_layout(title='Baseline Depression vs Change in Depression',
                       xaxis_title='Baseline Depression (T1)',
                       yaxis_title='Change in Depression',
                       xaxis_range=[35, 95],
                       yaxis_range=[-40, 10])

    fig3 = go.Figure(data=go.Scatter(x=csq8_values,
                                     y=chng_depression_moderated_values,
                                     mode='markers',
                                     marker=dict(color=t1_values,
                                                 colorscale='Plasma',
                                                 opacity=0.7,
                                                 size=10,
                                                 colorbar=dict(title='Baseline Depression'))))
    fig3.update_layout(title='Moderation Effect of Baseline Depression',
                       xaxis_title='CSQ8 Score',
                       yaxis_title='Change in Depression',
                       xaxis_range=[17, 33],
                       yaxis_range=[-40, 10])

    return fig1, fig2, fig3

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

def generate_achievement_plot(durations):
    model = IBDModel(50, durations)
    for _ in range(6):
        model.step()
    
    agent_data = model.datacollector.get_agent_vars_dataframe()
    timepoints = range(1, 7)
    averages = [agent_data.xs(t, level="Step")["Achievement"].mean() for t in timepoints]
    
    return {
        'data': [go.Scatter(
            x=timepoints,
            y=averages,
            mode='lines+markers',
            line=dict(color='royalblue')
        )],
        'layout': go.Layout(
            title='Average Achievement Over Time',
            xaxis={'title': 'Timepoint'},
            yaxis={'title': 'Achievement Score', 'range': [0, 350]},
            hovermode='closest'
        )
    }

if __name__ == '__main__':
    app.run_server(debug=True)
