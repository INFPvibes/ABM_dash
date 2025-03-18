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
        
        # Attributes for Anxiety Model
        self.anxiety_change = random.randint(-30, 50)

        # Attributes for IBD Model
        self.achievement = 0
        self.session_durations = [random.randint(70, 800) for _ in (5)]

    def step(self):
        pass

class UnifiedModel(Model):
    def __init__(self, N, width, height):
        self.num_agents = N
        self.grid = MultiGrid(width, height, True)
        self.agents = []
        for i in range(self.num_agents):
            a = MyAgent(i, self)
            self.agents.append(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def step(self):
        for agent in self.agents:
            agent.step()

def run_model(N=50, width=10, height=10, steps=100):
    model = UnifiedModel(N, width, height)
    for _ in range(steps):
        model.step()
    return model

# --- CSQ8 Moderation Effect Calculation ---
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

# --- Generate Depression Change Plots ---
def generate_plots(csq8_influence, baseline_depression_influence):
    num_agents = 50
    csq8_values_clustered = [max(18, min(32, int(np.random.normal(csq8_influence, 2)))) for _ in range(num_agents)]
    t1_depression_values_clustered = [max(40, min(90, int(np.random.normal(baseline_depression_influence, 5)))) for _ in range(num_agents)]

    chng_depression_csq8_values = [-1.2437 * (csq8 - 25) + np.random.normal(0, 2) for csq8 in csq8_values_clustered]
    chng_depression_t1_values = [-0.2412 * (t1_depression - 65) + np.random.normal(0, 2) for t1_depression in t1_depression_values_clustered]

    chng_depression_moderated_values = [
        calculate_depression_change(csq8, bd) for csq8, bd in zip(csq8_values_clustered, t1_depression_values_clustered)
    ]

    fig1 = go.Figure(data=go.Scatter(x=csq8_values_clustered,
                                     y=chng_depression_csq8_values,
                                     mode='markers',
                                     marker=dict(opacity=0.6)))
    fig1.update_layout(title='CSQ8 vs Change in Depression',
                       xaxis_title='CSQ8 Score',
                       yaxis_title='Change in Depression',
                       xaxis_range=[17, 33],
                       yaxis_range=[-40, 10])

    fig2 = go.Figure(data=go.Scatter(x=t1_depression_values_clustered,
                                     y=chng_depression_t1_values,
                                     mode='markers',
                                     marker=dict(opacity=0.6)))
    fig2.update_layout(title='Baseline Depression vs Change in Depression',
                       xaxis_title='Baseline Depression (T1)',
                       yaxis_title='Change in Depression',
                       xaxis_range=[35, 95],
                       yaxis_range=[-40, 10])

    fig3 = go.Figure(data=go.Scatter(x=csq8_values_clustered,
                                     y=chng_depression_moderated_values,
                                     mode='markers',
                                     marker=dict(color=t1_depression_values_clustered,
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

# --- Generate Anxiety TAB Plot ---
def generate_anxiety_plot_anxiety_tab(attendance_influence, csq8_influence):
    num_agents = 50
    # Generate CSQ8 values clustered around the slider value
    csq8_values = [max(18, min(32, int(np.random.normal(csq8_influence, 3)))) for _ in range(num_agents)]
    # Generate random attendance values for agents
    attendance_values = [max(1, min(10, int(np.random.normal(attendance_influence, 1)))) for _ in range(num_agents)]
    
    anxiety_change = []

    # Calculate anxiety change based on CSQ8 and attendance
    for csq8, attendance in zip(csq8_values, attendance_values):
        if attendance <= 4:
            effect = -3.2783  # Strong negative effect at low attendance
        else:
            effect = -3.2783 + (attendance - 4) * 0.5  # Weaken with higher attendance
        change = effect * (csq8 - 25) + np.random.normal(0, 10)
        anxiety_change.append(change)

    # Create the scatter plot with a color gradient based on attendance
    fig = go.Figure(data=go.Scatter(
        x=csq8_values,
        y=anxiety_change,
        mode='markers',
        marker=dict(
            color=attendance_values,  # Color by attendance
            colorscale='Viridis',     # Choose a colorscale
            colorbar=dict(title='Attendance'),
            opacity=0.7,
            size=10
        )
    ))
    
    fig.update_layout(
        title=f'CSQ8 and Anxiety Change Moderated by Attendance ({attendance_influence} Sessions)',
        xaxis_title='CSQ8 Score',
        yaxis_title='Anxiety Change',
        xaxis_range=[17, 33],
        yaxis_range=[-60, 60]
    )
    
    return fig

# --- Generate IBD Plot ---
def generate_ibd_plot(durations):
    """
    Generate the IBD plot using the unified agent structure.
    """
    num_timepoints = 6
    achievements = []
    cumulative_effect = 0
    momentum = 0
    decay_factor = 0.8  # How much previous achievement carries over if no new sessions

    for timepoint in range(1, num_timepoints + 1):
        duration_idx = min(timepoint - 1, len(durations) - 1)
        session_duration = durations[duration_idx]

        if timepoint < 3:
            b_x = 0.1294
        elif timepoint < 5:
            b_x = 0.2885
        else:
            b_x = 0.4476

        z_strength = timepoint * 0.07955

        # Calculate new achievement gain
        new_achievement = session_duration * b_x * z_strength

        # Update momentum (increases with consistent use, decreases with inactivity)
        if session_duration > 0:
            momentum = min(momentum + 0.1, 1.0)  # Cap momentum at 1.0
        else:
            momentum = max(momentum - 0.2, 0)  # Momentum decreases faster than it builds

        # Apply momentum to new achievement
        new_achievement *= (1 + momentum)

        # Add new achievement to cumulative effect
        cumulative_effect = (cumulative_effect * decay_factor) + new_achievement

        achievements.append(cumulative_effect)

    fixed_y_min = 0
    fixed_y_max = max(achievements) * 1.1

    fig = go.Figure(data=[
        go.Scatter(
            x=list(range(1, num_timepoints + 1)),
            y=achievements,
            mode='lines+markers'
        )
    ])

    fig.update_layout(
        title='Cumulative Achievement Over Time',
        xaxis_title='Timepoint',
        yaxis_title='Cumulative Achievement Score',
        yaxis=dict(range=[fixed_y_min, fixed_y_max])
    )

    return fig, np.mean(achievements)

# --- Generate Depression Change Plot ---
def generate_depression_plot(average_achievement):
    num_agents = 50
    achievement_values = np.random.normal(average_achievement, average_achievement * 0.1, num_agents)
    depression_change = -0.7 * achievement_values + np.random.normal(0, 10, num_agents)

    fig = go.Figure(data=go.Scatter(
        x=achievement_values,
        y=depression_change,
        mode='markers',
        marker=dict(opacity=0.6)
    ))

    fig.update_layout(
        title='Achievement vs Depression Change',
        xaxis_title='Achievement Score',
        yaxis_title='Depression Change',
        yaxis_range=[-40, 40]
    )

    return fig

# --- Generate Anxiety Change Plot ---
def generate_anxiety_plot(average_achievement):
    num_agents = 50
    achievement_values = np.random.normal(average_achievement, average_achievement * 0.1, num_agents)
    anxiety_change = -0.3 * achievement_values + np.random.normal(0, 15, num_agents)

    fig = go.Figure(data=go.Scatter(
        x=achievement_values,
        y=anxiety_change,
        mode='markers',
        marker=dict(opacity=0.6)
    ))

    fig.update_layout(
        title='Achievement vs Anxiety Change',
        xaxis_title='Achievement Score',
        yaxis_title='Anxiety Change',
        yaxis_range=[-40, 40]
    )

    return fig
# --- AGE on attendance, depression, anxiety, and Achievement ---
def generate_age_attendance_anxiety_plot(age_influence):
    num_agents = 50
    ages = np.random.normal(age_influence, 0.5, num_agents)  # Tighter clustering
    attendance = np.random.randint(1, 11, num_agents)
    anxiety_change = []

    for attend, age in zip(attendance, ages): #the way attend and age moderates eachother will effect anxiety change
        if age <= 15: #the x axis won't show age
            effect = 7.3124
        else:
            effect = -2.9477
        anxiety_change.append(effect * attend+ np.random.normal(0, 5))

    fig = go.Figure(data=go.Scatter(
        x=attendance, #age won't appear as the x axis variable
        y=anxiety_change,
        mode='markers',
        marker=dict(
            color=ages,
            colorscale='Viridis',
            colorbar=dict(title='Age'),
            opacity=0.7,
            size=10
        )
    ))
    fig.update_layout(
        title='Moderation Effect of Age on Attendence and Anxiety Change',
        xaxis_title='Attendance',
        yaxis_title='Anxiety Change',
        xaxis_range=[0, 11],
        yaxis_range=[-30,50]
    )
    return fig

def generate_age_anxiety_t1_plot(age_influence):
    num_agents = 50
    ages = np.random.normal(age_influence, 0.5, num_agents)  # Tighter clustering
    anxiety_t1 = [40 + 0.323 * age + np.random.normal(0, 5) for age in ages]

    fig = go.Figure(data=go.Scatter(x=ages, y=anxiety_t1, mode='markers'))
    fig.update_layout(
        title='Age vs Anxiety T1 (r = 0.323)',
        xaxis_title='Age',
        yaxis_title='Anxiety T1',
        xaxis_range=[13,18]
    )
    return fig

def generate_age_depression_t1_plot(age_influence):
    num_agents = 50
    ages = np.random.normal(age_influence, 0.5, num_agents)  # Tighter clustering
    depression_t1 = [40 + 0.419 * age + np.random.normal(0, 5) for age in ages]

    fig = go.Figure(data=go.Scatter(x=ages, y=depression_t1, mode='markers'))
    fig.update_layout(
        title='Age vs Depression T1 (r = 0.419)',
        xaxis_title='Age',
        yaxis_title='Depression T1',
        xaxis_range=[13,18]
    )
    return fig

def generate_age_attendance_satisfaction_plot(age_influence):
    num_agents = 50
    ages = np.random.normal(age_influence, 0.5, num_agents)  # Tighter clustering
    attendance = np.random.randint(1, 11, num_agents)
    satisfaction = []

    for age, attend in zip(ages, attendance):
        if attend <= 4: #age will no longer appear on the plot
            effect = -1.4488
        else:
            effect = 1.8749
        satisfaction.append(effect * attend + np.random.normal(0, 2)) #attend influences the satisfaction

    fig = go.Figure(data=go.Scatter(
        x=attendance, #age won't appear on the plot
        y=satisfaction,
        mode='markers',
        marker=dict(
            color=ages,
            colorscale='Viridis',
            colorbar=dict(title='Age'),
            opacity=0.7,
            size=10
        )
    ))
    fig.update_layout(
        title='Moderation Effect of Age on Attendence and Client Satisfaction',
        xaxis_title='Attendance',
        yaxis_title='Client Satisfaction',
        xaxis_range=[0, 11],
        yaxis_range=[18,32]
    )
    return fig

def generate_age_achievement_plot(age_influence):
    num_agents = 50
    ages = np.random.normal(age_influence, 0.5, num_agents)  # Tighter clustering
    achievements = [10 * age + np.random.normal(0, 20) for age in ages]

    fig = go.Figure(data=go.Scatter(x=ages, y=achievements, mode='markers'))
    fig.update_layout(
        title='Age vs Achievement Levels',
        xaxis_title='Age',
        yaxis_title='Achievement',
        xaxis_range=[13,18]
    )
    return fig

# --- Dash App ---
app = dash.Dash(__name__)
server = app.server  # For deployment

app.layout = html.Div([
    dcc.Tabs([
        # Depression Model Tab
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
        # Anxiety Model Tab
dcc.Tab(label='Anxiety Model', children=[
    html.H1("Agent-Based Model for Anxiety"),
    html.Label("Attendance (Number of Sessions)"),
    dcc.Slider(id='attendance-slider', min=1, max=10, step=1, value=4),
    html.Label("CSQ8 Influence"),
    dcc.Slider(id='csq8-slider-anxiety', min=18, max=32, step=1, value=25),  
    dcc.Graph(id='anxiety-plot', style={'width': '60%', 'margin': 'auto'})
]),
     
      # Age Analysis Tab
dcc.Tab(label='Age Analysis', children=[
    html.H1("Age-Related Analysis"),
    html.Label("Age Influence"),
    dcc.Slider(id='age-slider', min=13, max=18, step=1, value=15),  # Slider for age influence
    html.Div([
        dcc.Graph(id='age-attendance-anxiety-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='age-anxiety-t1-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='age-depression-t1-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='age-attendance-satisfaction-plot', style={'width': '50%', 'display': 'inline-block'}),
        dcc.Graph(id='age-achievement-plot', style={'width': '50%', 'display': 'inline-block'})
    ])
]),
        # IBD Model Tab
        dcc.Tab(label='IBD Model', children=[
            html.H1("Agent-Based Model for IBD"),
            html.Div([
                # Left column for sliders
                html.Div([
                    html.Div([
                        html.Label(f'Interval {i+1}'),
                        dcc.Slider(id=f'interval-slider-{i+1}', min=70, max=800, step=20, value=400)
                    ]) for i in range(5)
                ], style={'width': '40%', 'display': 'inline-block', 'vertical-align': 'top'}),
                # Right column for plots
                html.Div([
                    dcc.Graph(id='ibd-plot'),
                    dcc.Graph(id='Achievement-depression-plot'),
                    dcc.Graph(id='Achievement-anxiety-plot')
                ], style={'width': '60%', 'display': 'inline-block'})
            ])
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
    Output('anxiety-plot', 'figure'),
    Input('attendance-slider', 'value'),
    Input('csq8-slider-anxiety', 'value')
)
def update_anxiety_graph(attendance_influence, csq8_influence):
    fig = generate_anxiety_plot_anxiety_tab(attendance_influence, csq8_influence)
    return fig

@app.callback(
    [Output('ibd-plot', 'figure'),
     Output('Achievement-depression-plot', 'figure'),
     Output('Achievement-anxiety-plot', 'figure')],
    [Input(f'interval-slider-{i+1}', 'value') for i in range(5)]
)
def update_ibd_graph(*intervals):
    durations = list(intervals)
    fig_achievement, average_achievement = generate_ibd_plot(durations)
    fig_depression = generate_depression_plot(average_achievement)
    fig_anxiety = generate_anxiety_plot(average_achievement)
    return fig_achievement, fig_depression, fig_anxiety
    # Callback for Age Analysis Tab
# Callback for Age Analysis Tab
@app.callback(
    [Output('age-attendance-anxiety-plot', 'figure'),
     Output('age-anxiety-t1-plot', 'figure'),
     Output('age-depression-t1-plot', 'figure'),
     Output('age-attendance-satisfaction-plot', 'figure'),
     Output('age-achievement-plot', 'figure')],
    Input('age-slider', 'value')  # Input from the age slider
)
def update_age_plots(age_influence):  # Pass age_influence to the functions
    fig1 = generate_age_attendance_anxiety_plot(age_influence)
    fig2 = generate_age_anxiety_t1_plot(age_influence)
    fig3 = generate_age_depression_t1_plot(age_influence)
    fig4 = generate_age_attendance_satisfaction_plot(age_influence)
    fig5 = generate_age_achievement_plot(age_influence)
    return fig1, fig2, fig3, fig4, fig5
if __name__ == '__main__':
    app.run_server(debug=True)
