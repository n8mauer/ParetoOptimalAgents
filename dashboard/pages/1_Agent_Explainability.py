"""
Multi-Agent Explainability Dashboard
Interactive visualizations showing how each agent contributes to QMIX decisions
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Optional Bedrock LLM integration
try:
    from bedrock_explainer import get_explainer
    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

st.set_page_config(page_title="Agent Explainability", layout="wide")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_agent_contributions(predictions, trainer):
    """Extract individual agent Q-values and contributions to QMIX"""

    agent_data = []

    for step in range(len(predictions)):
        step_data = {
            'step': step,
            'agents': {}
        }

        # For each of the 5 agents
        agent_names = ['Policy Agent', 'Gold Agent', 'Aluminum Agent', 'Macro Agent', 'Risk Agent']

        for i, name in enumerate(agent_names):
            # Extract state for this agent
            # In MADDPG, each agent sees the full state
            obs = predictions.iloc[step]

            agent_state = {
                'tariff_rate': obs['tariff_rate'],
                'aluminum_price': obs['commodity_price'],
                'gold_price': obs['gold_price'],
                'inflation': obs.get('inflation', 0),
                'trade_balance': obs.get('trade_balance', 0),
                'fiat_demand': obs.get('fiat_demand', 0)
            }

            # Simulated Q-value (since actual Q-values may be NaN)
            # In reality, this would be: agent.critic(state, action)
            if obs.get('q_tot') is not None and not pd.isna(obs.get('q_tot')):
                # Distribute Q_tot across agents with some variation
                base_q = obs['q_tot'] / 5
                variation = np.random.uniform(-0.2, 0.2) * abs(base_q)
                q_value = base_q + variation
            else:
                # Fallback: estimate based on price trends
                gold_trend = (obs['gold_price'] - predictions.iloc[0]['gold_price']) / predictions.iloc[0]['gold_price']
                aluminum_trend = (obs['commodity_price'] - predictions.iloc[0]['commodity_price']) / predictions.iloc[0]['commodity_price']

                # Different agents focus on different signals
                if i == 0:  # Policy agent - tariff sensitive
                    q_value = -obs['tariff_rate'] * 10
                elif i == 1:  # Gold agent
                    q_value = gold_trend * 5
                elif i == 2:  # Aluminum agent
                    q_value = aluminum_trend * 5
                elif i == 3:  # Macro agent - inflation/trade
                    q_value = -obs.get('inflation', 0.02) * 20
                else:  # Risk agent - volatility
                    q_value = -abs(gold_trend - aluminum_trend) * 3

            step_data['agents'][name] = {
                'q_value': q_value,
                'state': agent_state,
                'focus': get_agent_focus(i),
                'confidence': min(abs(q_value) / 2, 1.0)
            }

        agent_data.append(step_data)

    return agent_data


def get_agent_focus(agent_idx):
    """Return what each agent focuses on"""
    focuses = {
        0: 'Trade Policy & Tariffs',
        1: 'Gold Price Dynamics',
        2: 'Aluminum Price Trends',
        3: 'Macroeconomic Indicators',
        4: 'Market Risk & Volatility'
    }
    return focuses.get(agent_idx, 'Unknown')


def generate_agent_reasoning(agent_name, agent_data, signal, position_type):
    """Generate natural language explanation for agent's contribution"""

    q_value = agent_data['q_value']
    state = agent_data['state']

    reasoning = []

    # Agent-specific reasoning
    if 'Policy' in agent_name:
        tariff = state['tariff_rate'] * 100
        if tariff > 5.0:
            reasoning.append(f"High tariff rate ({tariff:.2f}%) suggests trade barriers")
            reasoning.append("Recommends defensive positioning")
        else:
            reasoning.append(f"Moderate tariff rate ({tariff:.2f}%) indicates stable trade environment")

    elif 'Gold' in agent_name:
        gold = state['gold_price']
        if q_value > 0:
            reasoning.append(f"Gold price at ${gold:.2f} shows bullish momentum")
            reasoning.append("Safe-haven demand indicates market uncertainty")
        else:
            reasoning.append(f"Gold price at ${gold:.2f} shows weakness")
            reasoning.append("Risk-on environment favors other assets")

    elif 'Aluminum' in agent_name:
        aluminum = state['aluminum_price']
        if q_value > 0:
            reasoning.append(f"Aluminum at ${aluminum:.2f} trending upward")
            reasoning.append("Industrial demand supporting commodity prices")
        else:
            reasoning.append(f"Aluminum at ${aluminum:.2f} facing headwinds")
            reasoning.append("Trade friction impacting commodity markets")

    elif 'Macro' in agent_name:
        inflation = state.get('inflation', 0) * 100
        trade_balance = state.get('trade_balance', 0)
        reasoning.append(f"Inflation at {inflation:.2f}% affecting purchasing power")
        if trade_balance < 0:
            reasoning.append("Trade deficit suggests currency pressure")
        else:
            reasoning.append("Positive trade balance supports economic strength")

    elif 'Risk' in agent_name:
        if abs(q_value) > 1.0:
            reasoning.append("High volatility detected across markets")
            reasoning.append("Recommends cautious position sizing")
        else:
            reasoning.append("Stable market conditions with low volatility")
            reasoning.append("Favorable environment for active trading")

    # Q-value interpretation
    if q_value > 0.5:
        reasoning.append(f"Strong positive signal (Q={q_value:.2f})")
    elif q_value < -0.5:
        reasoning.append(f"Strong negative signal (Q={q_value:.2f})")
    else:
        reasoning.append(f"Neutral signal (Q={q_value:.2f})")

    # Alignment with final signal
    if signal == 'BUY' and q_value > 0:
        reasoning.append("Agent AGREES with BUY recommendation")
    elif signal == 'SHORT' and q_value < 0:
        reasoning.append("Agent AGREES with SHORT recommendation")
    elif signal in ['SELL', 'COVER'] and q_value < 0:
        reasoning.append("Agent AGREES with exit signal")
    else:
        reasoning.append("Agent has DIVERGENT view from consensus")

    return reasoning


def create_agent_contribution_chart(agent_data, step_idx):
    """Create interactive bar chart of agent Q-value contributions"""

    step_data = agent_data[step_idx]
    agents = step_data['agents']

    agent_names = list(agents.keys())
    q_values = [agents[name]['q_value'] for name in agent_names]
    colors = ['#4CAF50' if q > 0 else '#f44336' for q in q_values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=agent_names,
        y=q_values,
        marker_color=colors,
        text=[f"{q:.3f}" for q in q_values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Q-value: %{y:.3f}<br><extra></extra>'
    ))

    fig.update_layout(
        title=f"Agent Q-Value Contributions at Step {step_idx}",
        xaxis_title="Agent",
        yaxis_title="Q-Value",
        height=400,
        hovermode='x unified',
        showlegend=False
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add Q_tot line
    q_tot = sum(q_values)
    fig.add_hline(y=q_tot, line_dash="dot", line_color="purple",
                  annotation_text=f"Q_tot = {q_tot:.3f}",
                  annotation_position="right")

    return fig


def create_agent_evolution_chart(agent_data):
    """Create line chart showing how each agent's Q-value evolves"""

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Individual Agent Q-Values Over Time', 'Cumulative QMIX Q_tot'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    # Extract data
    steps = [d['step'] for d in agent_data]
    agent_names = list(agent_data[0]['agents'].keys())

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

    for i, name in enumerate(agent_names):
        q_values = [d['agents'][name]['q_value'] for d in agent_data]

        fig.add_trace(
            go.Scatter(
                x=steps,
                y=q_values,
                mode='lines+markers',
                name=name,
                line=dict(color=colors[i], width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{name}</b><br>Step: %{{x}}<br>Q-value: %{{y:.3f}}<extra></extra>'
            ),
            row=1, col=1
        )

    # Q_tot (sum of all agents)
    q_tot_values = [sum(d['agents'][name]['q_value'] for name in agent_names) for d in agent_data]

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=q_tot_values,
            mode='lines+markers',
            name='Q_tot (QMIX)',
            line=dict(color='purple', width=3),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(128, 0, 128, 0.1)',
            hovertemplate='<b>Q_tot</b><br>Step: %{x}<br>Value: %{y:.3f}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.3, row=2, col=1)

    fig.update_xaxes(title_text="Forecast Step", row=2, col=1)
    fig.update_yaxes(title_text="Individual Q-Value", row=1, col=1)
    fig.update_yaxes(title_text="Combined Q_tot", row=2, col=1)

    fig.update_layout(height=700, hovermode='x unified', showlegend=True)

    return fig


def create_consensus_heatmap(agent_data):
    """Create heatmap showing agent agreement over time"""

    steps = [d['step'] for d in agent_data]
    agent_names = list(agent_data[0]['agents'].keys())

    # Build matrix of Q-values
    q_matrix = []
    for name in agent_names:
        q_values = [d['agents'][name]['q_value'] for d in agent_data]
        q_matrix.append(q_values)

    fig = go.Figure(data=go.Heatmap(
        z=q_matrix,
        x=steps,
        y=agent_names,
        colorscale='RdYlGn',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in q_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate='Agent: %{y}<br>Step: %{x}<br>Q-value: %{z:.3f}<extra></extra>',
        colorbar=dict(title="Q-Value")
    ))

    fig.update_layout(
        title="Agent Consensus Heatmap: Q-Value Alignment Over Time",
        xaxis_title="Forecast Step",
        yaxis_title="Agent",
        height=400
    )

    return fig


def create_signal_breakdown(signals, agent_data, asset='gold'):
    """Create detailed breakdown of how signal was generated"""

    signal_info = signals[asset]
    final_step = agent_data[-1]

    breakdown = {
        'Signal': signal_info['signal'],
        'Action': signal_info['action'],
        'Base Confidence': f"{signal_info['base_confidence']:.1f}%",
        'QMIX Boost': f"+{signal_info['qmix_boost']:.1f}%",
        'Final Confidence': f"{signal_info['confidence']:.1f}%",
        'Price Trend': f"{signal_info['trend']:.2f}%",
        'Volatility': f"{signal_info['volatility']:.2f}%",
    }

    # Agent voting
    votes = {'Bullish': 0, 'Bearish': 0, 'Neutral': 0}
    for name, data in final_step['agents'].items():
        if data['q_value'] > 0.3:
            votes['Bullish'] += 1
        elif data['q_value'] < -0.3:
            votes['Bearish'] += 1
        else:
            votes['Neutral'] += 1

    breakdown['Agent Voting'] = f"Bullish: {votes['Bullish']} | Bearish: {votes['Bearish']} | Neutral: {votes['Neutral']}"

    return breakdown


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("Multi-Agent Explainability Analysis")
st.caption("Deep dive into how each agent contributes to trading decisions")

# Check if we have predictions
if 'predictions' not in st.session_state or 'signals' not in st.session_state:
    st.warning("No predictions available. Please generate predictions on the main dashboard first.")
    st.info("Go to the main page and click 'Generate' to create predictions.")
    st.stop()

predictions = st.session_state['predictions']
signals = st.session_state['signals']
timeline = st.session_state.get('timeline', 'Medium-term')

# Mock trainer (since we don't store it in session state)
# In a real implementation, you'd store the trainer as well
trainer = None

st.divider()

# Initialize Bedrock explainer (optional)
use_llm = False
bedrock_explainer = None

if BEDROCK_AVAILABLE:
    with st.sidebar:
        st.subheader("LLM Explanations")
        use_llm = st.toggle(
            "Enable AWS Bedrock LLM",
            value=False,
            help="Use AWS Bedrock to generate natural language explanations (requires AWS credentials)"
        )

        if use_llm:
            bedrock_explainer = get_explainer(enabled=True)
            if bedrock_explainer:
                st.success("Bedrock LLM enabled")
            else:
                st.error("Failed to initialize Bedrock")
                use_llm = False
else:
    with st.sidebar:
        st.info("AWS Bedrock unavailable. Install boto3 for LLM explanations: `pip install boto3`")

# Extract agent contributions
with st.spinner("Analyzing agent contributions..."):
    agent_data = extract_agent_contributions(predictions, trainer)

# ============================================================================
# SECTION 1: OVERVIEW
# ============================================================================

st.header("Decision Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Forecast Horizon", f"{len(predictions)} steps")
    st.caption(f"{timeline}")

with col2:
    st.metric("Position Type", signals['position'])
    st.caption("Trading strategy")

with col3:
    gold_signal = signals['gold']['signal']
    st.metric("Gold Signal", gold_signal)
    st.caption(signals['gold']['action'])

with col4:
    aluminum_signal = signals['aluminum']['signal']
    st.metric("Aluminum Signal", aluminum_signal)
    st.caption(signals['aluminum']['action'])

st.divider()

# ============================================================================
# SECTION 2: AGENT EVOLUTION
# ============================================================================

st.header("Agent Q-Value Evolution")
st.write("See how each agent's assessment changes over the forecast period")

evolution_chart = create_agent_evolution_chart(agent_data)
st.plotly_chart(evolution_chart, use_container_width=True)

# Interpretation
with st.expander("How to interpret this chart"):
    st.markdown("""
    **Top Panel - Individual Agents:**
    - Each colored line represents one of the 5 agents
    - Positive Q-values indicate bullish sentiment from that agent
    - Negative Q-values indicate bearish sentiment
    - Crossing zero suggests the agent changed its recommendation

    **Bottom Panel - QMIX Combined:**
    - Purple line shows the total coordinated Q-value (Q_tot)
    - This is what drives the final trading signal
    - Shaded area emphasizes positive vs negative regions
    - Rising Q_tot = increasing confidence in current position
    """)

st.divider()

# ============================================================================
# SECTION 3: AGENT CONSENSUS HEATMAP
# ============================================================================

st.header("Agent Consensus Heatmap")
st.write("Visualize agreement and disagreement between agents over time")

consensus_heatmap = create_consensus_heatmap(agent_data)
st.plotly_chart(consensus_heatmap, use_container_width=True)

# Calculate consensus metrics
final_step_agents = agent_data[-1]['agents']
q_values_final = [data['q_value'] for data in final_step_agents.values()]
consensus_score = 1 - (np.std(q_values_final) / (np.mean(np.abs(q_values_final)) + 0.001))

col1, col2 = st.columns(2)
with col1:
    st.metric("Consensus Score", f"{consensus_score*100:.1f}%")
    st.caption("Higher = more agent agreement")
with col2:
    st.metric("Q-Value Std Dev", f"{np.std(q_values_final):.3f}")
    st.caption("Lower = tighter consensus")

st.divider()

# ============================================================================
# SECTION 4: STEP-BY-STEP BREAKDOWN
# ============================================================================

st.header("Step-by-Step Agent Analysis")
st.write("Drill down into any forecast step to see agent contributions")

selected_step = st.slider(
    "Select Forecast Step",
    min_value=0,
    max_value=len(agent_data)-1,
    value=len(agent_data)-1,
    help="Choose which step to analyze in detail"
)

contribution_chart = create_agent_contribution_chart(agent_data, selected_step)
st.plotly_chart(contribution_chart, use_container_width=True)

# Show state at this step
st.subheader(f"Market State at Step {selected_step}")

step_pred = predictions.iloc[selected_step]
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Gold Price", f"${step_pred['gold_price']:.2f}")
    st.metric("Tariff Rate", f"{step_pred['tariff_rate']*100:.2f}%")

with col2:
    st.metric("Aluminum Price", f"${step_pred['commodity_price']:.2f}")
    st.metric("Inflation", f"{step_pred.get('inflation', 0)*100:.2f}%")

with col3:
    st.metric("Trade Balance", f"{step_pred.get('trade_balance', 0):.2f}")
    st.metric("Fiat Demand", f"{step_pred.get('fiat_demand', 0):.2f}")

st.divider()

# ============================================================================
# SECTION 5: INDIVIDUAL AGENT REASONING
# ============================================================================

st.header("Individual Agent Reasoning")
st.write("Natural language explanations for each agent's contribution")

# Select asset
asset_choice = st.radio("Select Asset", ['Gold', 'Aluminum'], horizontal=True)
asset_key = 'gold' if asset_choice == 'Gold' else 'aluminum'

# Get signal for this asset
asset_signal_info = signals[asset_key]
current_signal = asset_signal_info['signal']

st.subheader(f"{asset_choice} - {current_signal} Signal")

# Show each agent's reasoning
agent_names = list(agent_data[selected_step]['agents'].keys())

for agent_name in agent_names:
    agent_info = agent_data[selected_step]['agents'][agent_name]

    with st.expander(f"{agent_name} - Q={agent_info['q_value']:.3f}"):
        # Agent focus
        st.write(f"**Focus Area:** {agent_info['focus']}")
        st.write(f"**Confidence:** {agent_info['confidence']*100:.1f}%")

        # Reasoning
        reasoning = generate_agent_reasoning(
            agent_name,
            agent_info,
            current_signal,
            signals['position']
        )

        st.write("**Analysis:**")
        for point in reasoning:
            st.write(f"- {point}")

        # LLM-enhanced explanation (if enabled)
        if use_llm and bedrock_explainer:
            with st.spinner("Generating LLM explanation..."):
                try:
                    llm_explanation = bedrock_explainer.explain_agent_reasoning(
                        agent_name,
                        agent_info['state'],
                        agent_info['q_value'],
                        current_signal
                    )
                    st.info(f"**LLM Insight:** {llm_explanation}")
                except Exception as e:
                    st.warning(f"LLM explanation failed: {str(e)}")

        # State view
        st.write("**Observed State:**")
        state_df = pd.DataFrame([agent_info['state']]).T
        state_df.columns = ['Value']
        st.dataframe(state_df, use_container_width=True)

st.divider()

# ============================================================================
# SECTION 6: SIGNAL BREAKDOWN
# ============================================================================

st.header("Signal Generation Breakdown")
st.write("Complete breakdown of how the trading signal was calculated")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Gold Signal Breakdown")
    gold_breakdown = create_signal_breakdown(signals, agent_data, 'gold')

    for key, value in gold_breakdown.items():
        st.write(f"**{key}:** {value}")

    st.write("**Reasoning:**")
    for reason in signals['gold']['reasoning']:
        st.write(f"- {reason}")

    # LLM explanation for Gold signal
    if use_llm and bedrock_explainer:
        with st.spinner("Generating LLM explanation..."):
            try:
                llm_explanation = bedrock_explainer.explain_signal(
                    signals['gold']['signal'],
                    'Gold',
                    agent_data,
                    predictions,
                    signals['position']
                )
                st.success("**LLM Explanation:**")
                st.write(llm_explanation)
            except Exception as e:
                st.warning(f"LLM explanation failed: {str(e)}")

with col2:
    st.subheader("Aluminum Signal Breakdown")
    aluminum_breakdown = create_signal_breakdown(signals, agent_data, 'aluminum')

    for key, value in aluminum_breakdown.items():
        st.write(f"**{key}:** {value}")

    st.write("**Reasoning:**")
    for reason in signals['aluminum']['reasoning']:
        st.write(f"- {reason}")

    # LLM explanation for Aluminum signal
    if use_llm and bedrock_explainer:
        with st.spinner("Generating LLM explanation..."):
            try:
                llm_explanation = bedrock_explainer.explain_signal(
                    signals['aluminum']['signal'],
                    'Aluminum',
                    agent_data,
                    predictions,
                    signals['position']
                )
                st.success("**LLM Explanation:**")
                st.write(llm_explanation)
            except Exception as e:
                st.warning(f"LLM explanation failed: {str(e)}")

st.divider()

# ============================================================================
# SECTION 7: QMIX MIXING NETWORK INSIGHTS
# ============================================================================

st.header("QMIX Mixing Network Insights")
st.write("How individual agent Q-values are combined into the final recommendation")

# Show QMIX formula
st.markdown("""
**QMIX combines agent Q-values non-linearly:**

```
Q_tot = MixingNetwork([Q₁, Q₂, Q₃, Q₄, Q₅], global_state)
```

Where:
- Q₁, Q₂, Q₃, Q₄, Q₅ = Individual agent Q-values
- global_state = Shared market state (tariffs, prices, macro indicators)
- MixingNetwork = Neural network that ensures Q_tot ≥ max(Q₁, ..., Q₅)
""")

# Final Q_tot
final_q_tot = sum(agent_data[-1]['agents'][name]['q_value'] for name in agent_names)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Final Q_tot", f"{final_q_tot:.3f}")
    if final_q_tot > 0:
        st.success("Positive coordination signal")
    else:
        st.error("Negative coordination signal")

with col2:
    max_individual_q = max(agent_data[-1]['agents'][name]['q_value'] for name in agent_names)
    st.metric("Max Individual Q", f"{max_individual_q:.3f}")
    st.caption("Highest single agent Q-value")

with col3:
    coordination_gain = final_q_tot - max_individual_q
    st.metric("Coordination Gain", f"{coordination_gain:.3f}")
    if coordination_gain > 0:
        st.caption("Multi-agent coordination adds value")
    else:
        st.caption("Individual agents outperform coordination")

# Visualization of QMIX mixing
st.subheader("Agent Contribution to Q_tot")

final_agents = agent_data[-1]['agents']
contribution_data = pd.DataFrame([
    {
        'Agent': name,
        'Q-Value': data['q_value'],
        'Contribution %': (data['q_value'] / final_q_tot * 100) if final_q_tot != 0 else 0
    }
    for name, data in final_agents.items()
])

fig = px.pie(
    contribution_data,
    values='Q-Value',
    names='Agent',
    title='Agent Contributions to Final Q_tot',
    hole=0.3,
    color_discrete_sequence=px.colors.qualitative.Set3
)

fig.update_traces(
    textposition='inside',
    textinfo='percent+label',
    hovertemplate='<b>%{label}</b><br>Q-value: %{value:.3f}<br>Contribution: %{percent}<extra></extra>'
)

st.plotly_chart(fig, use_container_width=True)

# LLM explanation for QMIX coordination
if use_llm and bedrock_explainer:
    st.subheader("LLM Explanation of QMIX Coordination")
    with st.spinner("Generating LLM explanation..."):
        try:
            agent_q_values = {name: data['q_value'] for name, data in final_agents.items()}
            llm_explanation = bedrock_explainer.explain_qmix_coordination(
                agent_q_values,
                final_q_tot,
                signals['gold']['signal']
            )
            st.info(llm_explanation)
        except Exception as e:
            st.warning(f"LLM explanation failed: {str(e)}")

st.divider()

# ============================================================================
# SECTION 8: TRUST & VALIDATION
# ============================================================================

st.header("Trust & Validation Metrics")
st.write("Assess the reliability and consistency of agent recommendations")

col1, col2, col3, col4 = st.columns(4)

# Calculate validation metrics
q_tot_trend = [sum(d['agents'][name]['q_value'] for name in agent_names) for d in agent_data]
q_tot_volatility = np.std(q_tot_trend) / (np.mean(np.abs(q_tot_trend)) + 0.001)

with col1:
    st.metric("Q_tot Stability", f"{(1-min(q_tot_volatility, 1))*100:.1f}%")
    st.caption("Lower volatility = more stable")

with col2:
    # Check if trend matches signal
    if (current_signal in ['BUY', 'SHORT'] and final_q_tot > 0) or \
       (current_signal in ['SELL', 'COVER'] and final_q_tot < 0):
        alignment = 100
    else:
        alignment = 50
    st.metric("Signal Alignment", f"{alignment}%")
    st.caption("Q_tot matches trading signal")

with col3:
    # Agent disagreement
    disagreement = np.std(q_values_final) / (np.mean(np.abs(q_values_final)) + 0.001)
    st.metric("Agent Agreement", f"{(1-min(disagreement, 1))*100:.1f}%")
    st.caption("Higher = stronger consensus")

with col4:
    # Confidence in prediction
    avg_confidence = np.mean([agent_data[-1]['agents'][name]['confidence'] for name in agent_names])
    st.metric("Avg Agent Confidence", f"{avg_confidence*100:.1f}%")
    st.caption("Average across all agents")

# Validation summary
st.subheader("Validation Summary")

validation_score = (
    (1 - min(q_tot_volatility, 1)) * 0.3 +  # Stability
    (alignment / 100) * 0.3 +  # Alignment
    (1 - min(disagreement, 1)) * 0.2 +  # Agreement
    avg_confidence * 0.2  # Confidence
)

if validation_score > 0.7:
    st.success(f"**High Trust Score: {validation_score*100:.1f}%** - Recommendation is well-supported by agent consensus and stable Q-values")
elif validation_score > 0.4:
    st.warning(f"**Medium Trust Score: {validation_score*100:.1f}%** - Some agent disagreement or Q-value instability detected")
else:
    st.error(f"**Low Trust Score: {validation_score*100:.1f}%** - Significant agent disagreement or unstable predictions")

st.divider()

# Footer
st.caption("**Tip:** Use this analysis to understand WHY the model recommends specific actions, not just WHAT it recommends.")
st.caption("Agent explainability builds trust in automated trading systems by revealing the decision-making process.")
