import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page config
st.set_page_config(
    page_title="Robot Adviser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load data from previous chunks
@st.cache_data
def load_data():
    try:
        selected_funds = pd.read_csv('data/selected_funds.csv', header=None).iloc[:, 0].tolist()
        mean_returns = pd.read_csv('data/mean_returns.csv', index_col=0).iloc[:, 0]
        cov_matrix = pd.read_csv('data/cov_matrix.csv', index_col=0)
        
        with open('data/risk_free_rate.txt', 'r') as f:
            risk_free_rate = float(f.read().strip())
            
        return selected_funds, mean_returns, cov_matrix, risk_free_rate
    except FileNotFoundError as e:
        st.error(f"Error: Required data files not found. Please run Chunks 1 and 2 first.")
        st.stop()

# Calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
    returns = np.sum(mean_returns * weights)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / volatility
    return returns, volatility, sharpe_ratio

# Calculate efficient frontier for interactive plot
@st.cache_data
def calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, points=100, allow_short=False):
    num_assets = len(mean_returns)
    
    # Portfolio volatility function
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Portfolio return function
    def portfolio_return(weights):
        return np.sum(weights * mean_returns)
    
    # Negative Sharpe function for optimization
    def negative_sharpe(weights):
        returns, volatility, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        return -sharpe  # Negative because we're minimizing
    
    # Set bounds based on whether short sales are allowed
    bounds = tuple((-1, 1) if allow_short else (0, 1) for _ in range(num_assets))
    
    # Constraint that weights sum to 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    
    # Get minimum volatility portfolio
    min_vol_result = minimize(
        portfolio_volatility, 
        np.array([1/num_assets] * num_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    min_vol_weights = min_vol_result['x']
    min_vol_return = portfolio_return(min_vol_weights)
    min_vol_volatility = portfolio_volatility(min_vol_weights)
    
    # Get maximum Sharpe portfolio (tangency portfolio)
    max_sharpe_result = minimize(
        negative_sharpe,
        np.array([1/num_assets] * num_assets),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    max_sharpe_weights = max_sharpe_result['x']
    max_sharpe_return, max_sharpe_volatility, max_sharpe_ratio = portfolio_performance(
        max_sharpe_weights, mean_returns, cov_matrix, risk_free_rate
    )
    
    # Calculate the efficient frontier points
    target_returns = np.linspace(min_vol_return, max(mean_returns), points)
    efficient_portfolios = []
    
    for target in target_returns:
        target_constraints = constraints.copy()
        target_constraints.append({
            'type': 'eq', 
            'fun': lambda x: portfolio_return(x) - target
        })
        
        result = minimize(
            portfolio_volatility, 
            np.array([1/num_assets] * num_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=target_constraints
        )
        
        if result['success']:
            weights = result['x']
            returns, volatility, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
            efficient_portfolios.append({
                'return': returns,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'weights': weights
            })
    
    # Add the tangency portfolio (max Sharpe ratio) explicitly
    tangency_portfolio = {
        'return': max_sharpe_return,
        'volatility': max_sharpe_volatility,
        'sharpe_ratio': max_sharpe_ratio,
        'weights': max_sharpe_weights
    }
    
    return {
        'efficient_portfolios': efficient_portfolios,
        'min_vol_portfolio': {
            'return': min_vol_return,
            'volatility': min_vol_volatility,
            'weights': min_vol_weights,
            'sharpe_ratio': portfolio_performance(min_vol_weights, mean_returns, cov_matrix, risk_free_rate)[2]
        },
        'tangency_portfolio': tangency_portfolio
    }

# NEW UTILITY FUNCTION
def utility_function(expected_return, volatility, risk_aversion):
    """
    Calculate investor utility based on expected return, volatility, and risk aversion.
    U = r - (σ²A)/2
    """
    return expected_return - 0.5 * risk_aversion * (volatility ** 2)

# NEW FUNCTION: Find portfolio that maximizes utility for a given risk aversion
def get_portfolio_for_utility(efficient_frontier_data, risk_level):
    """
    Find portfolio that maximizes utility for given risk level (1-10)
    
    Args:
        efficient_frontier_data (dict): Dictionary containing portfolios on efficient frontier
        risk_level (int): Risk level from 1-10
        
    Returns:
        dict: Portfolio with maximum utility
    """
    # Convert risk level (1-10) to risk aversion parameter
    # Lower risk level = higher risk aversion = more concerned about risk
    # Higher risk level = lower risk aversion = more willing to take risk
    risk_aversion = 11 - risk_level  # Invert the scale: risk_level 1 → risk_aversion 10, risk_level 10 → risk_aversion 1
    
    # Find portfolio with maximum utility
    portfolio = max(
        efficient_frontier_data['efficient_portfolios'],
        key=lambda p: utility_function(p['return'], p['volatility'], risk_aversion)
    )
    
    # Calculate and add utility to portfolio info
    portfolio['utility'] = utility_function(portfolio['return'], portfolio['volatility'], risk_aversion)
    portfolio['risk_aversion'] = risk_aversion
    
    return portfolio, risk_aversion

# Map risk tolerance (1-10) directly to a portfolio on the efficient frontier
def get_portfolio_for_risk_level(efficient_frontier_data, risk_level):
    # Get all portfolios on the efficient frontier
    portfolios = efficient_frontier_data['efficient_portfolios']
    
    # Calculate min and max volatility
    min_vol = efficient_frontier_data['min_vol_portfolio']['volatility']
    max_vol = max(p['volatility'] for p in portfolios)
    
    # Map risk level to volatility
    # Risk level 1 = min volatility
    # Risk level 10 = max volatility
    target_volatility = min_vol + (max_vol - min_vol) * (risk_level - 1) / 9
    
    # Find closest portfolio
    closest_portfolio = min(portfolios, key=lambda p: abs(p['volatility'] - target_volatility))
    return closest_portfolio, target_volatility

# Generate a heatmap for the covariance matrix
def plot_correlation_heatmap(cov_matrix):
    # Create correlation matrix from covariance matrix
    corr_matrix = pd.DataFrame(
        data=np.zeros(cov_matrix.shape),
        index=cov_matrix.index,
        columns=cov_matrix.columns
    )
    
    # Convert covariance to correlation
    for i in range(len(cov_matrix.index)):
        for j in range(len(cov_matrix.columns)):
            corr_matrix.iloc[i, j] = cov_matrix.iloc[i, j] / (
                np.sqrt(cov_matrix.iloc[i, i]) * np.sqrt(cov_matrix.iloc[j, j])
            )
    
    # Plot using Plotly
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        text_auto='.2f'
    )
    
    fig.update_layout(
        title='Correlation Matrix Heatmap',
        height=600,
        width=800
    )
    
    return fig

# Risk appetite questionnaire
def risk_appetite_questionnaire():
    # Define risk questionnaire button
    if st.sidebar.button("Complete Risk Profile Questionnaire"):
        st.session_state.show_questionnaire = True
    
    # Initialize risk level in session state if not present
    if 'risk_level' not in st.session_state:
        st.session_state.risk_level = 5  # Default risk level
    
    # Show questionnaire in sidebar if state is active
    if st.session_state.get('show_questionnaire', False):
        st.sidebar.markdown("## Risk Profile Questionnaire")
        st.sidebar.markdown("Answer these questions to determine your optimal investment strategy.")
        
        # Define questions and options with their associated scores
        questions = {
            "investment_objective": {
                "question": "1. What is your primary investment objective?",
                "options": {
                    "A": "Maximum capital growth, accepting high volatility",
                    "B": "Strong capital growth, tolerating significant volatility",
                    "C": "Moderate growth with some income, accepting moderate volatility",
                    "D": "Income generation with some growth, preferring limited volatility",
                    "E": "Capital preservation with minimal risk"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "time_horizon": {
                "question": "2. What is your investment time horizon?",
                "options": {
                    "A": "Over 15 years",
                    "B": "10-15 years",
                    "C": "6-10 years",
                    "D": "3-5 years",
                    "E": "Less than 3 years"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "loss_reaction": {
                "question": "3. If your portfolio suddenly lost 20% of its value, what would you do?",
                "options": {
                    "A": "Invest substantially more to take advantage of lower prices",
                    "B": "Invest a small amount more",
                    "C": "Hold my position and wait for recovery",
                    "D": "Sell a small portion to reduce further risk",
                    "E": "Sell most or all of my investments"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "investment_experience": {
                "question": "4. How much investment experience do you have?",
                "options": {
                    "A": "Very experienced, including complex investment products",
                    "B": "Experienced with stocks, bonds, and mutual funds",
                    "C": "Some experience, mainly with mutual funds or managed investments",
                    "D": "Limited experience, mainly bank deposits and low-risk investments",
                    "E": "No investment experience"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "investment_percentage": {
                "question": "5. What percentage of your liquid net worth are you investing?",
                "options": {
                    "A": "Less than 20%",
                    "B": "20-40%",
                    "C": "41-60%",
                    "D": "61-80%",
                    "E": "More than 80%"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "income_stability": {
                "question": "6. How stable is your current and expected future income?",
                "options": {
                    "A": "Very stable with high growth potential",
                    "B": "Stable with some growth potential",
                    "C": "Moderately stable",
                    "D": "Somewhat unstable",
                    "E": "Very unstable or declining"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "portfolio_preference": {
                "question": "7. Which portfolio would you be most comfortable with over 10 years?",
                "options": {
                    "A": "Portfolio A: Potential return 12%+, potential loss 25%+",
                    "B": "Portfolio B: Potential return 10%, potential loss 20%",
                    "C": "Portfolio C: Potential return 8%, potential loss 15%",
                    "D": "Portfolio D: Potential return 6%, potential loss 10%",
                    "E": "Portfolio E: Potential return 4%, potential loss 5%"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "check_frequency": {
                "question": "8. How often do you check your investment performance?",
                "options": {
                    "A": "Rarely, maybe annually",
                    "B": "Quarterly",
                    "C": "Monthly",
                    "D": "Weekly",
                    "E": "Daily or multiple times daily"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "withdrawal_timeline": {
                "question": "9. How many years until you plan to begin making withdrawals?",
                "options": {
                    "A": "20+ years",
                    "B": "11-20 years",
                    "C": "5-10 years",
                    "D": "1-4 years",
                    "E": "Already withdrawing"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            },
            "investment_attitude": {
                "question": "10. When making a long-term investment, I am...",
                "options": {
                    "A": "Focused solely on maximum returns regardless of risk",
                    "B": "Primarily concerned with growth, secondarily with risk",
                    "C": "Equally concerned with risk and return",
                    "D": "Primarily concerned with risk, secondarily with growth",
                    "E": "Focused solely on minimizing risk"
                },
                "weights": {"A": 9, "B": 7, "C": 5, "D": 3, "E": 1}
            }
        }
        
        # Create form for questionnaire
        with st.sidebar.form("risk_questionnaire_form"):
            # Initialize responses
            responses = {}
            
            # Display questions - no default selections
            for q_id, q_data in questions.items():
                st.markdown(f"**{q_data['question']}**")
                
                # No default index - force user to make a selection
                response = st.radio(
                    f"Select one option:",
                    options=list(q_data['options'].keys()),
                    format_func=lambda x: q_data['options'][x],
                    key=f"question_{q_id}",
                    label_visibility="collapsed"
                )
                responses[q_id] = response
                
                # Add some spacing
                st.markdown("")
            
            # Submit button
            submitted = st.form_submit_button("Calculate My Risk Profile")
            
            if submitted:
                # Calculate weights for responses
                weights = {}
                for q_id, response in responses.items():
                    weights[q_id] = questions[q_id]['weights'][response]
                
                # Calculate total score
                total_score = sum(weights.values())
                max_possible_score = sum([max(q_data['weights'].values()) for q_data in questions.values()])
                min_possible_score = sum([min(q_data['weights'].values()) for q_data in questions.values()])
                
                # Calculate risk level (1-10 scale)
                risk_level = 1 + (total_score - min_possible_score) / (max_possible_score - min_possible_score) * 9
                risk_level = round(risk_level)
                
                # Ensure risk level is within bounds
                risk_level = max(1, min(10, risk_level))
                
                # Store risk level in session state
                st.session_state.risk_level = risk_level
                
                # Get risk profile based on score ranges
                st.session_state.risk_profile = get_risk_profile(total_score)
                
                # Hide questionnaire after submission
                st.session_state.show_questionnaire = False
                st.rerun()
    
    # Always display the calculated risk level if available
    if hasattr(st.session_state, 'risk_profile'):
        risk_profile = st.session_state.risk_profile
        
        # Display risk level with simple styling (better visibility)
        st.sidebar.markdown("## Your Risk Assessment")
        st.sidebar.markdown(
            f"""
            <div style='background-color:#f0f2f6; padding:10px; border-radius:5px; border-left: 4px solid #4b78e6;'>
                <h3 style='margin:0; color:#0e1117;'>Risk Level: {st.session_state.risk_level}/10</h3>
                <h4 style='margin-top:5px; color:#0e1117;'>{risk_profile["profile"]}</h4>
                <p style='color:#0e1117;'>{risk_profile["description"]}</p>
                <p style='color:#0e1117;'><b>Risk Aversion:</b> {11 - st.session_state.risk_level}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display recommended time horizon
        st.sidebar.markdown(f"**Recommended Time Horizon:** {risk_profile['time_horizon']}")

    # Manual override option always available
    st.sidebar.markdown("## Manual Risk Level")
    manual_risk_level = st.sidebar.slider(
        "Adjust your risk level (1-10):",
        min_value=1,
        max_value=10,
        value=st.session_state.get('risk_level', 5),
        help="1 = Lowest Risk, 10 = Highest Risk"
    )
    
    if manual_risk_level != st.session_state.get('risk_level', 5):
        st.session_state.risk_level = manual_risk_level
    
    return st.session_state.risk_level

# Get risk profile description based on total score
def get_risk_profile(score):
    if score < 19:
        return {
            'profile': 'Defensive',
            'description': 'You prioritize capital protection over growth.',
            'time_horizon': 'Minimum 2 years'
        }
    elif score < 40:
        return {
            'profile': 'Conservative',
            'description': 'You seek income with some growth potential, accepting lower returns for stability.',
            'time_horizon': 'Minimum 3 years'
        }
    elif score < 65:
        return {
            'profile': 'Balanced',
            'description': 'You balance growth and security, accepting some short-term risk for long-term gains.',
            'time_horizon': 'Minimum 5 years'
        }
    elif score < 89:
        return {
            'profile': 'Assertive',
            'description': 'You focus on maximizing growth and take calculated risks to achieve higher returns.',
            'time_horizon': 'Minimum 7 years'
        }
    else:
        return {
            'profile': 'Aggressive',
            'description': 'You seek maximum growth potential and accept higher volatility for potential long-term returns.',
            'time_horizon': 'Minimum 9 years'
        }

# Main app
def main():
    st.title("Interactive Robot Adviser")
    
    # Load data
    selected_funds, mean_returns, cov_matrix, risk_free_rate = load_data()
    
    # Calculate efficient frontiers (cached for performance)
    with st.spinner("Calculating efficient frontier..."):
        ef_no_short = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, points=100, allow_short=False)
        ef_with_short = calculate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, points=100, allow_short=True)
    
    # Add questionnaire to sidebar and get risk level
    risk_level = risk_appetite_questionnaire()
    
    # Get portfolio based on risk level (original approach)
    risk_mapped_portfolio, target_volatility = get_portfolio_for_risk_level(ef_no_short, risk_level)
    
    # NEW: Get portfolio that maximizes utility based on risk aversion
    utility_portfolio, risk_aversion = get_portfolio_for_utility(ef_no_short, risk_level)
    
    # Create tabs for Portfolio Recommendation and Technical Analysis
    tab1, tab2, tab3 = st.tabs(["Portfolio Recommendation", "Technical Analysis", "Utility Analysis"])
    
    with tab1:
        # Create interactive plot
        st.header("Efficient Frontier")
        
        # Extract data for plotting
        ef_no_short_vols = [p['volatility'] for p in ef_no_short['efficient_portfolios']]
        ef_no_short_returns = [p['return'] for p in ef_no_short['efficient_portfolios']]
        
        ef_with_short_vols = [p['volatility'] for p in ef_with_short['efficient_portfolios']]
        ef_with_short_returns = [p['return'] for p in ef_with_short['efficient_portfolios']]
        
        # Get tangency portfolio (max Sharpe ratio)
        tangency_portfolio = ef_no_short['tangency_portfolio']
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add efficient frontier without short sales
        fig.add_trace(go.Scatter(
            x=ef_no_short_vols,
            y=ef_no_short_returns,
            mode='lines',
            name='Efficient Frontier (No Short Sales)',
            line=dict(color='green', width=2),
            hovertemplate='Volatility: %{x:.4f}<br>Return: %{y:.4f}'
        ))
        
        # Add efficient frontier with short sales
        fig.add_trace(go.Scatter(
            x=ef_with_short_vols,
            y=ef_with_short_returns,
            mode='lines',
            name='Efficient Frontier (With Short Sales)',
            line=dict(color='blue', width=2),
            hovertemplate='Volatility: %{x:.4f}<br>Return: %{y:.4f}'
        ))
        
        # Add individual funds
        fund_returns = mean_returns.values
        fund_vols = np.array([np.sqrt(cov_matrix.iloc[i, i]) for i in range(len(selected_funds))])
        
        fig.add_trace(go.Scatter(
            x=fund_vols,
            y=fund_returns,
            mode='markers+text',
            marker=dict(
                size=10,
                color='darkblue',
            ),
            text=selected_funds,
            textposition="top center",
            name='Individual Funds'
        ))
        
        # Add selected portfolio based on risk level
        fig.add_trace(go.Scatter(
            x=[utility_portfolio['volatility']],
            y=[utility_portfolio['return']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star'
            ),
            name=f'Your Portfolio (Risk Level {risk_level})'
        ))
        
        # Add minimum volatility portfolio
        min_vol = ef_no_short['min_vol_portfolio']
        fig.add_trace(go.Scatter(
            x=[min_vol['volatility']],
            y=[min_vol['return']],
            mode='markers',
            marker=dict(
                size=12,
                color='purple',
                symbol='circle'
            ),
            name='Global Minimum Variance Portfolio (Without Short Sales)'
        ))
        
        # Add minimum volatility portfolio with short sales
        min_vol_short = ef_with_short['min_vol_portfolio']
        fig.add_trace(go.Scatter(
            x=[min_vol_short['volatility']],
            y=[min_vol_short['return']],
            mode='markers',
            marker=dict(
                size=12,
                color='purple',
                symbol='triangle-up'
            ),
            name='Global Minimum Variance Portfolio (With Short Sales)'
        ))
        
        # Add tangency portfolio (max Sharpe) for no shorts
        fig.add_trace(go.Scatter(
            x=[tangency_portfolio['volatility']],
            y=[tangency_portfolio['return']],
            mode='markers',
            marker=dict(
                size=12,
                color='orange',
                symbol='diamond'
            ),
            name='Tangency Portfolio (No Short Sales)'
        ))
        
        # Add tangency portfolio with short sales
        tangency_short = ef_with_short['tangency_portfolio']
        fig.add_trace(go.Scatter(
            x=[tangency_short['volatility']],
            y=[tangency_short['return']],
            mode='markers',
            marker=dict(
                size=12,
                color='darkorange',
                symbol='diamond-open'
            ),
            name='Tangency Portfolio (With Short Sales)'
        ))
        
        # Draw the Capital Market Line as a tangent from risk-free rate through tangency portfolio
        # Ensure it extends far enough to be visible
        max_vol_display = max(max(ef_no_short_vols), max(ef_with_short_vols)) * 1.5
        
        # Calculate the slope (Sharpe ratio)
        cml_slope = (tangency_portfolio['return'] - risk_free_rate) / tangency_portfolio['volatility']
        
        # Calculate the second point
        cml_x = [0, max_vol_display]
        cml_y = [risk_free_rate, risk_free_rate + cml_slope * max_vol_display]
        
        fig.add_trace(go.Scatter(
            x=cml_x,
            y=cml_y,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Capital Market Line (Tangent)'
        ))
        
        # Add the risk-free point explicitly
        fig.add_trace(go.Scatter(
            x=[0],
            y=[risk_free_rate],
            mode='markers',
            marker=dict(
                size=10,
                color='green',
                symbol='cross'
            ),
            name='Risk-Free Rate'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Efficient Frontier (Risk-Free Rate: {risk_free_rate:.2%})',
            xaxis_title='Expected Volatility',
            yaxis_title='Expected Return',
            height=600,
            hovermode='closest'
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
        
        # Display portfolio details based on risk level
        st.header("Your Optimal Portfolio")
        
        # Show the corresponding information
        st.info(f"""
        **Your selected risk level is {risk_level}/10, which corresponds to a risk aversion of {risk_aversion}**
        
        This portfolio is optimized to maximize your utility function: U = r - (σ²A)/2
        """)
        
        # Create 4 columns for metrics (including utility)
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        metrics_col1.metric(
            "Expected Annual Return", 
            f"{utility_portfolio['return']:.2%}",
            delta=f"{utility_portfolio['return'] - risk_free_rate:.2%} vs Risk-Free"
        )
        metrics_col2.metric(
            "Expected Volatility", 
            f"{utility_portfolio['volatility']:.2%}"
        )
        metrics_col3.metric(
            "Sharpe Ratio", 
            f"{utility_portfolio['sharpe_ratio']:.2f}"
        )
        metrics_col4.metric(
            "Utility Value",
            f"{utility_portfolio['utility']:.4f}"
        )
        
        # Portfolio weights
        st.subheader("Portfolio Allocation")
        weights_dict = dict(zip(selected_funds, utility_portfolio['weights']))
        sorted_weights = {k: v for k, v in sorted(weights_dict.items(), key=lambda item: item[1], reverse=True) if v > 0.01}
        
        # Create columns for chart and table
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create bar chart for weights
            fig_weights = px.bar(
                x=list(sorted_weights.keys()),
                y=list(sorted_weights.values()),
                labels={'x': 'Fund', 'y': 'Weight'},
                title='Portfolio Weights',
                color=list(sorted_weights.values()),
                color_continuous_scale='Viridis'
            )
            
            fig_weights.update_layout(
                xaxis_tickangle=-45,
                height=400,
                coloraxis_showscale=False
            )
            
            st.plotly_chart(fig_weights, use_container_width=True)
        
        with col2:
            # Show exact allocations as a table
            st.subheader("Exact Allocations")
            allocation_df = pd.DataFrame({
                'Fund': list(sorted_weights.keys()),
                'Weight': [f"{w*100:.2f}%" for w in sorted_weights.values()]
            })
            st.dataframe(allocation_df, hide_index=True, height=400)
        
        # Portfolio insights
        st.subheader("Portfolio Insights")
        
        # Potential loss based on volatility
        potential_loss = -(utility_portfolio['volatility'] * 2.33 * 100)  # Approximation based on normal distribution
        
        # Display insights
        st.markdown("### Risk and Return Profile")
        st.markdown(f"""
        - **Expected Annual Return**: {utility_portfolio['return']:.2%}
        - **Expected Volatility**: {utility_portfolio['volatility']:.2%}
        - **Potential Loss (Bad Year)**: Around {potential_loss:.1f}%
        - **Utility Value**: {utility_portfolio['utility']:.4f}
        """)
        
        # Explanation section
        st.markdown("---")
        st.header("How This Works")
        st.markdown("""
        ### Comprehensive Risk Profiling
        
        Your risk profile can be determined through a detailed questionnaire that assesses:
        
        - Your current life stage and financial situation
        - Investment goals and timeframe
        - Income stability and financial needs
        - Behavioral responses to market fluctuations
        - Your preferred portfolio characteristics
        
        ### Key Points on the Chart
        
        - **Green line**: Efficient frontier without short sales
        - **Blue line**: Efficient frontier with short sales allowed
        - **Purple circle**: Global Minimum Variance Portfolio without short sales (lowest possible risk)
        - **Purple triangle**: Global Minimum Variance Portfolio with short sales allowed
        - **Orange diamond**: Tangency Portfolio (highest Sharpe ratio) without short sales
        - **Orange open diamond**: Tangency Portfolio with short sales allowed
        - **Red dashed line**: Capital Market Line (tangent from risk-free rate through tangency portfolio)
        - **Blue dots**: Individual funds
        - **Red star**: Your recommended portfolio based on your risk profile
        
        The Capital Market Line represents the optimal risk-return tradeoff when combining the risk-free asset with the tangency portfolio.
        
        ### Portfolio Construction
        
        Your portfolio is positioned on the efficient frontier to maximize your utility function:
        
        **U = r - (σ²A)/2**
        
        Where:
        - **r** is the expected return
        - **σ** is the volatility (standard deviation)
        - **A** is your risk aversion parameter
        
        This represents the mathematical tradeoff between risk and return based on your personal risk tolerance.
        """)
    
    with tab2:
        st.header("Technical Analysis")
        
        # Display mean returns table
        st.subheader("Mean Returns of Funds")
        mean_returns_df = pd.DataFrame({
            'Fund': selected_funds,
            'Mean Return': [f"{r:.4f}" for r in mean_returns.values]
        })
        st.dataframe(mean_returns_df, hide_index=True)
        
        # Display covariance matrix as a heatmap
        st.subheader("Variance-Covariance Matrix Visualization")
        
        # Create two tabs for different ways to view the covariance data
        cov_tab1, cov_tab2 = st.tabs(["Heatmap Visualization", "Raw Data"])
        
        with cov_tab1:
            # Generate and display the heatmap
            corr_heatmap = plot_correlation_heatmap(cov_matrix)
            st.plotly_chart(corr_heatmap, use_container_width=True)
            
            st.markdown("""
            **Note**: The heatmap shows the **correlation matrix** derived from the covariance matrix. 
            Values closer to +1 (dark blue) indicate strong positive correlation, 
            values closer to -1 (dark red) indicate strong negative correlation, 
            and values near 0 (white) indicate little to no correlation.
            """)
        
        with cov_tab2:
            # Display the raw data with formatting
            st.dataframe(cov_matrix.style.background_gradient(cmap='coolwarm', axis=None).format("{:.4f}"), height=400)
        
        # Add information about both GMVPs
        st.subheader("Global Minimum Variance Portfolios")
        
        # Create columns for with/without short sales
        gmvp_col1, gmvp_col2 = st.columns(2)
        
        with gmvp_col1:
            st.markdown("#### Without Short Sales")
            st.markdown(f"**Return:** {ef_no_short['min_vol_portfolio']['return']:.4f}")
            st.markdown(f"**Volatility:** {ef_no_short['min_vol_portfolio']['volatility']:.4f}")
            st.markdown(f"**Sharpe Ratio:** {ef_no_short['min_vol_portfolio']['sharpe_ratio']:.4f}")
            
            # Display weights
            gmvp_weights = ef_no_short['min_vol_portfolio']['weights']
            gmvp_weights_df = pd.DataFrame({
                'Fund': selected_funds,
                'Weight': [f"{w*100:.2f}%" for w in gmvp_weights]
            })
            st.dataframe(gmvp_weights_df, hide_index=True)
        
        with gmvp_col2:
            st.markdown("#### With Short Sales")
            st.markdown(f"**Return:** {ef_with_short['min_vol_portfolio']['return']:.4f}")
            st.markdown(f"**Volatility:** {ef_with_short['min_vol_portfolio']['volatility']:.4f}")
            st.markdown(f"**Sharpe Ratio:** {ef_with_short['min_vol_portfolio']['sharpe_ratio']:.4f}")
            
            # Display weights
            gmvp_weights = ef_with_short['min_vol_portfolio']['weights']
            gmvp_weights_df = pd.DataFrame({
                'Fund': selected_funds,
                'Weight': [f"{w*100:.2f}%" for w in gmvp_weights]
            })
            st.dataframe(gmvp_weights_df, hide_index=True)
        
        # Display tangency portfolio details with both options (with and without short sales)
        st.subheader("Tangency Portfolios (Maximum Sharpe Ratio)")
        
        # Create columns for both tangency portfolios
        tang_col1, tang_col2 = st.columns(2)
        
        with tang_col1:
            st.markdown("#### Without Short Sales")
            st.markdown(f"**Return:** {ef_no_short['tangency_portfolio']['return']:.4f}")
            st.markdown(f"**Volatility:** {ef_no_short['tangency_portfolio']['volatility']:.4f}")
            st.markdown(f"**Sharpe Ratio:** {ef_no_short['tangency_portfolio']['sharpe_ratio']:.4f}")
            
            # Display weights
            tang_weights = ef_no_short['tangency_portfolio']['weights']
            tang_weights_df = pd.DataFrame({
                'Fund': selected_funds,
                'Weight': [f"{w*100:.2f}%" for w in tang_weights]
            })
            st.dataframe(tang_weights_df, hide_index=True)
        
        with tang_col2:
            st.markdown("#### With Short Sales")
            st.markdown(f"**Return:** {ef_with_short['tangency_portfolio']['return']:.4f}")
            st.markdown(f"**Volatility:** {ef_with_short['tangency_portfolio']['volatility']:.4f}")
            st.markdown(f"**Sharpe Ratio:** {ef_with_short['tangency_portfolio']['sharpe_ratio']:.4f}")
            
            # Display weights
            tang_weights = ef_with_short['tangency_portfolio']['weights']
            tang_weights_df = pd.DataFrame({
                'Fund': selected_funds,
                'Weight': [f"{w*100:.2f}%" for w in tang_weights]
            })
            st.dataframe(tang_weights_df, hide_index=True)
            
    # NEW TAB: Utility Analysis
    with tab3:
        st.header("Utility Analysis")
        
        st.markdown(f"""
        ## Investor Utility Function
        
        Your utility function represents your preference for risk and return:
        
        **U = r - (σ²A)/2**
        
        Where:
        - **r** is the expected return
        - **σ** is the volatility (standard deviation)
        - **A** is the risk aversion parameter (currently {risk_aversion})
        
        With your risk level of {risk_level}/10:
        - Higher risk level (10) = Lower risk aversion (1) = More willing to take risks
        - Lower risk level (1) = Higher risk aversion (10) = Less willing to take risks
        
        This function helps us find the portfolio that maximizes your satisfaction based on your risk tolerance.
        """)
        
        # Compare risk-mapped portfolio vs utility-optimized portfolio
        st.subheader("Portfolio Comparison")
        
        # Calculate utility for risk-mapped portfolio for fair comparison
        risk_mapped_utility = utility_function(
            risk_mapped_portfolio['return'],
            risk_mapped_portfolio['volatility'],
            risk_aversion
        )
        
        # Create a comparison dataframe
        comparison_data = {
            "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", "Utility Value"],
            "Risk-Mapped Portfolio": [
                f"{risk_mapped_portfolio['return']:.2%}",
                f"{risk_mapped_portfolio['volatility']:.2%}",
                f"{risk_mapped_portfolio['sharpe_ratio']:.2f}",
                f"{risk_mapped_utility:.4f}"
            ],
            "Utility-Optimized Portfolio": [
                f"{utility_portfolio['return']:.2%}",
                f"{utility_portfolio['volatility']:.2%}",
                f"{utility_portfolio['sharpe_ratio']:.2f}",
                f"{utility_portfolio['utility']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.table(comparison_df)
        
        # Explanation of utility maximization
        st.markdown("""
        ### Understanding Utility Maximization
        
        The utility-optimized portfolio directly maximizes your satisfaction based on your risk aversion parameter.
        This approach differs from the risk-mapped portfolio in that:
        
        1. **Mathematically Optimal**: It finds the exact portfolio that maximizes the utility function
        
        2. **Personalized**: It accounts for your specific risk aversion parameter
        
        3. **Theoretically Sound**: It's based on economic theory of rational choice under uncertainty
        
        The higher the utility value, the better the portfolio matches your personal preferences for the risk-return tradeoff.
        """)
        
        # Create a utility visualization across the efficient frontier
        st.subheader("Utility Across the Efficient Frontier")
        
        # Calculate utilities for all portfolios on the efficient frontier
        utilities = []
        for idx, p in enumerate(ef_no_short['efficient_portfolios']):
            util = utility_function(p['return'], p['volatility'], risk_aversion)
            utilities.append({
                'Portfolio': idx,
                'Return': p['return'],
                'Volatility': p['volatility'],
                'Utility': util
            })
        
        # Create a DataFrame for visualization
        utilities_df = pd.DataFrame(utilities)
        
        # Create a scatter plot to show utility values across the efficient frontier
        fig_utility = px.scatter(
            utilities_df,
            x='Volatility',
            y='Return',
            color='Utility',
            color_continuous_scale='Viridis',
            title=f'Utility Values Across the Efficient Frontier (Risk Aversion = {risk_aversion})'
        )
        
        # Mark the optimal portfolio
        fig_utility.add_trace(go.Scatter(
            x=[utility_portfolio['volatility']],
            y=[utility_portfolio['return']],
            mode='markers',
            marker=dict(
                size=15,
                color='red',
                symbol='star',
                line=dict(width=2, color='black')
            ),
            name='Optimal Portfolio'
        ))
        
        # Update layout
        fig_utility.update_layout(
            xaxis_title='Volatility',
            yaxis_title='Expected Return',
            height=500,
            coloraxis_colorbar=dict(title='Utility')
        )
        
        st.plotly_chart(fig_utility, use_container_width=True)
        
        # Explanation
        st.markdown("""
        ### Key Insights
        
        * The color gradient shows utility values across the efficient frontier
        * The red star marks the portfolio that maximizes your utility
        * As your risk aversion changes, the optimal portfolio will move along the efficient frontier
        * Higher risk aversion (more risk-averse) will shift the optimal portfolio toward lower volatility
        * Lower risk aversion (more risk-tolerant) will shift the optimal portfolio toward higher returns
        """)

if __name__ == "__main__":
    # Initialize session state for questionnaire
    if 'show_questionnaire' not in st.session_state:
        st.session_state.show_questionnaire = False
    
    main()