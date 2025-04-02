import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
            "life_stage": {
                "question": "1. Which best describes your current stage of life?",
                "options": {
                    1: "Few financial burdens, accumulating wealth",
                    2: "Establishing home, may have children",
                    3: "Own home with mortgage & regular costs",
                    4: "Peak earning years, thinking of retirement",
                    5: "Preparing for retirement, few burdens",
                    6: "Retired, relying on existing funds"
                },
                "weights": {1: 9, 2: 3, 3: 1, 4: 5, 5: 3, 6: 2}
            },
            "investment_purpose": {
                "question": "2. What is your primary investment purpose?",
                "options": {
                    1: "Long-term capital growth",
                    2: "To meet income needs",
                    3: "Both growth and income",
                    4: "Capital security"
                },
                "weights": {1: 7, 2: 3, 3: 5, 4: 1}
            },
            "income_security": {
                "question": "3. How secure is your current/future income?",
                "options": {
                    1: "Not secure",
                    2: "Somewhat secure",
                    3: "Fairly secure",
                    4: "Very secure"
                },
                "weights": {1: 1, 2: 3, 3: 5, 4: 7}
            },
            "investment_timeframe": {
                "question": "4. How long will you invest before needing access?",
                "options": {
                    1: "2 years or less",
                    2: "Within 3-5 years",
                    3: "Within 6-10 years",
                    4: "Not for 10+ years"
                },
                "weights": {1: 1, 2: 3, 3: 5, 4: 7}
            },
            "market_drop_reaction": {
                "question": "5. If your $100,000 investment fell to $85,000, would you:",
                "options": {
                    1: "Sell all investments - avoid risk",
                    2: "Sell some and invest in more secure assets",
                    3: "Hold all investments, expecting improvement",
                    4: "Buy more at the lower price"
                },
                "weights": {1: 1, 2: 3, 3: 5, 4: 7}
            },
            "portfolio_volatility": {
                "question": "6. Which portfolio matches your risk comfort level?",
                "options": {
                    1: "25% growth, 3.5% return, -7.9% potential loss",
                    2: "42% growth, 4.4% return, -12.7% potential loss",
                    3: "66% growth, 5.5% return, -17.3% potential loss",
                    4: "79% growth, 6.1% return, -20.9% potential loss",
                    5: "98% growth, 7.1% return, -25.9% potential loss"
                },
                "weights": {1: 1, 2: 3, 3: 5, 4: 7, 5: 9}
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
    
    # Get portfolio based on risk level
    portfolio, target_volatility = get_portfolio_for_risk_level(ef_no_short, risk_level)
    
    # Create tabs for Portfolio Recommendation and Technical Analysis
    tab1, tab2 = st.tabs(["Portfolio Recommendation", "Technical Analysis"])
    
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
            x=[portfolio['volatility']],
            y=[portfolio['return']],
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
            name='Global Minimum Variance Portfolio'
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
            name='GMVP (With Short Sales)'
        ))
        
        # Add tangency portfolio (max Sharpe)
        fig.add_trace(go.Scatter(
            x=[tangency_portfolio['volatility']],
            y=[tangency_portfolio['return']],
            mode='markers',
            marker=dict(
                size=12,
                color='orange',
                symbol='diamond'
            ),
            name='Tangency Portfolio (Max Sharpe Ratio)'
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
        
        # Show the corresponding volatility as read-only information
        st.info(f"""
        **Your selected risk level is {risk_level}/10, which corresponds to a volatility of {target_volatility:.2f}**
        
        This portfolio is optimized to give you the highest expected return for this level of risk.
        """)
        
        # Create 3 columns for metrics
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        metrics_col1.metric(
            "Expected Annual Return", 
            f"{portfolio['return']:.2%}",
            delta=f"{portfolio['return'] - risk_free_rate:.2%} vs Risk-Free"
        )
        metrics_col2.metric(
            "Expected Volatility", 
            f"{portfolio['volatility']:.2%}"
        )
        metrics_col3.metric(
            "Sharpe Ratio", 
            f"{portfolio['sharpe_ratio']:.2f}"
        )
        
        # Portfolio weights
        st.subheader("Portfolio Allocation")
        weights_dict = dict(zip(selected_funds, portfolio['weights']))
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
        potential_loss = -(portfolio['volatility'] * 2.33 * 100)  # Approximation based on normal distribution
        
        # Display insights
        st.markdown("### Risk and Return Profile")
        st.markdown(f"""
        - **Expected Annual Return**: {portfolio['return']:.2%}
        - **Expected Volatility**: {portfolio['volatility']:.2%}
        - **Potential Loss (Bad Year)**: Around {potential_loss:.1f}%
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
        - **Orange diamond**: Tangency Portfolio (highest Sharpe ratio)
        - **Red dashed line**: Capital Market Line (tangent from risk-free rate through tangency portfolio)
        - **Blue dots**: Individual funds
        - **Red star**: Your recommended portfolio based on your risk profile
        
        The Capital Market Line represents the optimal risk-return tradeoff when combining the risk-free asset with the tangency portfolio.
        
        ### Portfolio Construction
        
        Your portfolio is positioned on the efficient frontier to maximize expected return for your specific risk tolerance level. It represents the mathematically optimal allocation of funds based on Modern Portfolio Theory.
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
        
        # Display variance-covariance matrix
        st.subheader("Variance-Covariance Matrix")
        st.dataframe(cov_matrix.style.format("{:.4f}"), height=400)
        
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
        
        # Display tangency portfolio details
        st.subheader("Tangency Portfolio (Maximum Sharpe Ratio)")
        st.markdown(f"**Return:** {tangency_portfolio['return']:.4f}")
        st.markdown(f"**Volatility:** {tangency_portfolio['volatility']:.4f}")
        st.markdown(f"**Sharpe Ratio:** {tangency_portfolio['sharpe_ratio']:.4f}")
        
        # Display weights
        tangency_weights = tangency_portfolio['weights']
        tangency_weights_df = pd.DataFrame({
            'Fund': selected_funds,
            'Weight': [f"{w*100:.2f}%" for w in tangency_weights]
        })
        st.dataframe(tangency_weights_df, hide_index=True)

if __name__ == "__main__":
    # Initialize session state for questionnaire
    if 'show_questionnaire' not in st.session_state:
        st.session_state.show_questionnaire = False
    
    main()