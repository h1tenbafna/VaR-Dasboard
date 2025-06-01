import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from scipy.optimize import minimize
import yfinance as yf
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
import datetime
import pytz
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced VaR Risk Management Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #e0e2e6 !important;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #333333 !important;
        font-weight: 500;
    }
    .risk-high {
        background-color: #ffebee !important;
        border-left: 5px solid #f44336;
        color: #333333 !important;
    }
    .risk-medium {
        background-color: #fff3e0 !important;
        border-left: 5px solid #ff9800;
        color: #333333 !important;
    }
    .risk-low {
        background-color: #e8f5e8 !important;
        border-left: 5px solid #4caf50;
        color: #333333 !important;
    }
    .metric-container h4, .metric-container p {
        color: #333333 !important;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitVaRModel:
    """
    Streamlit-optimized VaR Model with caching and interactive features
    """
    
    def __init__(self):
        self.data = None
        self.returns = None
        self.portfolio_returns = None
        self.weights = None
        self.regimes = None
        self.regime_probs = None
    
    @st.cache_data
    def fetch_data(_self, tickers, start_date, end_date, _progress_bar=None):
        """Fetch stock data with caching, progress tracking, and retry mechanism"""
        try:
            if _progress_bar:
                _progress_bar.progress(20, "Downloading market data...")
            
            # Convert datetime.date to datetime.datetime if necessary
            if isinstance(start_date, datetime.date) and not isinstance(start_date, datetime.datetime):
                start_date = datetime.datetime.combine(start_date, datetime.time(0, 0))
            if isinstance(end_date, datetime.date) and not isinstance(end_date, datetime.datetime):
                end_date = datetime.datetime.combine(end_date, datetime.time(0, 0))
            
            # Convert to string format for yfinance to avoid timezone issues
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            # Retry mechanism for yfinance
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    data = yf.download(
                        tickers,
                        start=start_date_str,
                        end=end_date_str,
                        progress=False,
                        timeout=30
                    )['Close']
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        st.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                        continue
                    raise Exception(f"Failed to fetch data after {max_retries} attempts: {str(e)}")
            
            if isinstance(data, pd.Series):
                data = pd.DataFrame(data, columns=tickers)
            
            if data.empty:
                raise ValueError("No data returned for the selected tickers and date range.")
            
            if _progress_bar:
                _progress_bar.progress(50, "Processing returns...")
            
            returns = data.pct_change().dropna(how='all')
            
            if returns.empty:
                raise ValueError("No valid returns data after processing. Check date range or tickers.")
            
            if _progress_bar:
                _progress_bar.progress(100, "Data loaded successfully!")
            
            # Debug output
            print("Data shape:", data.shape)
            print("Returns shape:", returns.shape)
            
            return data, returns
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None, None
    
    def detect_market_regimes(self, returns, n_regimes=3):
        """AI-Enhanced Market Regime Detection"""
        if returns.empty:
            raise ValueError("Returns data is empty. Cannot perform regime detection.")
        
        feature_data = pd.DataFrame(index=returns.index)
        portfolio_returns = returns.mean(axis=1)
        
        if portfolio_returns.empty:
            raise ValueError("Portfolio returns are empty. Check input data.")
        
        feature_data['returns'] = portfolio_returns
        feature_data['volatility'] = portfolio_returns.rolling(21).std()
        feature_data['momentum'] = portfolio_returns.rolling(10).mean()
        feature_data['mean_reversion'] = (portfolio_returns - portfolio_returns.rolling(30).mean()) / portfolio_returns.rolling(30).std()
        
        feature_data = feature_data.dropna()
        
        if feature_data.empty:
            raise ValueError("Feature data is empty after preprocessing.")
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(feature_data)
        
        gmm = GaussianMixture(n_components=n_regimes, random_state=42, covariance_type='full')
        gmm.fit(features_scaled)
        
        regime_labels = gmm.predict(features_scaled)
        regime_probabilities = gmm.predict_proba(features_scaled)
        
        regime_df = pd.DataFrame(index=feature_data.index)
        regime_df['regime'] = regime_labels
        
        for i in range(n_regimes):
            regime_df[f'prob_regime_{i}'] = regime_probabilities[:, i]
        
        return regime_df, portfolio_returns
    
    def calculate_var_metrics(self, returns, confidence_level=0.05):
        """Calculate all VaR metrics"""
        if not isinstance(returns, (np.ndarray, pd.Series)):
            raise TypeError("Returns must be a NumPy array or pandas Series")
        
        if len(returns) == 0:
            raise ValueError("Input returns are empty")
        
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)
        
        returns_clean = returns.dropna()
        
        print("returns_clean shape:", returns_clean.shape)
        print("returns_clean:", returns_clean)
        
        if len(returns_clean) == 0:
            raise ValueError("No valid data after removing NaNs")
        
        if not 0 < confidence_level < 1:
            raise ValueError(f"Invalid confidence_level: {confidence_level}. Must be between 0 and 1.")
        
        var_historical = np.percentile(returns_clean, confidence_level * 100)
        returns_mean = returns_clean.mean()
        returns_std = returns_clean.std()
        var_parametric = stats.norm.ppf(confidence_level, returns_mean, returns_std)
        np.random.seed(42)
        simulated_returns = np.random.normal(returns_mean, returns_std, 10000)
        var_monte_carlo = np.percentile(simulated_returns, confidence_level * 100)
        es_historical = returns_clean[returns_clean <= var_historical].mean()
    
        return {
            'VaR_Historical': var_historical,
            'VaR_Parametric': var_parametric,
            'VaR_MonteCarlo': var_monte_carlo,
            'Expected_Shortfall': es_historical
        }
    
    def backtest_var(self, returns, var_value, confidence_level=0.05):
        """Comprehensive backtesting"""
        if len(returns) == 0:
            raise ValueError("Backtest returns are empty")
        
        violations = (returns < var_value).sum()
        total_obs = len(returns)
        violation_rate = violations / total_obs if total_obs > 0 else 0
        expected_violations = total_obs * confidence_level
        
        if violations > 0:
            lr_stat = 2 * (violations * np.log(violation_rate / confidence_level) + 
                         (total_obs - violations) * np.log((1 - violation_rate) / (1 - confidence_level)))
            p_value_kupiec = 1 - stats.chi2.cdf(lr_stat, 1)
        else:
            lr_stat = 0
            p_value_kupiec = 1
        
        if violations <= expected_violations:
            traffic_light = "🟢 Green"
        elif violations <= expected_violations * 1.5:
            traffic_light = "🟡 Yellow"
        else:
            traffic_light = "🔴 Red"
        
        return {
            'violations': violations,
            'violation_rate': violation_rate,
            'expected_violations': expected_violations,
            'kupiec_p_value': p_value_kupiec,
            'traffic_light': traffic_light
        }

def main():
    st.markdown('<h1 class="main-header">Advanced VaR Risk Management Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("### AI-Enhanced Value at Risk Analysis for Institutional Portfolio Management")
    
    model = StreamlitVaRModel()
    
    st.sidebar.header("Portfolio Configuration")
    
    default_tickers = ['SPY', 'QQQ', 'IWM', 'EFA']
    
    portfolio_option = st.sidebar.radio(
        "Select Portfolio Type:",
        ["Predefined ETF Portfolio", "Custom Portfolio", "Single Asset Analysis"]
    )
    
    if portfolio_option == "Predefined ETF Portfolio":
        tickers = st.sidebar.multiselect(
            "Select ETFs:",
            ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'BND', 'GLD', 'TLT'],
            default=default_tickers
        )
    elif portfolio_option == "Custom Portfolio":
        ticker_input = st.sidebar.text_input(
            "Enter tickers (comma-separated):",
            "AAPL,GOOGL,MSFT,AMZN"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
    else:
        single_ticker = st.sidebar.text_input("Enter single ticker:", "SPY")
        tickers = [single_ticker.upper()] if single_ticker else ['SPY']
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start Date", pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input("End Date", pd.to_datetime('2024-01-01'))
    
    # Validate date range
    if start_date >= end_date:
        st.error("End date must be after start date.")
        return
    if end_date > datetime.date.today():
        st.error("End date cannot be in the future.")
        return
    
    st.sidebar.header("Risk Parameters")
    confidence_level = st.sidebar.selectbox(
        "Confidence Level:",
        [0.01, 0.05, 0.10],
        index=1,
        format_func=lambda x: f"{(1-x)*100:.0f}% ({x*100:.0f}% VaR)"
    )
    
    portfolio_value = st.sidebar.number_input(
        "Portfolio Value ($):",
        min_value=100000,
        max_value=1000000000,
        value=1000000,
        step=100000,
        format="%d"
    )
    
    use_ai_regimes = st.sidebar.checkbox(
        "Enable AI Regime Detection",
        value=True,
        help="Use machine learning to detect market regimes"
    )
    
    if use_ai_regimes:
        n_regimes = st.sidebar.slider("Number of Regimes:", 2, 4, 3)
    
    if st.sidebar.button("Run VaR Analysis", type="primary"):
        if not tickers:
            st.error("Please select at least one ticker.")
            return
        
        progress_bar = st.progress(0, "Starting analysis...")
        
        try:
            # Validate tickers
            invalid_tickers = []
            for ticker in tickers:
                try:
                    test_data = yf.Ticker(ticker).info
                    if not test_data or 'symbol' not in test_data:
                        invalid_tickers.append(ticker)
                except:
                    invalid_tickers.append(ticker)
            
            if invalid_tickers:
                st.error(f"Invalid or delisted tickers detected: {', '.join(invalid_tickers)}. Please check and try again.")
                return
            
            data, returns = model.fetch_data(tickers, start_date, end_date, progress_bar)
            
            if data is None or returns is None or returns.empty:
                st.error("Failed to fetch valid data. Possible reasons: invalid tickers, date range too short, or Yahoo Finance API issues. Try different tickers or date range.")
                return
            
            progress_bar.progress(60, "Calculating risk metrics...")
            
            if len(tickers) > 1:
                weights = np.array([1/len(tickers)] * len(tickers))
                portfolio_returns = (returns * weights).sum(axis=1)
            else:
                portfolio_returns = returns.iloc[:, 0]
            
            print("Portfolio returns shape:", portfolio_returns.shape)
            print("Portfolio returns:", portfolio_returns)
            
            if portfolio_returns.empty or portfolio_returns.isna().all():
                st.error("Portfolio returns are empty or all NaN. Check data integrity or try a different date range.")
                return
            
            st.session_state['data'] = data
            st.session_state['returns'] = returns
            st.session_state['portfolio_returns'] = portfolio_returns
            st.session_state['tickers'] = tickers
            st.session_state['portfolio_value'] = portfolio_value
            st.session_state['confidence_level'] = confidence_level
            
            progress_bar.progress(100, "Analysis complete!")
            st.success("Analysis completed successfully! ✅")
        
        except Exception as e:
            st.error(f"Error processing portfolio returns: {str(e)}")
            return
    
    if 'portfolio_returns' in st.session_state:
        portfolio_returns = st.session_state['portfolio_returns']
        data = st.session_state['data']
        returns = st.session_state['returns']
        tickers = st.session_state['tickers']
        portfolio_value = st.session_state['portfolio_value']
        confidence_level = st.session_state['confidence_level']
        
        st.header("Portfolio Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            ann_return = portfolio_returns.mean() * 252
            st.metric(
                "Annual Return",
                f"{ann_return:.2%}",
                delta=f"{ann_return - 0.08:.2%}" if ann_return > 0.08 else None
            )
        
        with col2:
            ann_vol = portfolio_returns.std() * np.sqrt(252)
            st.metric(
                "Annual Volatility",
                f"{ann_vol:.2%}",
                delta=f"{0.15 - ann_vol:.2%}" if ann_vol < 0.15 else f"{0.15 - ann_vol:.2%}"
            )
        
        with col3:
            sharpe = ann_return / ann_vol if ann_vol > 0 else 0
            st.metric(
                "Sharpe Ratio",
                f"{sharpe:.2f}",
                delta=f"{sharpe - 1.0:.2f}" if sharpe > 1.0 else None
            )
        
        with col4:
            max_drawdown = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min()
            st.metric(
                "Max Drawdown",
                f"{max_drawdown:.2%}",
                delta=None
            )
        
        with col5:
            skewness = portfolio_returns.skew()
            st.metric(
                "Skewness",
                f"{skewness:.2f}",
                delta=f"{skewness:.2f}" if skewness < 0 else f"+{skewness:.2f}"
            )
        
        st.header("Value at Risk Analysis")
        
        try:
            var_metrics = model.calculate_var_metrics(portfolio_returns, confidence_level)
        except Exception as e:
            st.error(f"Error calculating VaR metrics: {str(e)}")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("VaR Estimates")
            
            for method, value in var_metrics.items():
                dollar_var = abs(value * portfolio_value)
                
                if abs(value) > 0.05:
                    risk_class = "risk-high"
                elif abs(value) > 0.03:
                    risk_class = "risk-medium"
                else:
                    risk_class = "risk-low"
                
                st.markdown(f"""
                <div class="metric-container {risk_class}">
                    <h4>{method.replace('_', ' ')}</h4>
                    <p><strong>Percentage:</strong> {value:.4f} ({value*100:.2f}%)</p>
                    <p><strong>Dollar Amount:</strong> ${dollar_var:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("VaR Comparison Chart")
            
            methods = list(var_metrics.keys()) 
            values = [abs(v * portfolio_value) for v in var_metrics.values()]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=methods,
                    y=values,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
                    text=[f'${v:,.0f}' for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title=f"VaR Comparison ({(1-confidence_level)*100:.0f}% Confidence)",
                xaxis_title="Method",
                yaxis_title="VaR (USD)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        if use_ai_regimes:
            st.header("AI-Enhanced Regime Analysis")
            
            try:
                with st.spinner("Detecting market regimes using machine learning..."):
                    regime_df, _ = model.detect_market_regimes(returns, n_regimes)
            
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Regime Statistics")
                    
                    regime_stats = []
                    # Align portfolio_returns to regime_df's index
                    aligned_returns = portfolio_returns.loc[regime_df.index]

                    for regime in range(n_regimes):
                        regime_mask = regime_df['regime'] == regime
                        regime_returns = aligned_returns[regime_mask]
                        
                        if len(regime_returns) > 10:
                            regime_stats.append({
                                'Regime': f'Regime {regime}',
                                'Observations': len(regime_returns),
                                'Ann Return': f"{regime_returns.mean() * 252:.2%}",
                                'Ann Vol': f"{regime_returns.std() * np.sqrt(252):.2%}",
                                'VaR': f"{np.percentile(regime_returns, confidence_level*100):.2%}"
                            })
                    
                    regime_stats_df = pd.DataFrame(regime_stats)
                    st.dataframe(regime_stats_df, use_container_width=True)
                
                with col2:
                    st.subheader("Regime Timeline")
                    
                    fig = go.Figure()
                    
                    colors = ['red', 'blue', 'green', 'orange']
                    for regime in range(n_regimes):
                        regime_mask = regime_df['regime'] == regime
                        regime_data = aligned_returns[regime_mask]
                        
                        fig.add_trace(go.Scatter(
                            x=regime_data.index,
                            y=regime_data,
                            mode='markers',
                            name=f'Regime {regime}',
                            marker=dict(color=colors[regime], size=3),
                            opacity=0.7
                        ))
                    
                    fig.update_layout(
                        title="Portfolio Returns by Market Regime",
                        xaxis_title="Date",
                        yaxis_title="Returns",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.warning(f"Regime detection failed: {str(e)}. Skipping regime analysis.")
        
        st.header("Model Backtesting")
        
        col1, col2 = st.columns(2)
        
        with col1:
            backtest_days = st.selectbox(
                "Backtesting Period:",
                [252, 504, 756],
                format_func=lambda x: f"Last {x} days ({x//252} year{'s' if x//252 > 1 else ''})"
            )
        
        with col2:
            var_method_to_test = st.selectbox(
                "VaR Method to Test:",
                list(var_metrics.keys())
            )
        
        try:
            test_returns = portfolio_returns.tail(backtest_days)
            var_to_test = var_metrics[var_method_to_test]
            backtest_results = model.backtest_var(test_returns, var_to_test, confidence_level)
        except Exception as e:
            st.error(f"Backtesting failed: {str(e)}")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Violations",
                f"{backtest_results['violations']}/{int(backtest_results['expected_violations']):.0f}",
                delta=f"{backtest_results['violations'] - backtest_results['expected_violations']:.0f}"
            )
        
        with col2:
            st.metric(
                "Violation Rate",
                f"{backtest_results['violation_rate']:.2%}",
                delta=f"{(backtest_results['violation_rate'] - confidence_level)*100:.1f}pp"
            )
        
        with col3:
            st.metric(
                "Kupiec Test p-value",
                f"{backtest_results['kupiec_p_value']:.4f}",
                help="p-value > 0.05 indicates model is acceptable"
            )
        
        st.subheader("Regulatory Assessment")
        st.markdown(f"**Traffic Light Status:** {backtest_results['traffic_light']}")
        
        if "Green" in backtest_results['traffic_light']:
            st.success("Model passes regulatory backtesting requirements ✅")
        elif "Yellow" in backtest_results['traffic_light']:
            st.warning("Model shows some concerns - monitor closely ⚠️")
        else:
            st.error("Model fails regulatory requirements - needs recalibration ❌")
        
        st.header("Advanced Visualizations")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(['Portfolio Performance', 'Return Distribution', 
                           'Rolling VaR', 'Drawdown Analysis']),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns, 
                      name='Cumulative Returns', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Histogram(x=portfolio_returns, nbinsx=50, name='Return Distribution',
                        marker_color='lightblue', opacity=0.7),
            row=1, col=2
        )
        
        fig.add_vline(x=var_metrics['VaR_Historical'], line_dash="dash", 
                     line_color="red", row=1, col=2)
        
        rolling_var = portfolio_returns.rolling(window=63).quantile(confidence_level)
        fig.add_trace(
            go.Scatter(x=rolling_var.index, y=rolling_var, 
                      name='Rolling VaR (3M)', line=dict(color='red')),
            row=2, col=1
        )
        
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown, 
                      name='Drawdown', fill='tonexty', 
                      line=dict(color='red'), fillcolor='rgba(255,0,0,0.3)'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.header("Stress Testing")
        
        stress_scenarios = {
            '2008 Financial Crisis': -0.20,
            'COVID-19 Crash': -0.35,
            'Flash Crash': -0.15,
            'Severe Recession': -0.40
        }
        
        stress_results = []
        for scenario, shock in stress_scenarios.items():
            stressed_loss = portfolio_value * shock
            stress_results.append({
                'Scenario': scenario,
                'Market Shock': f"{shock:.1%}",
                'Portfolio Loss': f"${abs(stressed_loss):,.0f}",
                'Loss %': f"{abs(shock):.1%}"
            })
        
        stress_df = pd.DataFrame(stress_results)
        st.dataframe(stress_df, use_container_width=True)
        
        st.header("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            summary_report = f"""
# VaR Analysis Report

## Portfolio Summary
- Tickers: {', '.join(tickers)}
- Portfolio Value: ${portfolio_value:,}
- Analysis Period: {start_date} to {end_date}
- Confidence Level: {(1-confidence_level)*100:.0f}%

## Risk Metrics
- Historical VaR: {var_metrics['VaR_Historical']:.4f} (${abs(var_metrics['VaR_Historical'] * portfolio_value):,.0f})
- Expected Shortfall: {var_metrics['Expected_Shortfall']:.4f} (${abs(var_metrics['Expected_Shortfall'] * portfolio_value):,.0f})

## Backtesting Results
- Violations: {backtest_results['violations']}/{int(backtest_results['expected_violations']):.0f}
- Traffic Light: {backtest_results['traffic_light']}
- Kupiec p-value: {backtest_results['kupiec_p_value']:.4f}
            """
            
            st.download_button(
                "📄 Download Report",
                summary_report,
                file_name=f"var_report_{'-'.join(tickers)}_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )
        
        with col2:
            export_data = pd.DataFrame({
                'Date': portfolio_returns.index,
                'Portfolio_Returns': portfolio_returns.values,
                'Cumulative_Returns': (1 + portfolio_returns).cumprod().values
            })
            
            csv = export_data.to_csv(index=False)
            st.download_button(
                "📊 Download Data",
                csv,
                file_name=f"portfolio_data_{'-'.join(tickers)}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col3:
            model_params = {
                'tickers': tickers,
                'portfolio_value': portfolio_value,
                'confidence_level': confidence_level,
                'var_historical': var_metrics['VaR_Historical'],
                'expected_shortfall': var_metrics['Expected_Shortfall']
            }
            
            st.download_button(
                "⚙️ Download Config",
                str(model_params),
                file_name=f"model_config_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()