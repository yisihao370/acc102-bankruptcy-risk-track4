"""
ACC102 Track 4: US Listed Company Default Risk Predictor
Streamlit Application for Investor Risk Assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 处理 WRDS 可选导入
try:
    import wrds
    WRDS_AVAILABLE = True
except ImportError:
    WRDS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Bankruptcy Risk Predictor | ACC102",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .risk-safe { background-color: #d4edda; padding: 10px; border-radius: 5px; color: #155724; }
    .risk-grey { background-color: #fff3cd; padding: 10px; border-radius: 5px; color: #856404; }
    .risk-distress { background-color: #f8d7da; padding: 10px; border-radius: 5px; color: #721c24; font-weight: bold; }
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #2E86AB; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def fetch_wrds_data(tickers, _username, _password):
    """Cached WRDS data fetcher (only works if WRDS_AVAILABLE)"""
    if not WRDS_AVAILABLE:
        return pd.DataFrame()
    
    try:
        conn = wrds.Connection(wrds_username=_username)
        
        if isinstance(tickers, str):
            tickers = [tickers]
        ticker_str = "','".join(tickers).upper()
        
        query = f"""
        SELECT a.gvkey, a.datadate, a.fyear, a.tic, a.conm as company_name,
               a.at as total_assets, a.wcap as working_capital, a.re as retained_earnings,
               a.ebit as ebit, a.lt as total_liabilities, a.sale as sales,
               COALESCE(a.mkvalt, a.prcc_f * a.csho) as market_value
        FROM comp.funda a
        WHERE a.tic IN ('{ticker_str}')
        AND a.fyear >= 2019 AND a.fyear <= 2024
        AND a.indfmt = 'INDL' AND a.datafmt = 'STD' AND a.consol = 'C'
        AND a.at > 0
        ORDER BY a.tic, a.datadate
        """
        
        df = conn.raw_sql(query)
        if not df.empty:
            df['datadate'] = pd.to_datetime(df['datadate'])
            for col in ['total_assets', 'working_capital', 'retained_earnings', 
                       'ebit', 'total_liabilities', 'sales', 'market_value']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"WRDS Connection Error: {e}")
        return pd.DataFrame()

def calculate_z_score(df):
    """Calculate Altman Z-Score"""
    df = df.copy()
    df['x1'] = df['working_capital'] / df['total_assets']
    df['x2'] = df['retained_earnings'] / df['total_assets']
    df['x3'] = df['ebit'] / df['total_assets']
    df['x4'] = df['market_value'] / df['total_liabilities']
    df['x5'] = df['sales'] / df['total_assets']
    
    df['z_score'] = 1.2*df['x1'] + 1.4*df['x2'] + 3.3*df['x3'] + 0.6*df['x4'] + 1.0*df['x5']
    
    def classify(z):
        if pd.isna(z): return 'Unknown'
        return 'Safe' if z > 2.99 else 'Grey' if z > 1.81 else 'Distress'
    
    df['risk_zone'] = df['z_score'].apply(classify)
    return df

def forecast_3years(hist_df):
    """Predict next 3 years using linear trend"""
    hist = hist_df.dropna(subset=['z_score', 'fyear'])
    if len(hist) < 3:
        return pd.DataFrame(), {}
    
    X = hist['fyear'].values.reshape(-1, 1)
    y = hist['z_score'].values
    
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    r2 = model.score(X, y)
    
    last_year = hist['fyear'].max()
    future_years = np.arange(last_year + 1, last_year + 4)
    predictions = model.predict(future_years.reshape(-1, 1))
    
    forecast_df = pd.DataFrame({
        'fyear': future_years,
        'z_score_pred': predictions,
        'type': 'Forecast'
    })
    
    metrics = {
        'slope': slope,
        'r2': r2,
        'current_z': hist['z_score'].iloc[-1],
        'final_z': predictions[-1],
        'current_zone': 'Safe' if hist['z_score'].iloc[-1] > 2.99 else 'Grey' if hist['z_score'].iloc[-1] > 1.81 else 'Distress',
        'future_zone': 'Safe' if predictions[-1] > 2.99 else 'Grey' if predictions[-1] > 1.81 else 'Distress',
        'worsening': (hist['z_score'].iloc[-1] > 1.81 and predictions[-1] < 1.81)
    }
    return forecast_df, metrics

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_risk_card(ticker, metrics):
    """Render colored risk status card"""
    zone = metrics['current_zone']
    color_class = f"risk-{zone.lower()}"
    
    st.markdown(f"""
    <div class="{color_class}">
        <h3>{ticker}: {zone} Zone</h3>
        <p>Current Z-Score: <b>{metrics['current_z']:.2f}</b> | 
           3-Year Prediction: <b>{metrics['future_zone']}</b> (Z={metrics['final_z']:.2f})</p>
        {'⚠️ WARNING: Risk of entering Distress Zone within 3 years!' if metrics['worsening'] else '✅ Trend is stable or improving'}
    </div>
    """, unsafe_allow_html=True)

def create_interactive_chart(hist_df, forecast_df, ticker):
    """Create Plotly chart for Streamlit"""
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=hist_df['fyear'], y=hist_df['z_score'],
        mode='lines+markers', name='Historical',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8)
    ))
    
    # Forecast
    if not forecast_df.empty:
        # Connector
        fig.add_trace(go.Scatter(
            x=[hist_df['fyear'].iloc[-1], forecast_df['fyear'].iloc[0]],
            y=[hist_df['z_score'].iloc[-1], forecast_df['z_score_pred'].iloc[0]],
            mode='lines', line=dict(color='gray', dash='dash', width=1),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['fyear'], y=forecast_df['z_score_pred'],
            mode='lines+markers', name='3-Year Forecast',
            line=dict(color='#A23B72', dash='dash'),
            marker=dict(symbol='diamond', size=10)
        ))
    
    # Thresholds
    fig.add_hline(y=1.81, line_dash="dot", line_color="red", 
                  annotation_text="Distress (1.81)", annotation_position="right")
    fig.add_hline(y=2.99, line_dash="dot", line_color="green", 
                  annotation_text="Safe (2.99)", annotation_position="right")
    
    # Layout
    fig.update_layout(
        title=f"{ticker} - Z-Score Trajectory & Default Risk Forecast",
        xaxis_title="Fiscal Year",
        yaxis_title="Altman Z-Score",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis=dict(range=[0, max(4, hist_df['z_score'].max() + 0.5)])
    )
    
    # Add risk zones
    fig.add_hrect(y0=0, y1=1.81, fillcolor="red", opacity=0.1, line_width=0)
    fig.add_hrect(y0=1.81, y1=2.99, fillcolor="yellow", opacity=0.1, line_width=0)
    fig.add_hrect(y0=2.99, y1=10, fillcolor="green", opacity=0.1, line_width=0)
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("⚠️ US Listed Company Default Risk Predictor")
    st.subtitle("Track 4: AI-Driven Bankruptcy Risk Analysis for Investors")
    
    # Sidebar controls
    with st.sidebar:
        st.header("Control Panel")
        
        # 如果没有 WRDS，强制 Demo 模式
        if not WRDS_AVAILABLE:
            st.info("ℹ️ WRDS library not installed. Running in Demo Mode only.")
            data_mode = "Demo Mode (Pre-loaded Data)"
        else:
            # Data Source Toggle - Demo 默认在前
            data_mode = st.radio(
                "Data Mode",
                ["Demo Mode (Pre-loaded Data)", "Live WRDS (Requires Account)"],
                help="Demo mode uses cached data for testing without WRDS login"
            )
            
            # Live WRDS 可用性检查
            if data_mode == "Live WRDS (Requires Account)":
                if not WRDS_AVAILABLE:
                    st.error("❌ WRDS library not installed. Please use Demo Mode.")
                    st.stop()
        
        # Live WRDS 凭证输入
        wrds_username = ""
        wrds_password = ""
        if data_mode == "Live WRDS (Requires Account)" and WRDS_AVAILABLE:
            wrds_username = st.text_input("WRDS Username", type="default")
            wrds_password = st.text_input("Password", type="password")
            st.info("ℹ️ Your credentials are not stored; used only for this session")
        
        st.divider()
        
        # Stock Input
        tickers_input = st.text_input(
            "Enter Stock Tickers (comma-separated)",
            "AAPL, TSLA, F, XOM",
            help="Examples: AAPL (Apple), TSLA (Tesla), F (Ford)"
        )
        
        analyze_btn = st.button("🔍 Analyze Risk", type="primary", use_container_width=True)
        
        st.divider()
        st.caption("Powered by: WRDS Compustat | Altman Z-Score | Linear Forecasting")
        st.caption("⚠️ Disclaimer: Academic demo only; not investment advice.")
    
    # Main content area
    if analyze_btn:
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if len(tickers) == 0:
            st.error("Please enter at least one ticker")
            return
        
        if len(tickers) > 5:
            st.warning("For performance, analyzing first 5 tickers only")
            tickers = tickers[:5]
        
        # DEMO MODE: Load from CSV
        try:
            demo_data = pd.read_csv('z_score_analysis.csv')
            demo_data['datadate'] = pd.to_datetime(demo_data['datadate'])
            data_available = True
        except FileNotFoundError:
            st.error("Demo data not found. Please ensure z_score_analysis.csv is uploaded.")
            return
        
        # Progress bar
        progress_bar = st.progress(0)
        
        # Analyze each ticker
        results = []
        for i, ticker in enumerate(tickers):
            progress_bar.progress((i + 1) / len(tickers))
            
            with st.spinner(f"Analyzing {ticker}..."):
                # 关键修改：如果没有WRDS或选择Demo，用本地数据
                if data_mode == "Demo Mode (Pre-loaded Data)" or not WRDS_AVAILABLE:
                    raw_df = demo_data[demo_data['tic'] == ticker].copy()
                else:
                    raw_df = fetch_wrds_data(ticker, wrds_username, wrds_password)
                
                if raw_df.empty:
                    st.error(f"No data found for {ticker}")
                    continue
                
                # Calculate Z-score
                z_df = calculate_z_score(raw_df)
                
                # Forecast
                latest = z_df.sort_values('fyear').groupby('tic').last().reset_index()
                hist_clean = z_df.sort_values('fyear')
                forecast_df, metrics = forecast_3years(hist_clean)
                
                results.append({
                    'ticker': ticker,
                    'data': z_df,
                    'forecast': forecast_df,
                    'metrics': metrics,
                    'latest': latest
                })
        
        progress_bar.empty()
        
        if not results:
            st.error("No valid data retrieved. Please check tickers and try again.")
            return
        
        # Display Results
        st.divider()
        st.header("📊 Risk Analysis Results")
        
        # Summary table
        summary_data = []
        for r in results:
            m = r['metrics']
            summary_data.append({
                'Ticker': r['ticker'],
                'Current Z': f"{m['current_z']:.2f}",
                'Risk Level': m['current_zone'],
                '3-Year Outlook': m['future_zone'],
                'Trend': "📉 Declining" if m['slope'] < -0.1 else "📈 Improving" if m['slope'] > 0.1 else "➡️ Stable"
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Detailed charts
        for r in results:
            ticker = r['ticker']
            m = r['metrics']
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Risk Status")
                render_risk_card(ticker, m)
                
                st.markdown("### Key Metrics")
                st.metric("Current Z-Score", f"{m['current_z']:.2f}", 
                         delta=f"{m['slope']:+.2f}/year")
                st.metric("3-Year Prediction", f"{m['final_z']:.2f}")
                st.metric("Model Reliability (R²)", f"{m['r2']:.2f}")
                
                # Investment recommendation
                if m['current_zone'] == 'Distress' or m['worsening']:
                    st.error("🔴 **High Risk**: Consider divestment or hedging")
                elif m['current_zone'] == 'Grey':
                    st.warning("🟡 **Monitor Closely**: Review quarterly filings")
                else:
                    st.success("🟢 **Lower Risk**: Maintain current position")
            
            with col2:
                fig = create_interactive_chart(r['data'], r['forecast'], ticker)
                st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
        
        # Export option
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=pd.concat([r['data'] for r in results]).to_csv(index=False).encode('utf-8'),
            file_name='risk_analysis_report.csv',
            mime='text/csv'
        )

    else:
        # Landing page content
        st.info("👈 Enter stock tickers in the sidebar and click 'Analyze Risk' to begin")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Safe Zone", "Z > 2.99", "Low default risk")
        with col2:
            st.metric("Grey Zone", "1.81 < Z < 2.99", "Moderate risk")
        with col3:
            st.metric("Distress Zone", "Z < 1.81", "High bankruptcy risk")
        
        st.markdown("""
        ### About This Tool
        This **Track 4 Data Product** predicts bankruptcy risk for US listed companies using:
        - **Data Source**: WRDS Compustat Fundamentals (2019-2024)
        - **Methodology**: Altman Z-Score + Linear Trend Extrapolation
        - **Prediction**: 3-year forward risk trajectory based on historical financial trends
        
        **Target Users**: Individual investors, portfolio managers, credit analysts
        
        **Limitations**: 
        - Z-Score model designed for manufacturing firms; may misclassify tech/financial companies
        - Predictions assume historical trends continue; cannot predict black swan events
        - Requires minimum 3 years of financial data
        """)

if __name__ == "__main__":
    main()