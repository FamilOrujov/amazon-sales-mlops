import streamlit as st

from amazon_sales_ml.ui.api_client import get_client
from amazon_sales_ml.ui.views import regression


# Page Configuration
st.set_page_config(
    page_title="Amazon Sales ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Available Pages (Add new model pages here)
PAGES = {
    "Sales Amount Prediction": {
        "icon": "📈",
        "module": regression,
        "description": "Predict total sales amount (Regression)",
    },
}


# Sidebar Navigation
def render_sidebar():
    """Render sidebar with navigation and project info."""
    
    st.sidebar.markdown("## Amazon Sales ML")
    st.sidebar.caption("MLOps Project Demo")
    
    st.sidebar.markdown("---")
    
    # Navigation
    st.sidebar.markdown("### Navigation")
    
    page_names = list(PAGES.keys())
    selected_page = st.sidebar.radio(
        "Select Model",
        page_names,
        format_func=lambda x: f"{PAGES[x]['icon']} {x}",
        label_visibility="collapsed",
    )
    
    st.sidebar.markdown("---")
    
    return selected_page


# Main App
def main():
    """Main application entry point."""
    
    # Initialize API client
    client = get_client()
    
    selected_page = render_sidebar()
    
    page_config = PAGES[selected_page]
    page_module = page_config["module"]
    
    page_module.render(client)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "Amazon Sales Prediction MLOps Project."
    )


if __name__ == "__main__":
    main()

