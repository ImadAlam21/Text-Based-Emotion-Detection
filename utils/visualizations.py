import plotly.graph_objects as go
import altair as alt
import pandas as pd
import streamlit as st

def create_emotion_radar_chart(emotions_dict):
    """Create a radar chart for emotion probabilities."""
    categories = list(emotions_dict.keys())
    values = list(emotions_dict.values())
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(76, 175, 80)'),
            fillcolor='rgba(76, 175, 80, 0.5)'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def create_pos_bar_chart(pos_counts):
    """Create a bar chart for part-of-speech distribution."""
    df = pd.DataFrame(list(pos_counts.items()), columns=['POS', 'Count'])
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('POS:N', sort='-y'),
        y='Count:Q',
        color=alt.value('#4CAF50')
    ).properties(
        height=300
    )
    
    return chart

def create_sentiment_gauge(sentiment_score):
    """Create a gauge chart for sentiment score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [-1, 1]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [-1, -0.3], 'color': "lightgray"},
                {'range': [-0.3, 0.3], 'color': "gray"},
                {'range': [0.3, 1], 'color': "darkgray"}
            ]
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def display_text_stats(text_stats):
    """Display text statistics in a formatted way."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Word Count", text_stats['word_count'])
    with col2:
        st.metric("Sentence Count", text_stats['sentence_count'])
    with col3:
        st.metric("Average Word Length", f"{text_stats['avg_word_length']:.1f}")

def create_confidence_indicator(confidence_score):
    """Create a confidence indicator visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#4CAF50"},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"},
                {'range': [70, 100], 'color': "darkgray"}
            ]
        }
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

def create_historical_emotions_chart(history_data):
    """Create a line chart showing emotion trends over time."""
    if not history_data:
        return None
        
    df = pd.DataFrame(history_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    emotions = df['emotion'].unique()
    chart_data = []
    
    for emotion in emotions:
        emotion_data = df[df['emotion'] == emotion]
        chart_data.append(
            alt.Chart(emotion_data).mark_line().encode(
                x='timestamp:T',
                y='confidence:Q',
                color=alt.value('#4CAF50' if emotion == df['emotion'].iloc[-1] else 'gray')
            )
        )
    
    chart = alt.layer(*chart_data).properties(
        height=300
    ).interactive()
    
    return chart 