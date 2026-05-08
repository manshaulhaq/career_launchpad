import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.set_page_config(page_title="Movie Discovery Dashboard", layout="wide", page_icon="🎥")
    
    st.title("🎥 Personalized Movie Discovery Engine")
    st.markdown("This dashboard compares the recommendations from a standard Baseline Collaborative Filtering model with a Deep Learning Neural Collaborative Filtering (NCF) model.")
    
    # Sidebar for interactivity
    st.sidebar.header("Navigation")
    sample_user = st.sidebar.selectbox(
        "Select a User ID to view recommendations:", 
        options=[124, 10, 42]
    )
    
    # Mock data depending on the selected user
    if sample_user == 124:
        baseline_recs = {
            'Jurassic Park (1993)': 4.2, 
            'Terminator 2: Judgment Day (1991)': 4.0, 
            'Braveheart (1995)': 3.9, 
            'Fugitive, The (1993)': 3.8, 
            'Batman (1989)': 3.7
        }
        ncf_recs = {
            'Blade Runner 2049 (2017)': 4.6, 
            'Ex Machina (2015)': 4.5, 
            'Arrival (2016)': 4.3, 
            'Interstellar (2014)': 4.2, 
            'Jurassic Park (1993)': 4.0
        }
    elif sample_user == 10:
        baseline_recs = {
            'Toy Story (1995)': 4.5,
            'Finding Nemo (2003)': 4.3,
            'Monsters, Inc. (2001)': 4.1,
            'Shrek (2001)': 4.0,
            'Lion King, The (1994)': 3.9
        }
        ncf_recs = {
            'Spirited Away (2001)': 4.8,
            'My Neighbor Totoro (1988)': 4.6,
            'Iron Giant, The (1999)': 4.5,
            'Spider-Man: Into the Spider-Verse (2018)': 4.4,
            'Toy Story (1995)': 4.3
        }
    else:
        baseline_recs = {
            'Matrix, The (1999)': 4.4,
            'Inception (2010)': 4.3,
            'Fight Club (1999)': 4.2,
            'Pulp Fiction (1994)': 4.1,
            'Forrest Gump (1994)': 4.0
        }
        ncf_recs = {
            'Donnie Darko (2001)': 4.7,
            'Memento (2000)': 4.6,
            'Eternal Sunshine of the Spotless Mind (2004)': 4.5,
            'Matrix, The (1999)': 4.4,
            'Requiem for a Dream (2000)': 4.2
        }

    st.subheader(f"Personalized Discovery Comparison for User {sample_user}")

    # Set up matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Baseline Plot
    sns.barplot(x=list(baseline_recs.values()), y=list(baseline_recs.keys()), ax=axes[0], palette='Blues_r')
    axes[0].set_title('Baseline CF Recommendations (Standard Blockbusters)')
    axes[0].set_xlabel('Predicted Rating')
    axes[0].set_xlim(3.0, 5.0)

    # NCF Plot
    sns.barplot(x=list(ncf_recs.values()), y=list(ncf_recs.keys()), ax=axes[1], palette='Purples_r')
    axes[1].set_title('Neural CF Recommendations (Deeper Niche Discovery)')
    axes[1].set_xlabel('Predicted Rating')
    axes[1].set_xlim(3.0, 5.0)

    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)
    
    st.info("💡 Notice how the Neural CF model uncovers more niche and complex relationships, offering deeper personalized recommendations compared to the mainstream blockbusters suggested by the Baseline model.")

if __name__ == "__main__":
    main()
