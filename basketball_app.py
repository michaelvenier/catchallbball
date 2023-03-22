import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu


st.title('NBA Player Stats')
st.write(str(option_menu.__version__))
#Top menu
selected_page = option_menu(
    menu_title=None,
    options=["Player Stats","Team Stats"],
    default_index=0,
    orientation='horizontal'
)

st.markdown("""
This is my model. Here's how to use it.
""")

#SECTION: GETTING AND FILTERING DATA

#Getting the model data
# @st.cache #???
excel_file = 'Summary.xlsx'
df2 = pd.read_excel(excel_file,sheet_name=1)
df2.drop('Unnamed: 0',axis=1,inplace=True) #delete 1st column which is there mistakenly 
df2.drop([0,1],axis=0,inplace=True) #delete 0th and 1st rows which are blank.
#Rename all the columns
df2.rename(columns = {'Unnamed: 1':'Scenario',
'Unnamed: 2':'Year', 
'Unnamed: 3':'Player',
'Unnamed: 4':'Position',
'Unnamed: 5':'Age',
'Unnamed: 6':'Team',
'Unnamed: 7':'Games Played',
'Unnamed: 8':'Games Started',
'Unnamed: 9':'Total Minutes',
'Unnamed: 10':'Scoring',
'Unnamed: 11':'Passing',
'Unnamed: 12':'Rebounds',
'Unnamed: 13':'Total Offense',
'Unnamed: 14':'Total Defense',
'Unnamed: 15':'Total Score',
'Unnamed: 16':'MP Threshold'},
inplace=True
)
df2.drop('MP Threshold',axis=1,inplace=True) #Delete unneeded column.

#User input year and filtering model data by selected year (Sidebar)
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950,2024))))
df2 = df2[df2['Year']==selected_year] #Filter by selected year

# Web scraping of NBA player stats from basketball-reference.com
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating headers in content
    raw = raw.fillna(0)
    df1 = raw.drop(['Rk'], axis=1)
    return df1
df1 = load_data(selected_year)


# # Sidebar - Team selection
# sorted_unique_team = sorted(playerstats.Tm.unique())
# selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# User input Position selection and filtering data by positions
unique_pos = ['C','PF','SF','PG','SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
df2 = df2[df2.Position.isin(selected_pos)]
df1 = df1[df1.Pos.isin(selected_pos)]

#Select regular season or playoffs and filter data accordingly
list_of_scenarios = ['Regular Season']
if 'Playoffs' in set(df2['Scenario']): #ie if the year has gotten to the playoffs yet.
    list_of_scenarios.append('Playoffs')
scenario = st.sidebar.selectbox('Scenario',list_of_scenarios)
df2 = df2[df2['Scenario']==scenario] #Filter by scenario

# Sidebar - Team selection
sorted_unique_team = sorted(df2.Team.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
df1 = df1[df1.Tm.isin(selected_team)]
df2 = df2[df2.Team.isin(selected_team)]

#Minutes Played Filter (We don't always want players with low minutes to be included)
max_games_played = min([82,max(df2['Games Played'])]) #Roughly how many games have been played in the season so far? 
default_mins = int((max_games_played/82)*1200)
if scenario=='Playoffs': #In the playoffs we start with 0 as the minimum since lots of teams play very few games.
    mp = int(st.sidebar.text_input('Minimum Minutes Played','0'))
else:
    mp = int(st.sidebar.text_input('Minimum Minutes Played',str(default_mins)))
df2 =df2[df2["Total Minutes"] >= mp] #Filter df2 by MP. 

#SECTION: DATA EXPLORATION

#Team Data function. 
def byTeam():
    data = []
    for i in range(len(selected_team)):
        filt = df2['Team']==selected_team[i]
        df = df2[filt]
        avgTot = ((df['Total Score']*df['Total Minutes']).sum())/(df['Total Minutes'].sum()) #Weighted average of total weighted by MP. 
        avgOff = ((df['Total Offense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum()) #Weighted avg of offense weighted by MP.
        avgDef = ((df['Total Defense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum()) #Weighted avg of defense weighted by MP. 
        data.append((selected_team[i],avgTot,avgOff,avgDef))
    df = pd.DataFrame(data)
    df.columns = ['0','1','2','3']
    df.rename(columns = {'0':'Team','1':'Avg Total','2':'Avg Off','3':'Avg Def'},inplace=True)
    return df

players = [] #Players are manually selected (via user search) in line 154
def byPlayer():
    filt = df2.Player.isin(players)
    df = df2[filt]
    return df

def byX(year,teams,positions,mp,x): #A team stat that graphs by X like below. NEEDS WORK *&$#(* &^# *&^# (*&#^ )*$&^()*&@ )*(&^# )(*^&@)
    listt = []
    for i in range(min(df[x])):
        filt = df2['Age']==i
        df = df2[filt]
        avgTot = ((df['Total Score']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        avgOff = ((df['Total Offense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        avgDef = ((df['Total Defense']*df['Total Minutes']).sum())/(df['Total Minutes'].sum())
        listt.append((teams[i],avgTot,avgOff,avgDef))
    df = pd.DataFrame(listt)
    df.columns = ['0','1','2','3']
    df.rename(columns = {'0':'Team','1':'Avg Total','2':'Avg Off','3':'Avg Def'},inplace=True)
    return df

#SECTION: DISPLAYING DATA ON THE SITE. 

#Player stats
if selected_page=='Player Stats':
    st.header('Standard Stats from Basketball Reference')
    st.write('Data Dimension: ' + str(df1.shape[0]) + ' rows and ' + str(df1.shape[1]) + ' columns.')
    st.dataframe(df1)
    st.header('The Model Data')
    st.dataframe(df2)

    #Search by player
    st.header('Search By Player')
    players = st.multiselect('Players',list(df2['Player']))
    st.dataframe(byPlayer())

    #Visualization
    st.header('Visualization of Various Parameters Vs. Total Score')
    X = st.selectbox('X Axis',['Age','Team','Total Minutes','Scoring','Passing','Rebounds','Total Offense','Total Defense'])
    fig = px.scatter(df2,x=X,y='Total Score',hover_data=['Player'])
    st.plotly_chart(fig)

#Team stats
if selected_page=='Team Stats':
    st.header('Average Total Score for Teams')
    st.markdown("""
    Weighted by minutes played
    """)

    st.dataframe(byTeam())

    #Plot Teams Data
    dummy = byTeam()
    fig = px.scatter(dummy,x='Team',y='Avg Total', labels={'x':'Index','y':'Avg Score'},text='Team',title="Plot")
    fig.update_layout(yaxis_title='Avg Value')
    fig.update_layout(xaxis=dict(showticklabels=False))
    fig.update_traces(marker_opacity=0)
    st.header('Visualization of Various Parameters Vs. Total Score')
    st.plotly_chart(fig)

    # Download NBA player stats data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df1), unsafe_allow_html=True)

# Heatmap - Move this. 
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df1.to_csv('output.csv',index=False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
    
#django for csv update.
