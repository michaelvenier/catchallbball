import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import requests
import datetime
from streamlit_option_menu import option_menu
from google.oauth2 import service_account
from gsheetsdb import connect
from gspread_pandas import Spread, Client

current_year = int(datetime.date.today().year)

st.title('NBA Analytical Model')

st.sidebar.write("Contact: catchallbball@gmail.com")

# Top menu - 3 sections, individual players, team, and testing.
selected_page = option_menu(
    menu_title=None,
    options=["Player Stats", "Team Stats", 'Testing The Model'],
    default_index=0,
    orientation='horizontal'
)

# SECTION: GETTING, CLEANING AND FILTERING DATA

# Functions/variables ending in 2 are for the model data.
# Functions/vars ending in _exp are for the Testing the Model section.
# Functions/vars ending in nothing or 1 are standard stats from the web.

# Get data from the google sheet.
# @st.experimental_memo
# def load_data2():
#     # Create a connection object.
#     credentials = service_account.Credentials.from_service_account_info(
#         st.secrets["gcp_service_account"],
#         scopes=[
#             "https://www.googleapis.com/auth/spreadsheets",
#         ],
#     )
#     conn = connect(credentials=credentials)
#     sheet_url = st.secrets["private_gsheets_url"]
#     query = f'SELECT * FROM "{sheet_url}"'
#     rows = conn.execute(query, headers=0).fetchall()
#
#     # Convert data to a Pandas DataFrame.
#     df = pd.DataFrame.from_records(rows, columns=rows[0])
#     return df
#
# df2 = load_data2()  # Raw model data.

@st.experimental_memo
def load_data2():
    excel_file = 'catchall_April_2023.xlsx'
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


df2 = load_data2()

# Filtering df2: Rename columns
df2.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
df2.rename(columns={'0': 'Scenario',
                    '1': 'Year', '2': 'Player',
                    '3': 'Position',
                    '4': 'Age',
                    '5': 'Team',
                    '6': 'G',
                    '7': 'GS',
                    '8': 'MP',
                    '9': 'MP/G',
                    '10': 'Scoring',
                    '11': 'Passing',
                    '12': 'Rebounds',
                    '13': 'Total Offense',
                    '14': 'Total Defense',
                    '15': 'Total Score',
                    '16': 'MP Threshold'}
           , inplace=True
           )
# Make data type appropriate
df2['Year'] = df2['Year'].astype(int)
df2['Age'] = df2['Age'].astype(int)
df2['G'] = df2['G'].astype(int)
df2['GS'] = df2['GS'].astype(int)
df2['MP'] = df2['MP'].astype(int)

df2.drop('MP Threshold', axis=1, inplace=True)  # Delete unneeded column.

# User input year and filtering model data by selected year (Sidebar). This needs to be above load_data, a function of year.
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(2014, current_year + 1))))
df_exp = df2  # This unfiltered data will be used when running the experiment to test the model. See line
df2 = df2[df2['Year'] == selected_year]  # Filter by selected year

# Web scraping of NBA player stats from basketball-reference.com
@st.cache
def load_data(year):
    url = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    df1 = raw.drop(['Rk'], axis=1)
    return df1

df1 = load_data(selected_year)

@st.cache
def load_data_playoffs(year):
    url = "https://www.basketball-reference.com/playoffs/NBA_" + str(year) + "_per_game.html"
    response = requests.get(url)
    if response.status_code == 404:  # Accounting for if the playoffs haven't happened yet ie the link doesn't exist.
        return None
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  # Deletes repeating headers in content
    raw = raw.fillna(0)
    df1 = raw.drop(['Rk'], axis=1)
    return df1

dfTemp = load_data_playoffs(selected_year)
if load_data_playoffs(selected_year) is not None:
    dfP = load_data_playoffs(selected_year)

@st.cache
def load_data_exp(
        year=current_year):  # Counting the number of wins for each time since 2016. Will be combined with byTeam_exp
    url = 'https://www.basketball-reference.com/leagues/NBA_' + str(2016) + '_standings.html'

    # Read the HTML table into a list of DataFrames
    dfs = pd.read_html(url, header=0)

    # Select the second DataFrame (index 1)
    df0 = dfs[0]
    df0 = df0[['Eastern Conference', 'W']]
    df0.columns = ['0', '1']
    df0.rename(columns={'0': 'Team', '1': 'W'}, inplace=True)
    df1 = dfs[1]
    df1 = df1[['Western Conference', 'W']]
    df1.columns = ['0', '1']
    df1.rename(columns={'0': 'Team', '1': 'W'}, inplace=True)
    df = pd.concat([df0, df1])
    df.sort_values(by=['Team'], axis=0, ascending=True, inplace=True)
    df['Team'] = ['ATL',
                  'BOS',
                  'BRK',
                  'CHO',
                  'CHI',
                  'CLE',
                  'DAL',
                  'DEN',
                  'DET',
                  'GSW',
                  'HOU',
                  'IND',
                  'LAC',
                  'LAL',
                  'MEM',
                  'MIA',
                  'MIL',
                  'MIN',
                  'NOP',
                  'NYK',
                  'OKC',
                  'ORL',
                  'PHI',
                  'PHO',
                  'POR',
                  'SAC',
                  'SAS',
                  'TOR',
                  'UTA',
                  'WAS'
                  ]
    df.sort_values(by=['Team'], axis=0, ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    for i in range(2017, year + 1):
        url = 'https://www.basketball-reference.com/leagues/NBA_' + str(i) + '_standings.html'
        dfs = pd.read_html(url, header=0)

        # Select the second DataFrame (index 1)
        df0 = dfs[0]
        df0 = df0[['Eastern Conference', 'W']]
        df0.columns = ['0', '1']
        df0.rename(columns={'0': 'Team', '1': 'W'}, inplace=True)
        df1 = dfs[1]
        df1 = df1[['Western Conference', 'W']]
        df1.columns = ['0', '1']
        df1.rename(columns={'0': 'Team', '1': 'W'}, inplace=True)
        dfTemp = pd.concat([df0, df1])
        dfTemp.sort_values(by=['Team'], axis=0, ascending=True, inplace=True)
        dfTemp['Team'] = ['ATL',
                          'BOS',
                          'BRK',
                          'CHO',
                          'CHI',
                          'CLE',
                          'DAL',
                          'DEN',
                          'DET',
                          'GSW',
                          'HOU',
                          'IND',
                          'LAC',
                          'LAL',
                          'MEM',
                          'MIA',
                          'MIL',
                          'MIN',
                          'NOP',
                          'NYK',
                          'OKC',
                          'ORL',
                          'PHI',
                          'PHO',
                          'POR',
                          'SAC',
                          'SAS',
                          'TOR',
                          'UTA',
                          'WAS'
                          ]
        dfTemp.sort_values(by=['Team'], axis=0, ascending=True, inplace=True)
        dfTemp.reset_index(drop=True, inplace=True)
        df['W'] += dfTemp['W']
    return df

# User input Position selection and filtering data by positions
unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)
df2 = df2[df2.Position.isin(selected_pos)]
df1 = df1[df1.Pos.isin(selected_pos)]

# Select regular season or playoffs and filter data accordingly
list_of_scenarios = ['Regular Season']
if 'Playoffs' in set(df2['Scenario']):  # ie if the year has gotten to the playoffs yet.
    list_of_scenarios.append('Playoffs')
scenario = st.sidebar.selectbox('Scenario', list_of_scenarios)
df2 = df2[df2['Scenario'] == scenario]  # Filter by scenario

# Sidebar - Team selection
sorted_unique_team = sorted(df2.Team.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
df1 = df1[df1.Tm.isin(selected_team)]
df2 = df2[df2.Team.isin(selected_team)]

# Minutes Played Filter (We don't always want players with low minutes to be included)
max_games_played = min([82, max(df2['G'])])  # Roughly how many games have been played in the season so far?
default_mins = int((max_games_played / 82) * 1200)
if scenario == 'Playoffs':  # In the playoffs we start with 0 as the minimum since lots of teams play very few games.
    mp = int(st.sidebar.text_input('Minimum Minutes Played', '0'))
else:
    mp = int(st.sidebar.text_input('Minimum Minutes Played', str(default_mins)))
df2 = df2[df2["MP"] >= mp]  # Filter df2 by MP.

# Sort data by Total Score and reindex.j
df2.sort_values(by=['Total Score'], axis=0, ascending=False, inplace=True)
df2.reset_index(drop=True, inplace=True)
df2.index = np.arange(1, len(df2) + 1)

# SECTION: DATA EXPLORATION

# Team Data function.
@st.experimental_memo
def byTeam():
    data = []
    for i in range(len(selected_team)):
        filt = df2['Team'] == selected_team[i]
        df = df2[filt]
        if df['MP'].sum() == 0:
            (avgTot, avgOff, avgDef, avgPass, avgScore, avgRebound, avgAge) = (0, 0, 0, 0, 0, 0, 0)
        else:
            avgTot = ((df['Total Score'] * df['MP']).sum()) / (
                df['MP'].sum())  # Weighted average of total weighted by MP.
            avgOff = ((df['Total Offense'] * df['MP']).sum()) / (
                df['MP'].sum())  # Weighted avg of offense weighted by MP.
            avgDef = ((df['Total Defense'] * df['MP']).sum()) / (
                df['MP'].sum())  # Weighted avg of defense weighted by MP.
            avgPass = ((df['Passing'] * df['MP']).sum()) / (df['MP'].sum())
            avgScore = ((df['Scoring'] * df['MP']).sum()) / (df['MP'].sum())
            avgRebound = ((df['Rebounds'] * df['MP']).sum()) / (df['MP'].sum())
            avgAge = ((df['Age'] * df['MP']).sum()) / (df['MP'].sum())
        data.append((selected_team[i], avgTot, avgOff, avgDef, avgPass, avgScore, avgRebound, avgAge))
    df = pd.DataFrame(data)
    df.columns = ['0', '1', '2', '3', '4', '5', '6', '7']
    df.rename(
        columns={'0': 'Team', '1': 'Avg Total', '2': 'Avg Off', '3': 'Avg Def', '4': 'Avg Passing', '5': 'Avg Scoring',
                 '6': 'Avg Rebounds', '7': 'Avg Age'}, inplace=True)
    return df

sorted_unique_team = pd.DataFrame(sorted_unique_team)
sorted_unique_team.columns = ['0']

# This is similar to above, but uses df_exp, which hasn't been filtered by year. This way we can average over several years.
# This will be combined with load_data_exp for the Testing section of the site.
@st.experimental_memo
def byTeam_exp():
    data = []
    teams = sorted_unique_team[sorted_unique_team['0'] != 'TOT']
    teams.reset_index(drop=True, inplace=True)
    for i in range(len(teams)):
        filt = df_exp['Team'] == teams['0'][i]
        df = df_exp[filt]
        df = df[df['Year'].isin(list(range(2016, current_year + 1)))]
        if df['MP'].sum() == 0:
            avgTot = 0
        else:
            avgTot = ((df['Total Score'] * df['MP']).sum()) / (
                df['MP'].sum())  # Weighted average of total weighted by MP.
        data.append((selected_team[i], avgTot))
    df = pd.DataFrame(data)
    df.columns = ['0', '1']
    df.rename(columns={'0': 'Team', '1': 'Avg Total'}, inplace=True)
    df.sort_values(by='Team', axis=0, ascending=True, inplace=True)
    return df

# Combining byTeam_exp (Model avg) and load_data_exp (number of wins). We'll see how these things correlate to test the model.
@st.cache(allow_output_mutation=True)
def load_combined_exp():
    df1 = byTeam_exp()  # Model avg score over several years
    df2 = load_data_exp(current_year)  # Number of wins over the past several years, to compare to.
    df2['W'] = df2['W'].astype(int)
    df1['W'] = df2['W'].astype(int)
    return df1

dfCombined = load_combined_exp()
corr = dfCombined['Avg Total'].corr(dfCombined['W'])

players = []  # Players are manually selected (via user search)
@st.experimental_memo
def byPlayer():
    filt = df2.Player.isin(players)
    df = df2[filt]
    return df

# SECTION: DISPLAYING DATA ON THE SITE. Displays everything that isn't needed directly for filtering. Things needed for filtering were displayed
# as they were introduced.

# Player stats
if selected_page == 'Player Stats':
    st.markdown("""
    Explore data produced by the catchall analytical model! There are also standard NBA stats for comparison.
    """)
    st.header('Standard Stats from Basketball Reference')
    st.write('Data Dimension: ' + str(df1.shape[0]) + ' rows and ' + str(df1.shape[1]) + ' columns.')
    if scenario == 'Regular Season':
        st.dataframe(df1)
    if scenario == 'Playoffs':  # Make this an else.
        st.dataframe(dfP)
    st.header('The Model Data')
    st.dataframe(df2)
    # st.dataframe(df2)

    # Search by player
    st.header('Search By Player')
    players = st.multiselect('Players', list(df2['Player']))
    st.dataframe(byPlayer())

    # Visualization
    st.header('Visualization of Various Parameters Vs. Total Score')
    X = st.selectbox('X Axis',
                     ['Age', 'Team', 'MP', 'Scoring', 'Passing', 'Rebounds', 'Total Offense', 'Total Defense'])
    fig = px.scatter(df2, x=X, y='Total Score', hover_data=['Player'])
    st.plotly_chart(fig)

    # Heatmap. Edit
    if st.button('Intercorrelation Heatmap'):
        st.header('Intercorrelation Matrix Heatmap')
        df1.to_csv('output.csv', index=False)
        dfHeat1 = pd.read_csv('output.csv')

        corrHeat1 = dfHeat1.corr()
        mask1 = np.zeros_like(corrHeat1)
        mask1[np.triu_indices_from(mask1)] = True
        with sns.axes_style("white"):
            f1, ax1 = plt.subplots(figsize=(7, 5))
            ax1 = sns.heatmap(corrHeat1, mask=mask1, vmax=1, square=True)
        plt.title('Standard Data')
        st.pyplot(f1)

        df2.to_csv('output.csv', index=False)
        dfHeat2 = pd.read_csv('output.csv')

        corrHeat2 = dfHeat2.corr()
        mask2 = np.zeros_like(corrHeat2)
        mask2[np.triu_indices_from(mask2)] = True
        with sns.axes_style("white"):
            f2, ax2 = plt.subplots(figsize=(7, 5))
            ax2 = sns.heatmap(corrHeat2, mask=mask2, vmax=1, square=True)
        plt.title('Model Data')
        st.pyplot(f2)

# Team stats
if selected_page == 'Team Stats':
    st.header('Average Total Score for Teams')
    st.markdown("""
    Weighted by minutes played
    """)

    st.dataframe(byTeam())

    # Plot Teams Data
    st.header('Visualization of Various Parameters Vs. Total Score')
    X2 = st.selectbox('X Axis',
                      ['Team', 'Avg Total', 'Avg Off', 'Avg Def', 'Avg Passing', 'Avg Scoring', 'Avg Rebounds',
                       'Avg Age'])
    fig = px.scatter(byTeam(), x=X2, y='Avg Total', labels={'x': 'Index', 'y': 'Avg Score'}, text='Team',
                     title="Weighted by Minutes Played")
    fig.update_layout(yaxis_title='Avg Total')
    fig.update_layout(xaxis=dict(showticklabels=True))
    if X2 == 'Team':
        fig.update_layout(xaxis=dict(showticklabels=False))
    fig.update_traces(marker_opacity=0)

    st.plotly_chart(fig)

# Testing the model
if selected_page == 'Testing The Model':
    st.header('Testing the model by comparing to each teams number of wins.')
    st.markdown("""
        In the following Table, we have each team's average score averaged over all seasons since the 2015-2016 season. 
        We also have the number of wins the team has in that time. If these data sets correlate strongly, it is indication
        that the model is working well.
        """)
    st.dataframe(load_combined_exp())
    st.write('correlation = ' + str(corr))
    st.markdown("""
        -1 indicates perfect inverse correlation, 0 indicates no correlation and 1 indicates perfect correlation.
        The closer the number is to 1 the better.
        """)
    # Download NBA player stats data
    # https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806

def filedownload2(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download the model data CSV File (current data being displayed)</a>'
    return href

def filedownload_exp(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download the model data CSV File (Data for all years and players)</a>'
    return href

st.markdown(filedownload2(df1), unsafe_allow_html=True)
st.markdown(filedownload_exp(df_exp), unsafe_allow_html=True)
