import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\jfbaa\PycharmProjects\ds2500\2500 project\matches.csv")

#checking column types and null values
df.info()
df.isna().sum()

#setting season to be the start year of a season
df['season'] = df.season.str.split('/').str[0]


#creating home and away score
df[['home_team', 'away_team']] = df.match_name.str.split(' - ', expand = True)
df[['home_score', 'away_score']] = df.result.str.split(':', expand = True)


#creates new winner collumn with the winner of the match comparing the home and away score
df['winner'] = np.where(df.home_score > df.away_score, 'HOME_TEAM', np.where(df.away_score > df.home_score, 'AWAY_TEAM', 'DRAW'))

# drop results as it is not needed anymore
df.drop(columns = ['result'], inplace = True)


to_int = ['season','home_score', 'away_score']
to_float = ['a_odd', 'd_odd', 'h_odd']

#turning columns into integers and floats
for col in to_int:
    df[col] = df[col].astype(int)

for col in to_float:
    df[col] = df[col].str.replace('-', '0')
    df[col] = df[col].astype(float)


#changing from date to datetime
df['date'] = pd.to_datetime(df.date, format ='mixed')

#home points made in each match
df['home_match_pts'] = np.where(df['winner'] == 'HOME_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))

#away points made in each match
df['away_match_pts'] = np.where(df['winner'] == 'AWAY_TEAM', 3 , np.where(df['winner'] == 'DRAW',1, 0))


cols_order = ['season', 'date', 'match_name', 'home_team', 'away_team', 'winner', 'home_score', 'away_score',
              'h_odd', 'd_odd', 'a_odd', 'home_match_pts', 'away_match_pts']

df = df[cols_order]

df.head()


def get_rank(x, team, delta_year):
    full_season_df = df[(df.season == (x.season - delta_year))]


    # Select columns to sum *before* calling .sum()
    full_home_df = full_season_df.groupby(['home_team'])[['home_match_pts', 'home_score', 'away_score']].sum().reset_index()
    full_home_df.columns = ['team', 'season_pts', 'season_gs', 'season_gc']

    # Select columns to sum *before* calling .sum()
    full_away_df = full_season_df.groupby(['away_team'])[['away_match_pts', 'away_score', 'home_score']].sum().reset_index()
    full_away_df.columns = ['team', 'season_pts', 'season_gs', 'season_gc']

    # --- Rest of the function remains the same ---
    rank_df = pd.concat([full_home_df, full_away_df], ignore_index = True)
    rank_df['season_gd'] = rank_df.season_gs - rank_df.season_gc
    rank_df = rank_df.groupby(['team']).sum().reset_index() # This sum is okay because only numeric columns are left
    rank_df = rank_df.sort_values(by = ['season_pts', 'season_gd', 'season_gs'], ascending = False)
    rank_df['rank'] = rank_df.season_pts.rank(method = 'first', ascending = False).astype(int)

    # Handle cases where the team might not be in the season's rank_df
    team_rank_series = rank_df[rank_df.team == team]['rank']
    if team_rank_series.empty:
        # Decide what rank to return if the team didn't play/exist that season
        # Returning NaN or a large number might be appropriate
        team_rank = np.nan # Or perhaps 999 or some other indicator
    else:
        team_rank = team_rank_series.min()

    return team_rank

def get_match_stats(x, team):
    #home df filter
    home_df = df[(df.home_team == team) & (df.date < x.date) & (df.season == x.season)]

    #away df filter
    away_df = df[(df.away_team == team) & (df.date < x.date) & (df.season == x.season)]

    #points
    home_table = home_df.groupby(['date']).sum()[['home_match_pts', 'home_score', 'away_score']].reset_index()
    home_table.columns = ['date', 'match_pts', 'match_gs', 'match_gc']
    home_table['match_gd'] = home_table.match_gs - home_table.match_gc
    home_table['host'] = 'home'

    away_table = away_df.groupby(['date']).sum()[['away_match_pts', 'away_score', 'home_score']].reset_index()
    away_table.columns = ['date', 'match_pts', 'match_gs', 'match_gc']
    away_table['match_gd'] = away_table.match_gs - away_table.match_gc
    away_table['host'] = 'away'

    full_table = pd.concat([home_table, away_table], ignore_index = True)
    full_table = full_table.sort_values('date', ascending = True)

    #get streaks
    full_table['start_of_streak'] = full_table.match_pts.ne(full_table.match_pts.shift())
    full_table['streak_id'] = full_table['start_of_streak'].cumsum()
    full_table['streak_counter'] = full_table.groupby('streak_id').cumcount() + 1

    #make exponentially weighted average
    full_table['ewma_pts'] = full_table.match_pts.ewm(span=3, adjust=False).mean()
    full_table['ewma_gs'] = full_table.match_gs.ewm(span=3, adjust=False).mean()
    full_table['ewma_gc'] = full_table.match_gc.ewm(span=3, adjust=False).mean()

    streak_table = full_table[full_table.date == full_table.date.max()]

    if not streak_table.empty: # Check if streak_table is not empty before accessing .min()
        if streak_table.match_pts.min() == 3:
            win_streak = streak_table.streak_counter.sum()
            loss_streak = 0
            draw_streak = 0
        elif streak_table.match_pts.min() == 0:
            win_streak = 0
            loss_streak = streak_table.streak_counter.sum()
            draw_streak = 0
        else: # points == 1
            win_streak = 0
            loss_streak = 0
            draw_streak = streak_table.streak_counter.sum()
    else: # Handle case where there are no previous matches in the season yet
         win_streak = 0
         loss_streak = 0
         draw_streak = 0


    home_season_pts = home_table.match_pts.sum()
    home_season_gs = home_table.match_gs.sum()
    home_season_gc = home_table.match_gc.sum()
    home_season_wins = len(home_table[home_table.match_pts == 3])
    home_season_draws = len(home_table[home_table.match_pts == 1])
    home_season_losses = len(home_table[home_table.match_pts == 0])

    away_season_pts = away_table.match_pts.sum()
    away_season_gs = away_table.match_gs.sum()
    away_season_gc = away_table.match_gc.sum()
    away_season_wins = len(away_table[away_table.match_pts == 3])
    away_season_draws = len(away_table[away_table.match_pts == 1])
    away_season_losses = len(away_table[away_table.match_pts == 0])

    #total points stats
    curr_season_pts = home_season_pts + away_season_pts
    curr_season_gs = home_season_gs + away_season_gs
    curr_season_gc = home_season_gc + away_season_gc
    curr_season_wins = home_season_wins + away_season_wins
    curr_season_draws = home_season_draws + away_season_draws
    curr_season_losses = home_season_losses + away_season_losses

    #getting data for a given delta (last 3 matches)
    # Check if there are at least 3 matches before slicing
    num_matches = len(full_table)
    if num_matches >= 3:
        full_table_delta = full_table.iloc[-3:] # Use iloc for position-based slicing
    elif num_matches > 0:
        full_table_delta = full_table.iloc[-num_matches:] # Use fewer if less than 3 exist
    else:
        full_table_delta = pd.DataFrame() # Empty dataframe if no matches

    # Calculate last 3 stats only if full_table_delta is not empty
    if not full_table_delta.empty:
        home_pts_last_3 = full_table_delta[full_table_delta.host == 'home'].match_pts.sum()
        away_pts_last_3 = full_table_delta[full_table_delta.host == 'away'].match_pts.sum()

        # Use actual goals from the delta, not season totals, for avg_gs and avg_gc
        home_gs_last_3 = full_table_delta[full_table_delta.host == 'home'].match_gs.sum()
        away_gs_last_3 = full_table_delta[full_table_delta.host == 'away'].match_gs.sum()
        home_gc_last_3 = full_table_delta[full_table_delta.host == 'home'].match_gc.sum()
        away_gc_last_3 = full_table_delta[full_table_delta.host == 'away'].match_gc.sum()

        # Correct averaging logic: divide by number of matches in delta (up to 3)
        delta_match_count = len(full_table_delta)
        avg_pts_last_3 = (home_pts_last_3 + away_pts_last_3) / delta_match_count
        avg_gs_last_3 = (home_gs_last_3 + away_gs_last_3) / delta_match_count
        avg_gc_last_3 = (home_gc_last_3 + away_gc_last_3) / delta_match_count

        # EWMA values are typically taken from the most recent point in time
        ewma_pts_last_3 = full_table.iloc[-1]['ewma_pts']
        ewma_gs_last_3 = full_table.iloc[-1]['ewma_gs']
        ewma_gc_last_3 = full_table.iloc[-1]['ewma_gc']

    else: # Handle case with no previous matches
        avg_pts_last_3 = 0
        avg_gs_last_3 = 0
        avg_gc_last_3 = 0
        ewma_pts_last_3 = 0
        ewma_gs_last_3 = 0
        ewma_gc_last_3 = 0


    return curr_season_pts, avg_pts_last_3, ewma_pts_last_3, curr_season_gs, avg_gs_last_3, ewma_gs_last_3, curr_season_gc, avg_gc_last_3, ewma_gc_last_3, curr_season_wins, curr_season_draws, curr_season_losses, win_streak, loss_streak, draw_streak

def get_days_ls_match(x, team):

    #filtering last game of the team and getting date
    last_date = df[(df.date < x.date) & (df.season == x.season) & (df.match_name.str.contains(team))].date.max()

    # Handle NaT case (no previous match found)
    if pd.isna(last_date):
        days_since_last = np.nan # Or perhaps a large number?
    else:
        days_since_last = (x.date - last_date)/np.timedelta64(1,'D')

    return days_since_last

def get_ls_winner(x):
    temp_df = df[(df.date < x.date) & (df.match_name.str.contains(x.home_team)) & (df.match_name.str.contains(x.away_team))]
    if not temp_df.empty:
        temp_df = temp_df[temp_df.date == temp_df.date.max()]

    #checking if there was a previous match
    if temp_df.empty:
        last_h2h_winner = None # Or maybe 'NONE' or 'FIRST_MEETING'
    elif temp_df.winner.iloc[0] == 'DRAW': # Use iloc[0] as there should only be one row
        last_h2h_winner = 'DRAW'
    elif temp_df.home_team.iloc[0] == x.home_team:
         # If current home team was home team last time, winner is direct
        last_h2h_winner = temp_df.winner.iloc[0] # 'HOME_TEAM' or 'AWAY_TEAM'
    else:
        # If current home team was away team last time, flip the winner
        if temp_df.winner.iloc[0] == 'HOME_TEAM':
             last_h2h_winner = 'AWAY_TEAM' # Winner was the other team (now away)
        else: # Winner was 'AWAY_TEAM'
             last_h2h_winner = 'HOME_TEAM' # Winner was the other team (now home)

    return last_h2h_winner

def create_main_cols(x, team):

    #get current and last delta (years) rank
    curr_team_rank = get_rank(x, team, 0)
    prev_team_rank = get_rank(x, team, 1)

    #get main match stats
    curr_season_pts, avg_pts_last_3, ewma_pts_last_3, curr_season_gs, avg_gs_last_3, ewma_gs_last_3, curr_season_gc, avg_gc_last_3, ewma_gc_last_3, curr_season_wins, curr_season_draws, curr_season_losses, win_streak, loss_streak, draw_streak = get_match_stats(x, team)

    #get days since last match
    days_since_last = get_days_ls_match(x, team)

    return curr_team_rank, prev_team_rank, days_since_last, curr_season_pts, avg_pts_last_3, ewma_pts_last_3, curr_season_gs, avg_gs_last_3, ewma_gs_last_3, curr_season_gc, avg_gc_last_3, ewma_gc_last_3, curr_season_wins, curr_season_draws, curr_season_losses, win_streak, loss_streak, draw_streak

# Define base column names for features calculated per team
cols = ['curr_season_rank', 'prev_season_rank', 'days_since_last', 'curr_season_pts',
        'avg_pts_last_3', 'ewma_pts_last_3', 'curr_season_gs', 'avg_gs_last_3', 'ewma_gs_last_3',
        'curr_season_gc', 'avg_gc_last_3', 'ewma_gc_last_3', 'curr_season_wins', 'curr_season_draws',
        'curr_season_losses', 'win_streak', 'loss_streak', 'draw_streak']

# Create prefixed column names for home and away teams
home_cols = ['home_' + col for col in cols]
away_cols = ['away_' + col for col in cols]

# Apply the function to calculate features for the home team
df[home_cols] = pd.DataFrame(
    df.apply(
        lambda x: create_main_cols(x, x.home_team), axis = 1).to_list(), index = df.index)

# Apply the function to calculate features for the away team
df[away_cols] = pd.DataFrame(
    df.apply(
        lambda x: create_main_cols(x, x.away_team), axis = 1).to_list(), index = df.index)

# Calculate the winner of the last head-to-head match
df['last_h2h_winner'] = df.apply(lambda x: get_ls_winner(x), axis = 1)


# Define the path and filename for your output CSV
output_filepath = r"C:\Users\jfbaa\PycharmProjects\ds2500\2500 project\processed_matches_features.csv" # Using a descriptive name
# Save the DataFrame to CSV
df.to_csv(output_filepath, index=False)
print(f"DataFrame with engineered features saved successfully to: {output_filepath}")
