import pandas as pd
import numpy as np
from utilities import *

def removeWinnerLoserReference (data):
    """
    Remove all the references of winner and loser, return only useful columns
    @params:
        data    - Original dataframe (Dataframe)
    """
    neededCols = ['Date', 'Location', 'Tournament', 'Series', 'Court', 'Surface',
                  'Round', 'Winner', 'Loser', 'WRank', 'LRank', 'WPts', 'LPts',
                  'Comment', 'B365W', 'B365L', 'PSW', 'PSL', 'AvgW', 'AvgL']
    data = data[neededCols]

    data.columns = ['Date', 'Location', 'Tournament', 'Series', 'Court', 'Surface',
                  'Round', 'Player0', 'Player1', 'Rank0', 'Rank1', 'Pts0', 'Pts1',
                  'Comment', 'B3650', 'B3651', 'PS0', 'PS1', 'Avg0', 'Avg1']
    return data


def findOddsForRow (row,  df):
    """
    Call in a loop to create terminal progress bar
    @params:
        row   - Selected row (Series)
        df    - Original dataframe (Dataframe)
    """
    # Search a row with similar ranks
    foundRows = pd.DataFrame()
    nearRank = 10
    while foundRows.empty and nearRank <= 100:
        foundRows = df[((row.Rank0 - nearRank) < df.Rank0) & (df.Rank0 < (row.Rank0 + nearRank))
                       & ((row.Rank1 - nearRank) < df.Rank1) & (df.Rank1 < (row.Rank1 + nearRank))]
        nearRank += 10

    return ( foundRows["Avg0"].mean(), foundRows["Avg1"].mean() ) if not foundRows.empty else ( None, None )


def expectedScore(A, B):
    """
    Calculate expected score of A in a match against B
    @params:
        A   - Elo rating for player A
        B   - Elo rating for player B
    """
    return 1 / (1 + 10 ** ((B - A) / 400))

def eloRating(old_elo, expected_score, actual_score, k_factor = 32):
    """
    Calculate the new Elo rating for a player
    @params:
        old_elo         - The previous Elo rating
        expected_score  - The expected score for this match
        actual_score    - The actual score for this match
        k_factor        - The k-factor for Elo (default: 32)
    """
    return old_elo + k_factor * (actual_score - expected_score)

def addEloRatingFeature(X, defaultElo = 1500):
    """
    Add the Elo Rating for each match of the dataset
    K-factor for players below 2100, between 2100–2400 and above 2400 of 32, 24 and 16, respectively
    @params:
        X            - The dataset
        kFactor      - The k-factor for Elo (default: 32)
        defaultElo   - The initial value for each player
    """
    players = pd.concat([X.Player0, X.Player1]).unique()
    oldEloRatings = pd.Series(np.ones(players.size) * defaultElo, index=players)
    kFactor = 32 if players.size < 2100 else (24 if 2100 <= players.size <= 2400 else 16)

    # Player 0 wins always
    PLAYER_0_SCORE = 1
    PLAYER_1_SCORE = 0

    # New feature columns
    player0EloRating = pd.Series(np.ones(X.shape[0]) * defaultElo)
    player1EloRating = pd.Series(np.ones(X.shape[0]) * defaultElo)

    printProgressBar(0, X.shape[0], prefix='Progress:', suffix='Complete')
    for i, row in X.iterrows():
        # First assign the rating, then update it for the next matches with the current match information
        player0EloRating[i] = oldEloRatingPlayer0 = oldEloRatings[row.Player0]
        player1EloRating[i] = oldEloRatingPlayer1 = oldEloRatings[row.Player1]

        expectedScorePlayer0 = expectedScore(oldEloRatingPlayer0, oldEloRatingPlayer1)
        expectedScorePlayer1 = expectedScore(oldEloRatingPlayer1, oldEloRatingPlayer0)

        oldEloRatings[row.Player0] = eloRating(oldEloRatingPlayer0, expectedScorePlayer0,
                                               PLAYER_0_SCORE, k_factor=kFactor)
        oldEloRatings[row.Player1] = eloRating(oldEloRatingPlayer1, expectedScorePlayer1,
                                               PLAYER_1_SCORE, k_factor=kFactor)

        printProgressBar(i+1, X.shape[0], prefix='Progress:', suffix='Complete')

    return X.assign(EloRating0 = player0EloRating.values, EloRating1 = player1EloRating.values)


def addMatchesPlayedAndWonFeatures(X, yearZeroForFeatures, years):
    """
    Add these features:
        * Number of matches played during the last solar year
        * Percentage of matches won during the last solar year
    @params:
        X                       - The dataset
        yearZeroForFeatures     - Dataset of the year before the first one
        years                   - Years considered
    """
    data = pd.concat([yearZeroForFeatures, X], sort=False)

    # New feature columns
    matchesPlayed0 = pd.Series(np.zeros(X.shape[0]))
    matchesPlayed1 = pd.Series(np.zeros(X.shape[0]))
    matchesWon0 = pd.Series(np.zeros(X.shape[0]))
    matchesWon1 = pd.Series(np.zeros(X.shape[0]))

    # Fill matrix with players as rows and years as cols
    players = pd.concat([data.Player0, data.Player1]).unique()
    matchesPlayed = pd.DataFrame(columns=years, index=players)
    matchesWon = pd.DataFrame(columns=years, index=players)
    for y in years:
        currentYear = data[data['Date'].dt.year == y]
        wonMatches = currentYear.Player0.value_counts()
        lostMatches = currentYear.Player1.value_counts()
        totalMatches = wonMatches + lostMatches

        matchesPlayed[y] = totalMatches
        matchesWon[y] = wonMatches / totalMatches * 100

    printProgressBar(0, X.shape[0], prefix='Progress:', suffix='Complete')
    for i, row in X.iterrows():
        matchesPlayed0[i] = matchesPlayed[row.Date.year - 1][row.Player0]
        matchesPlayed1[i] = matchesPlayed[row.Date.year - 1][row.Player1]
        matchesWon0[i] = matchesWon[row.Date.year - 1][row.Player0]
        matchesWon1[i] = matchesWon[row.Date.year - 1][row.Player1]

        printProgressBar(i+1, X.shape[0], prefix='Progress:', suffix='Complete')

    matchesPlayed0.fillna(0, inplace=True)
    matchesPlayed1.fillna(0, inplace=True)
    matchesWon0.fillna(0, inplace=True)
    matchesWon1.fillna(0, inplace=True)

    return X.assign(MatchesPlayed0 = matchesPlayed0.values, MatchesPlayed1 = matchesPlayed1.values,
                    MatchesWon0 = matchesWon0.values, MatchesWon1 = matchesWon1.values)


def addInjuriesAndWinningStreakFeatures(X, yearZeroForFeatures, years):
    """
    Add these features:
        * Injuries: number matches in witch the player retired or walkover in the past 3 months
        * Winning streak: current sequence of won games
    @params:
        X                       - The dataset
        yearZeroForFeatures     - Dataset of the year before the first one
        years                   - Years considered
    """
    data = pd.concat([yearZeroForFeatures, X], ignore_index=True, sort=False)

    # New features
    injuries0 = pd.Series(np.zeros(X.shape[0]))
    injuries1 = pd.Series(np.zeros(X.shape[0]))
    winningStreak0 = pd.Series(np.zeros(X.shape[0]))
    winningStreak1 = pd.Series(np.zeros(X.shape[0]))

    players = pd.concat([data.Player0, data.Player1]).unique()
    injuries = pd.DataFrame(columns=['Date', 'Player'])
    currentStreak = pd.Series(np.zeros(players.size), index=players)

    printProgressBar(0, data.shape[0], prefix='Progress:', suffix='Complete')
    k = 0
    for i, row in data.iterrows():
        if row.Date.year <= years[0]: # Skip year zero (used only for historical data)
            winningStreak0[k] = currentStreak[row.Player0]
            winningStreak1[k] = currentStreak[row.Player1]
            injuries0[k] = injuries[(injuries.Date >= (row.Date - pd.DateOffset(months=3)))
                                    & (injuries.Player == row.Player0)].count().max()
            injuries1[k] = injuries[(injuries.Date >= (row.Date - pd.DateOffset(months=3)))
                                    & (injuries.Player == row.Player1)].count().max()
            k += 1

        currentStreak[row.Player0] = float(currentStreak[row.Player0]) + 1
        currentStreak[row.Player1] = 0

        if row.Comment != "Completed":
            injuries.loc[len(injuries)] = [row.Date, row.Player1]

        printProgressBar(i+1, data.shape[0], prefix='Progress:', suffix='Complete')

    return X.assign(Injuries0=injuries0.values, Injuries1=injuries1.values, WinningStreak0=winningStreak0.values,
                    WinningStreak1=winningStreak1.values)
