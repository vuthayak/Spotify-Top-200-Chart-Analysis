# Github User: vuthayak
# Project name: Spotify Top 200 Charts Analysis
# Date: 2024-04-05
# Description: This program aims to demonstrate applications of data science + machine learning through
#              python libraries (pandas, numpy, matplotlib, seaborn, tensorflow, scikit-learn) and
#              principals (neural networks, feature engineering, data visualization). Dataset used is
#              'Spotify Top Songs and Audio Features' (V1) by Juliano Orlandi on Kaggle.

# DataFrame Libraries
import pandas as pd
import numpy as np
import random as rnd

# Visualization Libraries
import matplotlib.pyplot as plt
from pandasgui import show
import seaborn as sns

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Read in Data
spotify_df = pd.read_csv('spotify_top_songs_audio_features.csv',index_col="id")

# Clean Data
    # Dropping source, mode, key, time_signature (no/little correlation to features)
spotify_df.drop(['source','mode', 'key', 'time_signature'],axis=1,inplace=True)

    # Mapping outlier in artist_names (Tyler, The Creator -> Tyler The Creator) 
def tyler_map(artist_names):
    if 'Tyler, The Creator' in artist_names:
        return artist_names.replace('Tyler, The Creator','Tyler The Creator')
    else:
        return artist_names

spotify_df['artist_names'] = spotify_df['artist_names'].apply(tyler_map)

    # Splitting artist names into lists of each artist + making dummies for each artist
spotify_df['artist_names'] = spotify_df['artist_names'].apply(lambda x:x.split(", "))

artist_dummy = pd.get_dummies(data=spotify_df['artist_names'].explode(),drop_first=True).groupby(level=0).sum()

    # Concat dummies to original list (without artist_names)
spotify_df = pd.concat([spotify_df.drop('artist_names',axis=1),artist_dummy],axis=1)

# Collect Action from User
invalid_input = True
while invalid_input:
    try:
        action = int(input("""Enter the corresponding input to the action you would like to complete:
                        \n\t[0] Show all tracks in Spotify Top 200 Charts for an artist
                        \n\t[1] Show songs with the longest time on Spotify Top 200 Charts
                        \n\t[2] Predict how long a song will stay on Spotify Top 200 Charts based on features
                        \n\t[3] Predict how long a song will stay on Spotify Top 200 Charts based on the artist
                        \n\t[4] Visualize comparisons of 2 features
                        \n\tInput: """))
        if action in [0,1,2,3,4]:
            invalid_input = False
        else:
            print("\nInput out of Range. Try again.\n")

    except:
        print("\nInvalid input. Try again.\n")

    # Show all {artist} tracks
if action == 0:
    artist = input("\nEnter the name of the artist whose songs you would like to view:\n\n\tInput: ")
    artist_list = list(zip(spotify_df[spotify_df[artist]==1]['track_name'],spotify_df[spotify_df[artist]==1]['weeks_on_chart']))
    for row in artist_list:
        print(f'Track Name: {row[0]}, \tWeeks on Chart: {row[1]}')

    # Show {x} songs w/ longest time on charts
if action == 1:
    invalid_limit = True
    while invalid_limit:
        try:
            limit = int(input("\nEnter the corresponding input to the action you would like to complete:\n\n\tInput: "))
            invalid_limit = False
        except:
            print("\nInvalid input. Try again.\n")
    
    sorted_df = spotify_df.sort_values(by='weeks_on_chart',ascending=False).head(limit)[['track_name','weeks_on_chart']]
    sorted_list = list(zip(sorted_df['track_name'],sorted_df['weeks_on_chart']))
    for row in sorted_list:
        print(f'Track Name: {row[0]}, \tWeeks on Chart: {row[1]}')

    # Predict how long song will last on chart based on {x} features
if action == 2:

        # Collect features
    feature_list = np.append(spotify_df.columns[1:11],spotify_df.columns[12])
    print("\nChoose the features you would like to predict on. Be sure to type the features *exactly* how you see it.\n")
    for feature in feature_list:
        print (feature)

    chosen_features_incomplete = True
    chosen_features = []
    while chosen_features_incomplete:
        feature = input(f"""\nEnter a feature. Type '0' to stop.\n\tCurrent features: {chosen_features}
                       \n\tInput: """)
        if feature != '0':
            if feature in feature_list and feature not in chosen_features:
                chosen_features.append(feature)
            else:
                print("\nFeature not in feature list or already entered. Try again.")
        else:
            chosen_features_incomplete = False
    
        # Train based on those features (x) + weeks on chart (y) -- neural net
    X = spotify_df[chosen_features]
    y = spotify_df['weeks_on_chart']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Creating linear regression model
    model = LinearRegression()

        # Standardize data
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.fit(X_test)
    
        # Fit model to training data
    print("\nTraining predictive model now...\n")
    model.fit(X_train,y_train)
    print("\nPredictive model training complete!\n")

        # Collect inputs for each feature
    pred_chosen = []
    for feature in chosen_features:
        data_loop = True
        while data_loop:
            try:
                data = float(input(f'\nEnter data for {feature}: '))
                pred_chosen.append(data)
                data_loop = False
            except:
                print("\nInvalid input. Try again.")
        
        # Output result
    test_pred = model.predict(X_test)
    pred = model.predict(scaler.transform(pd.Series(pred_chosen).values.reshape(-1,len(pred_chosen))))

    print(f'\nWith the values and features you provided, a song is predicted to last {pred} weeks on the charts.')

        # Output R2 score
    print("\nR2 Score of Prediction Model:\n")
    print(r2_score(y_test,test_pred))

    # Predict how long song will last on chart based on who wrote it
if action == 3:

        # Train model on artist dummies (x) + weeks on chart (y) -- neural net
    X = spotify_df.iloc[:,13:]
    y = spotify_df['weeks_on_chart']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Standardize data + fit train and test data
    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

        # Creating neural network
    model = Sequential()
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

    model.add(Dense(234, activation = 'relu'))
    model.add(Dropout(0.1))

    for i in range(2):
        model.add(Dense(78, activation = 'relu'))
        model.add(Dropout(0.1))

        model.add(Dense(78, activation = 'relu'))
        model.add(Dropout(0.2))

    for i in range(5):
        model.add(Dense(39, activation = 'relu'))
        model.add(Dropout(0.1))

        model.add(Dense(39, activation = 'relu'))
        model.add(Dropout(0.2))

    for i in range(3):
        model.add(Dense(13, activation = 'relu'))
        model.add(Dropout(0.1))

        model.add(Dense(13, activation = 'relu'))
        model.add(Dropout(0.2))

    model.add(Dense(1, activation = 'linear'))

    model.compile(optimizer='adam',loss='mse')

        # Fit model to training data
    print("\nTraining predictive model now...\n")
    model.fit(X_train,y_train,epochs=600,validation_data=(X_test,y_test),batch_size=50,verbose=0,callbacks=[early_stop])
    print("\nPredictive model training complete!\n")

        # Collect test input (artist)
    artist_incomplete = True
    accepted_artists = spotify_df.columns[13:]
    while artist_incomplete:
        artist = input(f"""\nEnter the name of an artist you'd like to predict for.
                    \n\tInput: """)
        if artist in accepted_artists:
            artist_incomplete = False
        else:
            print("\nArtist not in charts history. Try again.")


        # Output result
    pred_df=(X[X[artist]==1])
    pred_row=pred_df.iloc[rnd.randint(0,len(pred_df))]
    pred = model.predict(scaler.transform(pred_row.values.reshape(1,2045)),verbose=0)
    test_pred = model.predict(X_test)

    print(f'\nA song by {artist} is predicted to last {round(pred[0][0],3)} weeks on the charts.')

        # Output R2 score
    print("\nR2 Score of Prediction Model:\n")
    print(r2_score(y_test,test_pred))

    # Visualize feature of choice vs artists of choice
if action == 4:
    feature_list = spotify_df.columns[1:13]
    print("\nChoose 2 of the following features to visualize. Be sure to type the features *exactly* how you see it.\n")
    for feature in feature_list:
        print (feature)
    
        # Collect 'x' feature
    invalid_feature_1 = True
    while invalid_feature_1:
        feature_1 = input("\nEnter the first feature:\n\n\tInput: ")
        if feature_1 in feature_list:
            invalid_feature_1 = False
        else:
            print("\nFeature not in feature list. Try again.")

        # Collect 'y' feature
    invalid_feature_2 = True
    while invalid_feature_2:
        feature_2 = input("\nEnter the second feature:\n\n\tInput: ")
        if feature_2 in feature_list:
            invalid_feature_2 = False
        else:
            print("\nFeature not in feature list. Try again.")

        # Collect artist for hue
    artist_incomplete = True
    accepted_artists = spotify_df.columns[13:]
    while artist_incomplete:
        artist = input(f"""\nEnter the name of an artist you'd like to compare these features against.
                       \n\tInput: """)
        if artist in accepted_artists:
            artist_incomplete = False
        else:
            print("\nArtist not in charts history. Try again.")
    
        # Plot scatterplot
    plt.title(f'{feature_1} vs {feature_2} for {artist}')
    sns.scatterplot(data=spotify_df[spotify_df[artist]==1],x=feature_1,y=feature_2,hue=artist)
    plt.legend([],[],frameon=False)
    plt.show()



