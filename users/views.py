from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def TrainModel(request):
    import os
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    from django.conf import settings
    from matplotlib import pyplot as plt
    activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

    activity_color_map = {activity_codes_mapping['A']: 'lime',
                        activity_codes_mapping['B']: 'red',
                        activity_codes_mapping['C']: 'blue',
                        activity_codes_mapping['D']: 'orange',
                        activity_codes_mapping['E']: 'yellow',
                        activity_codes_mapping['F']: 'lightgreen',
                        activity_codes_mapping['G']: 'greenyellow',
                        activity_codes_mapping['H']: 'magenta',
                        activity_codes_mapping['I']: 'gold',
                        activity_codes_mapping['J']: 'cyan',
                        activity_codes_mapping['K']: 'purple',
                        activity_codes_mapping['L']: 'lightgreen',
                        activity_codes_mapping['M']: 'violet',
                        activity_codes_mapping['O']: 'limegreen',
                        activity_codes_mapping['P']: 'deepskyblue',   
                        activity_codes_mapping['Q']: 'mediumspringgreen',
                        activity_codes_mapping['R']: 'plum',
                        activity_codes_mapping['S']: 'olive'}
        
    def show_accel_per_activity(device, df, act, interval_in_sec = None):
        ''' Plots acceleration time history per activity '''

        df1 = df.loc[df.activity == act].copy()
        df1.reset_index(drop = True, inplace = True)

        df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds

        if interval_in_sec == None:
            ax = df1[:].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)
        else:
            ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)

        ax.set_xlabel('duration  (sec)', fontsize = 15)
        ax.set_ylabel('acceleration  (m/sec^2)',fontsize = 15)
        ax.set_title('Acceleration:   Device: ' + device + '      Activity:  ' + act, fontsize = 15)
        # plt.show()
        
        
    def show_ang_velocity_per_activity(device, df, act, interval_in_sec = None):
        ''' Plots angular volocity time history per activity '''

        df1 = df.loc[df.activity == act].copy()
        df1.reset_index(drop = True, inplace = True)

        df1['duration'] = (df1['timestamp'] - df1['timestamp'].iloc[0])/1000000000 # nanoseconds --> seconds

        if interval_in_sec == None:
            ax = df1[:].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)
        else:
            ax = df1[:interval_in_sec*20].plot(kind='line', x='duration', y=['x','y','z'], figsize=(25,7), grid = True) # ,title = act)

        ax.set_xlabel('duration  (sec)', fontsize = 15)
        ax.set_ylabel('angular velocity  (rad/sec)',fontsize = 15)
        ax.set_title('Angular velocity:  Device: ' + device + '      Activity:  ' + act, fontsize = 15)
        
    datasetpath = os.path.join(settings.MEDIA_ROOT,'wisdm-dataset')
    
    #accel_phone
        
    raw_par_10_phone_accel = pd.read_csv(datasetpath + '/' + 'raw/phone/accel/data_1610_accel_phone.txt', names = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z'], index_col=None, header=None)
    print('-'*100)
    print(raw_par_10_phone_accel)
    raw_par_10_phone_accel.z = raw_par_10_phone_accel.z.str.strip(';')
    raw_par_10_phone_accel.z = pd.to_numeric(raw_par_10_phone_accel.z)

    raw_par_10_phone_accel['activity'] = raw_par_10_phone_accel['activity_code'].map(activity_codes_mapping)

    raw_par_10_phone_accel = raw_par_10_phone_accel[['participant_id', 'activity_code', 'activity', 'timestamp', 'x', 'y', 'z']]

    print(raw_par_10_phone_accel)
    
    for key in activity_codes_mapping:
        show_accel_per_activity('Phone', raw_par_10_phone_accel, activity_codes_mapping[key], 10)
        
    #accel_watch
        
    raw_par_20_watch_accel = pd.read_csv(datasetpath + '/' + 'raw/watch/accel/data_1620_accel_watch.txt', names = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z'], index_col=None, header=None)

    raw_par_20_watch_accel.z = raw_par_20_watch_accel.z.str.strip(';')
    raw_par_20_watch_accel.z = pd.to_numeric(raw_par_20_watch_accel.z)

    raw_par_20_watch_accel['activity'] = raw_par_20_watch_accel['activity_code'].map(activity_codes_mapping)

    raw_par_20_watch_accel = raw_par_20_watch_accel[['participant_id', 'activity_code', 'activity', 'timestamp', 'x', 'y', 'z']]

    print(raw_par_20_watch_accel)
    for key in activity_codes_mapping:
        show_accel_per_activity('Watch', raw_par_20_watch_accel, activity_codes_mapping[key], 50)
        
        
    #gyro_phone
    raw_par_35_phone_ang_vel = pd.read_csv(datasetpath + '/' + 'raw/phone/gyro/data_1635_gyro_phone.txt', names = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z'], index_col=None, header=None)

    raw_par_35_phone_ang_vel.z = raw_par_35_phone_ang_vel.z.str.strip(';')
    raw_par_35_phone_ang_vel.z = pd.to_numeric(raw_par_35_phone_ang_vel.z)

    raw_par_35_phone_ang_vel['activity'] = raw_par_35_phone_ang_vel['activity_code'].map(activity_codes_mapping)

    raw_par_35_phone_ang_vel = raw_par_35_phone_ang_vel[['participant_id', 'activity_code', 'activity', 'timestamp', 'x', 'y', 'z']]

    print(raw_par_35_phone_ang_vel)
    
    for key in activity_codes_mapping:
        show_ang_velocity_per_activity('Phone', raw_par_35_phone_ang_vel, activity_codes_mapping[key])
        
        
    #gyro_watch
    raw_par_45_watch_ang_vel = pd.read_csv(datasetpath + '/' + 'raw/watch/gyro/data_1635_gyro_watch.txt', names = ['participant_id' , 'activity_code' , 'timestamp', 'x', 'y', 'z'], index_col=None, header=None)

    raw_par_45_watch_ang_vel.z = raw_par_45_watch_ang_vel.z.str.strip(';')
    raw_par_45_watch_ang_vel.z = pd.to_numeric(raw_par_45_watch_ang_vel.z)

    raw_par_45_watch_ang_vel['activity'] = raw_par_45_watch_ang_vel['activity_code'].map(activity_codes_mapping)

    raw_par_45_watch_ang_vel = raw_par_45_watch_ang_vel[['participant_id', 'activity_code', 'activity', 'timestamp', 'x', 'y', 'z']]

    print(raw_par_45_watch_ang_vel)
    
    for key in activity_codes_mapping:
        show_ang_velocity_per_activity('Watch', raw_par_45_watch_ang_vel, activity_codes_mapping[key])
        
        
    features = ['ACTIVITY',
            'X0', # 1st bin fraction of x axis acceleration distribution
            'X1', # 2nd bin fraction ...
            'X2',
            'X3',
            'X4',
            'X5',
            'X6',
            'X7',
            'X8',
            'X9',
            'Y0', # 1st bin fraction of y axis acceleration distribution
            'Y1', # 2nd bin fraction ...
            'Y2',
            'Y3',
            'Y4',
            'Y5',
            'Y6',
            'Y7',
            'Y8',
            'Y9',
            'Z0', # 1st bin fraction of z axis acceleration distribution
            'Z1', # 2nd bin fraction ...
            'Z2',
            'Z3',
            'Z4',
            'Z5',
            'Z6',
            'Z7',
            'Z8',
            'Z9',
            'XAVG', # average sensor value over the window (per axis)
            'YAVG',
            'ZAVG',
            'XPEAK', # Time in milliseconds between the peaks in the wave associated with most activities. heuristically determined (per axis)
            'YPEAK',
            'ZPEAK',
            'XABSOLDEV', # Average absolute difference between the each of the 200 readings and the mean of those values (per axis)
            'YABSOLDEV',
            'ZABSOLDEV',
            'XSTANDDEV', # Standard deviation of the 200 window's values (per axis)  ***BUG!***
            'YSTANDDEV',
            'ZSTANDDEV',
            'XVAR', # Variance of the 200 window's values (per axis)   ***BUG!***
            'YVAR',
            'ZVAR',
            'XMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'XMFCC1',
            'XMFCC2',
            'XMFCC3',
            'XMFCC4',
            'XMFCC5',
            'XMFCC6',
            'XMFCC7',
            'XMFCC8',
            'XMFCC9',
            'XMFCC10',
            'XMFCC11',
            'XMFCC12',
            'YMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'YMFCC1',
            'YMFCC2',
            'YMFCC3',
            'YMFCC4',
            'YMFCC5',
            'YMFCC6',
            'YMFCC7',
            'YMFCC8',
            'YMFCC9',
            'YMFCC10',
            'YMFCC11',
            'YMFCC12',
            'ZMFCC0', # short-term power spectrum of a wave, based on a linear cosine transform of a log power spectrum on a non-linear mel scale of frequency (13 values per axis)
            'ZMFCC1',
            'ZMFCC2',
            'ZMFCC3',
            'ZMFCC4',
            'ZMFCC5',
            'ZMFCC6',
            'ZMFCC7',
            'ZMFCC8',
            'ZMFCC9',
            'ZMFCC10',
            'ZMFCC11',
            'ZMFCC12',
            'XYCOS', # The cosine distances between sensor values for pairs of axes (three pairs of axes)
            'XZCOS',
            'YZCOS',
            'XYCOR', # The correlation between sensor values for pairs of axes (three pairs of axes)
            'XZCOR',
            'YZCOR',
            'RESULTANT', # Average resultant value, computed by squaring each matching x, y, and z value, summing them, taking the square root, and then averaging these values over the 200 readings
            'PARTICIPANT'] # Categirical: 1600 -1650

    import glob

    #the duplicate files to be ignored; all identical to 1600
    duplicate_files = [str(i) for i in range(1611, 1618)] # '1611',...'1617'

    # path = r'media/wisdm-dataset/arff_files/phone/accel'
    path = datasetpath + '/' + 'arff_files/phone/accel'
    all_files = glob.glob(path + "/*.arff")

    list_dfs_phone_accel = []

    for filename in all_files:

        if any(dup_fn in filename for dup_fn in duplicate_files):
            continue #ignore the duplicate files
        df = pd.read_csv(filename, names = features, skiprows = 96, index_col=None, header=0)
        list_dfs_phone_accel.append(df)

    all_phone_accel = pd.concat(list_dfs_phone_accel, axis=0, ignore_index=True, sort=False)

    print(all_phone_accel)
    
    print(all_phone_accel.info())
    
    all_phone_accel_breakpoint = all_phone_accel.copy()
    # all_phone_accel['ACTIVITY'].map(activity_codes_mapping).value_counts()
    
    
    # _ = all_phone_accel['ACTIVITY'].map(activity_codes_mapping).value_counts().plot(kind = 'bar', figsize = (15,5), color = 'purple', title = 'row count per activity', legend = True, fontsize = 15)
        
    all_phone_accel.drop('PARTICIPANT', axis = 1, inplace = True)
    
    
    from sklearn.model_selection import train_test_split

    y = all_phone_accel.ACTIVITY
    X = all_phone_accel.drop('ACTIVITY', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        train_size = 0.75, 
                                                        test_size = 0.25,
                                                        shuffle = True, 
                                                        stratify = all_phone_accel.ACTIVITY)
    
    print('-----X_train-------')
    print(X_train)
    print('-----Y Train-------')
    print(y_train)
    
    X_train.insert(0, 'Y', y_train)
    print('-----X_train-------')
    print(X_train)
    
    y_train = X_train['Y']
    print('-----Y Train-------')
    print(y_train)
    
    X_train.drop(['Y'], axis = 1, inplace = True)
    
    from sklearn.preprocessing import MaxAbsScaler

    scaling_transformer = MaxAbsScaler().fit(X_train[['XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'RESULTANT']])
    X_train[['XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'RESULTANT']] = scaling_transformer.transform(X_train[['XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'RESULTANT']])
    X_test = X_test.copy()
    X_test[['XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'RESULTANT']] = scaling_transformer.transform(X_test[['XAVG', 'YAVG', 'ZAVG', 'XPEAK', 'YPEAK', 'ZPEAK', 'XABSOLDEV', 'YABSOLDEV', 'ZABSOLDEV', 'RESULTANT']])
    
    print('-------X_test-------')
    print(X_test)
    
    X_train.reset_index(drop = True, inplace = True)
    print('-------X_train--------')
    print(X_train)
    
    X_test.reset_index(drop = True, inplace = True)
    print('-------X_test-------')
    print(X_test)
    print('len - ',len(X_test))
    print('type - ',type(X_test))
    X_test.to_csv('TestDataFrame.csv')
    
    y_train.reset_index(drop = True, inplace = True)
    print('-----Y Train-------')
    print(y_train)
    
    y_test.reset_index(drop = True, inplace = True)
    print('-----Y test----------')
    print(y_test)
    
    
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.tree import DecisionTreeClassifier
    import pickle
    from sklearn.model_selection import StratifiedShuffleSplit
    
    
    _ = y_train.value_counts(sort = False).plot(kind = 'bar', figsize = (15,5), color = 'red', title = 'row count per activity', legend = True, fontsize = 15)
    # plt.show()
    
    
        

    my_cv = StratifiedShuffleSplit(n_splits=5, train_size=0.7, test_size=0.3)
    
    dt_classifier = DecisionTreeClassifier()
    
    my_param_grid = {'min_samples_leaf': [6, 10, 20, 40],
                 'min_weight_fraction_leaf': [0.01, 0.02, 0.05],
                 'criterion': ['entropy'],
                 'min_impurity_decrease': [1e-2, 7e-3]}
    
    dt_model_gs = GridSearchCV(estimator=dt_classifier, 
                           param_grid=my_param_grid, 
                           cv=my_cv, 
                           scoring='accuracy',
                           verbose = 0,
                           return_train_score = True)
    
    dt_model_gs.fit(X_train, y_train)
    print('-------Fit Done---------')
    
    print(dt_model_gs.best_params_)
    dt_best_classifier = dt_model_gs.best_estimator_
    
    
    pickle.dump(dt_best_classifier, open('Ajmodel.pkl', 'wb'))
    print('-------Pickling Model Dumped------')
    
    y_test_pred = dt_best_classifier.predict(X_test)
    
    
    classification_report = classification_report(y_true=y_test,y_pred=y_test_pred,output_dict=True)
    
    classification_report = pd.DataFrame(classification_report).transpose().to_html()
    
    
    return render(request, 'users/TrainModel.html', {'classification_report':classification_report})


def Predict(request):
    if request.method == 'POST':
        
        
        activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

        
        import os
        from django.conf import settings
        import pickle
        import pandas as pd
        index_no = request.POST.get('index_no')
        print(index_no)
        print('type ----> ',type(index_no))
        modelPath = os.path.join(settings.MEDIA_ROOT,'Ajmodel.pkl')
        testDataPath = os.path.join(settings.MEDIA_ROOT,'TestDataFrame.csv')
        
        pickled_model = pickle.load(open(modelPath, 'rb'))
        testData = pd.read_csv(testDataPath)
        pred_val = testData.iloc[int(index_no)]
        print(pred_val)
        pred_result = pickled_model.predict([pred_val])
        SampleTestData = testData.head(100).to_html
        print('type of pred_result --> ', type(pred_result))
        print(pred_result)
        print('type ---> ',type(activity_codes_mapping))
        
        activity = activity_codes_mapping.get(pred_result[0])
        
        return render(request, 'users/prediction.html', {'testData':SampleTestData(index=False),'activity':activity})
        
    else:
        return render(request, 'users/prediction.html', {})
            