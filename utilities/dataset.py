import pandas as pd
import numpy as np
from scipy import stats
import urllib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class Dataset:
    def __init__(self, x=None, y=None, feature_names=None, class_names=None, categorical_features=None):
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
    
    def rul_finder(self, mapper):
        RULs = []
        RULsInd = []
        cnt = 1
        for m in mapper:
            RULsInd.append(cnt)
            cnt += 1
            RULs.append(m[1])
        return RULsInd, RULs

    def load_data_turbofan(self):
        import sys
        path = sys.path[-1]
        feature_names = ['u', 't', 'os_1', 'os_2', 'os_3'] #u:unit, t:time, s:sensor
        feature_names += ['s_{0:02d}'.format(s + 1) for s in range(26)]
        fm = {}
        for i in range(4):
            p = path+'\\datasets\\CMAPSSData\\train_FD00'+ str(i+1) +'.txt'
            df_train = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            mapper = {}
            for unit_nr in df_train['u'].unique():
                mapper[unit_nr] = df_train['t'].loc[df_train['u'] == unit_nr].max()
            df_train['RUL'] = df_train['u'].apply(lambda nr: mapper[nr]) - df_train['t']
            p = path+'\\datasets\\CMAPSSData\\test_FD00'+ str(i+1) +'.txt'
            df_test = pd.read_csv(p, sep= ' ', header=None, names=feature_names, index_col=False)
            p = path+'\\datasets\\CMAPSSData\\RUL_FD00'+ str(i+1) +'.txt'
            df_RUL = pd.read_csv(p, sep= ' ', header=None, names=['RUL_actual'], index_col=False)
            temp_mapper = {}
            for unit_nr in df_test['u'].unique():
                temp_mapper[unit_nr] = df_test['t'].loc[df_test['u'] == unit_nr].max()#max time einai to rul tou
            mapper_test = {}
            cnt = 1
            for mt in df_RUL.values:
                mapper_test[cnt]=mt[0]+temp_mapper[cnt]
                cnt += 1
            df_test['RUL'] = df_test['u'].apply(lambda nr: mapper_test[nr]) - df_test['t']
            s = 'FaultMode'+ str(i+1) +''
            fm[s] = {'df_train': df_train, 'df_test': df_test}
        feature_names.append('RUL')
                
        fm1_train = fm['FaultMode1']['df_train']
        fm1_train_target = fm1_train['RUL'].values
        fm1_test= fm['FaultMode1']['df_test']
        fm1_test_target = fm1_test['RUL'].values

        LSTM_train = fm1_train.drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])
        LSTM_test = fm1_test.drop(columns=['t', 'os_1', 'os_2', 'os_3', 's_01', 's_05', 's_06', 's_10', 's_16', 's_18', 's_19', 's_22', 's_23', 's_24', 's_25', 's_26'])
        train_units = set(LSTM_train['u'].values)
        test_units = set(LSTM_test['u'].values)
        sensors = ['s_02', 's_03', 's_04', 's_07', 's_08', 's_09', 's_11', 's_12',
                    's_13', 's_14', 's_15', 's_17', 's_20', 's_21']
        scalers = {}
        from sklearn.preprocessing import MinMaxScaler
        for column in sensors:
            scaler = MinMaxScaler(feature_range=(0.1,1))
            LSTM_train[column] = scaler.fit_transform(LSTM_train[column].values.reshape(-1,1))
            LSTM_test[column] = scaler.transform(LSTM_test[column].values.reshape(-1,1))
            scalers[column] = scaler
        window = 50
        temp_LSTM_x_train = []
        LSTM_y_train = []
        for unit in train_units:
            temp_unit = LSTM_train[LSTM_train['u']==unit].drop(columns=['u','RUL']).values
            temp_unit_RUL = LSTM_train[LSTM_train['u']==unit]['RUL'].values
            for i in range(len(temp_unit) - window + 1):#elekse edw an len temp_unit - window > 0
                temp_instance = []
                for j in range(window):
                    temp_instance.append(temp_unit[i+j])
                temp_LSTM_x_train.append(np.array(temp_instance))
                LSTM_y_train.append(temp_unit_RUL[i+window-1])
        LSTM_y_train = np.array(LSTM_y_train)
        LSTM_x_train = np.array(temp_LSTM_x_train)
        temp_LSTM_x_test = []
        LSTM_y_test = []
        for unit in test_units:
            temp_unit = LSTM_test[LSTM_test['u']==unit].drop(columns=['u','RUL']).values
            temp_unit_RUL = LSTM_test[LSTM_test['u']==unit]['RUL'].values
            for i in range(len(temp_unit) - window + 1):#elekse edw an len temp_unit - window > 0
                temp_instance = []
                for j in range(window):
                    temp_instance.append(temp_unit[i+j])
                temp_LSTM_x_test.append(np.array(temp_instance))
                LSTM_y_test.append(temp_unit_RUL[i+window-1])
        LSTM_y_test = np.array(LSTM_y_test)
        LSTM_x_test = np.array(temp_LSTM_x_test)
        return LSTM_x_train, LSTM_y_train, LSTM_x_test, LSTM_y_test, sensors

    def load_credit_approval(self, original_data_available=False, extra_clean=True):
        #Source: https://www.kaggle.com/rikdifos/credit-card-approval-prediction
        if original_data_available:
            from sklearn.base import TransformerMixin
            from sklearn.pipeline import Pipeline
            from imblearn.under_sampling import TomekLinks, NeighbourhoodCleaningRule, NearMiss,RandomUnderSampler
            from collections import Counter

            class DataFrameSelector(TransformerMixin):
                def __init__(self,attribute_names):
                    self.attribute_names = attribute_names
                def fit(self,X,y = None):
                    return self
                def transform(self,X):
                    return X[self.attribute_names]

            class dummies(TransformerMixin):
                def __init__(self,cols):
                    self.cols = cols
                
                def fit(self,X,y = None):
                    return self
                
                def transform(self,X):
                    df = pd.get_dummies(X)
                    df_new = df[df.columns.difference(cols)]
                    return df_new
                   
            import sys
            data = pd.read_csv(sys.path[-1]+'\\datasets\\application_record.csv', encoding = 'utf-8')
            record = pd.read_csv(sys.path[-1]+'\\datasets\\credit_record.csv', encoding = 'utf-8')

            #data = pd.read_csv("/kaggle/input/credit-card-approval-prediction/application_record.csv", encoding = 'utf-8')
            #record = pd.read_csv("/kaggle/input/credit-card-approval-prediction/credit_record.csv", encoding = 'utf-8')

            # find all users' account open month.
            begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
            begin_month=begin_month.rename(columns={'MONTHS_BALANCE':'months_balance'})
            new_data=pd.merge(data,begin_month,how="left",on="ID") #merge to record data

            record['dep_value'] = None
            record['dep_value'][record['STATUS'] =='2']='Yes'
            record['dep_value'][record['STATUS'] =='3']='Yes'
            record['dep_value'][record['STATUS'] =='4']='Yes'
            record['dep_value'][record['STATUS'] =='5']='Yes'

            cpunt=record.groupby('ID').count()
            cpunt['dep_value'][cpunt['dep_value'] > 0]='Yes'
            cpunt['dep_value'][cpunt['dep_value'] == 0]='No'
            cpunt = cpunt[['dep_value']]
            new_data=pd.merge(new_data,cpunt,how='inner',on='ID')
            new_data['target']=new_data['dep_value']
            new_data.loc[new_data['target']=='Yes','target']=1
            new_data.loc[new_data['target']=='No','target']=0

            new_data.drop(['OCCUPATION_TYPE', 'NAME_INCOME_TYPE', 'FLAG_OWN_REALTY',
                           'FLAG_PHONE', 'FLAG_EMAIL', 'FLAG_MOBIL', 'dep_value'], axis='columns', inplace=True)

            new_data.rename(columns={'CODE_GENDER':'gender',
                                     'FLAG_OWN_CAR':'car',
                                     'AMT_INCOME_TOTAL':'income',
                                     'NAME_EDUCATION_TYPE':'education',
                                     'NAME_FAMILY_STATUS':'family-status',
                                     'CNT_CHILDREN': '#-children',
                                     'NAME_HOUSING_TYPE':'house-type',
                                     'FLAG_WORK_PHONE':'work-phone',
                                     'CNT_FAM_MEMBERS':'family-size',
                                     'DAYS_BIRTH':'age',
                                     'DAYS_EMPLOYED':'years_employed',
                                    },inplace=True)

            def in_years(x):
                return int(abs(x)/365)
            
            def in_years_f(x):
                if x > 0:
                    return 0
                return abs(x)/365

            def abs_months(x):
                return abs(x)

            from scipy import stats
            if extra_clean:
                new_data['age'] = new_data['age'].apply(in_years)
                new_data['years_employed'] = new_data['years_employed'].apply(in_years_f)
                new_data['months_balance'] = new_data['months_balance'].apply(abs_months)

                new_data = new_data[(np.abs(stats.zscore(new_data['income'])) < 3)]
                new_data = new_data[(np.abs(stats.zscore(new_data['#-children'])) < 3)]

            new_data['gender'].replace('F' ,2, regex=True, inplace = True)
            new_data['gender'].replace('M' ,1, regex=True, inplace = True)
            new_data['car'].replace('Y' ,2, regex=True, inplace = True)
            new_data['car'].replace('N' ,1, regex=True, inplace = True)

            #new_data['family-status'].replace('Civil marriage' ,'married', regex=True, inplace = True)
            #new_data['family-status'].replace('Married' ,'married', regex=True, inplace = True)
            #new_data['family-status'].replace('Single / not married' ,'Single/not-married', regex=True, inplace = True)
            #new_data['family-status'].replace('Widow' ,'separated/widowed', regex=True, inplace = True)
            #new_data['family-status'].replace('Separated' ,'separated/widowed', regex=True, inplace = True)

            new_data['family-status'].replace('Civil marriage' ,2, regex=True, inplace = True)
            new_data['family-status'].replace('Married' ,2, regex=True, inplace = True)
            new_data['family-status'].replace('Single / not married' ,1, regex=True, inplace = True)
            new_data['family-status'].replace('Widow' ,3, regex=True, inplace = True)
            new_data['family-status'].replace('Separated' ,3, regex=True, inplace = True)

            new_data['education'].replace('Lower secondary', 1, regex=True, inplace = True)
            new_data['education'].replace('Secondary / secondary special', 2, regex=True, inplace = True)
            new_data['education'].replace('Incomplete higher', 3, regex=True, inplace = True)
            new_data['education'].replace('Higher education', 4, regex=True, inplace = True)
            new_data['education'].replace('Academic degree', 5, regex=True, inplace = True)

            new_data['house-type'].replace('Co-op apartment', 2, regex=True, inplace = True)
            new_data['house-type'].replace('Rented apartment', 2, regex=True, inplace = True)
            new_data['house-type'].replace('With parents', 1, regex=True, inplace = True)
            new_data['house-type'].replace('House / apartment', 3, regex=True, inplace = True)
            new_data['house-type'].replace('Municipal apartment', 3, regex=True, inplace = True)
            new_data['house-type'].replace('Office apartment', 3, regex=True, inplace = True)

            #for h in ['House / apartment', 'Municipal apartment', 'Office apartment']:
            #    new_data['house-type'].replace(h, 'apartment (house/municipal/office)', regex=True, inplace = True)

            #new_data = new_data.dropna()

            #cols = []
            #cat_col_new = ['family-status', 'house-type']
            #pipeline_cat=Pipeline([('selector',DataFrameSelector(cat_col_new)),
            #                       ('dummies',dummies(cols))])
            #cat_df = pipeline_cat.fit_transform(new_data)

            #new_data.drop(['family-status', 'house-type', 'ID'], axis='columns', inplace=True)
            new_data.drop(['ID'], axis='columns', inplace=True)

            #cat_df['id'] = pd.Series(range(cat_df.shape[0]))
            new_data['id'] = pd.Series(range(new_data.shape[0]))
            
            #final_df = pd.merge(cat_df,new_data,how = 'inner', on = 'id')
            final_df = new_data
            final_df.drop(['id'], axis='columns', inplace=True)
            print(f"Number of observations in final dataset: {final_df.shape}")

            y = final_df['target'].values
            y = [i for i in y]
            final_df.drop(labels = ['target'],axis = 1,inplace = True)
            X = final_df.values
            feature_names = final_df.columns

            print('Original dataset shape %s' % Counter(y))

            tl = TomekLinks()
            X_res, y_res = tl.fit_resample(X, y)
            print('TomekLinks: Resampled dataset shape %s' % Counter(y_res))
            ncr = NeighbourhoodCleaningRule()
            X_res, y_res = ncr.fit_resample(X_res, y_res)
            print('NC: Resampled dataset shape %s' % Counter(y_res))
            nm = NearMiss(version=3)
            X_res, y_res = nm.fit_resample(X_res, y_res)
            print('NM: Resampled dataset shape %s' % Counter(y_res))
            rus = RandomUnderSampler()
            X_res, y_res = rus.fit_resample(X_res, y_res)
            print('Random: Resampled dataset shape %s' % Counter(y_res))

            return X_res, X, y_res, y, feature_names
        else:
            import pickle
            import sys
            path = sys.path[-1]
            with open(path+'\\datasets\\credit_score.pkl', 'rb') as f:
                data = pickle.load(f)
            X_res = data[0]
            X = data[1]
            y_res = data[2]
            y = data[3]
            feature_names = data[4]
            return X_res, X, y_res, y, feature_names

    def load_hurricane(self):
        from keras.preprocessing.image import load_img
        from keras.preprocessing.image import img_to_array
        from pathlib import Path
        input_path = 'C:\\Users\\iamollas\\Desktop\\Altruist New\\datasets\\Huriccan Damage Estimation'
        image_df = pd.DataFrame({'path': list(Path(input_path).glob('**/*.jp*g'))})

        image_df['damage'] = image_df['path'].map(lambda x: x.parent.stem)
        image_df['data_split'] = image_df['path'].map(lambda x: x.parent.parent.stem)
        image_df['location'] = image_df['path'].map(lambda x: x.stem)
        image_df['lon'] = image_df['location'].map(lambda x: float(x.split('_')[0]))
        image_df['lat'] = image_df['location'].map(lambda x: float(x.split('_')[-1]))
        image_df['path'] = image_df['path'].map(lambda x: str(x)) # convert the path back to a string

        # get the train-validation-test splits
        image_df_train = image_df[image_df['data_split']=='train_another'].copy()
        image_df_val = image_df[image_df['data_split']=='validation_another'].copy()
        image_df_test = image_df[image_df['data_split']=='test_another'].copy()

        # sort to ensure reproducible behaviour
        image_df_train.sort_values('lat', inplace=True)
        image_df_val.sort_values('lat', inplace=True)
        image_df_test.sort_values('lat', inplace=True)
        image_df_train.reset_index(drop=True,inplace=True)
        image_df_val.reset_index(drop=True,inplace=True)
        image_df_test.reset_index(drop=True,inplace=True)

        # paths
        train_path = image_df_train['path'].copy().values
        val_path = image_df_val['path'].copy().values
        test_path = image_df_test['path'].copy().values

        # labels
        train_labels = np.zeros(len(image_df_train), dtype=np.int8)
        train_labels[image_df_train['damage'].values=='damage'] = 1

        val_labels = np.zeros(len(image_df_val), dtype=np.int8)
        val_labels[image_df_val['damage'].values=='damage'] = 1

        test_labels = np.zeros(len(image_df_test), dtype=np.int8)
        test_labels[image_df_test['damage'].values=='damage'] = 1

        train_images = []
        for i in train_path:
            image = load_img(i,(128, 128, 3))
            numpy_image = img_to_array(image)
            train_images.append(numpy_image)
        test_images = []
        for i in test_path:
            image = load_img(i,(128, 128, 3))
            numpy_image = img_to_array(image)
            test_images.append(numpy_image)
        val_images = []
        for i in val_path:
            image = load_img(i,(128, 128, 3))
            numpy_image = img_to_array(image)
            val_images.append(numpy_image)

        train_images = np.array(train_images)
        test_images = np.array(test_images)
        val_images = np.array(val_images)

        #augmentation
        
        augmented_images = []
        augmented_labels = []
        for i in range(len(train_labels)):
            image = train_images[i]
            label = train_labels[i]
            #Flips right-left when random gives 0 and up-down when 1
            if np.random.randint(2, size=1)[0] == 0:
                augmented_images.append(np.flip(image, np.random.randint(2, size=1)[0]))
                augmented_labels.append(label)
            else:
                augmented_images.append(np.rot90(image, np.random.randint(2, 5, size=1)[0]))
                augmented_labels.append(label)    
        [augmented_images.append(image) for image in train_images]
        [augmented_labels.append(label) for label in train_labels]
        train_images = np.array(augmented_images)
        train_labels = np.array(augmented_labels)

        augmented_images = []
        augmented_labels = []
        for i in range(len(val_labels)):
            image = val_images[i]
            label = val_labels[i]
            #Flips right-left when random gives 0 and up-down when 1
            if np.random.randint(2, size=1)[0] == 0:
                augmented_images.append(np.flip(image, np.random.randint(2, size=1)[0]))
                augmented_labels.append(label)
            else:
                augmented_images.append(np.rot90(image, np.random.randint(2, 5, size=1)[0]))
                augmented_labels.append(label)    
        [augmented_images.append(image) for image in val_images]
        [augmented_labels.append(label) for label in val_labels]
        val_images = np.array(augmented_images)
        val_labels = np.array(augmented_labels)
        
        return train_images, train_labels, test_images, test_labels, val_images, val_labels
        
