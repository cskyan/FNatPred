import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import os
import re
import pandas as pd
pd.options.display.max_columns = 200

script_dir = os.path.dirname("/home/hdm/Fmodel/code")

## Import data
# folder_path = 'data/TumorvsNormal/TumorvsNormal__df_psRep200_HiSeq_Fungi_DecontamV2_BCM_species'
# folder_path = 'data/TumorvsNormal/TumorvsNormal__df_psRep200_HiSeq_Fungi_DecontamV2_Broad_WGS_species'
# folder_path = 'data/TumorvsNormal/TumorvsNormal__df_psRep200_HiSeq_Fungi_DecontamV2_CMS_species'
# folder_path = 'data/TumorvsNormal/TumorvsNormal__df_psRep200_HiSeq_Fungi_DecontamV2_UNC_species'
folder_path = 'data/TumorvsNormal/TumorvsNormal__df_psRep200_HiSeq_Fungi_DecontamV2_HMS_species'
folder_path = os.path.join(script_dir, folder_path)

# Initialize an array of stored file names
file_names = []
dataset = 'HMS'
train_datay = None
train_datax = None
roc_df = pd.DataFrame(columns=['Model', 'auroc', 'aupr', 'diseaseType', 'datasetName'])
results_cv = []  # store model's results

# Iterate through all the files in the folder
for i, filename in enumerate(sorted(os.listdir(folder_path))):
    if os.path.isfile(os.path.join(folder_path, filename)) and 'Features' in filename:

        name_parts = filename.split('--')
        if len(name_parts) > 0:
            extracted_name = name_parts[0]
            file_names.append(extracted_name)
        file_path = os.path.join(folder_path, filename)
        train_datax = pd.read_csv(file_path)
        print(f"read file：{file_path}")
    elif os.path.isfile(os.path.join(folder_path, filename)) and 'trainY' in filename:
        file_path = os.path.join(folder_path, filename)
        train_datay = pd.read_csv(file_path)
        print(f"read file：{file_path}")
    print(file_names)
    if train_datay is not None and train_datax is not None:
        print(file_names[i // 2])
        # print(train_datay)
        train_data_y = train_datay['predY']
        train_X = train_datax.iloc[:, 1:]
        train_data_X = train_X
        original_index = train_data_y.index
        train_data_y_encoded = [0 if label == "PrimaryTumor" else 1 for label in train_data_y]
        train_data_y_encoded_series = pd.Series(train_data_y_encoded, index=original_index)
        np.random.seed(42)
        X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y_encoded_series,
                                                            test_size=0.2,
                                                            stratify=train_data_y)
        print("----------------------------xgboost----------------------------")
        xgb_reg = xgb.XGBRegressor(max_depth=8,
                                   learning_rate=0.1,
                                   n_estimators=300,
                                   n_jobs=4,
                                   colsample_bytree=0.8,
                                   subsample=0.8,
                                   random_state=32,
                                   )
        xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
                    verbose=5, eval_metric='auc')
        xgb_predict = xgb_reg.predict(X_test)



        y_pred = xgb_reg.predict(X_test)
        xgb_classes = y_pred.round().astype(int)
        y_true = y_test
        xgb_classes = np.where(xgb_classes >= 0.5, 1, 0)
        classification_rep = classification_report(y_true, xgb_classes)

        auc_value_xgb = roc_auc_score(y_true, xgb_classes)
        pr_value_xgb = average_precision_score(y_test, xgb_classes)
        print(classification_rep)
        print("xgb's AUC Value:", auc_value_xgb)
        # 5_flod
        print("---------------------------catboost----------------------------")
        cat_reg = ctb.CatBoostRegressor(learning_rate=0.1,
                                        depth=8,
                                        random_seed=32,
                                        )
        cat_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
                    verbose=5)
        cat_predict = cat_reg.predict(X_test)

        cat_classes = cat_predict.round().astype(int)
        classification_rep = classification_report(y_test, cat_classes)
        print(classification_rep)
        auc_value_cat = roc_auc_score(y_test, cat_classes)
        pr_value_cat = average_precision_score(y_test, cat_classes)
        print("cat AUC Score:", auc_value_cat)
        print("---------------------------lightgbm----------------------------")
        param_grid = {
            'n_estimators': [150],
            'max_depth': [3],
            'learning_rate': [0.1],
            'min_child_samples': [1]
        }
        lgb_reg = lgb.LGBMRegressor()
        grid_search = GridSearchCV(estimator=lgb_reg, param_grid=param_grid, cv=5)
        grid_search.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
                        verbose=5, eval_metric='auc')
        lgb_predict = grid_search.predict(X_test)

        lgb_classes = lgb_predict.round().astype(int)
        classification_rep = classification_report(y_test, lgb_classes)
        print(classification_rep)
        auc_value_gbm = roc_auc_score(y_test, lgb_classes)
        pr_value_gbm = average_precision_score(y_test, lgb_classes)
        print("gbm AUC Score:", auc_value_gbm)
        print("----------------------------random forest----------------------------------")
        rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_train)
        rf_predict = rf_reg.predict(X_test)
        auc_value_rf = roc_auc_score(y_test, rf_predict)
        pr_value_rf = average_precision_score(y_test, rf_predict)
        print(classification_report(y_test, rf_predict))
        print("rf AUC Score:", auc_value_rf)

        ## create our model: Fmodel
        print("--------------------------Fmodel------------------------------------")

        X_train1, X_test1, y_train1, y_test1 = train_test_split(train_data_X,
                                                                train_data_y_encoded_series,
                                                                test_size=0.5,
                                                                stratify=train_data_y_encoded_series,
                                                                random_state=42)


        oof_lgb = np.zeros(len(X_train1))
        oof_xgb = np.zeros(len(X_train1))
        oof_cat = np.zeros(len(X_train1))
        oof_svm = np.zeros(len(X_train1))
        oof_rf = np.zeros(len(X_train1))
        test_output_df = pd.DataFrame()
        print("初始化test_output_df", test_output_df)
        test_output_df = pd.DataFrame(columns=['lgb', 'xgb', 'cat', 'svm', 'rf'], index=range(X_test1.shape[0]))
        test_output_df = test_output_df.fillna(0)
        print("初始化test_output_df", test_output_df)
        svm_reg = SVC(kernel='linear', C=1.0)

        # stacking+5-fold
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(X_train1), 1):
            train_x, train_y = X_train1.iloc[train_idx], y_train1.iloc[train_idx]
            valid_x, valid_y = X_train1.iloc[valid_idx], y_train1.iloc[valid_idx]
            # train_x = X_train[train_idx]
            # train_y = y_train[train_idx]
#             print("Train Index:", train_idx)
#             print("Validation Index:", valid_idx)
#             print("valid_y:", valid_y)
            print("---")
            grid_search.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],early_stopping_rounds=30)
            xgb_reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
            cat_reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
            rf_reg.fit(train_x, train_y)
            svm_reg.fit(train_x, train_y)
            oof_lgb[valid_idx] = grid_search.predict(valid_x)
            oof_xgb[valid_idx] = xgb_reg.predict(valid_x)
            oof_cat[valid_idx] = cat_reg.predict(valid_x)
            oof_rf[valid_idx] = rf_reg.predict(valid_x)
            oof_svm[valid_idx] = svm_reg.predict(valid_x)

            test_output_df['lgb'] += grid_search.predict(X_test1)
            test_output_df['xgb'] += xgb_reg.predict(X_test1)
            test_output_df['cat'] += cat_reg.predict(X_test1)
            test_output_df['rf'] += rf_reg.predict(X_test1)
            test_output_df['svm'] += svm_reg.predict(X_test1)

        test_output_df['lgb'] = test_output_df['lgb'] / 5
        test_output_df['xgb'] = test_output_df['xgb'] / 5
        test_output_df['cat'] = test_output_df['cat'] / 5
        test_output_df['rf'] = test_output_df['rf'] / 5
        test_output_df['svm'] = test_output_df['svm'] / 5
        print("cv得到的test_output_df:", test_output_df)

        oof_df = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat, 'rf': oof_rf, 'svm': oof_svm})
        # the second layer
        x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(test_output_df, y_test1, test_size=0.2,
                                                                                stratify=y_test1)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(x_data_train, y_data_train)
        stacking_test_pred = rf_model.predict(x_data_test)
        auc_value_fmodel = roc_auc_score(y_data_test, stacking_test_pred)
        pr_value_fmodel = average_precision_score(y_data_test, stacking_test_pred)
        print(classification_report(y_data_test, stacking_test_pred))
        print("fmodel AUC Score:", auc_value_fmodel)

        print("--------------------------save------------------------------------")
        model_names = ['XGBoost', 'CatBoost', 'LightGBM[1]', 'Random Forest[2]', 'Fmodel']
        roc_values = [auc_value_xgb, auc_value_cat, auc_value_gbm, auc_value_rf, auc_value_fmodel]
        pr_values = [pr_value_xgb, pr_value_cat, pr_value_gbm, pr_value_rf, pr_value_fmodel]
        disease = [file_names[i//2]] * len(model_names)
        datasetList = [dataset] * len(model_names)

        # save
        for model, roc, pr, diseaseType, dataSet in zip(model_names, roc_values, pr_values, disease, datasetList):
            roc_df = roc_df.append({'Model': model, 'auroc': roc, 'aupr': pr, 'diseaseType': diseaseType, 'datasetName': dataSet},
                                   ignore_index=True)
        print(roc_df)

        train_datay = None
        train_datax = None

output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Before saving roc_df to CSV")
output_file1 = os.path.join(output_folder, 'TumorvsNormal_FmodelvsOthers_value_HMS.csv')
roc_df.to_csv(output_file1, index=False)

print("after saving roc_df to CSV")