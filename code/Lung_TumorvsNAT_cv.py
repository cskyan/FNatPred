import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as ctb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import os
import numpy as np
from scipy import interpolate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc
pd.options.display.max_columns = 200

script_dir = os.path.dirname("/home/hdm/Fmodel/code")
# 构建文件的相对路径
file_name = "data/TumorvsNAT/features__weizmannSpeciesShared_Nonzero/"
file_path = os.path.join(script_dir, file_name)
train_datax = pd.read_csv(file_path + '/lung -- tumor vs nat -- Features.csv')
train_datay = pd.read_csv(file_path + '/lung -- tumor vs nat -- trainY.csv')
# train_datax = pd.read_csv('data/TumorvsNAT/features__weizmannSpeciesShared_Nonzero/breast -- tumor vs nat -- Features.csv')
# train_datay = pd.read_csv('data/TumorvsNAT/features__weizmannSpeciesShared_Nonzero/breast -- tumor vs nat -- trainY.csv')
# train_datax = pd.read_csv('data/features__weizmannSpeciesShared_Nonzero/lung -- tumor vs nat -- Features.csv')
# train_datay = pd.read_csv('data/features__weizmannSpeciesShared_Nonzero/lung -- tumor vs nat -- trainY.csv')
# train_datax = pd.read_csv('data/features__weizmannSpecies/breast -- tumor vs nat -- Features.csv')
# train_datay = pd.read_csv('data/features__weizmannSpecies/breast -- tumor vs nat -- trainY.csv')

# Initialize
fold_result_dfs = pd.DataFrame(columns=['model', 'auroc', 'aupr', 'rep', 'diseaseType', 'sampleType'])

train_data_y = train_datay['predY']
train_X = train_datax.iloc[:, 1:]
X_train = train_X
# train_data_X = train_X
original_index = train_data_y.index
train_data_y_encoded = [0 if label == "nat" else 1 for label in train_data_y]
# train_data_y_encoded = LabelEncoder().fit_transform(train_data_y)
train_data_y_encoded_series = pd.Series(train_data_y_encoded, index=original_index)
y_train = train_data_y_encoded_series
np.random.seed(42)
# X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y_encoded_series,
#                                                     test_size=0.2,
#                                                     stratify=train_data_y)


fig = plt.figure(figsize=(10, 7))

# 5 base models
print("---------------------------catboost----------------------------")
cat_reg = ctb.CatBoostRegressor(learning_rate=0.1,
                                depth=8,
                                random_seed=32,
                                )

print("----------------------------xgboost----------------------------")
xgb_reg = xgb.XGBRegressor(max_depth=8,
                           learning_rate=0.1,
                           n_estimators=13,
                           n_jobs=4,
                           colsample_bytree=0.8,
                           subsample=0.8,
                           random_state=32,)

print("---------------------------lightgbm----------------------------")
param_grid = {
    'n_estimators': [150],
    'max_depth': [3],
    'learning_rate': [0.1],
    'min_child_samples': [1]
}
lgb_reg = lgb.LGBMRegressor()
grid_search = GridSearchCV(estimator=lgb_reg, param_grid=param_grid, cv=5)

print("##--------------------------随机森林----------------------------------")
rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)

svm_reg = SVC(kernel='linear', C=1.0)

# calculate AUC values

oof_lgb = np.zeros(len(X_train))
oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_svm = np.zeros(len(X_train))
oof_rf = np.zeros(len(X_train))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
results_cv = []  # store model's results
for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(X_train), 1):

    train_x, train_y = X_train.iloc[train_idx], y_train.iloc[train_idx]
    valid_x, valid_y = X_train.iloc[valid_idx], y_train.iloc[valid_idx]

    grid_search.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
    xgb_reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
    cat_reg.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
    rf_reg.fit(train_x, train_y)
    # svm_reg.fit(train_x, train_y)

    pred_gbm = grid_search.predict(valid_x)
    pred_xgb = xgb_reg.predict(valid_x)
    pred_cat = cat_reg.predict(valid_x)
    pred_rf = rf_reg.predict(valid_x)
    # pred_svm = svm_reg.predict(valid_x)

    pred_gbm = np.where(pred_gbm >= 0.5, 1, 0)
    pred_xgb = np.where(pred_xgb >= 0.5, 1, 0)
    pred_cat = np.where(pred_cat >= 0.5, 1, 0)
    pred_rf = np.where(pred_cat >= 0.5, 1, 0)

    auc_score_gbm = roc_auc_score(valid_y, pred_gbm)
    auc_score_xgb = roc_auc_score(valid_y, pred_xgb)
    auc_score_cat = roc_auc_score(valid_y, pred_cat)
    auc_score_rf = roc_auc_score(valid_y, pred_rf)

    pr_value_gbm = average_precision_score(valid_y, pred_gbm)
    pr_value_xgb = average_precision_score(valid_y, pred_xgb)
    pr_value_cat = average_precision_score(valid_y, pred_cat)
    pr_value_rf = average_precision_score(valid_y, pred_rf)

    row_gbm = {
        'model': 'Gradient Boosting',
        'auroc': auc_score_gbm,
        'aupr': pr_value_gbm,
        'rep': f'Fold{fold_num}',
        'diseaseType': 'lung',
        'sampleType': 'tumor vs nat'
    }
    row_xgb = {
        'model': 'XGBoost',
        'auroc': auc_score_xgb,
        'aupr': pr_value_xgb,
        'rep': f'Fold{fold_num}',
        'diseaseType': 'lung',
        'sampleType': 'tumor vs nat'
    }
    row_cat = {
        'model': 'CatBoost',
        'auroc': auc_score_cat,
        'aupr': pr_value_cat,
        'rep': f'Fold{fold_num}',
        'diseaseType': 'lung',
        'sampleType': 'tumor vs nat'
    }
    row_rf = {
        'model': 'Random Forest',
        'auroc': auc_score_rf,
        'aupr': pr_value_rf,
        'rep': f'Fold{fold_num}',
        'diseaseType': 'lung',
        'sampleType': 'tumor vs nat'
    }
    results_cv.append(row_gbm)
    results_cv.append(row_xgb)
    results_cv.append(row_cat)
    results_cv.append(row_rf)

results_cv_df = pd.DataFrame(results_cv)
print(results_cv)



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_train, train_data_y_encoded_series, test_size=0.5,
                                                    stratify=train_data_y_encoded_series,
                                                    random_state=42)

# total_samples = X_train.shape[0]
# split_index = total_samples // 2
# X_train1 = X_train[:split_index]
# X_test1 = X_train[split_index:]
# y_train1 = train_data_y_encoded_series[:split_index]
# y_test1 = train_data_y_encoded_series[split_index:]

oof_lgb = np.zeros(len(X_train1))
oof_xgb = np.zeros(len(X_train1))
oof_cat = np.zeros(len(X_train1))
oof_svm = np.zeros(len(X_train1))
oof_rf = np.zeros(len(X_train1))
test_output_df = pd.DataFrame(columns=['lgb', 'xgb', 'cat', 'svm', 'rf'], index=range(X_test1.shape[0]))
test_output_df = test_output_df.fillna(0)

# stacking+5-fold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(X_train1), 1):

    train_x, train_y = X_train1.iloc[train_idx], y_train1.iloc[train_idx]
    valid_x, valid_y = X_train1.iloc[valid_idx], y_train1.iloc[valid_idx]

    print("Train Index:", train_idx)
    print("Validation Index:", valid_idx)
    print("---")
    grid_search.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], early_stopping_rounds=30)
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

print(test_output_df)

oof_df = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat, 'rf': oof_rf, 'svm': oof_svm})


# the second layer
x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(test_output_df, y_test1, test_size=0.2, stratify=y_test1, random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_data_train, y_data_train)
stacking_test_pred = rf_model.predict(x_data_test)
auc_values_fmodel = roc_auc_score(y_data_test, stacking_test_pred)
pr_value_fmodel = average_precision_score(y_data_test, stacking_test_pred)
print(classification_report(y_data_test, stacking_test_pred))
print("fmodel AUC Score:", auc_values_fmodel)
print("fmodel pr Score:", pr_value_fmodel)

# predictions_fmodel = rf_model.predict_proba(x_data_test)[:, 1]
# fpr_fmodel, tpr_fmodel, _ = roc_curve(y_data_test, predictions_fmodel)
# plt.plot(fpr_fmodel, tpr_fmodel, 'darkorange', label='fmodel = %0.2f' % auc_values_fmodel)
#
# model_names = ['xgb', 'catboost', 'gbm', 'rf', 'fmodel']
# roc_values = [auc_value_xgb, auc_value_cat, auc_value_gbm, auc_value_rf, auc_values_fmodel]
# pr_values = [pr_value_xgb, pr_value_cat, pr_value_gbm, pr_value_rf, pr_value_fmodel]
#
# roc_df = pd.DataFrame(columns=['Model', 'auroc', 'aupr'])
# for model, roc, pr in zip(model_names, roc_values, pr_values):
#     roc_df = roc_df.append({'Model': model, 'auroc': roc, 'aupr': pr}, ignore_index=True)
# print(roc_df)
#
# output_folder = "output"
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# print("Before saving roc_df to CSV")
# # output_file = os.path.join(output_folder, 'roc_values_breast_0.9.csv')
# # roc_df.to_csv(output_file, index=False)
# # print("after saving roc_df to CSV")
#
# print("--------------Drawing roc curve--------------")
# ###############################roc auc公共设置##################################
# plt.title('Lung ROC Validation')
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('Lung_multi_models_roc_0.9.png')
# plt.show()
#
#
print("----rf+ 5-fold validation")
for fold_num, (train_idx, valid_idx) in enumerate(kfold.split(test_output_df), 1):

    train_x_rf, train_y_rf = test_output_df.iloc[train_idx], y_test1.iloc[train_idx]
    valid_x_rf, valid_y_rf = test_output_df.iloc[valid_idx], y_test1.iloc[valid_idx]
    rf_model.fit(train_x_rf, train_y_rf)
    pred_rf = rf_model.predict(valid_x_rf)
    pred_rf = np.where(pred_rf >= 0.5, 1, 0)
    print("Fold", fold_num)
    print(classification_report(valid_y_rf,  pred_rf))
    auc_score = roc_auc_score(valid_y_rf, pred_rf)
    pr_value = average_precision_score(valid_y_rf, pred_rf)
    print("rf AUC Score:", auc_score)

    row_fmodel = {
        'model': 'Fmodel',
        'auroc': auc_score,
        'aupr': pr_value,
        'rep': f'Fold{fold_num}',
        'diseaseType': 'lung',
        'sampleType': 'tumor vs nat',
    }
    results_cv.append(row_fmodel)
results_cv_df = pd.DataFrame(results_cv)
print(results_cv_df)
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
print("Before saving roc_df to CSV")
output_file = os.path.join(output_folder, 'results_cv_Lung.csv')
results_cv_df.to_csv(output_file, index=False)
print("after saving roc_df to CSV")
