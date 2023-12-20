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
from scipy.interpolate import interp1d
from scipy import interpolate
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, auc, precision_recall_curve
pd.options.display.max_columns = 200

script_dir = os.path.dirname("/home/hdm/Fmodel/code")

file_name = "data/TumorvsNAT/features__weizmannSpeciesShared_Nonzero/"
file_path = os.path.join(script_dir, file_name)
train_datax = pd.read_csv(file_path + '/lung -- tumor vs nat -- Features.csv')
train_datay = pd.read_csv(file_path + '/lung -- tumor vs nat -- trainY.csv')


train_data_y = train_datay['predY']
train_X = train_datax.iloc[:, 1:]
train_data_X = train_X
original_index = train_data_y.index
train_data_y_encoded = [0 if label == "nat" else 1 for label in train_data_y]
# train_data_y_encoded = LabelEncoder().fit_transform(train_data_y)
train_data_y_encoded_series = pd.Series(train_data_y_encoded, index=original_index)
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y_encoded_series,
                                                    test_size=0.2,
                                                    stratify=train_data_y)


fig = plt.figure(figsize=(10, 7))


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
print("cat pr Score:", pr_value_cat)

precision_cat, recall_cat, _ = precision_recall_curve(y_test, cat_classes)
plt.plot(recall_cat, precision_cat, lw=2, alpha=0.5, color='g', label='CatBoost (AUPR={:.2f})'.format(pr_value_cat))

# predictions_cat = cat_reg.predict(X_test)
# fpr_cat, tpr_cat, _ = roc_curve(y_test, predictions_cat,)
# plt.plot(fpr_cat, tpr_cat, lw=2, alpha=0.5, color='g', label='CatBoost (AUROC={:.2f})'.format(auc_value_cat))

print("----------------------------xgboost----------------------------")
xgb_reg = xgb.XGBRegressor(max_depth=8,
                           learning_rate=0.1,
                           n_estimators=13,
                           n_jobs=4,
                           colsample_bytree=0.8,
                           subsample=0.8,
                           random_state=32,)
xgb_reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], early_stopping_rounds=30,
                    verbose=5, eval_metric='auc')
xgb_predict = xgb_reg.predict(X_test)
xgb_classes = xgb_predict.round().astype(int)
classification_rep = classification_report(y_test, xgb_classes)
auc_value_xgb = roc_auc_score(y_test, xgb_classes)
pr_value_xgb = average_precision_score(y_test, xgb_classes)
print(classification_rep)
print("xgb's AUC Value:", auc_value_xgb)
print("xgb's pr Value:", pr_value_xgb)
# predictions_xgb = xgb_reg.predict(X_test)
# fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb)
# plt.plot(fpr_xgb, tpr_xgb, lw=2, color='r', label='XGBoost (AUROC={:.2f})'.format(auc_value_xgb))

precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_classes)
plt.plot(recall_xgb, precision_xgb, lw=2, color='r', label='XGBoost (AUPR={:.2f})'.format(pr_value_xgb))

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
print("gbm pr Score:", pr_value_gbm)


precision_gbm, recall_gbm, _ = precision_recall_curve(y_test, lgb_classes)
plt.plot(recall_gbm, precision_gbm, lw=1, color='b', label='LightGBM[1](AUPR={:.2f})'.format(pr_value_gbm))


# predictions_gbm = grid_search.predict(X_test)
# fpr_gbm, tpr_gbm, _ = roc_curve(y_test, predictions_gbm)
# plt.plot(fpr_gbm, tpr_gbm, lw=1, color='b', label='LightGBM[1] (AUROC={:.2f})'.format(auc_value_gbm))


print("##--------------------------随机森林----------------------------------")
rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rf_predict = rf_reg.predict(X_test)
auc_value_rf = roc_auc_score(y_test, rf_predict)
pr_value_rf = average_precision_score(y_test, rf_predict)
print(classification_report(y_test, rf_predict))
print("rf AUC Score:", auc_value_rf)
print("rf pr Score:", pr_value_rf)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_predict)
plt.plot(recall_rf, precision_rf, lw=1, color='c', label='RandomForest[2] (AUPR={:.2f})'.format(pr_value_rf))


# predictions_rf = rf_reg.predict_proba(X_test)[:, 1]
# fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_rf)
# plt.plot(fpr_rf, tpr_rf, lw=1, color='c', label='Random forest[2] (AUROC={:.2f})'.format(auc_value_rf))

svm_reg = SVC(kernel='linear', C=1.0)


X_train1, X_test1, y_train1, y_test1 = train_test_split(train_data_X, train_data_y_encoded_series, test_size=0.5,
                                                        stratify=train_data_y_encoded_series,
                                                        random_state=42)


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

precision_fmodel, recall_fmodel, _ = precision_recall_curve(y_data_test, stacking_test_pred)
plt.plot(recall_fmodel, precision_fmodel, lw=2, color='m', label='Fmodel (AUPR={:.2f})'.format(pr_value_fmodel))

# predictions_fmodel = rf_model.predict_proba(x_data_test)[:, 1]
# fpr_fmodel, tpr_fmodel, _ = roc_curve(y_data_test, predictions_fmodel)
# plt.plot(fpr_fmodel, tpr_fmodel, lw=2, color='m', label='Fmodel (AUROC={:.2f})'.format(auc_values_fmodel))



# print("--------------Drawing roc curve--------------")
# ###############################roc auc公共设置##################################
# plt.title('Tumor vs NAT | Species Level | Lung')
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('Lung_multi_models_roc.png')
# plt.show()

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve |Tumor vs NAT | Species Level | Lung')
plt.plot([0, 1], [1, 0], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('Lung_multi_models_pr.png')
plt.show()

