import sys
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

# Reading datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit = pd.read_csv("Sample_Submission_Tm9Lura.csv")

# Saving id variables to create final submission
ids_test = test['User_ID'].copy()
product_ids_test = test['Product_ID'].copy()

# Reducing boundaries to decrease RMSE
cutoff_purchase = np.percentile(train['Purchase'], 99.9)  # 99.9 percentile
train.ix[train['Purchase'] > cutoff_purchase, 'Purchase'] = cutoff_purchase

# Label Encoding User_IDs
le = LabelEncoder()
train['User_ID'] = le.fit_transform(train['User_ID'])
test['User_ID'] = le.transform(test['User_ID'])

# Label Encoding Product_IDs
new_product_ids = list(set(pd.unique(test['Product_ID'])) - set(pd.unique(train['Product_ID'])))

le = LabelEncoder()
train['Product_ID'] = le.fit_transform(train['Product_ID'])
test.ix[test['Product_ID'].isin(new_product_ids), 'Product_ID'] = -1
new_product_ids.append(-1)

test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'] = le.transform(test.ix[~test['Product_ID'].isin(new_product_ids), 'Product_ID'])

# NOTES:
# 3631 unique values in train - Product_ID
# 3491 unique values in test - Product_ID
# 46 new product ids in test set and 186 products absent in test
# Only Product_Category_2 and Product_Category_3 have missing values.

y = train['Purchase']
train.drop(['Purchase', 'Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)
test.drop(['Product_Category_2', 'Product_Category_3'], inplace=True, axis=1)

train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Modeling
dtrain = xgb.DMatrix(train.values, label=y, missing=np.nan)

param = {'objective': 'reg:linear', 'booster': 'gbtree', 'silent': 1,
		 'max_depth': 10, 'eta': 0.1, 'nthread': 4,
		 'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 20,
		 'max_delta_step': 0, 'gamma': 0}
num_round = 690

# xgb.cv(param, dtrain, num_round, nfold=4, seed=2244, show_progress=True)
# exit()
# [690]   cv-test-rmse:2487.3809205+9.82125332763 - 10 690 and 20 - produt category 2,3 removed - v3

seeds = [1122, 2244, 3366, 4488, 5500]
test_preds = np.zeros((len(test), len(seeds)))

for run in range(len(seeds)):
	sys.stdout.write("\rXGB RUN:{}/{}".format(run+1, len(seeds)))
	sys.stdout.flush()
	param['seed'] = seeds[run]
	clf = xgb.train(param, dtrain, num_round)
	dtest = xgb.DMatrix(test.values, missing=np.nan)
	test_preds[:, run] = clf.predict(dtest)

test_preds = np.mean(test_preds, axis=1)

# Submission file
submit = pd.DataFrame({'User_ID': ids_test, 'Product_ID': product_ids_test, 'Purchase': test_preds})
submit = submit[['User_ID', 'Product_ID', 'Purchase']]

submit.ix[submit['Purchase'] < 0, 'Purchase'] = 12  # changing min prediction to min value in train
submit.to_csv("final_solution.csv", index=False)
