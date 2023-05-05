import optuna.integration.lightgbm as tlgb
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# from pyfm import pylibfm
# from scipy.sparse import csr_matrix


def run_lgbm(
    cfg, X_train, y_train, X_test, y_test, categorical_cols=[], use_optuna=False
):
    y_preds = []
    models = []
    oof_train = np.zeros((len(X_train),))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["seed"])

    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting": "gbdt",
        "learning_rate": 0.1,
        "verbosity": -1,
        "random_state": cfg["seed"],
    }

    for fold_id, (train_index, valid_index) in enumerate(cv.split(X_train, y_train)):
        X_tr = X_train[train_index, :]
        X_val = X_train[valid_index, :]
        y_tr = y_train[train_index]
        y_val = y_train[valid_index]

        if use_optuna:
            lgb_train = tlgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

            lgb_eval = tlgb.Dataset(
                X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
            )

            model = tlgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(period=100),
                ],
            )

        else:
            lgb_train = lgb.Dataset(X_tr, y_tr, categorical_feature=categorical_cols)

            lgb_eval = lgb.Dataset(
                X_val, y_val, reference=lgb_train, categorical_feature=categorical_cols
            )

            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_train, lgb_eval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(period=100),
                ],
            )

        oof_train[valid_index] = model.predict(
            X_val, num_iteration=model.best_iteration
        )

        models.append(model)

    # testデータでの性能を評価
    y_pred = np.mean([model.predict(X_test) for model in models], axis=0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return rmse


# def fm_fit_predict(cfg, X_train, y_train, X_test, y_test):
#     # csr_matrixに変換
#     X_train = csr_matrix(X_train, dtype=np.float64)
#     X_test = csr_matrix(X_test, dtype=np.float64)

#     fm = pylibfm.FM(
#         num_factors=20,
#         num_iter=100,
#         task="regression",
#         initial_learning_rate=0.001,
#         learning_rate_schedule="optimal",
#         verbose=False,
#     )
#     fm.fit(X_train, y_train)

#     # testデータでの性能を評価
#     y_pred = fm.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))

#     return rmse
