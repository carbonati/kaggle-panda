from sklearn.metrics import cohen_kappa_score


def compute_qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_provider_scores(df_pred, target_col='isup_grade', pred_col='prediction'):
    provider_scores = df_pred.groupby('data_provider').apply(
        lambda x: cohen_kappa_score(x[target_col], x[pred_col], weights='quadratic')
    )
    return provider_scores
