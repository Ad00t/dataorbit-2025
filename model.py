import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

ic_cols = {
    # "KIDSDRIV": {"dtype": "numeric"},
    "AGE": {"dtype": "numeric"},
    # "HOMEKIDS": {"dtype": "numeric"},
    # "YOJ": {"dtype": "numeric"},
    "INCOME": {"dtype": "money"},
    # "PARENT1": {"dtype": "categorical"},
    # "HOME_VAL": {"dtype": "money"},
    # "MSTATUS": {"dtype": "categorical"},
    # "GENDER": {"dtype": "categorical"},
    # "EDUCATION": {"dtype": "categorical"},
    # "OCCUPATION": {"dtype": "categorical"},
    "TRAVTIME": {"dtype": "numeric"},
    # "CAR_USE": {"dtype": "categorical"},
    "BLUEBOOK": {"dtype": "money"},
    # "TIF": {"dtype": "numeric"},
    "CAR_TYPE": {"dtype": "categorical"},
    # "RED_CAR": {"dtype": "categorical"},
    # "OLDCLAIM": {"dtype": "money"},
    # "CLM_FREQ": {"dtype": "numeric"},
    "REVOKED": {"dtype": "categorical"},
    "MVR_PTS": {"dtype": "numeric"},
    "CAR_AGE": {"dtype": "numeric"},
    "URBANICITY": {"dtype": "categorical"},
}

cm_cols = {
    "Duration": {"dtype": "numeric"},
    "Coverage": {"dtype": "categorical"},
    "SubReason": {"dtype": "categorical"},
    "Company": {"dtype": "categorical"},
}

def generate_embeddings(column: pd.Series):
    return { cat: i for i, cat in enumerate(column.unique()) }

def money_to_numeric(column: pd.Series):
    return column.str.replace(r"\D", "", regex=True)

class Model:

    def __init__(self, initial_claims_path="data/initial_claims.csv", complaints_path="data/complaints.csv") -> None:
        self.ic_train_test_df = pd.read_csv(initial_claims_path)
        self.ic_embeddings = {}
        self.cm_train_test_df = pd.read_csv(complaints_path)
        self.cm_embeddings = {}

    def preprocess(self, is_pred: bool, ic_df: pd.DataFrame, cm_df: pd.DataFrame = None) -> tuple:
        # Preprocess initial claims df
        ic_df = ic_df.dropna()
        for col_name, properties in ic_cols.items():
            if properties["dtype"] == "money":
                ic_df.loc[:, col_name] = pd.to_numeric(money_to_numeric(ic_df[col_name]), errors="coerce")
            elif properties["dtype"] == "categorical":
                if not is_pred:
                    self.ic_embeddings[col_name] = generate_embeddings(ic_df[col_name])
                ic_df.loc[:, col_name] = ic_df[col_name].map(self.ic_embeddings[col_name])

        # Preprocess complaints df
        if (type(cm_df) != type(None)):
            cm_df = cm_df.fillna("<None>")
            if not is_pred:
                cm_df["Opened"] = pd.to_datetime(cm_df["Opened"].replace("<None>", pd.NaT))
                cm_df["Closed"] = pd.to_datetime(self.cm_train_test_df["Closed"].replace("<None>", pd.NaT))
                cm_df["Duration"] = (cm_df["Closed"] - cm_df["Opened"]).dt.total_seconds() / (60 * 60 * 24)
            # self.cm_df = pd.get_dummies(self.cm_df, columns=["Coverage", "SubReason", "Company"]) # SubReason is a very important variable for the accuracy score
            for cat_col_name in [ "Coverage", "SubReason", "Company" ]:
                if not is_pred:
                    self.cm_embeddings[cat_col_name] = generate_embeddings(cm_df[cat_col_name])
                cm_df.loc[:, cat_col_name] = cm_df[cat_col_name].map(self.cm_embeddings[cat_col_name])
            if not is_pred:
                cm_df["Approved"] = (cm_df["Recovery"] > 0).map({ True: 1, False: 0 })
            return (ic_df, cm_df)
        else:
            return (ic_df, None)

    def train_test(self):
        self.unique_subreasons = self.cm_train_test_df["SubReason"].unique()
        self.ic_train_test_df, self.cm_train_test_df = self.preprocess(False, self.ic_train_test_df, self.cm_train_test_df)

        # Train & test initial complaints model
        X_ic = self.ic_train_test_df[sorted(list(ic_cols.keys()))]
        Y_ic = self.ic_train_test_df["CLAIM_FLAG"]
        X_train_ic, X_test_ic, Y_train_ic, Y_test_ic = train_test_split(X_ic, Y_ic, test_size=0.2, random_state=42)

        self.ic_model = LogisticRegression(max_iter=10000)
        self.ic_model.fit(X_train_ic, Y_train_ic)

        Y_pred_ic = self.ic_model.predict(X_test_ic)
        accuracy_ic = accuracy_score(Y_test_ic, Y_pred_ic)
        report_ic = classification_report(Y_test_ic, Y_pred_ic)
        print(f"Initial Claims Model Accuracy: {accuracy_ic:.2f}")
        print("Classification Report:\n", report_ic)

        # Train & test complaints model
        X_cm = self.cm_train_test_df[sorted(list(cm_cols.keys()))]
        Y_cm = self.cm_train_test_df["Approved"]
        X_train_cm, X_test_cm, Y_train_cm, Y_test_cm = train_test_split(X_cm, Y_cm, test_size=0.2, random_state=1)

        self.cm_model = RandomForestClassifier(n_estimators=100, random_state=1)
        self.cm_model.fit(X_train_cm, Y_train_cm)

        Y_pred_cm = self.cm_model.predict(X_test_cm)
        accuracy_cm = accuracy_score(Y_test_cm, Y_pred_cm)
        report_cm = classification_report(Y_test_ic, Y_pred_ic)
        print(f"Complaints Model Accuracy: {accuracy_cm:.2f}")
        print("Classification Report:\n", report_cm)

    def predict(self, api_submission: dict):
        print(api_submission)
        X_pred_ic_raw = pd.DataFrame({ key: [api_submission[key]] for key in api_submission if key in list(ic_cols.keys()) })
        X_pred_cm_raw = pd.DataFrame({ key: [api_submission[key]] for key in api_submission if key in list(cm_cols.keys()) })

        if api_submission['complaint_included']:
            X_pred_ic, X_pred_cm = self.preprocess(True, X_pred_ic_raw, X_pred_cm_raw)
            X_pred_ic = X_pred_ic[sorted(X_pred_ic.columns)]
            X_pred_cm = X_pred_cm[sorted(X_pred_cm.columns)]
            
            Y_pred_ic = self.ic_model.predict_proba(X_pred_ic)[0, 1]
            Y_pred_cm = self.cm_model.predict_proba(X_pred_cm)[0, 1]
            return {
                "p_initial_claim_approved": Y_pred_ic,
                "p_complaint_approved": Y_pred_cm,
                "p_final_approved": Y_pred_ic + (1 - Y_pred_ic) * Y_pred_cm
            }
        else:
            try:
                n_sr = len(self.unique_subreasons)
                X_pred_cm_raw = X_pred_cm_raw.loc[X_pred_cm_raw.index.repeat(n_sr)].reset_index(drop=True)
                X_pred_cm_raw['Coverage'] = np.repeat(['A & H'], n_sr)
                X_pred_cm_raw['SubReason'] = np.tile(self.unique_subreasons, 1)
                X_pred_ic, X_pred_cm = self.preprocess(True, X_pred_ic_raw, X_pred_cm_raw)
                X_pred_ic = X_pred_ic[sorted(X_pred_ic.columns)]
                X_pred_cm = X_pred_cm[sorted(X_pred_cm.columns)]

                Y_pred_ic = self.ic_model.predict_proba(X_pred_ic)[0, 1]
                Y_pred_cm = self.cm_model.predict_proba(X_pred_cm)[:, 1].mean()

                return {
                    "p_initial_claim_approved": Y_pred_ic,
                    "p_complaint_approved": Y_pred_cm,
                    "p_final_approved": Y_pred_ic + (1 - Y_pred_ic) * Y_pred_cm
                }
            except Exception as e:
                print(e)


