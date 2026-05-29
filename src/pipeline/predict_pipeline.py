import joblib
import pandas as pd


class PredictPipeline:
    def __init__(self):
        """
        Load all required
        inference artifacts.
        """

        self.model = joblib.load("../artifacts/churn/models/xgb_model.pkl")

        self.scaler = joblib.load("../artifacts/churn/scalers/scaler.pkl")

        self.feature_columns = joblib.load(
            "../artifacts/churn/features/feature_columns.pkl"
        )

    def predict(
        self,
        input_data: dict,
    ):
        """
        Predict churn probability
        for a new customer.
        """

        # CONVERT TO DATAFRAME

        input_df = pd.DataFrame([input_data])

        # FEATURE ALIGNMENT

        input_df = input_df[self.feature_columns]

        # SCALING

        scaled_data = self.scaler.transform(input_df)

        # PREDICTION

        prediction = self.model.predict(scaled_data)[0]

        probability = self.model.predict_proba(scaled_data)[0][1]

        return {
            "Prediction": int(prediction),
            "Churn_probability": float(
                round(
                    probability,
                    4,
                )
            ),
        }
