# Dependencies

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class HotelBookingModelTrainer:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_and_prep_data(self):
        """Load data, drop fitur yg irrelevan, dan split fitur/target"""
        data = pd.read_csv(self.data_path)
        data = data.drop(['Booking_ID', 'arrival_date'], axis=1)  # Drop unique identifiers

        target = data.columns[len(data.columns)-1] #column terakhit
        self.X = data.drop(target, axis=1)
        self.y = self.label_encoder.fit_transform(data[target]) #langsung encode

        return self.X, self.y
    
    def preprocess(self):
        """Missing value dan apply feature processing"""
        num_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_cols = self.X.select_dtypes(include=['object']).columns.tolist()

        self.X[num_cols] = self.X[num_cols].fillna(self.X[num_cols].median()) #Pembeda hasil oop ipynb dan oop py
        self.X[cat_cols] = self.X[cat_cols].fillna(self.X[cat_cols].mode().iloc[0])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
            ]
        )
        
        # Apply preprocessing
        self.X_processed = self.preprocessor.fit_transform(self.X)

        return self.X_processed
    
    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y, 
            test_size=self.test_size, 
            random_state=self.random_state
        )

        models = {
            'RandomForest': RandomForestClassifier(random_state=self.random_state),
            'XGBoost': xgb.XGBClassifier(random_state=self.random_state)
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'report': report
            }

            print(f"\n{name} Performance:")
            print(f"Accuracy: {accuracy:.4f}")
            print("Classification Report:\n", report)

        # Best model
        self.best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest Model: {self.best_model[0]} (Accuracy: {self.best_model[1]['accuracy']:.4f})")     
        return self.best_model

    # def save_pipeline(self, output_path='hotel_booking_pipeline.joblib'):
    #     """Save the preprocessing pipeline and best model for future use."""
    #     pipeline = {
    #         'preprocessor': self.preprocessor,
    #         'model': self.best_model[1]['model'],
    #         'label_encoder': self.label_encoder
    #     }
        
    #     joblib.dump(pipeline, output_path)
    #     print(f"\nPipeline saved to {output_path}")

if __name__ == "__main__":
    model_trainer = HotelBookingModelTrainer(data_path="Dataset_B_hotel.csv")

    # Tahap Load dan data Prep
    X, y = model_trainer.load_and_prep_data()
    
    # Tahap Preprocess
    X_preprocessed = model_trainer.preprocess()

    # Tahap train model dan best model
    best_model = model_trainer.train_model()

    # Simpan pipeline untuk penggunaan kedepanya
    # model_trainer.save_pipeline()