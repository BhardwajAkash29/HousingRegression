from utils import load_data, split_data, evaluate_model, save_model
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, r2 = evaluate_model(y_test, y_pred)
    print(f"{name}: MSE = {mse:.2f}, R² = {r2:.2f}")
    save_model(model, f"{name}.joblib")
