from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        income = float(request.form['Income'])
        recency = float(request.form['Recency'])
        mnt_wines = float(request.form['MntWines'])
        mnt_fruits = float(request.form['MntFruits'])
        mnt_meat_products = float(request.form['MntMeatProducts'])
        mnt_fish_products = float(request.form['MntFishProducts'])
        mnt_sweet_products = float(request.form['MntSweetProducts'])
        mnt_gold_prods = float(request.form['MntGoldProds'])
        num_deals_purchases = float(request.form['NumDealsPurchases'])
        num_web_purchases = float(request.form['NumWebPurchases'])
        num_catalog_purchases = float(request.form['NumCatalogPurchases'])
        num_store_purchases = float(request.form['NumStorePurchases'])
        num_web_visits_month = float(request.form['NumWebVisitsMonth'])
        age = float(request.form['Age'])

        features = np.array([[income, recency, mnt_wines, mnt_fruits, mnt_meat_products, mnt_fish_products,
                              mnt_sweet_products, mnt_gold_prods, num_deals_purchases, num_web_purchases,
                              num_catalog_purchases, num_store_purchases, num_web_visits_month, age]])

        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        cluster_descriptions = {
            0: 'Cluster 0 :Income: Low, Recency: High (customers recently purchased), MntWines: Low to Medium, MntFruits: Low, MntMeatProducts: Low, MntFishProducts: Low, MntSweetProducts: Low, MntGoldProds: Low, NumDealsPurchases: High, NumWebPurchases: Low, NumCatalogPurchases: Low, NumStorePurchases: High, NumWebVisitsMonth: High, Age: Young',
            1: 'Cluster 1:Income:, High, Recency: Low (customers have not purchased recently), MntWines: High, MntFruits: High, MntMeatProducts: High, MntFishProducts: High, MntSweetProducts: High, MntGoldProds: High,   NumDealsPurchases: Low, NumWebPurchases: High, NumCatalogPurchases: High, NumStorePurchases: Low, NumWebVisitsMonth: Low, Age: Older',
            2: 'Cluster 2: Income: Medium, Recency: Medium, MntWines: Medium, MntFruits: Medium, MntMeatProducts: Medium, MntFishProducts: Medium, MntSweetProducts: Medium, MntGoldProds: Medium, NumDealsPurchases: Medium, NumWebPurchases: Medium, NumCatalogPurchases: Medium, NumStorePurchases: Medium, NumWebVisitsMonth: Medium, Age: Middle-aged',
            3: 'Cluster 3: Income: Low to Medium, Recency: High, MntWines: Low, MntFruits: Low, MntMeatProducts: Low, MntFishProducts: Low, MntSweetProducts: Low, MntGoldProds: Low, NumDealsPurchases: Medium, NumWebPurchases: Low, NumCatalogPurchases: Low, NumStorePurchases: Low, NumWebVisitsMonth: High, Age: Young'
        }
        
        prediction_text = cluster_descriptions[prediction]
        
        return render_template('index.html', prediction_text=prediction_text)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
