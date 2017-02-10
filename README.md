# RegresionHousePrices
[House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

In this challenge we are being asked to produce a prediction about sale prices of residential homes in Ames,
Iowa. Train set and test set needed for the prediction is provided as two separated CSV file, we want to produce
another CSV file containing for each row the Id of the house and a prediction for the sale price. Follow the
procedure adopted.

To launch the script execute the follow statement
```{r, engine='bash', run_script}
python3 main.py "data/train.csv" "data/test.csv"
```
The prediction will write on file pred.csv into data folder.

### Check _HousePrices.ipynb_ to watch workflow of the script, enjoy!