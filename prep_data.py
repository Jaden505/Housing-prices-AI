import os
import urllib
import pandas as pd
import tarfile
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


class PrepData:
    def __init__(self):
        self.DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        self.HOUSING_PATH = os.path.join("datasets", "housing")
        self.HOUSING_URL = self.DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    def get_housing_data(self):
        if not os.path.isdir(self.HOUSING_PATH):
            # Create path and insert data
            os.makedirs(self.HOUSING_PATH)
            
            tgz_path = os.path.join(self.HOUSING_PATH, "housing.tgz")
            urllib.request.urlretrieve(self.HOUSING_UR, tgz_path)
            housing_tgz = tarfile.open(tgz_path)
            housing_tgz.extractall(path=self.HOUSING_PATH)
            housing_tgz.close()

        csv_path = os.path.join(self.HOUSING_PATH, "housing.csv")
        return pd.read_csv(csv_path)

    def clean(self, housing):
        # Split to x and y
        housing_labels = housing["median_house_value"].to_numpy()
        housing = housing.drop("median_house_value", axis=1)
        
        # Fill median for missing values
        housing_num = housing.drop("ocean_proximity", axis=1)
        X = SimpleImputer(strategy='median').fit_transform(housing_num) 
        print(pd.DataFrame(X, columns=housing_num.columns).info())
                
        # Strings to numeric classses one-hot 
        housing_cat_1hot = OneHotEncoder().fit_transform(housing[["ocean_proximity"]])
        print(housing_cat_1hot.toarray())
        
        return housing, housing_labels

    def scale(self):
        
        
if __name__ == '__main__':
    p = PrepData() 
    housing_data = p.get_housing_data()
    housing_data = p.clean(housing_data)