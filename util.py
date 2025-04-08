import json
import pickle
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location , sqft , bath , bhk):
    x = np.zeros(len(__data_columns)-1)
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x[1] = bhk
    x[2] = sqft
    x[3] = bath
    if loc_index >= 0:
        x[loc_index] = 1
    print("Estimated Price : ")
    return __model.predict([x])[0]
def get_location_names():
    load_saved_artifacts()
    return __locations


def load_saved_artifacts():
    print("Loading Saved Artifacts")
    global __locations, __data_columns
    with open("./artifacts/cols_for_real_estate_banglore" , 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[7:]

    with open("./artifacts/real_estate_prediction",'rb') as f:
        global __model
        __model = pickle.load(f)
        print("loading saved artifacts")

if __name__ == "__main__":
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
