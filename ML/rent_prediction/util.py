from joblib import load
import json
import numpy as np

__locations = None
__model = None
__data_columns = None
__types = None


def get_predicted_rent(com_sc, room_sv_sc, mem, size, type_, room_sc, att_wr, loc, elec, wifi, sec):
    global __model , __types , __locations
    __model = get_model()
    __locations = get_location_names()
    __types = get_types()

    x = np.zeros(18)
    x[0] = com_sc
    x[1] = room_sv_sc
    x[2] = mem
    x[3] = size
    x[4] = room_sc
    x[5] = att_wr
    x[6] = elec
    x[7] = wifi
    x[8] = sec

    locs = list(__locations)
    loc_start_index = 12  # Start of location one-hot encoding
    if loc in locs:
        idx = locs.index(loc)
        x[loc_start_index + idx] = 1

    types = list(__types)
    type_start_index = 9  # Start of type one-hot encoding
    if type_ in types:
        idx = types.index(type_)
        x[type_start_index + idx] = 1

    return __model.predict([x])[0]

def get_location_names():
    global __data_columns
    global __locations
    __data_columns = get_attributes()
    __locations = __data_columns[12:]
    return __locations

def get_attributes():
    global __data_columns
    with open("server/artifacts/cols_for_rent_prediction", 'r') as f:
        data = json.load(f)
        __data_columns = data['data_columns']
    return __data_columns

def get_model():
    global __model
    with open("server/artifacts/rent_prediction.joblib", 'rb') as f:
        __model =load(f)
    return __model

def get_types():
    global __types , __data_columns
    __data_columns = get_attributes()
    __types = __data_columns[9:12]
    return __types

if __name__ == "__main__":
    print(get_location_names())