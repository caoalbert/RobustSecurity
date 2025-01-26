import numpy as np

# calc_cum_arrive: calculate cumulative number of passenger arrivals at each time step
def calc_cum_arrive(array):
    arrival = []
    num = 0
    for i in range(24*4):
        num_of_pax = np.sum(array == i)
        num += num_of_pax
        arrival.append(num)

    return np.array(arrival)

# calc_cum_depart: calculate cumulative number of passenger departures at each time step
def calc_cum_depart(array, capacity):
    departure = []
    cumulative_departure = 0
    queue = 0
    for i in range(24*4):
        num_pax = np.sum(array == i)
        queue += num_pax

        departed = min(capacity, queue)
        queue -= departed

        cumulative_departure += departed
        departure.append(cumulative_departure)

    return np.array(departure)

