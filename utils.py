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

def calc_prob_den(solution, inst):
    np.random.seed(2)
    prob = np.zeros(shape=(solution.shape[0], solution.shape[1]))
    for i in range(solution.shape[0]):
        beta_i = []
        for j in range(solution.shape[1]):
            diff = inst.latest_arrival_time[i] - j - 1
            if diff >= 0 and diff <= 15:
                beta_i.append(inst.pmf[diff])
            else:
                beta_i.append(0)
        beta_i = np.array(beta_i)
        prob_i = np.zeros(shape=(solution.shape[1]))
        for t in range(solution.shape[1]):
            alpha_t = np.random.normal(inst.alpha[i,t], inst.sigma)
            prob_i[t] += alpha_t * solution[i,t]
            for s in range(solution.shape[1]):
                prob_i[s] += solution[i,t] * (1-alpha_t) * beta_i[s]

        prob[i] = prob_i
    return prob

def generate_assigned_arrivals(realized, df):
    np.random.seed(2)
    realized[realized < 1e-10] = 0
    arrival_time = np.array([])
    for i in range(realized.shape[0]):
        arrival_time_sample = np.random.choice(np.arange(realized.shape[1]), size=int(df['Aircraft Capacity'][i]), p=realized[i]/realized[i].sum())
        arrival_time = np.concatenate([arrival_time, arrival_time_sample])

    return arrival_time
