import numpy as np


def closestNumber(n, m):
    # Find the quotient
    q = int(n / m)
    # 1st possible closest number
    n1 = m * q
    # 2nd possible closest number
    if (n * m) > 0:
        n2 = m * (q + 1)
    else:
        n2 = m * (q - 1)
    # if true, then n1 is the required closest number
    # if (abs(n - n1) < abs(n - n2)) :
    # return n1
    # else n2 is the required closest number
    return n2


def calculate_costs_gcf(list_execution_times, offset=1):

    mem_list = [512, 2048, 4096, 8192]
    compute_list = [800, 2400, 4800, 4800]
    GB_ms_cost_tier_1 = 0.0000025 / 1000
    GHz_ms_cost_tier_1 = 0.0000100 / 1000
    INVOCATIONS_UNIT_PRICE = 0.0000004
    # per_mil = 1000000
    per_mil = 1
    # cost_per_mil = 0.40
    cost_per_mil = 0
    list_cost = np.array([])
    for i in range(len(list_execution_times)):
        time = closestNumber(int(list_execution_times[i]) * 1000, 100)
        gb_s = (mem_list[offset] / 1024) * time * per_mil * GB_ms_cost_tier_1
        gh_s = (compute_list[offset] / 1000) * time * per_mil * GHz_ms_cost_tier_1
        # cost_val = gb_s+gh_s+cost_per_mil
        cost_val = gb_s + gh_s + INVOCATIONS_UNIT_PRICE
        # print(cost_val)
        list_cost = np.append(list_cost, cost_val)
    return list_cost
