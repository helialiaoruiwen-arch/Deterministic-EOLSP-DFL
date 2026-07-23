import pandas as pd
import numpy as np
from types import SimpleNamespace
import sys, time
cplex_path = r'/opt/ibm/ILOG/CPLEX_Studio221/cplex/python/3.10/x86-64_linux'
if cplex_path not in sys.path:
    sys.path.append(cplex_path)
import cplex


class CplexOptimizer:
    def __init__(self, days, products):
        instance = initialization(days, products)
        # initialize the parameters
        self.T = instance.T
        self.R = instance.R
        self.days = instance.days
        self.length_microperiod = instance.length_microperiod
        self.R_ens = instance.R_ens
        self.J = instance.J
        self.D = instance.D
        self.initial_stock = instance.initial_stock
        self.capacity_unit = instance.capacity_unit
        self.capacity = instance.capacity
        self.startup_cost = instance.startup_cost
        self.holding_cost = instance.holding_cost
        self.lost_cost = instance.lost_cost
        self.energy_product = instance.energy_product
        self.energy_startup = instance.energy_startup
        self.gen_elec = np.array(instance.gen_elec)
        self.battery_cap = instance.battery_cap
        self.charge_lim = instance.charge_lim
        self.discharge_lim = instance.discharge_lim
        self.charge_efficiency = instance.charge_efficiency
        self.discharge_efficiency = instance.discharge_efficiency
        self.trans_efficiency = instance.trans_efficiency
        self.energy_purchase_price_daily = instance.energy_purchase_price_daily
        self.energy_purchase_price = instance.energy_purchase_price
        

        # 1. Initialize the CPLEX instance
        self.detModel = cplex.Cplex()

        # Turn off the output stream
        self.detModel.set_results_stream(None)
        self.detModel.set_warning_stream(None)
        self.detModel.set_error_stream(None)
        self.detModel.set_log_stream(None)

        self.detModel.objective.set_sense(self.detModel.objective.sense.minimize)

        self.startup_var = {}
        self.setup_var = {}
        self.prod_var = {}
        self.inventory_var = {}
        self.lost_var = {}
        self.elec_charge_var = {}
        self.elec_discharge_var = {}
        self.elec_purchase_var = {}
        self.elec_offer_var = {}
        self.elec_battery_var = {}
        self.elec_use_var = {}
        self.status_charge_var = {}
        self.status_discharge_var = {}
        
        # 2. Setup Base Model (Variables & Constants)
        # Defining variables
        # including the artificial product at the end
        for j in range(self.J + 1):
            for r in range(1, self.R +1):
                self.startup_var[(j,r)] = self.detModel.variables.add(types = self.detModel.variables.type.binary,
                names = [f'startup_{j}_{r}'],
                obj = [self.startup_cost[j]]
                )[0]
            
                self.prod_var[(j,r)] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
                names = [f'production_{j}_{r}'],
                lb = [0]
                )[0]

        for j in range(self.J + 1):
            for r in range(self.R+1):
                self.setup_var[(j,r)] = self.detModel.variables.add(types = self.detModel.variables.type.binary,
                names = [f'setup_{j}_{r}']
                )[0]

        for j in range(self.J):
            for t in range(self.T+1):
                self.inventory_var[(j,t)] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
                names = [f'inventory_{j}_{t}'],
                obj = [self.holding_cost[j]]
                )[0]

                self.lost_var[j,t] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
                names = [f'lost_var_{j}_{t}'],
                obj = [self.lost_cost[j]]
                )[0]

        

        for r in range(1,self.R+1):
            self.elec_charge_var[r] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
            names = [f'electricity_charged_{r}'],
            lb = [0]
            )[0]

            self.elec_discharge_var[r] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
            names = [f'electricity_discharged_{r}'],
            lb = [0]
            )[0]

            self.elec_purchase_var[r] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
            names = [f'electricity_purchased_{r}'],
            lb = [0],
            obj = [self.energy_purchase_price[r-1]]
            )[0]

            self.elec_use_var[r] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
            names = [f'electricity_use_{r}'],
            lb = [0]
            )[0]

        for r in range(self.R+1):
            self.elec_battery_var[r] = self.detModel.variables.add(types = self.detModel.variables.type.continuous,
            names = ['electricity_battery_' + str(r)],
            lb = [0],
            ub = [self.battery_cap]
            )[0]
        
        # initialization
        for j in range(self.J):
            self.detModel.linear_constraints.add(
                lin_expr= [
                    cplex.SparsePair(ind = [self.inventory_var[j,0]],
                    val = [1])
                ],
                senses = ['E'],
                rhs = [self.initial_stock[j]],
                names = [f'initial_inventory_{j}']
            )

            self.detModel.linear_constraints.add(
                lin_expr= [
                    cplex.SparsePair(ind = [self.inventory_var[j,self.T]],
                    val = [1])
                ],
                senses = ['G'],
                rhs = [self.initial_stock[j]],
                names = [f'final_stock_{j}']
            )

        for j in range(self.J):
            self.detModel.linear_constraints.add(
                lin_expr= [
                    cplex.SparsePair(ind = [self.setup_var[j,0]],
                    val = [1])
                ],
                senses = ['E'],
                rhs = [0],
                names = [f'initial_setup_{j}']
            )

        self.detModel.linear_constraints.add(
                lin_expr= [
                    cplex.SparsePair(ind = [self.setup_var[self.J, 0]],
                    val = [1])
                ],
                senses = ['E'],
                rhs = [1],
                names = [f'initial_setup_{self.J}']
            )

        self.detModel.linear_constraints.add(
            lin_expr= [
                cplex.SparsePair(ind = [self.elec_battery_var[0]],
                val = [1])
            ],
            senses = ['E'],
            rhs = [0],
            names = [f'initial_battery']
        )

        # Constraints
        # inventory balance constraint
        for j in range(self.J):    
            for t in range(1, self.T+1):
                self.detModel.linear_constraints.add(
                    lin_expr = [
                        cplex.SparsePair(ind = [self.inventory_var[(j,t)]] + [self.inventory_var[(j, t-1)]] 
                        + [self.prod_var[(j,r)] for r in self.R_ens[t]] + [self.lost_var[j,t]],
                        val = [1] + [-1] + [-1 for r in self.R_ens[t]] + [-1])
                    ],
                    senses = ['E'],
                    rhs = [-self.D[j][t-1]],
                    names = [f'invent_balance_{j}_{t}']
                )

        for j in range(self.J):
            for r in range(1, self.R+1):
                # at most two products can be produced in one microperiod
                self.detModel.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind = [self.prod_var[(j,r)]] + [self.setup_var[(j,r-1)]] + [self.setup_var[(j,r)]],
                        val = [self.capacity_unit[j]] + [-self.length_microperiod] +[-self.length_microperiod])
                    ],
                    senses = ['L'],
                    rhs = [0],
                    names = [f'products_constraint_{j}_{r}']
                )

        for j in range(self.J+1):
            for r in range(1, self.R+1):
                # relationship between startup and setup variables
                self.detModel.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind = [self.startup_var[(j,r)]] + [self.setup_var[(j,r)]] + [self.setup_var[(j,r-1)]],
                        val = [1] + [-1] + [1])
                    ],
                    senses = ['G'],
                    rhs = [0],
                    names = [f'relat_startup_setup_{j}_{r}']
                )

        for r in range(1, self.R+1):
            # capacity limitation for the machine
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.prod_var[(j,r)] for j in range(self.J)],
                    val = [self.capacity_unit[j] for j in range(self.J)])
                ],
                senses = ['L'],
                rhs = [self.length_microperiod],
                names = [f'capacity_lim_{r}']
            )

            # the machine can be set up for only one product at the beginning or at the end of a microperiod
        for r in range(self.R+1):
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.setup_var[(j,r)] for j in range(self.J+1)],
                    val = [1 for j in range(self.J+1)])
                ],
                senses = ['E'],
                rhs = [1],
                names = [f'setup_constraint_{r}']
            )

        
        # --- Energy_related constraints ---
        for r in range(1, self.R+1):
            # the energy balance for the whole system
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.elec_use_var[r]] + [self.elec_purchase_var[r]]  + [self.elec_discharge_var[r]] + [self.elec_charge_var[r]],
                    val = [1] + [-self.trans_efficiency] + [-self.discharge_efficiency] + [1/self.charge_efficiency])
                ],
                senses = ['E'],
                rhs = [float(self.gen_elec[r-1])],
                names = [f'energy_balance_fac_{r}']
            )

            # energy balance in the battery
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.elec_battery_var[r]] + [self.elec_battery_var[r-1]] + [self.elec_charge_var[r]] + [self.elec_discharge_var[r]],
                    val = [1] + [-1] + [-1] + [1])
                ],
                senses = ['E'],
                rhs = [0],
                names = [f'energy_balance_battery_{r}']
            )

            # the battery can be charged or discharged only if the status is set to charging or discharging
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.elec_charge_var[r]],
                    val = [1])
                ],
                senses = ['L'],
                rhs = [self.charge_lim],
                names = [f'battery_charging_{r}']
            )

            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.elec_discharge_var[r]],
                    val = [1])
                ],
                senses = ['L'],
                rhs = [self.discharge_lim],
                names = [f'battery_discharging_{r}']
            )

            # energy used in each microperiod
            self.detModel.linear_constraints.add(
                lin_expr=[
                    cplex.SparsePair(ind = [self.elec_use_var[r]] + [self.startup_var[(j,r)] for j in range(self.J)] + [self.prod_var[(j,r)] for j in range(self.J)],
                    val = [1] + [-self.energy_startup[j] for j in range(self.J)] + [-self.energy_product[j] for j in range(self.J)])
                ],
                senses = ['E'],
                rhs = [0],
                names = [f'energy_used_{r}']
            )

        # valid inequalities for the setup and startup
        for j in range(self.J+1):
            for r in range(1,self.R+1):
                self.detModel.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind=[self.startup_var[j,r]] + [self.setup_var[j,r-1]],
                        val = [1] + [1])
                    ],
                    senses=['L'],
                    rhs = [1],
                    names = [f'valid_inequalities_startset_{j}_{r}']
                )

                self.detModel.linear_constraints.add(
                    lin_expr=[
                        cplex.SparsePair(ind=[self.startup_var[j,r]] + [self.setup_var[j,r]],
                        val = [1] + [-1])
                    ],
                    senses = ['L'],
                    rhs = [0],
                    names = [f'startup_less_setup_{j}_{r}']
                )


        # # Keep track of temporary constraint names
        # self.temp_constraints = []

    def generate(self, n_samples):
        inputs = []
        outputs = []
        relax_values = []

        for i in range(n_samples):
            s = self.generate_random_scenario()
            cons_updates = self.update_param(s)
            obj_updates = self.update_objective(s)
            ub_updates = self.update_ub(s)

            # update the constraints, objective, upperbound
            self.detModel.linear_constraints.set_rhs(cons_updates)
            self.detModel.objective.set_linear(obj_updates)
            self.detModel.variables.set_upper_bounds(ub_updates)

            # self.detModel.write('md.lp')
            relax = self.solve_with_relaxation()
            relax_values.append(relax)

            # log_file = open("cplex_log.txt", "w")
            # self.detModel.set_log_stream(log_file)
            # self.detModel.set_results_stream(log_file)
            
            # record the solving time 
            start_time = time.perf_counter()
            # solve the model with the new generated parameters
            self.detModel.parameters.timelimit.set(1200)
            self.detModel.solve()

            end_time = time.perf_counter()
            resolution_time = end_time - start_time

            # Get the raw status code
            status = self.detModel.solution.get_status()

            valid_statuses = [
                self.detModel.solution.status.optimal,            # 1
                self.detModel.solution.status.MIP_optimal,        # 101
                self.detModel.solution.status.optimal_tolerance,    # 102
                self.detModel.solution.status.MIP_time_limit_feasible, # 107
            ]
            
            if status in valid_statuses:
                inputs.append(s)
                # 1. Collect all indices first
                startup_indices = [self.startup_var[j,r] for j in range(self.J+1) for r in range(1, self.R+1)]
                setup_indices   = [self.setup_var[j,r] for j in range(self.J+1) for r in range(self.R+1)]
                prod_indices   = [self.prod_var[j,r] for j in range(self.J) for r in range(1,self.R+1)]
                lost_indices   = [self.lost_var[j,t] for j in range(self.J) for t in range(1,self.T+1)]

                # 2. Make just TWO calls to CPLEX (super fast)
                opt_sol = SimpleNamespace()
                opt_sol.X = np.array(self.detModel.solution.get_values(startup_indices), dtype=np.float32)
                opt_sol.Y = np.array(self.detModel.solution.get_values(setup_indices), dtype=np.float32)
                opt_sol.Q = np.array(self.detModel.solution.get_values(prod_indices), dtype=np.float32)
                opt_sol.L = np.array(self.detModel.solution.get_values(lost_indices), dtype=np.float32)
                opt_sol.obj = self.detModel.solution.get_objective_value()
                opt_sol.mip_gap = self.detModel.solution.MIP.get_mip_relative_gap()
                opt_sol.best_LB = self.detModel.solution.MIP.get_best_objective()
                opt_sol.resolution_time = resolution_time
                outputs.append(opt_sol)
                # print('status is',self.detModel.solution.get_status())
                # print('objective value is',self.detModel.solution.get_objective_value())
                # print('best bound is',self.detModel.solution.MIP.get_best_objective())
                # print('demand', s.D)
                
            else:
                print(f"Skipping scenario: Status is {status}")
                
            if i % 10 == 0:
                print(f"Solved {i}/{n_samples}...")


        return inputs, outputs, relax_values

    # Call this when you are completely finished with the optimizer
    def terminate(self):
        if hasattr(self, 'detModel'):
            self.detModel.end()

    def generate_random_scenario(self):
        scenario = SimpleNamespace()
        # generate random parameters for the constraints
        scenario.D = self.generate_demand(np.random.uniform(0.7, 0.9, size=None)) + [[0]*self.T]
        scenario.PV = self.generate_elec(np.random.uniform(1, 5, size=None))
        scenario.initial_stock = self.generate_stock(scenario.D) + [0]
        scenario.initial_setup = [0]*self.J + [1]

        scenario.capacity = self.capacity
        scenario.capacity_unit = self.capacity_unit
        scenario.energy_startup = self.energy_startup
        scenario.energy_product = self.energy_product

        scenario.length_microperiod = self.length_microperiod

        # generate random upper bounds for variables
        scenario.battery_cap = np.random.uniform(100, 500, size=None)
        lim = np.random.uniform(0.8*scenario.battery_cap, scenario.battery_cap, size=None)
        scenario.charge_lim, scenario.discharge_lim = lim, lim

        # generate radom objective coefficients
        scenario.startup_cost = list(np.random.uniform(50, 400, size=(self.J,))) + [0]
        scenario.holding_cost = list(np.random.uniform(0.01, 0.1, size=(self.J,))) + [0]
        scenario.lost_cost = list(np.random.uniform(500,2000, size=(self.J,))) + [0]
        energy_purchase_price_day = self.energy_purchase_price_daily
        scenario.energy_purchase_price = np.array(list(energy_purchase_price_day)*self.days)/np.random.uniform(1,10, size=None)
        # print(len(scenario.energy_purchase_price))
        return scenario


    def update_param(self, scenario):
        updates = [(f'invent_balance_{j}_{t}',-scenario.D[j][t-1]) for j in range(self.J) for t in range(1,self.T+1)] \
        + [(f'energy_balance_fac_{r}', -scenario.PV[r-1]) for r in range(1,self.R+1)] \
        + [(f'initial_inventory_{j}', scenario.initial_stock[j]) for j in range(self.J)] \
        + [(f'final_stock_{j}', scenario.initial_stock[j]) for j in range(self.J)] \
        + [(f'initial_setup_{j}', scenario.initial_setup[j]) for j in range(self.J)]

        return updates 


    def update_ub(self, scenario):
        updates = [(self.elec_battery_var[r], scenario.battery_cap) for r in range(self.R+1)] \
            + [(self.elec_charge_var[r], scenario.charge_lim) for r in range(1, self.R+1)] \
            + [(self.elec_discharge_var[r], scenario.discharge_lim) for r in range(1, self.R+1)]

        return updates


    def update_objective(self, scenario, penalize=False):
        updates = \
        [(self.startup_var[j,r], scenario.startup_cost[j]) for j in range(self.J) for r in range(1,self.R+1)] \
        + [(self.inventory_var[j,t], scenario.holding_cost[j]) for j in range(self.J) for t in range(self.T+1)] \
        + [(self.lost_var[j,t], scenario.lost_cost[j]) for j in range(self.J) for t in range(self.T+1)] \
        + [(self.elec_purchase_var[r], scenario.energy_purchase_price[r-1]) for r in range(1,self.R+1)]

        if penalize == False:
            return updates

        ### Soft penalization
        if penalize == True:
            # add term in the objective function: M((1-p)y + p(1-y)) = M((1-2p)y + p)
            big_M = 100.0

            cnst = np.sum(self.y_pred)
            offset = cnst * big_M
            for j in range(self.J+1):
                for r_idx in range(self.R):
                    # r is 1-indexed in your CplexOptimizer setup_var keys
                    r_cplex = r_idx + 1 

                    var_index = self.setup_var[(j, r_cplex)]

                    updates.append((var_index, big_M*(1-2*self.y_pred[j][r_idx])))


            # cnst = -np.sum(np.log(1-self.y_pred + 1e-5))
            # offset = cnst * big_M
            # for j in range(self.J+1):
            #     for r_idx in range(self.R):
            #         # r is 1-indexed in your CplexOptimizer setup_var keys
            #         r_cplex = r_idx + 1 

            #         var_index = self.setup_var[(j, r_cplex)]

            #         coef = np.log(1-self.y_pred[j][r_idx] + 1e-5) - np.log(self.y_pred[j][r_idx]+1e-5)
            #         updates.append((var_index, big_M*coef))


        ### Penalization with the term M(1-y)
        if penalize == 'Hard':
            big_M = 100
            cnst = sum(np.array(self.confident_mask).astype(int))
            offset = cnst * big_M
            for r_idx in range(self.R):
                # r is 1-indexed in your CplexOptimizer setup_var keys
                r_cplex = r_idx + 1 

                # fix only values whose probability is above a certain threshold
                if self.confident_mask[r_idx]:
                    chosen_j = int(self.prod_indices[r_idx])

                    for j in range(self.J + 1):
                        var_index = self.setup_var[(j, r_cplex)]

                        # If this product was chosen by GNN, fix to 1. Else, fix to 0.
                        if j == chosen_j:
                            updates.append((var_index, -big_M))
        
        
        return updates, offset

        
    def generate_demand(self, utilization_rate):
        D_expected = int(self.capacity * utilization_rate / self.J)
        # Initialize a 2D array/list immediately
        D = [[int(max(np.random.normal(D_expected, D_expected/3), 0)) 
            for _ in range(self.T)] for _ in range(self.J)]
        return D

    def generate_elec(self, num_panels, season='winter'):
        gen_elec = [] 
        
        if season == 'winter':
            daily = [1, 4, 10, 18, 25, 27, 30, 30, 25, 15, 5, 2, 0, 0, 0, 0]
        if season == 'summer':
            daily = [1, 4, 10, 18, 25, 27, 30, 30, 25, 15, 5, 2, 0, 0, 0, 0]
            
        gen_elec_day = np.array(daily) * num_panels
        
        for _ in range(self.days):
            for r in range(16):
                # Calculate the value
                val = max(np.random.normal(loc=gen_elec_day[r], scale=gen_elec_day[r]), 0)
                gen_elec.append(val)
                
        return gen_elec

    def generate_stock(self, D):
        return [np.random.randint(0, max(2*int(D[j][0]),1)) for j in range(self.J)]

    def solve_with_relaxation(self):
        model = self.detModel
        original_types = model.variables.get_types()

        model.variables.set_types([(i, model.variables.type.continuous) for i in range(len(original_types))])
        model.set_problem_type(model.problem_type.LP)
        model.parameters.timelimit.set(200)
        model.solve()

        startup_indices = [self.startup_var[j,r] for j in range(self.J+1) for r in range(1, self.R+1)]
        setup_indices   = [self.setup_var[j,r] for j in range(self.J+1) for r in range(self.R+1)]
        prod_indices   = [self.prod_var[j,r] for j in range(self.J) for r in range(1,self.R+1)]
        lost_indices   = [self.lost_var[j,t] for j in range(self.J) for t in range(1,self.T+1)]

        con_invent_balance_indices = [f'invent_balance_{j}_{t}' for j in range(self.J) for t in range(1,self.T+1)]
        con_products_constraint_indices = [f'products_constraint_{j}_{r}' for j in range(self.J) for r in range(1, self.R+1)]
        con_relat_startup_setup_indices = [f'relat_startup_setup_{j}_{r}' for j in range(self.J+1) for r in range(1, self.R+1)]
        con_capacity_lim_indices = [f'capacity_lim_{r}' for r in range(1, self.R+1)]
        con_setup_constraint_indices = [f'setup_constraint_{r}' for r in range(1, self.R+1)]


        relax_point = SimpleNamespace(
            X = np.array(model.solution.get_values(startup_indices), dtype=np.float32),
            Y = np.array(model.solution.get_values(setup_indices), dtype=np.float32),
            Q = np.array(model.solution.get_values(prod_indices), dtype=np.float32),
            L = np.array(model.solution.get_values(lost_indices), dtype=np.float32),
            con_invent_balance = np.array(model.solution.get_dual_values(con_invent_balance_indices), dtype=np.float32),
            con_products_constraint = np.array(model.solution.get_dual_values(con_products_constraint_indices), dtype=np.float32),
            con_relat_startup_setup = np.array(model.solution.get_dual_values(con_relat_startup_setup_indices), dtype=np.float32),
            con_capacity_lim = np.array(model.solution.get_dual_values(con_capacity_lim_indices), dtype=np.float32),
            con_setup_constraint = np.array(model.solution.get_dual_values(con_setup_constraint_indices), dtype=np.float32),
        )

        model.variables.set_types([(i, original_types[i]) for i in range(len(original_types))])

        return relax_point





def initialization(days,J):
    ins = SimpleNamespace()
    macroperiods = 2
    microperiods = 8
    ins.days = days
    ins.T = days * macroperiods
    ins.R = days * macroperiods * microperiods

    ins.R_ens = {}
    for i in range(1,ins.T+1):
        ins.R_ens[i] = list(range((i-1)*microperiods+1, i*microperiods+1))

    capacity_unit_0 = 0.05
    startup_cost_0 = 200
    production_energy_0 = 0.05
    holding_cost_0 = 0.05
    startup_energy_0 = 10
    lost_cost_0 = 1000

    ins.length_microperiod = 60
    length_macroperiod = ins.length_microperiod*microperiods

    ins.capacity = length_macroperiod/capacity_unit_0
    ins.capacity_unit = [capacity_unit_0]*J

    # add an artificial product that represent the idle state of the machine
    ins.startup_cost = [startup_cost_0]*J + [0]
    ins.holding_cost = [holding_cost_0]*J + [0]
    ins.lost_cost = [lost_cost_0]*J + [0]

    ins.energy_startup = [startup_energy_0]*J + [0]
    ins.energy_product = [production_energy_0]*J + [0]
    # if J == 3: 
    #     energy_product = [0.05, 0.13, 0.21]
    # if J == 5:
    #     energy_product = [0.03, 0.08, 0.13, 0.18, 0.23]
    if len(ins.energy_product) != J+1:
        print("input not coincident")


    ins.D = np.zeros((J,ins.T))
    ins.initial_stock = [0]*J + [0]
    ins.initial_setup = [0]*J + [1]

    ins.gen_elec_daily = [1, 4, 10, 18, 25, 27, 30, 30, 25, 15, 5, 2, 0, 0, 0, 0]
    ins.energy_purchase_price_daily = [4.8, 6.1, 6.3, 6.0, 5.6, 4.0, 3.7, 3.8, 4.5, 5.1, 5.4, 5.9, 6.4, 6.3, 5.5, 4.5]

    ins.gen_elec = list(ins.gen_elec_daily)*days
    ins.energy_purchase_price = list(ins.energy_purchase_price_daily)*days

    ins.battery_cap = 100
    ins.charge_lim = 250
    ins.discharge_lim = 250
    ins.charge_efficiency = 0.95
    ins.discharge_efficiency = 0.95
    ins.trans_efficiency = 0.95

    ins.J = J
    
    return ins



def solve_with_fixed_setups(scenario_data, prod_indices, confident_mask):
    """
    scenario_data: A dictionary or object containing raw_D, raw_sc, etc.
    gnn_predictions: A 1D array/tensor of the chosen product indices for each slot R.
    """

    # Initialize the existing class
    # We use the dimensions from the scenario data
    J = scenario_data.J
    days = scenario_data.days
    opt = CplexOptimizer(days, J)

    # Update the model with this specific instance's parameters
    opt.detModel.linear_constraints.set_rhs(opt.update_param(scenario_data))
    opt.detModel.objective.set_linear(opt.update_objective(scenario_data))
    opt.detModel.variables.set_upper_bounds(opt.update_ub(scenario_data))

    # FIX the setup variables based on GNN output
    # gnn_predictions should be shape [R] where each value is the chosen J
    R = scenario_data.R
    assert R == len(prod_indices)
    
    for r_idx in range(R):
        # r is 1-indexed in your CplexOptimizer setup_var keys
        r_cplex = r_idx + 1 

        # fix only values whose probability is above a certain threshold
        if confident_mask[r_idx]:
            chosen_j = int(prod_indices[r_idx])

            for j in range(J + 1):
                var_index = opt.setup_var[(j, r_cplex)]
                
                # If this product was chosen by GNN, fix to 1. Else, fix to 0.
                val = 1.0 if j == chosen_j else 0.0
                
                opt.detModel.variables.set_lower_bounds(var_index, val)
                opt.detModel.variables.set_upper_bounds(var_index, val)

    start_cplex = time.perf_counter()

    opt.detModel.parameters.timelimit.set(1200)
    # Solve the remaining Linear Program (LP)
    opt.detModel.solve()

    end_cplex = time.perf_counter()

    total_time = end_cplex - start_cplex
    
    status = opt.detModel.solution.get_status()
    if status in [1, 101, 102, 107]: # Optimal statuses
        return opt.detModel.solution.get_objective_value(), total_time
    else:
        # If GNN made an impossible choice (e.g., broke a hard constraint)
        return float('inf')



# Penalize with term M(1-y)
def solve_with_penalization_in_obj_hard(scenario_data, prod_indices, confident_mask):
    """
    scenario_data: A dictionary or object containing raw_D, raw_sc, etc.
    gnn_predictions: A 1D array/tensor of the chosen product indices for each slot R.
    """

    # Initialize the existing class
    # We use the dimensions from the scenario data
    J = scenario_data.J
    R = scenario_data.R
    days = scenario_data.days
    opt = CplexOptimizer(days, J)
    opt.prod_indices = prod_indices
    opt.confident_mask = confident_mask

    # Update the model with this specific instance's parameters
    opt.detModel.linear_constraints.set_rhs(opt.update_param(scenario_data))
    opt.detModel.variables.set_upper_bounds(opt.update_ub(scenario_data))

    obj_updates, offset = opt.update_objective(scenario_data, penalize='Hard')
    opt.detModel.objective.set_linear(obj_updates)
    opt.detModel.objective.set_offset(float(offset))
    # print('offset', offset)

    start_cplex = time.perf_counter()
    opt.detModel.parameters.timelimit.set(1200)

    # Solve the remaining Linear Program (LP)
    opt.detModel.solve()

    end_cplex = time.perf_counter()

    total_time = end_cplex - start_cplex
    
    status = opt.detModel.solution.get_status()
    if status in [1, 101, 102, 107]: # Optimal statuses
        setup_indices   = [opt.setup_var[j,r] for j in range(J+1) for r in range(R+1)]
        Y = np.array(opt.detModel.solution.get_values(setup_indices), dtype=np.float32)
        obj = opt.detModel.solution.get_objective_value()
        return obj, Y, total_time
    else:
        # If GNN made an impossible choice (e.g., broke a hard constraint)
        return float('inf')


# Soft penalization
def solve_with_penalization_in_obj(scenario_data, y_predictions):
    """
    scenario_data: A dictionary or object containing raw_D, raw_sc, etc.
    gnn_predictions: A 1D array/tensor of the chosen product indices for each slot R.
    """

    # Initialize the existing class
    # We use the dimensions from the scenario data
    J = scenario_data.J
    R = scenario_data.R
    days = scenario_data.days
    opt = CplexOptimizer(days, J)
    opt.y_pred = y_predictions

    # Update the model with this specific instance's parameters
    opt.detModel.linear_constraints.set_rhs(opt.update_param(scenario_data))
    opt.detModel.variables.set_upper_bounds(opt.update_ub(scenario_data))

    obj_updates, offset = opt.update_objective(scenario_data, penalize=True)
    opt.detModel.objective.set_linear(obj_updates)
    opt.detModel.objective.set_offset(offset)

    ene_idx = [opt.detModel.linear_constraints.get_indices(f"energy_balance_fac_{r}") for r in range(1, R+1)] \
        + [opt.detModel.linear_constraints.get_indices(f"energy_balance_battery_{r}") for r in range(1, R+1)] \
        + [opt.detModel.linear_constraints.get_indices(f"battery_charging_{r}") for r in range(1, R+1)] \
        + [opt.detModel.linear_constraints.get_indices(f"battery_discharging_{r}") for r in range(1, R+1)] \
        + [opt.detModel.linear_constraints.get_indices(f"energy_used_{r}") for r in range(1, R+1)]

    opt.detModel.linear_constraints.delete(ene_idx)

    start_cplex = time.perf_counter()

    # log_file = open("cplex_log_pen.txt", "w")
    # opt.detModel.set_log_stream(log_file)
    # opt.detModel.set_results_stream(log_file)

    opt.detModel.parameters.timelimit.set(1200)
    # Solve the remaining Linear Program (LP)
    opt.detModel.solve()

    end_cplex = time.perf_counter()

    total_time = end_cplex - start_cplex
    
    status = opt.detModel.solution.get_status()
    if status in [1, 101, 102, 107]: # Optimal statuses
        setup_indices   = [opt.setup_var[j,r] for j in range(J+1) for r in range(R+1)]
        Y = np.array(opt.detModel.solution.get_values(setup_indices), dtype=np.float32)
        obj = opt.detModel.solution.get_objective_value()
        return obj, Y, total_time
    else:
        # If GNN made an impossible choice (e.g., broke a hard constraint)
        # print()
        return float('inf')

def evaluate_solution(scenario_data, y_values):
    # Initialize the existing class
    # We use the dimensions from the scenario data
    J = scenario_data.J
    days = scenario_data.days
    opt = CplexOptimizer(days, J)

    # Update the model with this specific instance's parameters
    opt.detModel.linear_constraints.set_rhs(opt.update_param(scenario_data))
    opt.detModel.objective.set_linear(opt.update_objective(scenario_data))
    opt.detModel.variables.set_upper_bounds(opt.update_ub(scenario_data))

    # FIX the setup variables based on GNN output
    # gnn_predictions should be shape [R] where each value is the chosen J
    R = scenario_data.R
    y_values_reshape = np.array(y_values.reshape(J+1, R+1), dtype=np.float64)
    # print(y_values_reshape)
    
    for j in range(J+1):
        for r_idx in range(R):
            # r is 1-indexed in your CplexOptimizer setup_var keys
            r_cplex = r_idx + 1 

            var_index = opt.setup_var[(j, r_cplex)]
            
            opt.detModel.variables.set_lower_bounds(var_index, y_values_reshape[j][r_cplex])
            opt.detModel.variables.set_upper_bounds(var_index, y_values_reshape[j][r_cplex])

    start_cplex = time.perf_counter()
    # Solve the remaining Linear Program (LP)
    opt.detModel.solve()

    end_cplex = time.perf_counter()

    total_time = end_cplex - start_cplex
    
    status = opt.detModel.solution.get_status()
    if status in [1, 101, 102]: # Optimal statuses
        setup_indices   = [opt.setup_var[j,r] for j in range(J+1) for r in range(R+1)]
        Y = np.array(opt.detModel.solution.get_values(setup_indices), dtype=np.float32)
        return opt.detModel.solution.get_objective_value(), total_time, Y
    else:
        # If GNN made an impossible choice (e.g., broke a hard constraint)
        return float('inf')
    

if __name__ == '__main__':
    # Usage
    opt = CplexOptimizer(1,3)
    _, results = opt.generate(1)
    print(results)
