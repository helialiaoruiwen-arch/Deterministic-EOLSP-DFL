import os
import torch, random
from torch_geometric.data.hetero_data import HeteroData
from CplexModel import evaluate_solution

torch.serialization.add_safe_globals([HeteroData])

def Add_new_value(folder_path, new_key):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(folder_path, file_name)
            
            try:
                data = torch.load(file_path, weights_only=False)

                # Diagnostic print to track execution progress
                print(f"Processing {file_name}... ", end="")
                ave_Y = create_confident_optimal(data, 40)
                print("Calculated neighborhood average... ", end="")

                # Alternative storage technique to completely bypass PyG validation engines
                data['var_setup'].__dict__['_mapping']['aveY_'] = ave_Y
                
                torch.save(data, file_path)
                print("Saved successfully.")
                    
            except Exception as e:
                print(f"\nCRASHED on {file_name}!")
                print(f"Error Message: {e}")

def get_single_boundary_neighbor(matrix):
    num_products, num_periods = matrix.shape
    active_products = matrix.argmax(dim=0)
    
    boundaries = []
    for t in range(num_periods - 1):
        if active_products[t] != active_products[t+1]:
            boundaries.append(t)
            
    if not boundaries:
        return matrix.clone()
        
    t = random.choice(boundaries)
    prod_current = int(active_products[t])
    prod_next = int(active_products[t+1])
    
    neighbor = matrix.clone()
    if random.choice([True, False]):
        if (active_products == prod_next).sum() > 1:
            neighbor[prod_next, t+1] = 0.0
            neighbor[prod_current, t+1] = 1.0
    else:
        if (active_products == prod_current).sum() > 1:
            neighbor[prod_current, t] = 0.0
            neighbor[prod_next, t] = 1.0
            
    return neighbor

def get_combined_neighbor(matrix, k=3):
    current_solution = matrix.clone()
    for _ in range(k):
        current_solution = get_single_boundary_neighbor(current_solution)
    return current_solution

def calculate_alpha(delta_list, beta=1.0):
    deltas = torch.tensor(delta_list, dtype=torch.float64)
    numerator = torch.exp(-beta * deltas)
    denominator = torch.sum(numerator)
    return numerator / denominator

def create_confident_optimal(data, N):
    scenario = data.scenario
    J = scenario.J
    
    Y = data['var_setup'].y.reshape(scenario.J+1, -1).clone().float()
    
    initial_state = torch.zeros(J+1)
    initial_state[-1] = 1.0
    initial_column = initial_state.view(J+1, 1)

    Neighbors = [Y]
    Ecarts = [0.0]

    for i in range(N):
        neighbor_solution = get_combined_neighbor(Y, k=3)
        expanded_solution = torch.cat((initial_column, neighbor_solution), dim=1)
        expanded_solution = expanded_solution.flatten()
        
        try:
            # Let's verify exactly what evaluate_solution returns
            outputs = evaluate_solution(scenario, expanded_solution)
            
            # If evaluate_solution returns a raw float instead of a tuple, unpack manually
            if isinstance(outputs, tuple):
                obj_neighbor = outputs[0]
            else:
                obj_neighbor = outputs
                
            delta = (obj_neighbor - scenario.obj) / scenario.obj
            Neighbors.append(neighbor_solution)
            Ecarts.append(delta)
            
        except Exception as eval_error:
            raise RuntimeError(
                f"evaluate_solution cracked on neighbor iteration {i}.\n"
                f"Matrix Active Setup Count: {(neighbor_solution == 1).sum().item()}\n"
                f"Original Objective: {scenario.obj}\n"
                f"Internal Error: {eval_error}"
            )

    alpha = calculate_alpha(Ecarts)
    Y_stacked = torch.stack(Neighbors, dim=0)
    average_Y = torch.einsum('n,nij->ij', alpha.to(Y_stacked.dtype), Y_stacked)
    
    return average_Y.flatten()

if __name__ == '__main__':
    folder_path = './TrainingDataset_aveY/data_storage_var'
    new_key = "var_setup"
    
    # Run the correction routine
    Add_new_value(folder_path, new_key)
    
    # Verification Check
    torch.set_printoptions(precision=2, sci_mode=False)
    # print("\n--- Verification Check ---")
    # data = torch.load('test/graph_0.pt', weights_only=False)
    # print("Available keys in node type [var_setup]:", data['var_setup'].keys())
    # print("Stored tensor value:\n", data['var_setup'].aveY_)