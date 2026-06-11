import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax
import torch.optim as optim
from model_def import SchedulerForward, SchedulingDataset, SPOPlusFunction, FYFunction
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from CplexModel import solve_with_penalization_in_obj
from pathlib import Path

# Configuration
# DATA_PATH = "data_storage"
BATCH_SIZE = 16
LEARNING_RATE = 1e-3 # Slightly lower LR often helps with weighted loss
EPOCHS = 50

def criterion(y_hat_probs, y_true, num_slots, setup_weight=0.1):
    # 1. Imitation Loss 
    # calculate the error between the predicted y values and the true y values
    weights = torch.where(y_true == 1, 15.0, 1.0)
    imitation_loss = F.binary_cross_entropy(y_hat_probs, y_true, weight=weights)
    
    # 2. Setup Penalty (The "Differentiable Switch")
    # y_hat_probs is [Num_nodes, 1]
    
    slot_predictions = y_hat_probs.view(num_slots, -1)
    
    prob_t = slot_predictions[1:]
    prob_t_minus_1 = slot_predictions[:-1]
    
    # This measures how much the "Product Choice" shifted between slots
    setup_penalty = torch.mean(torch.abs(prob_t - prob_t_minus_1))
    
    return imitation_loss + (setup_weight * setup_penalty)


def train(train_data, test_data, model_path, feat_dims, edge_dims):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # checkpoint = torch.load("checkpoint_epoch_var_bce_19.pth")
    checkpoint = torch.load("checkpoints/var_150data_spo_moving_ave_3/checkpoint_epoch_49.pth")
    

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    model = SchedulerForward(hidden_channels=64, feat_dims=feat_dims, edge_dims_dict=edge_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    start_epoch = 0

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']

    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Scheduler
    # 'min' means it reacts when the loss stops decreasing
    # 'patience=5' means wait 5 epochs of no improvement before cutting LR
    # 'factor=0.5' means cut the LR in half (0.001 -> 0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 2e-4
    
    print(f"Training for SETUPS on {len(train_data)} graphs...")

    train_losses = []
    test_losses = []


    for i in range(EPOCHS):
        model.train()
        running_loss = 0.0

        epoch = i + start_epoch + 1

        # original_lr = optimizer.param_groups[0]['lr']
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = 2e-4
        
        for batch_data in train_loader:
            # Unified Device
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            
            # Forward Pass
            y_hat = model(batch_data) 

            # # We check the standard deviation of the features before the final Linear layer
            # feat_std = batch_data['var_setup'].x.std(dim=0).mean().item()
            # pred_std = y_hat.std().item()

            # print(f"Feature Variation: {feat_std:.4f} | Prediction Variation: {pred_std:.4f}")

            # Split to get the y_hat value and the y_true value for each graph
            node_batch_map = batch_data['var_setup'].batch
            counts = torch.bincount(node_batch_map, minlength=batch_data.num_graphs)
            y_hat_list = torch.split(y_hat, counts.tolist())

            y_true_list = torch.split(batch_data['var_setup'].y.float(), counts.tolist())

            # reverse the batching process, split the data into a list of individual heterodata object
            data_list = batch_data.to_data_list()

            if epoch >= 1:
                # if epoch == 20:
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = 2e-4

                spo_loss_list = []
                for i in range(len(y_hat_list)):
                    theta = y_hat_list[i]
                    scenario = data_list[i].scenario
                    theta_reshape = theta.reshape(scenario.J+1, scenario.R)
                    # obj, _, sol_time = solve_with_penalization_in_obj(scenario, theta_reshape)

                    y_true = y_true_list[i]
                    y_true_reshape = y_true.reshape(scenario.J+1, scenario.R)
                    # obj_true, _, sol_true_time = solve_with_penalization_in_obj(scenario, y_true_reshape)

                    # theta_perturbed = theta_reshape + torch.randn_like(theta_reshape) * 0.05
                    # spo_loss_list.append(SPOPlusFunction.apply(theta_perturbed, y_true_reshape, scenario))
                    # spo_loss_list.append(SPOPlusFunction.apply(theta_reshape, y_true_reshape, scenario, epoch, i))

                    spo_loss_list.append(SPOPlusFunction.apply(theta_reshape, y_true_reshape, scenario, epoch, i))

                    # if i == 0 and epoch%5 == 0:
                    #     print('theta', theta.detach().numpy())
                    #     print('y_true', y_true)
                    # print('obj and true obj', obj, obj_true)
                    # print('compare solution time', sol_time, sol_true_time)
                
                # print(spo_loss_list)
                spo_loss = torch.stack(spo_loss_list).mean()

                spo_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
                
                optimizer.step()
                
                running_loss += spo_loss.item()

               


            if epoch < 1:
                # Weighted BCE (Clamping targets for safety)
                # enforce the number to be strictly between 0 and 1
                targets = torch.clamp(batch_data['var_setup'].y.float(), 0.0, 1.0)

                # weights = torch.where(targets == 1, 15.0, 1.0)
                # loss = F.binary_cross_entropy(y_hat, targets, weight=weights)

                # total number of slots in a batch
                num_slots = batch_data['var_setup'].raw_r.size(0)
                # num_products = batch.raw_j.max()+1
                bce_loss = criterion(y_hat, targets, num_slots, setup_weight=1.0)
                
                # Backward Pass
                bce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                running_loss += bce_loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Avg Loss: {avg_loss:.4f}")


        ### ---- Evaluation Phase ---- ###
        if epoch % 5 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():

                # for batch in test_loader:
                #     batch = batch.to(device)
                #     y_hat = model(batch) 

                #     targets = torch.clamp(batch['var_setup'].y.float(), 0.0, 1.0)

                #     num_slots = batch['var_setup'].raw_r.size(0)
                #     # num_products = batch.raw_j.max()+1
                #     loss = criterion(y_hat, targets, num_slots, setup_weight=1.0)
                    
                #     test_loss += loss.item()

                # avg_loss = test_loss / len(test_loader)
                # test_losses.append(avg_loss)

                for batch in test_loader:
                    batch = batch.to(device)
                    y_hat = model(batch) 

                    node_batch_map = batch['var_setup'].batch
                    counts = torch.bincount(node_batch_map, minlength=batch.num_graphs)
                    y_hat_list = torch.split(y_hat, counts.tolist())

                    y_true_list = torch.split(batch['var_setup'].y.float(), counts.tolist())

                    # reverse the batching process, split the data into a list of individual heterodata object
                    data_list = batch.to_data_list()


                    spo_loss_list = []
                    for i in range(len(y_hat_list)):
                        theta = y_hat_list[i]
                        scenario = data_list[i].scenario
                        theta_reshape = theta.reshape(scenario.J+1, scenario.R)
                        # obj, _, sol_time = solve_with_penalization_in_obj(scenario, theta_reshape)

                        y_true = y_true_list[i]
                        y_true_reshape = y_true.reshape(scenario.J+1, scenario.R)
                        # obj_true, _, sol_true_time = solve_with_penalization_in_obj(scenario, y_true_reshape)

                        # theta_perturbed = theta_reshape + torch.randn_like(theta_reshape) * 0.05
                        # spo_loss_list.append(SPOPlusFunction.apply(theta_perturbed, y_true_reshape, scenario))
                        spo_loss_list.append(SPOPlusFunction.apply(theta_reshape, y_true_reshape, scenario, epoch, i))

                        # spo_loss_list.append(FYFunction.apply(theta_reshape, y_true_reshape, scenario, epoch, i))

                    spo_loss = torch.stack(spo_loss_list).mean()

                    test_loss += spo_loss.item()

                avg_loss = test_loss / len(test_loader)
                test_losses.append(avg_loss)
                
                scheduler.step(avg_loss)
                current_lr = optimizer.param_groups[0]['lr']

                print(f"Epoch {epoch:03d} | Avg Loss evaluation: {avg_loss:.4f} | LR: {current_lr}")
                


        checkpoint_dir = Path("checkpoints/var_150data_spo_CE_pen")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if epoch % 10 == 9:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                # 'scheduler_state_dict': scheduler.state_dict()
            }

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")


    # 3. Save the trained weights
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # save_loss_plot(train_losses)
    plot_mismatched_losses(train_losses, test_losses)

def monitor_gradients(model):
    print(f"{'Layer Name':<25} | {'Grad Norm':<10} | {'Weight Norm':<10}")
    print("-" * 50)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            weight_norm = torch.norm(param.data).item()
            print(f"{name:<25} | {grad_norm:<10.6f} | {weight_norm:<10.4f}")
        else:
            print(f"{name:<25} | {'None':<10} | {torch.norm(param.data).item():<10.4f}")
    print("\n")
    
def save_loss_plot(history, filename="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Training Loss', color='blue', linewidth=2)
    plt.title("GNN Scheduling Training Progress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Weighted BCE)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save the file
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Important: release memory


def plot_mismatched_losses(train_losses, test_losses, eval_interval=5):
    # 1. Define x-axis for training (every epoch)
    epochs_train = range(1, len(train_losses) + 1)
    
    # 2. Define x-axis for testing (every N epochs)
    # This starts at eval_interval and jumps by eval_interval
    epochs_test = range(eval_interval, (len(test_losses) * eval_interval) + 1, eval_interval)
    
    # Create the plot
    plt.plot(epochs_train, train_losses, 'b-', label='Train Loss', alpha=0.6)
    plt.plot(epochs_test, test_losses, 'r--o', label=f'Test Loss (every {eval_interval} eps)', markersize=4)
    
    # Standard Formatting
    plt.title('Training vs Testing Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Force 2 decimal places on the Y-axis
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    
    plt.tight_layout()
    plt.savefig('loss_comparison.png')
    print("Plot saved as loss_comparison.png")
    plt.close()


# from run_all import get_split_data
# DATA_PATH = "data_storage"
if __name__ == "__main__":
    full_dataset = SchedulingDataset("data_storage")
    # train_size = int(0.8 * len(full_dataset))
    # test_size = len(full_dataset) - train_size
    # train_set, test_set = torch.utils.data.random_split(
    #     full_dataset, 
    #     [train_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    # train(train_set)
    # print(len(full_dataset[0].x[:,1].unique()))