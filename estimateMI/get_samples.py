
import torch

from tqdm import tqdm

def get_samples(model, x, edge_index=None, train_idx=None):
    """
        Get samples from the DNN

            Parameters:
                model: DNN that returns the following:
                    - Representations learnt by the final neural network layer
                    - Class predictions
                    - List consisting of the representations learnt before being passed into AWGN channels
                x : torch.tensor of shape (num_examples, num_features)
                    Input tensor
                edge_index : torch.tensor of shape (2, num_edges), optional
                    Edge index
                train_idx : torch.tensor of shape (num_nodes), optional
                    Tensor indicating which nodes are part of the training set

            Returns:
                Us: dictionary
                    layer-wise unconditional samples
                Cs: dictionary
                    layer-wise conditional samples
    """
    Us, Cs = {}, {}
    total = x.size(0)

    model.eval()
    with torch.no_grad():
        print("Unconditional sampling...")
        _, _, uS = model(x) if edge_index is None else model(x, edge_index) 
        for i in range(len(uS)):
            Us[i] = uS[i] if train_idx is None else uS[i][train_idx]
        print("Done")

        print("Conditional sampling...")
        with tqdm(total=total) as pbar:
            for j in range(total):
                pbar.set_description(f"Conditional samples given x_{j+1}")
                _, _, cS = model(x) if edge_index is None else model(x, edge_index)
                for i in range(len(cS)):
                    if i not in Cs:
                        Cs[i] = [cS[i] if train_idx is None else cS[i][train_idx]]
                    else:
                        Cs[i].append(cS[i])
            pbar.update(1)
        pbar.close()
        for i in range(len(Cs)):
            Cs[i] = torch.stack(Cs[i], axis=1)
        print("Done")

    return Us, Cs