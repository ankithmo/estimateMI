
import torch

from tqdm import tqdm

def get_samples(model, x, edge_index=None, dataset=None):
    """
        Get samples from the DNN

        Args:
            - model: DNN model
            - x (num_examples, num_features): Datapoints
            - edge_index (2, num_edges): Edge index
            - dataset (str): Which dataset should the MI be estimated for?
                            Possible values: train, val, test

        Returns:
            - Us: Dictionary of layer-wise unconditional samples
            - Cs: Dictionary of layer-wise conditional samples
    """
    Us, Cs = {}, {}

    assert dataset in ["train", "val", "test"], 
        ValueError(f"Expected `dataset` to be in [train, val, test], got {dataset} instead.")

    total = x.size(0) if dataset is None else x[dataset].size(0)

    model.eval()
    with torch.no_grad():
        print("Unconditional sampling...")
        _, uS = model(x) if edge_index is None else model(x, edge_index) 
        for i in range(len(uS)):
            Us[i] = uS[i] if dataset is None else uS[i][dataset]
        print("Done")

        print("Conditional sampling...")
        with tqdm(total=total) as pbar:
            for j in range(total):
                pbar.set_description(f"Conditional samples given x_{j}")
                _, cS = model(x) if edge_index is None else model(x, edge_index)
                for i in range(len(cS)):
                  Cs[i] = cS[i] if dataset is None else cS[i][dataset]
            pbar.update(1)
        pbar.close()
        for i in range(len(Cs)):
            Cs[i] = torch.stack(Cs[i], axis=1)
            assert Cs[i].size() == (total, total, Cs[i].size(1))
        print("Done")

    return Us, Cs