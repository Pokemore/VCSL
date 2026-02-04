"""
init_sfd.py
Auxiliary script for initializing the SFD module.
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

try:
    from util.misc import nested_tensor_from_videos_list, NestedTensor
except ImportError:
    print("[Warning] init_sfd.py: Failed to import util.misc; the simplified processing mode will be attempted.")

def get_kmeans_centers(features_list, n_clusters):
    """
    Run K-Means clustering
    features_list: list of numpy arrays [N, C]
    """
    if not features_list:
        return None
    # Concatenate all data: [Total_Pixels, C]
    X = np.concatenate(features_list, axis=0)

    # Number of atoms as in the paper
    print(f"  > Clustering {X.shape[0]} feature points to generate {n_clusters} centers...")
    # MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=1024, n_init=3, random_state=42)
    kmeans.fit(X)

    return kmeans.cluster_centers_

def initialize_sfd_module(model, dataloader, device, num_batches=50):
    """
    Main function: Read data -> Extract features -> K-Means -> Assign to model
    """
    print("\n" + "="*50)
    print("[SFD-Init] Starting initialization of the SFD module...")
    print(f"[SFD-Init] Using {num_batches} batches of data to warm up the dictionary.")
    print("="*50)

    # 1. Check if SFD modules exist
    if not hasattr(model, 'sfd_modules') or model.sfd_modules is None:
        print("[SFD-Init] 'sfd_modules' not found in the model, skipping initialization.")
        return

    model.eval() # Switch to evaluation mode to prevent BN layer parameter changes
    model.to(device)

    # Prepare containers to store features
    # model.sfd_modules corresponds to the last few layers of the backbone output
    num_levels = len(model.sfd_modules)
    collected_features = [[] for _ in range(num_levels)]

    print("[SFD-Init] Extracting features (Vision Model Initialization)...")
    # 2. Iterate over data to extract features
    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches:
                break

            # (samples, targets) or (samples, captions, targets)
            if len(batch_data) == 3:
                samples, captions, targets = batch_data
            elif len(batch_data) == 2:
                samples, targets = batch_data
            else:
                samples = batch_data[0]

            # Move data to GPU

            if not isinstance(samples, NestedTensor):
                try:
                    # Try to convert, referencing util
                    if isinstance(samples, list): # If it's a list of tensors
                        samples = nested_tensor_from_videos_list(samples)
                    else:
                        mask = torch.zeros((samples.shape[0], samples.shape[2], samples.shape[3]), dtype=torch.bool)
                        samples = NestedTensor(samples, mask)
                except Exception as e:

                    pass

            if hasattr(samples, 'to'):
                samples = samples.to(device)

            # VCSL.backbone returns (features, pos)

            try:
                features, _ = model.backbone(samples)
            except Exception as e:
                print(f"[SFD-Init] Error: Backbone forward pass failed: {e}")
                return

            # We only care about the last num_levels layers, corresponding to SFD modules
            # For example, if features has 4 layers and sfd_modules has 3, take the last 3
            target_features = features[-num_levels:]

            for lvl, feat_nested in enumerate(target_features):
                #  tensor: [Batch*Time, C, H, W]
                tensor = feat_nested.tensors

                # convert to [Pixels, C]
                # permute(0, 2, 3, 1) -> [B, H, W, C]
                c = tensor.shape[1]
                flat_feat = tensor.permute(0, 2, 3, 1).reshape(-1, c)

                # randomly sample 50 points per image
                if flat_feat.shape[0] > 50:
                    idx = torch.randperm(flat_feat.shape[0])[:50]
                    flat_feat = flat_feat[idx]

                collected_features[lvl].append(flat_feat.cpu().numpy())

    # 3. Clustering and assignment
    print("\n[SFD-Init] Feature extraction completed, starting clustering and updating weights...")

    for lvl, sfd_module in enumerate(model.sfd_modules):

        feats_list = collected_features[lvl]
        if not feats_list:
            print(f"[SFD-Init] Level {lvl}: No features extracted, skipping.")
            continue

        # Get the number of atoms needed
        n_atoms = sfd_module.D.conv.in_channels   # This is the atoms dimension
        dim_backbone = sfd_module.D.conv.out_channels # This is the feature dimension

        centers = get_kmeans_centers(feats_list, n_clusters=n_atoms) # Shape: [atoms, dim_backbone]

        if centers is None:
            continue



        new_w = torch.from_numpy(centers.T).float().view(dim_backbone, n_atoms, 1, 1)

        # Normalize along the feature dimension (dim=0).
        new_w = torch.nn.functional.normalize(new_w, dim=0)

        # Assignment
        sfd_module.D.conv.weight.data = new_w.to(device)
        print(f"[SFD-Init] Level {lvl}: Initialization completed (Atoms={n_atoms}, Dim={dim_backbone}).")

    print("[SFD-Init] All SFD modules initialized!\n")