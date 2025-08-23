import torch
torch.cuda.empty_cache()
import numpy as np
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from Dataloader import FMRIDataset
from Model import FMRI2Embedding, SpatialFMRI2Embedding


dataset = FMRIDataset(
    fmri_path="../dataset/avatar.hf5",
    textgrid_path="../dataset/avatar.TextGrid"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp = SpatialFMRI2Embedding(dataset.X.shape[1], dataset.y.shape[1]).to(device)


EPOCHS = 50

# Hyperparameters for early stopping
patience = 5
min_delta = 1e-4  
best_loss = float('inf')
wait = 0

for epoch in range(EPOCHS):
    mlp.train()
    epoch_loss = 0
    total_batches = 0

    for [fmri_path, textgrid_path] in mlp.dataset_paths:
        dataset = FMRIDataset(fmri_path=fmri_path, textgrid_path=textgrid_path)

        # print("X mean:", dataset.X.mean(), "std:", dataset.X.std())
        # print("X min:", dataset.X.min(), "max:", dataset.X.max())

        for i, (xb, yb) in enumerate(dataset.train_loader):
            xb, yb = xb.to(device), yb.to(device)

            pred = mlp(xb)
            pred = F.normalize(pred, dim=-1)
            yb = F.normalize(yb, dim=-1)

            logits = torch.matmul(pred, yb.T)
            labels = torch.arange(logits.size(0), device=device)

            loss = mlp.clip_contrastive_loss(pred, yb, temperature=0.07)

            mlp.optimizer.zero_grad()
            loss.backward()
            mlp.optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

    epoch_loss /= total_batches
    mlp.scheduler.step()

    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f} | LR: {mlp.optimizer.param_groups[0]['lr']:.6f}")

    # ---- Early stopping logic ----
    if best_loss - epoch_loss > min_delta:
        best_loss = epoch_loss
        wait = 0
    else:
        wait += 1
        print(f"No improvement. Patience: {wait}/{patience}")
        if wait >= patience:
            print("Early stopping triggered.")
            break


mlp.eval()
sample_vec = torch.from_numpy(dataset.X_test[0]).unsqueeze(0).to(device)

with torch.no_grad():
    pred_embed = mlp(sample_vec).cpu().numpy()

sims = cosine_similarity(pred_embed, dataset.y_test) 
best_idx = np.argmax(sims)

print("\n Predicted sentence:")
print(dataset.test_sentences[best_idx]) 

print("\n Ground truth:")
print(dataset.test_sentences[0])  
from sklearn.metrics.pairwise import cosine_similarity

def get_top_k_similar(pred_vec, all_gt_vecs, all_gt_sentences, k=5):
    sims = cosine_similarity(pred_vec.reshape(1, -1), all_gt_vecs)[0]
    top_k = sims.argsort()[-k:][::-1]
    return [(all_gt_sentences[i], sims[i]) for i in top_k]
FMRI2Embedding
top_preds = get_top_k_similar(pred_embed, dataset.y_test, dataset.test_sentences)
for sent, score in top_preds:
    print(f"{score:.3f} | {sent}")

torch.save(mlp.state_dict(), "/home/ritik/Desktop/ML/Projects/models/spatial_mlp_model.pth")



