import torch
import h5py
import numpy as np
import pandas as pd
from textgrid import TextGrid
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset


from transformers import CLIPTokenizer, CLIPTextModel

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_device = torch.device("cpu")


class FMRIDataset():
    def __init__(self, fmri_path, textgrid_path, TR=1.5):
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to("cpu")
        self.TR = TR
        self.fmri = self._load_fmri(fmri_path)
        self.trans_df = self._load_transcript(textgrid_path)
        self.pairs = self._align_fmri_with_text()
        self.X, self.y = self._embed_pairs()
        self.train_loader, self.test_loader = self.get_dataloaders()

    def _load_fmri(self, path):
        with h5py.File(path, "r") as f:
            data = f["data"][:]
        # print("fMRI shape:", data.shape)
        return data

    def _load_transcript(self, path):
        tg = TextGrid.fromFile(path)
        word_tier = next((tier for tier in tg.tiers if "word" in tier.name.lower()), None)

        if word_tier is None:
            raise ValueError("No 'word' tier found in TextGrid.")

        transcript = [
            {
                "word": interval.mark.strip(),
                "start": interval.minTime,
                "end": interval.maxTime
            }
            for interval in word_tier.intervals
            if interval.mark and interval.mark.strip().lower() not in ["", "sp", "sil", "<unk>"]
        ]

        return pd.DataFrame(transcript)

    def _align_fmri_with_text(self):
        pairs = []
        n_TRs = self.fmri.shape[0]

        for t_idx in range(n_TRs):
            t_start = t_idx * self.TR
            t_end = t_start + self.TR

            words_in_tr = self.trans_df[
                            (self.trans_df['end'] > t_start) & 
                            (self.trans_df['start'] < t_end)
                        ]

            if not words_in_tr.empty:
                sentence = " ".join(words_in_tr['word'].tolist())
                brain_vec = self.fmri[t_idx]
                pairs.append((brain_vec, sentence))

        # print(f"Total aligned (brain, sentence) pairs: {len(pairs)}")
        return pairs

    def _embed_pairs(self):
        X = []
        sentences = []

        for brain_vec, sentence in self.pairs:
            X.append(brain_vec)
            sentences.append(sentence)

        self.sentences = sentences
        X = np.array(X, dtype=np.float32)

        inputs = self.clip_tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(clip_device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            y = outputs.pooler_output.cpu().numpy().astype(np.float32)

        y = y / np.linalg.norm(y, axis=1, keepdims=True)

        return X, y


    def get_dataloaders(self, batch_size=32, test_size=0.2, shuffle=True):
        self.X_train, self.X_test, self.y_train, self.y_test, self.train_sentences, self.test_sentences = train_test_split(
            self.X, self.y, self.sentences, test_size=test_size, random_state=42
        )

        train_dataset = TensorDataset(torch.from_numpy(self.X_train), torch.from_numpy(self.y_train))
        test_dataset = TensorDataset(torch.from_numpy(self.X_test), torch.from_numpy(self.y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        return train_loader, test_loader

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])
