import os
os.environ["PL_DISABLE_DYNAMO"] = "1"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_tcn import TCN  # pip install pytorch‑tcn :contentReference[oaicite:1]{index=1}

# ————————————————————————————————
# Streamlit UI
# ————————————————————————————————
st.title("Sliding‑Window Forecasting with TCN")

# 1) Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()
date_col   = st.selectbox("Date column",   df.columns)
target_col = st.selectbox("Target column", df.columns)

# Preview
st.write("## Data preview")
st.dataframe(df[[date_col, target_col]].head())

# Preprocess
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col).reset_index(drop=True)
series = df[target_col].values.astype(float)

# user‑selectable window & horizon
lookback = st.slider("Look‑back window size", 5, 50, 12)
horizon  = st.slider("Forecast horizon (steps)", 1, 252, 30)

# 2) Build sliding‑window dataset
X, y = [], []
for i in range(len(series) - lookback):
    X.append(series[i : i + lookback])
    y.append(series[i + lookback])
X = torch.tensor(np.stack(X), dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
dataset = TensorDataset(X, y)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# 3) Define TCN forecaster
class TCNForecaster(LightningModule):
    def __init__(self, in_channels=1, channels=[16,16,16], kernel_size=3, dropout=0.1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.tcn = TCN(
            num_inputs=in_channels,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
        )
        self.linear = torch.nn.Linear(channels[-1], 1)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        # x: (batch, seq_len) → (batch, 1, seq_len)
        x = x.unsqueeze(1)
        out = self.tcn(x)               # (batch, channel, seq_len)
        out = out[:, :, -1]             # take last timestep → (batch, channel)
        return self.linear(out).squeeze(-1)  # (batch,)

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# 4) Train Button & Control Flow
if "trained" not in st.session_state:
    st.session_state.trained = False

if not st.session_state.trained:
    if st.button("▶️ Train TCN"):
        st.session_state.trained = True
    else:
        st.info("Click **Train TCN** to start training.")
        st.stop()

with st.spinner("Training TCN model…"):
    model   = TCNForecaster()
    trainer = Trainer(
        max_epochs=10,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, loader)
st.success("Training complete!")

# 5) Iterative multi‑step forecast
window = series[-lookback:].tolist()
preds = []
for _ in range(horizon):
    x_in = torch.tensor(window[-lookback:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        next_val = model(x_in).item()
    preds.append(next_val)
    window.append(next_val)

# Build future dates
last_date = df[date_col].iloc[-1]
freq      = pd.infer_freq(df[date_col]) or "D"
future_dates = pd.date_range(last_date + pd.Timedelta(1, unit=freq[0]),
                             periods=horizon, freq=freq)

# 6) Plot history + forecast
plt.figure(figsize=(10, 5))
plt.plot(df[date_col], series, label="Historical")
plt.plot(future_dates, preds, "--", label="Forecast")
plt.xlabel("Date")
plt.ylabel(target_col)
plt.legend()
st.pyplot(plt.gcf())

# 7) In‑sample RMSE (no .numpy() at all)
from torchmetrics import MeanSquaredError

# switch to eval mode
model.eval()

# pick one: either use torchmetrics…
rmse_metric = MeanSquaredError(squared=False)   # √MSE

with torch.no_grad():
    # forward pass
    preds_tensor = model(X)                    # shape (n_samples,)
    # update metric (this also detaches internally)
    rmse_value = rmse_metric(preds_tensor, y).item()

# …or roll your own in pure PyTorch:
# with torch.no_grad():
#     preds_tensor = model(X)
#     mse = torch.mean((preds_tensor - y) ** 2)
#     rmse_value = torch.sqrt(mse).item()

st.write(f"**In‑sample RMSE**: {rmse_value:.4f}")

# 8) GenAI Trend Summary (unchanged)
st.subheader("GenAI Trend Summary")
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
hf_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    model_kwargs={"temperature":0.7, "max_new_tokens":200},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)
prompt = PromptTemplate(
    input_variables=["last_actual","first_forecast"],
    template=(
        "Last observed value: {last_actual:.2f}. "
        "First forecast step: {first_forecast:.2f}. "
        "Provide a concise, plain‑English summary of the trend."
    )
)
chain   = LLMChain(prompt=prompt, llm=hf_llm)
summary = chain.run(
    last_actual    = series[-1],
    first_forecast = preds[0]
)
st.write(summary)
