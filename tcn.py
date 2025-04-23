import os
os.environ["PL_DISABLE_DYNAMO"] = "1"  # Disable TorchDynamo compilation

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_tcn import TCN

# ————————————————————————————————
# Streamlit UI
# ————————————————————————————————
st.title("Sliding-Window Forecasting with TCN + GenAI Trend Debug")

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

# user-selectable window & horizon
lookback = st.slider("Look-back window size", 5, 50, 12)
horizon  = st.slider("Forecast horizon (steps)", 1, 252, 30)

# 2) Build sliding-window dataset
X_list, y_list = [], []
for i in range(len(series) - lookback):
    X_list.append(series[i : i + lookback])
    y_list.append(series[i + lookback])
X = torch.tensor(np.stack(X_list), dtype=torch.float32)
y = torch.tensor(y_list, dtype=torch.float32)
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
        x = x.unsqueeze(1)                # (batch, 1, seq_len)
        out = self.tcn(x)                 # (batch, channel, seq_len)
        out = out[:, :, -1]               # (batch, channel)
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

# 5) Iterative multi-step forecast
window = series[-lookback:].tolist()
preds = []
model.eval()
with torch.no_grad():
    for _ in range(horizon):
        x_in = torch.tensor(window[-lookback:], dtype=torch.float32).unsqueeze(0)
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

# 7) In-sample RMSE (pure PyTorch)
model.eval()
with torch.no_grad():
    preds_tensor = model(X)
    mse   = torch.mean((preds_tensor - y) ** 2)
    rmse  = torch.sqrt(mse).item()
st.write(f"**In-sample RMSE**: {rmse:.4f}")

# ——————————————————————————————————————————————————————————
# GenAI Trend Summary (debugging token & HTTP error)
# ——————————————————————————————————————————————————————————

# 1) List loaded secrets
st.write("All secrets keys:", list(st.secrets.keys()))

# 2) Retrieve and mask token
token = st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
st.write("Token present?", bool(token))
if token:
    st.write("Token prefix (first 8 chars):", token[:8] + "…")
else:
    st.error("No HuggingFace token found. Check .streamlit/secrets.toml.")
    st.stop()

# 3) Initialize the HuggingFaceHub LLM
from langchain.llms import HuggingFaceHub
hf_llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    task="text-generation",
    huggingfacehub_api_token=token,
    model_kwargs={"temperature":0.7, "max_new_tokens":200},
)

# 4) Build prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["last_actual","first_forecast"],
    template=(
        "Last observed value: {last_actual:.2f}. "
        "First forecast step: {first_forecast:.2f}. "
        "Provide a concise, plain-English summary of the trend."
    )
)
chain = LLMChain(prompt=prompt, llm=hf_llm)

# 5) Call the chain with error capture
try:
    summary = chain.run(
        last_actual    = df[target_col].iloc[-1],
        first_forecast = preds[0]
    )
    st.write(summary)
except Exception as e:
    st.error(f"HuggingFace API call failed: {e.__class__.__name__}: {e}")
    st.stop()
