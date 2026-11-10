# Hugging Face publication — release checklist

**Preconditions:** publish the GitHub repository first, and review `docs/huggingface/MODEL_CARD.md` for accuracy, project URL and a license decision. The author has already trained, evaluated, reloaded, and reproduced predictions from the local model artifact. The model is experimental.

## 1. Stage only approved public files (PowerShell)

From the repository root:

```powershell
$stage = Join-Path (Get-Location).Path "hf_release"
New-Item -ItemType Directory -Force -Path $stage | Out-Null
Copy-Item "docs/huggingface/MODEL_CARD.md" "$stage/README.md" -Force
Copy-Item "artifacts/isolation_forest.skops" "$stage/isolation_forest.skops" -Force
Copy-Item "artifacts/experiment.json" "$stage/experiment.json" -Force
Copy-Item "docs/results/isolation_forest.png" "$stage/isolation_forest.png" -Force
Copy-Item "docs/results/rolling_median_baseline.png" "$stage/rolling_median_baseline.png" -Force
Get-ChildItem $stage
Get-FileHash "$stage/isolation_forest.skops" -Algorithm SHA256
```

**Protect Git:** add `hf_release/` to `.gitignore` *before* `git add .`. Do not put API keys or Hugging Face access tokens into this directory.

## 2. Authenticate and upload

Create an account at https://huggingface.co/, generate a **write** access token via account settings, and authenticate interactively. Never paste your token into PowerShell history or a public issue.

```powershell
hf auth login
$hfUser = Read-Host "Your Hugging Face username"
$repoId = "$hfUser/service-anomaly-detector"
hf upload $repoId $stage . --repo-type model --commit-message "Publish experimental NAB anomaly detector"
```

The `hf upload` command may create the repository if it does not already exist. You can instead create a private model repository first on Hugging Face and then upload. **Check the repository visibility** before sharing the link publicly.

## 3. Verify published content

Confirm the Hugging Face page displays the Model Card and contains `isolation_forest.skops` and `experiment.json`. Share the model repository URL in the GitHub README only after verifying the upload.

## Publication caveats

- Uploading an artifact to the Hub does **not** deploy an inference endpoint.
- A `.skops` artifact requires **explicit trusted-type review** to load. The `sklearn.tree._tree.Tree` object is untrusted by default for documented memory-safety reasons.
- Re-training and/or a new model version are needed if input features, sampling interval, historical window or scikit-learn version change.
- The two NAB anomaly windows are too few to support broad performance claims. No official NAB score has been computed.
- This guide prepares publication, not automatic re-training or a production service.

Official references: [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) · [Model Cards](https://huggingface.co/docs/hub/model-cards) · [skops secure persistence](https://skops.readthedocs.io/en/stable/persistence.html).
