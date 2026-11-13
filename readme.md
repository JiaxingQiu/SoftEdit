## SoftEdit: One-Shot Calibration via Model Editing

*Under construction until acceptance.*

> [[Dataset]](https://huggingface.co/datasets/JJoy333/RankedVQA)

### Requirements

```bash
git clone https://github.com/JiaxingQiu/SoftEdit.git
cd SoftEdit
conda create -n your_venv python=3.10 -y
conda activate your_venv
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Usage

Edit a VLM on VQAv2:

```bash
cd softedit_vlm
python -m revlm.run.edit --editor softedit --model_name qwen3_4b --dataset_name vqav2
```

- Editors: `softedit`, `ft`, `mend`, `grace`, `balancedit`, `reasonedit`, `ike`, `baseline`
- Calibrators: `ts_soft`, `bc_soft`, `ats_soft`, `ts_ret_soft`, `bc_ret_soft`
- Models: `qwen3_4b`, `llava`, `blip`
- Dataset: `vqav2`

Edit an LLM on ProtoQA:

```bash
cd softedit_llm
python -m rellm.run.edit --editor softedit --model_name llama --dataset_name protoqa
```

- Editors: `softedit`, `ft`, `mend`, `grace`, `alphaedit`, `ike`, `baseline`
- Calibrators: `ts_soft`, `bc_soft`, `ats_soft`, `ts_ret_soft`, `bc_ret_soft`
- Models: `llama`, `mistral`, `qwen`
- Dataset: `protoqa`

`*_soft` uses the full answer distribution. `ft_retrain`, `mend_retrain`, `ts_soft`, `bc_soft`, and `ats_soft` take `--no-sequential`.

Table 1 scores from a saved eval file: `notebooks/main_table.ipynb`.
