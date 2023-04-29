# FactHunt: Retrieval-driven Fact Verification for Accurate Conclusions

### File Structure
- data: For Data Files
- outputs: Contains the log files of output
- src: Contains the code

### Data
Download data folder from [here](https://drive.google.com/drive/folders/1gLDT0aAMYGl9TZy6GxjigsTvDMkTePvx?usp=sharing).

### How to Run
- Prepare Data (Includes relevant row retrieval) \
`python prepare_data.py --selector contriever --parse_rows --parse_columns --output_path <path1>`

- Tokenise and pre-process for BERT \
`python tokenise_data.py --input_path <path1> --bert_model xlm-roberta-base`

- Train Model \
`python train_model.py --do_train --scan horizontal --bert_model xlm-roberta-base --input_save_dir <path1>`

### Authors
1. Mayank Kumar (19CS30029)
2. Shrinivas Khiste (19CS30043)
3. Ishan Goel (19CS30052)
4. Ashish Gupta (19IE10010)