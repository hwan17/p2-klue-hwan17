from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoConfig, AdamW, get_scheduler
from transformers.trainer_pt_utils import get_parameter_names

from load_data import *
from trainer import Trainer
from utils import *
import time
import wandb
import pandas as pd

from easydict import EasyDict
from prettyprinter import cpprint
from sklearn.model_selection import StratifiedKFold, train_test_split

def engine(cfg, args):
    seed_everything(cfg.values.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.values.model_name)

    ### 모델 정의
    config = AutoConfig.from_pretrained(cfg.values.model_name, num_labels = 42)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.values.model_name, config = config)

    if args.mode == 'train':
        dataset = load_data("/opt/ml/input/data/train/train.tsv")
        dataset_label = dataset['label'].values

        train_df, val_df = train_test_split(dataset, test_size = cfg.values.val_args.test_size, random_state = cfg.values.seed)

        tokenized_train = tokenized_dataset(train_df, tokenizer)
        tokenized_val = tokenized_dataset(val_df, tokenizer)

        train_dataset = RE_Dataset(tokenized_train, labels = train_df['label'].values)
        val_dataset = RE_Dataset(tokenized_val, labels = val_df['label'].values)

        ### optimizer, scheduler 정의
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': cfg.values.train_args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.values.train_args.lr,
            eps=cfg.values.train_args.adam_epsilon,
        )

        scheduler = get_scheduler(
            cfg.values.train_args.scheduler_name, optimizer, 
            num_warmup_steps = (train_dataset.__len__()// cfg.values.train_args.train_batch_size) * cfg.values.train_args.num_epochs, 
            num_training_steps = train_dataset.__len__() * cfg.values.train_args.num_epochs
            )

        best_valid_loss = float('inf')
        for epoch in range(cfg.values.train_args.num_epochs):
            start_time = time.time() # 시작 시간 기록

            trainer = Trainer(cfg, model, epoch, optimizer, scheduler, train_dataset, val_dataset)
            train_loss, train_acc = trainer.train()
            valid_loss, valid_acc = trainer.evaluate(args.mode)

            end_time = time.time() # 종료 시간 기록
            elapsed_time = end_time - start_time
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

            print(f'Time Spent : {elapsed_mins}m {elapsed_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {round(train_acc*100, 2)}%')
            print(f'\tValidation Loss: {valid_loss:.3f} | Val Acc: {round(valid_acc*100, 2)}%')

            wandb.log({"Train Acc": round(train_acc*100, 2), "Validation Acc": round(valid_acc*100, 2),
                    "Train Loss": train_loss, "Validation Loss": valid_loss})

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), f'/opt/ml/my_code/results/{cfg.values.model_name}.pt')
                print('\tBetter model found!! saving the model')
        print()
        print('='*50 + 'Training finished' + '='*50)

    elif args.mode == 'inference':
        ### load my model
        model.load_state_dict(torch.load(f"/opt/ml/my_code/results/{cfg.values.model_name}.pt"))
        # load test datset
        test_dataset_path = "/opt/ml/input/data/test/test.tsv"
        test_dataset, test_label = load_test_dataset(test_dataset_path, tokenizer)
        test_dataset = RE_Dataset(test_dataset, test_label)

        trainer = Trainer(cfg, model, test_dataset = test_dataset)
        pred_answer = trainer.evaluate(args.mode)

        output = pd.DataFrame(pred_answer, columns=['pred'])
        output.to_csv(f'/opt/ml/my_code/results/{cfg.values.model_name}_submission.csv', index=False)
        print()
        print('='*50 + 'Inference finished' + '='*50)



