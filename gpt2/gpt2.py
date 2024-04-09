from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("khmer_tokenizer")

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepare the dataset
train_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path="path_to_your_train_file.txt",
  block_size=128)

valid_dataset = TextDataset(
  tokenizer=tokenizer,
  file_path="path_to_your_validation_file.txt",
  block_size=128)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2-khmer",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=500,
    save_steps=500,
    warmup_steps=500,
    prediction_loss_only=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Start fine-tuning
trainer.train()
