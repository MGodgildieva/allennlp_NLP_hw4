local word_embedding_dim = 3;
local char_embedding_dim = 3;
local embedding_dim = 6;
local hidden_dim = 6;
local num_epochs = 100;
local patience = 10;
local batch_size = 5;
local learning_rate = 0.1;
{
  "dataset_reader": {
    "type": "name-reader"},
  "train_data_path": "../data/names/*.txt",
  "validation_data_path": "../data/names/*.txt",
  "model": {
    "type": "name-classifier",
        "word_embeddings": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": word_embedding_dim
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": char_embedding_dim,
                    },
                    "encoder": {
                        "type": "lstm",
                        "input_size": char_embedding_dim,
                        "hidden_size": char_embedding_dim
                    }
                }
            },
        },
        "encoder": {
            "type": "lstm",
            "input_size": embedding_dim,
            "hidden_size": hidden_dim
        }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"], ["token_characters", "num_token_characters"]],
    "batch_size": batch_size
  },
  "trainer": {
    "num_epochs": num_epochs,
    "patience": patience,
    "cuda_device": -1,
    "optimizer": {
            "type": "sgd",
            "lr": learning_rate
        },
  }
}
