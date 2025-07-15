#!/usr/bin/env python3
"""
Distributed reward model inference script that loads reward-bench dataset,
scores chosen/rejected pairs, and saves the results with scores.
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import accelerate
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from rewardbench import (
    REWARD_MODEL_CONFIG,
    check_tokenizer_chat_template,
    load_eval_dataset,
    torch_dtype_mapping,
)
from transformers import pipeline, PreTrainedTokenizerFast, LlamaPreTrainedModel, LlamaModel
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import List

# Custom model class for INF-ORM-Llama3.1-70B
class INFORMForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

# Model configuration for different reward models
REWARD_MODEL_CONFIGS = {
    "infly/INF-ORM-Llama3.1-70B": {
        "model_type": "INFORM",
        "model_class": INFORMForSequenceClassification,
        "tokenizer_class": PreTrainedTokenizerFast,
        "model_kwargs": {"num_labels": 1},
        "requires_attention_mask": False,
        "trust_remote_code": True,
    },
    "weqweasdas/RM-Gemma-7B": {
        "model_type": "pipeline",
        "pipeline_task": "sentiment-analysis",
        "pipeline_kwargs": {
            "top_k": None,
            "function_to_apply": "none",
            "batch_size": 1
        },
        "remove_bos_token": True,
        "trust_remote_code": False,
    },
    "allenai/Llama-3.1-70B-Instruct-RM-RB2": {
        "model_type": "standard",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "model_kwargs": {},
        "requires_attention_mask": False,
        "trust_remote_code": False,
    },
    "Skywork/Skywork-Reward-Gemma-2-27B": {
        "model_type": "standard",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "model_kwargs": {"num_labels": 1},
        "requires_attention_mask": False,
        "trust_remote_code": False,
    },
    "nicolinho/QRM-Gemma-2-27B": {
        "model_type": "QRM",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "model_kwargs": {},
        "requires_attention_mask": False,
        "trust_remote_code": True,
    },
    "LxzGordon/URM-LLaMa-3.1-8B": {
        "model_type": "URM",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "model_kwargs": {},
        "requires_attention_mask": True,
        "trust_remote_code": True,
    },
    "default": {
        "model_type": "standard",
        "model_class": AutoModelForSequenceClassification,
        "tokenizer_class": AutoTokenizer,
        "model_kwargs": {},
        "requires_attention_mask": False,
        "trust_remote_code": False,
    }
}

# Setup accelerator
accelerator = accelerate.Accelerator()

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login
    _login(token=HF_TOKEN, add_to_git_credential=False)


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="",
        metadata={"help": "The name or path of the reward model"},
    )
    model_name: str = field(
        default="",
        metadata={"help": "Model name for selecting the appropriate pipeline (if empty, uses model_name_or_path)"},
    )
    dataset_name_or_path: str = field(
        default="",
        metadata={"help": "The name or path of the dataset to score (if empty, loads reward-bench)"},
    )
    output_path: str = field(
        default="",
        metadata={"help": "The path to save the output dataset"},
    )
    chat_template: str = field(
        default="tulu",
        metadata={"help": "Chat template to use (ignored if tokenizer has chat template)"},
    )
    max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length for tokenization"},
    )
    torch_dtype: str = field(
        default="bfloat16",
        metadata={"help": "PyTorch dtype (float16, bfloat16, float32)"},
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "Attention implementation (eager, sdpa, flash_attention_2)"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code"},
    )
    pref_sets: bool = field(
        default=False,
        metadata={"help": "Use preference sets instead of core eval set"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Debug mode - process only first 10 samples"},
    )


class RMPipeline:
    def __init__(
        self,
        model_name_or_path: str,
        model_name: str = None,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
        truncation=True,
        max_length=4096,
        trust_remote_code=False,
    ):
        attn_implementation = None
        if device_map is None:
            device_map = {"": accelerator.process_index}
            
        self.model_name_or_path = model_name_or_path
        self.model_name = model_name or model_name_or_path
        self.truncation = truncation
        self.max_length = max_length
        
        # Get model configuration
        self.config = REWARD_MODEL_CONFIGS.get(self.model_name, REWARD_MODEL_CONFIGS["default"])
        
        # Override trust_remote_code if specified in config
        if self.config.get("trust_remote_code", False):
            trust_remote_code = True
            
        self.model_type = self.config["model_type"]
        
        # Initialize model and tokenizer based on type
        if self.model_type == "pipeline":
            self._init_pipeline_model(device_map, torch_dtype, trust_remote_code)
        else:
            self._init_direct_model(device_map, torch_dtype, attn_implementation, trust_remote_code)
            
    def _init_pipeline_model(self, device_map, torch_dtype, trust_remote_code):
        """Initialize pipeline-based model (e.g., weqweasdas/RM-Gemma-7B)"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        device = accelerator.device
        
        self.rm_pipe = pipeline(
            self.config["pipeline_task"],
            model=self.model_name_or_path,
            device=device,
            tokenizer=self.tokenizer,
            model_kwargs={"torch_dtype": torch_dtype}
        )
        
        self.pipe_kwargs = self.config["pipeline_kwargs"]
        self.remove_bos_token = self.config.get("remove_bos_token", False)
        
    def _init_direct_model(self, device_map, torch_dtype, attn_implementation, trust_remote_code):
        """Initialize direct model (standard, INFORM, QRM, URM)"""
        model_class = self.config["model_class"]
        tokenizer_class = self.config["tokenizer_class"]
        model_kwargs = self.config.get("model_kwargs", {})
        
        # Prepare model loading arguments
        load_args = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            **model_kwargs
        }
        
        # Add attention implementation if not pipeline
        if attn_implementation:
            load_args["attn_implementation"] = attn_implementation
            
        # Load model and tokenizer
        self.rm = model_class.from_pretrained(self.model_name_or_path, **load_args)
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_name_or_path, 
            use_fast=True,
            trust_remote_code=trust_remote_code,
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # For models whose config did not contain pad_token_id
        if hasattr(self.rm, 'config') and self.rm.config.pad_token_id is None:
            self.rm.config.pad_token_id = self.tokenizer.pad_token_id

    def __call__(self, conversation):
        """Score a single conversation"""
        if self.model_type == "pipeline":
            return self._score_pipeline(conversation)
        else:
            return self._score_direct(conversation)
            
    def _score_pipeline(self, conversation):
        """Score using pipeline (weqweasdas/RM-Gemma-7B)"""
        if isinstance(conversation, str):
            input_text = conversation
        else:
            input_text = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
        # Remove BOS token if required
        if self.remove_bos_token:
            input_text = input_text.replace(self.tokenizer.bos_token, "")
            
        pipe_outputs = self.rm_pipe([input_text], **self.pipe_kwargs)
        score = pipe_outputs[0][0]["score"]
        return {"score": score}
        
    def _score_direct(self, conversation):
        """Score using direct model inference"""
        # Handle both string and list of messages
        if isinstance(conversation, str):
            # If it's a string, use it directly
            inputs = self.tokenizer(
                conversation,
                return_tensors="pt",
                truncation=self.truncation,
                max_length=self.max_length,
                padding=True,
            )
        else:
            # If it's a list of messages, apply chat template
            if self.model_type == "INFORM":
                # For INFORM, use apply_chat_template with tokenize=True
                inputs = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    return_tensors="pt",
                    truncation=self.truncation,
                    max_length=self.max_length,
                )
                inputs = {"input_ids": inputs}
            else:
                # For other models, format then tokenize
                formatted_text = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False
                )
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    truncation=self.truncation,
                    max_length=self.max_length,
                    padding=True,
                )
        
        # Move inputs to device
        inputs = {k: v.to(self.rm.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            if self.model_type == "QRM":
                # nicolinho/QRM-Gemma-2-27B
                output = self.rm(**inputs)
                score = output.score.cpu().float().item()
            elif self.model_type == "URM" and self.config["requires_attention_mask"]:
                # LxzGordon/URM-LLaMa-3.1-8B
                output = self.rm(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                score = output.logits[0][0].item()
            else:
                # Standard models and INFORM
                if "attention_mask" in inputs:
                    output = self.rm(inputs["input_ids"], attention_mask=inputs["attention_mask"])
                else:
                    output = self.rm(inputs["input_ids"])
                score = output.logits.view(-1).cpu().item()
                
        return {"score": score}

    @torch.no_grad()
    def get_score(self, chosen, rejected):
        """Get scores for chosen and rejected responses"""
        return [self(chosen)["score"], self(rejected)["score"]]


def setup_logging():
    """Setup logging configuration"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    return logger


def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Setup logging
    logger = setup_logging()
    
    # Convert torch_dtype string to torch dtype
    torch_dtype = torch_dtype_mapping(script_args.torch_dtype)
    
    # Determine model name for configuration
    model_name = script_args.model_name or script_args.model_name_or_path
    
    logger.info(f"Running reward model on {script_args.model_name_or_path}")
    logger.info(f"Using model name: {model_name}")
    logger.info(f"Using torch dtype: {torch_dtype}")
    
    # Get model configuration
    config = REWARD_MODEL_CONFIGS.get(model_name, REWARD_MODEL_CONFIGS["default"])
    logger.info(f"Model type: {config['model_type']}")
    
    # Load dataset
    if script_args.dataset_name_or_path:
        logger.info(f"Loading dataset from {script_args.dataset_name_or_path}")
        ds = load_from_disk(script_args.dataset_name_or_path)
        subsets = ds.get("subset", [None] * len(ds))
        ids = ds.get("id", list(range(len(ds))))
    else:
        logger.info("Loading reward-bench dataset")
        # Load tokenizer for dataset preparation
        tokenizer_class = config.get("tokenizer_class", AutoTokenizer)
        tokenizer = tokenizer_class.from_pretrained(
            script_args.model_name_or_path, 
            trust_remote_code=script_args.trust_remote_code or config.get("trust_remote_code", False)
        )
        
        # Load the eval dataset
        from fastchat.conversation import get_conv_template
        conv = get_conv_template(script_args.chat_template)
        
        # Check if tokenizer has chat template
        custom_dialogue = not check_tokenizer_chat_template(tokenizer)
        
        ds, subsets = load_eval_dataset(
            core_set=not script_args.pref_sets,
            conv=conv if custom_dialogue else None,
            custom_dialogue_formatting=custom_dialogue,
            tokenizer=tokenizer if not custom_dialogue else None,
            logger=logger,
            keep_columns=["text_chosen", "text_rejected", "id"],
        )
        
        # Extract ids
        ids = ds["id"]
        ds = ds.remove_columns("id")
    
    # Debug mode - use only first 10 samples
    if script_args.debug:
        logger.info("Debug mode: processing only first 10 samples")
        ds = ds.select(range(10))
        subsets = subsets[:10]
        ids = ids[:10]
    
    # Initialize reward model
    logger.info("Initializing reward model...")
    rm = RMPipeline(
        model_name_or_path=script_args.model_name_or_path,
        model_name=model_name,
        torch_dtype=torch_dtype,
        attn_implementation=script_args.attn_implementation,
        max_length=script_args.max_length,
        trust_remote_code=script_args.trust_remote_code,
    )
    
    # Process dataset with distributed inference
    logger.info("Starting distributed inference...")
    accelerator.wait_for_everyone()
    
    with accelerator.split_between_processes(ds) as ds_shard:
        rewards = []
        for sample in tqdm(ds_shard, desc="Processing samples"):
            try:
                chosen_score, rejected_score = rm.get_score(
                    sample["text_chosen"], 
                    sample["text_rejected"]
                )
                rewards.append({
                    "chosen_reward": chosen_score, 
                    "rejected_reward": rejected_score
                })
            except Exception as e:
                logger.error(f"Error processing sample: {e}")
                # Add placeholder scores for failed samples
                rewards.append({
                    "chosen_reward": 0.0, 
                    "rejected_reward": 0.0
                })
    
    # Gather results from all processes
    logger.info("Gathering results from all processes...")
    rewards_gathered = accelerate.utils.gather_object([{"rewards": rewards}])
    
    # Save results on main process
    if accelerator.is_main_process:
        logger.info("Saving results...")
        # Flatten the gathered results
        all_rewards = [row for result in rewards_gathered for row in result["rewards"]]
        
        # Add reward columns to dataset
        ds = ds.add_column("chosen_reward", [r["chosen_reward"] for r in all_rewards])
        ds = ds.add_column("rejected_reward", [r["rejected_reward"] for r in all_rewards])
        
        # Add back the subset and id columns
        ds = ds.add_column("subset", subsets)
        ds = ds.add_column("id", ids)
        
        # Save to disk
        ds.save_to_disk(script_args.output_path)
        logger.info(f"Results saved to {script_args.output_path}")
        
        # Print some statistics
        logger.info(f"Processed {len(all_rewards)} samples")
        chosen_scores = [r["chosen_reward"] for r in all_rewards]
        rejected_scores = [r["rejected_reward"] for r in all_rewards]
        
        accuracy = sum(1 for c, r in zip(chosen_scores, rejected_scores) if c > r) / len(chosen_scores)
        logger.info(f"Accuracy (chosen > rejected): {accuracy:.4f}")
        logger.info(f"Average chosen score: {sum(chosen_scores) / len(chosen_scores):.4f}")
        logger.info(f"Average rejected score: {sum(rejected_scores) / len(rejected_scores):.4f}")


if __name__ == "__main__":
    main() 