# Fine-tuning Explanation for Assignment

## 1. Why full fine-tuning is expensive
Full fine-tuning involves updating all parameters of a large pre-trained model. This requires significant computational resources (high-end GPUs, large amounts of VRAM) and time, as the model size can be in billions of parameters. For instance, a 7B parameter model would require over 28GB of VRAM just to store the model weights in float32, let alone gradients and optimizer states.

## 2. What LoRA does
LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning (PEFT) technique. Instead of fine-tuning all parameters, LoRA injects small, trainable low-rank matrices into each layer of the pre-trained model. This drastically reduces the number of trainable parameters, making fine-tuning more efficient in terms of memory and computation, while maintaining or even improving performance. The base model weights remain frozen, and only the small LoRA adapters are trained.

## 3. What QLoRA does
QLoRA (Quantized LoRA) extends LoRA by quantizing the pre-trained model weights to 4-bit precision (specifically, 4-bit NormalFloat, NF4) during fine-tuning, while still using LoRA adapters for training. This further reduces the memory footprint of the base model, allowing for fine-tuning of very large models (e.g., 65B parameters) on more modest hardware.

## 4. Why QLoRA is useful on limited GPU
QLoRA's ability to quantize the base model to 4-bit significantly reduces the VRAM required to load and train large models. For example, a 7B parameter model that would take ~14GB in 8-bit can be loaded in ~3.5GB using 4-bit QLoRA. This makes it possible to fine-tune models with billions of parameters on consumer-grade GPUs or cloud instances with limited VRAM (like Colab's T4 GPUs), which would otherwise be impossible with full fine-tuning or even standard LoRA.

## 5. What is non-instruction fine-tuning?
Non-instruction fine-tuning, also known as continued pre-training or domain adaptation, involves training a pre-trained language model on a new dataset without explicit instructions. The goal is to adapt the model's knowledge to a specific domain or style of text. It teaches the model *what* to say based on the patterns in the new data, rather than *how* to follow instructions. During inference, it often requires 'few-shot' prompting where examples guide the model's output.
	### Stage 1: Non-Instruction Continued Pre-training (SFT on Raw Text)
	*   **Purpose:** To adapt the base model to the domain-specific vocabulary and patterns found in the YES Bank documentation, thereby enhancing its foundational understanding of the subject matter.
	*   **Data Used:**
		*   Source: `/content/Practical Fine-Tuning Assignment-04.pdf` (or similar raw text data).
		*   Description: Unstructured text extracted from relevant documents, containing facts and information about the credit card.
	*   **Techniques:**
		*   Supervised Fine-Tuning (SFT) on the raw text data.
		*   Model: `unsloth/tinyllama-bnb-4bit` (base model).
		*   LoRA configuration (R, Alpha, Dropout).
		*   Training parameters (batch size, learning rate, steps).
	*   **Expected Outcome:** The model gains a better grasp of the domain's language, but it's not yet capable of conversational Q&A.


## 6. What is instruction fine-tuning?
Instruction fine-tuning (IFT) trains a model to follow natural language instructions. The training data consists of examples where an instruction and an input are paired with a desired output (e.g., "Summarize this text: [input text] -> [summary]"). This stage makes the model conversational and capable of performing various tasks by understanding instructions, improving its ability to generalize to new, unseen instructions.
	### Stage 2: Instruction Fine-Tuning (SFT on Instruction Data)
	*   **Purpose:** To train the model to follow instructions and generate coherent, structured responses in a Q&A format. This stage makes the model "conversational ready."
	*   **Data Used:**
		*   Source: `yesbank_instruction_dataset.jsonl`.
		*   Description: Structured `(instruction, input, output)` pairs, guiding the model on how to answer specific questions based on given context.
	*   **Techniques:**
		*   Supervised Fine-Tuning (SFT) on instruction-formatted data.
		*   Model: Merged model from Stage 1 (`stage1_non_instruction_merged_model`).
		*   LoRA configuration (re-initialized or continued from Stage 1).
		*   Training parameters (batch size, learning rate, steps).
	*   **Expected Outcome:** The model can now understand and respond to instructions, generating relevant answers, but the quality and style of responses might not yet be optimal or aligned with specific preferences.


## 7. What is DPO?
DPO (Direct Preference Optimization) is a method for aligning language models with human preferences. Unlike traditional reinforcement learning from human feedback (RLHF) methods like PPO, DPO directly optimizes a policy to maximize the probability of chosen responses over rejected responses, given a prompt. It simplifies the alignment process by avoiding the need for a reward model and complex RL algorithms, making it more stable and computationally efficient.
	## Stage 3: Direct Preference Optimization (DPO)
	*   **Purpose:** To align the model's responses with human preferences, making its outputs more helpful, harmless, and honest (or specific quality criteria). This stage refines the model's style and avoids undesirable responses.
	*   **Data Used:**
		*   Source: `yesbank_preference_dataset.jsonl`.
		*   Description: `(prompt, chosen, rejected)` triplets, where `chosen` is a preferred response and `rejected` is a less preferred one for the same `prompt`.
	*   **Techniques:**
		*   Direct Preference Optimization (DPO).
		*   Model: Merged model from Stage 2 (`stage2_instruction_merged_model`).
		*   DPO-specific parameters (beta, max length).
		*   Training parameters (batch size, learning rate, steps).
	*   **Expected Outcome:** The final model produces high-quality, preferred responses that are accurate, concise, and aligned with the desired tone and format.


## 8. Difference between SFT and DPO
*   **SFT (Supervised Fine-Tuning)**: Primarily focuses on making the model follow instructions and produce relevant, coherent responses based on direct examples. It trains the model on structured datasets of (instruction, input, desired output) pairs. The goal is to teach the model *what* to say given an instruction.
*   **DPO (Direct Preference Optimization)**: Builds upon SFT. It refines the model's responses based on human preferences, ensuring the model's outputs are not only correct but also preferred (e.g., more helpful, safer, less biased). It trains on triplets of (prompt, chosen response, rejected response), directly optimizing the model to prefer the 'chosen' over the 'rejected' answer. The goal is to teach the model *how* to say it in a preferred way.

## 9. Parameter values used in the assignment:
*   **LoRA Rank (`LORA_R`):** `16`
*   **LoRA Alpha (`LORA_ALPHA`):** `32`
*   **LoRA Dropout (`LORA_DROPOUT`):** `0`
*   **Learning Rates:**
    *   Stage 1 (`STAGE1_LR`): `0.0002`
    *   Stage 2 (`STAGE2_LR`): `0.0001`
    *   Stage 3 (`STAGE3_LR`): `5e-05`
*   **Batch Size (`BATCH_SIZE`):** `1` (per device)

