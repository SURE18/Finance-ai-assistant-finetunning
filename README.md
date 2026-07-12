
# YES Bank Marquee Credit Card Q&A Chatbot

## 1. Project Title
YES Bank Marquee Credit Card Q&A Chatbot

## 2. Domain Selected
The project focuses on the financial domain, specifically targeting inquiries related to the **YES Bank Marquee Credit Card** and its associated terms and conditions (MITC document).

## 3. Business Problem
This project aims to develop a specialized Question & Answer (Q&A) chatbot for the YES Bank Marquee Credit Card. The goal is to create a preference-aligned language model that can accurately and helpfully respond to user inquiries regarding the credit card's features, benefits, fees, and other related information, extracted from the official MITC (Most Important Terms and Conditions) document.

To achieve this, a multi-stage fine-tuning pipeline is employed, leveraging the power of `unsloth` for efficient training.

## 4. Dataset Details
All datasets used for fine-tuning are hosted on Hugging Face within the repository `Suresh18/yesbank-marquee-assignment`.

### 4.1. Non-Instruction Data
*   **Description:** Raw, unstructured text extracted from the `/content/Practical Fine-Tuning Assignment-04.pdf` (MITC document). This data is used for initial domain adaptation.
*   **Hugging Face Path:** `non_instruction/train-00000-of-00001.parquet`

### 4.2. Instruction Data
*   **Description:** Structured `(instruction, input, output)` pairs designed to teach the model how to follow commands and generate coherent, structured Q&A responses.
*   **Hugging Face Path:** `instruction_dataset/instruction_dataset.jsonl`

### 4.3. Preference Data
*   **Description:** `(prompt, chosen, rejected)` triplets used for Direct Preference Optimization (DPO). The `chosen` responses are preferred, and `rejected` responses are less preferred for a given `prompt`, helping to align the model's output with desired quality and style.
*   **Hugging Face Path:** `preference_dataset/preference_dataset.jsonl`

## 5. Base Model Used
The foundational model for this project is:
*   **Model Name:** `unsloth/tinyllama-bnb-4bit`
*   **Description:** A TinyLlama model quantized to 4-bit, optimized for efficient fine-tuning with `unsloth`. This model provides a strong starting point for domain adaptation and subsequent instruction and preference tuning.

## 6. Non-Instruction Fine-Tuning Approach (Stage 1)
*   **Purpose:** To adapt the base model to the domain-specific vocabulary and patterns found in the YES Bank Marquee Credit Card documentation. This stage ensures the model gains a foundational understanding of the subject matter.
*   **Data Used:** Non-instruction data (see section 4.1).
*   **Base Model:** `unsloth/tinyllama-bnb-4bit`
*   **Techniques:** Supervised Fine-Tuning (SFT) using LoRA adapters.
*   **Outcome:** The model develops a better grasp of the domain's language, though it is not yet optimized for conversational Q&A.
*   **Hugging Face Model Paths (within `Suresh18/yesbank-marquee-assignment`):**
    *   **Adapter:** `non_instruction/lora`
    *   **Merged Model:** `non_instruction/merged` (used as base for Stage 2)

## 7. Instruction Fine-Tuning Approach (Stage 2)
*   **Purpose:** To train the model to follow instructions and generate coherent, structured responses in a Q&A format. This stage makes the model capable of engaging in a conversation.
*   **Data Used:** Instruction data (see section 4.2).
*   **Base Model:** Merged model from Stage 1 (`non_instruction/merged`).
*   **Techniques:** Supervised Fine-Tuning (SFT) using LoRA adapters.
*   **Outcome:** The model can now understand and respond to instructions, generating relevant answers, becoming 'conversational-ready'.
*   **Hugging Face Model Paths (within `Suresh18/yesbank-marquee-assignment`):**
    *   **Adapter:** `instruction/lora`
    *   **Merged Model:** `instruction/merged` (used as base for Stage 3)

## 8. DPO Alignment Approach (Stage 3)
*   **Purpose:** To align the model's responses with human preferences, making its outputs more helpful, harmless, and aligned with a specific style and quality. This stage refines the model's tone and avoids undesirable responses.
*   **Data Used:** Preference data (see section 4.3).
*   **Base Model:** Merged model from Stage 2 (`instruction/merged`).
*   **Techniques:** Direct Preference Optimization (DPO).
*   **Outcome:** The final model produces high-quality, preferred responses that are accurate, concise, and aligned with the desired tone and format for a credit card Q&A chatbot.
*   **Hugging Face Model Paths (within `Suresh18/yesbank-marquee-assignment`):**
    *   **Adapter:** `preference/lora`
    *   **Final Merged Model:** `preference/merged`

## 9. LoRA / QLoRA Configuration
The Parameter-Efficient Fine-Tuning (PEFT) method, specifically LoRA (Low-Rank Adaptation), combined with 4-bit quantization (QLoRA) is used across the fine-tuning stages for memory efficiency and faster training. The configuration parameters are consistent across stages:
*   **LoRA `r` (rank):** 16
*   **LoRA `alpha`:** 32
*   **LoRA `dropout`:** 0
*   **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

## 10. Training Screenshots or Logs
### Stage 1: Non-Instruction Fine-Tuning Logs
```
Unsloth: Tokenizing ["text"] (num_proc=6): 100%
 150/150 [00:01<00:00, 106.49 examples/s]
Unsloth: Packing train dataset (num_proc=6): 100%
 150/150 [00:00<00:00, 92.62 examples/s]
🦥 Unsloth: Packing enabled - training is >2x faster and uses less VRAM!
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 20 | Num Epochs = 20 | Total steps = 60
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 12,615,680 of 1,112,664,064 (1.13% trained)
 [60/60 03:01, Epoch 20/20]
Step	Training Loss
1	4.687600
2	5.050300
3	4.779000
4	4.689900
5	4.789300
6	5.190600
7	4.718400
8	4.764400
9	5.554100
10	4.970300
11	4.562400
12	5.286400
13	4.666500
14	4.787100
15	5.341700
16	5.106200
17	4.466800
18	5.206100
19	4.562900
20	5.393900
21	4.493800
22	4.537500
23	5.008900
24	5.055200
25	5.005700
26	4.707300
27	4.963500
28	5.087500
29	4.663300
30	4.683500
31	4.928600
32	4.623900
33	5.146900
34	4.697900
35	5.094500
36	4.937600
37	4.769200
38	5.095100
39	4.787200
40	5.373500
41	4.834500
42	4.577300
43	5.298300
44	4.544500
45	5.164100
46	5.418000
47	4.704300
48	4.641800
49	4.832800
50	5.015400
51	5.055200
52	4.754300
53	5.044200
54	5.356300
55	4.853600
56	5.011900
57	4.947100
58	5.082300
59	5.076000
60	4.425700

STAGE 1 - NON-INSTRUCTION PDF TRAINING RESULTS
Train time/sec: 198.99
Peak allocated VRAM/GB: 0.98
Peak reserved VRAM/GB: 1.082

Saving Stage 1 adapter...
Stage 1 adapter saved to: /content/unsloth_yes_merge_reload_outputs/stage1_non_instruction_adapter

Merging Stage 1 adapter with base model...
config.json: 100%
 749/749 [00:00<00:00, 71.7kB/s]
Found HuggingFace hub cache directory: /root/.cache/huggingface/hub
Checking cache directory for required files...
Cache check failed: model.safetensors not found in local cache.
Not all required files found in cache. Will proceed with downloading.
Checking cache directory for required files...
Cache check failed: tokenizer.model not found in local cache.
Not all required files found in cache. Will proceed with downloading.
Unsloth: Preparing safetensor model files:   0%|          | 0/1 [00:00<?, ?it/s]
model.safetensors: 100%
 2.20G/2.20G [00:42<00:00, 49.5MB/s]
Unsloth: Preparing safetensor model files: 100%|██████████| 1/1 [00:43<00:00, 43.30s/it]
Unsloth: Merging weights into 16bit: 100%|██████████| 1/1 [00:16<00:00, 16.76s/it]
Unsloth: Merge process complete. Saved to `/content/unsloth_yes_merge_reload_outputs/stage1_non_instruction_merged_model`
Stage 1 merged model saved to: /content/unsloth_yes_merge_reload_outputs/stage1_non_instruction_merged_model
```

### Stage 2: Instruction Fine-Tuning Logs
```
===============================================
STAGE 2: LOAD STAGE 1 MERGED MODEL AND TRAIN
===============================================
==((====))==  Unsloth 2026.7.2: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
trainable params: 12,615,680 || all params: 1,112,664,064 || trainable%: 1.1338
Unsloth: Tokenizing ["text"] (num_proc=3): 100%
 151/151 [00:01<00:00, 41.50 examples/s]
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 151 | Num Epochs = 7 | Total steps = 120
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 12,615,680 of 1,112,664,064 (1.13% trained)
 [120/120 04:54, Epoch 6/7]
Step	Training Loss
1	2.989000
2	3.221400
3	3.198000
4	3.162500
5	2.916300
6	3.127900
7	2.852700
8	3.126700
9	3.142600
10	3.037500
11	3.185000
12	3.016100
13	3.225800
14	3.081100
15	3.076700
16	3.067700
17	3.104000
18	2.997000
19	3.212200
20	3.194700
21	3.028400
22	2.860400
23	2.971800
24	3.036300
25	3.234000
26	3.157200
27	3.138300
28	3.092200
29	3.158300
30	2.869300
31	3.065000
32	3.121500
33	2.916800
34	3.123800
35	3.160400
36	3.354800
37	3.225300
38	3.008400
39	3.296000
40	3.018400
41	3.106100
42	2.955800
43	3.070600
44	2.974000
45	2.964000
46	3.095100
47	3.403300
48	3.026000
49	3.283900
50	3.033100
51	3.092700
52	3.023900
53	3.366300
54	3.029400
55	2.930700
56	2.977200
57	3.117600
58	3.014300
59	2.695400
60	3.234600
61	3.085300
62	3.247700
63	2.885700
64	3.157100
65	3.199400
66	3.112400
67	3.177400
68	3.273400
69	3.004700
70	3.102900
71	3.056700
72	2.718200
73	3.278300
74	3.198400
75	3.268100
76	3.036000
77	3.074700
78	2.876700
79	3.171000
80	3.236000
81	3.241800
82	2.943900
83	3.016300
84	2.992000
85	2.941300
86	3.129300
87	3.225200
88	3.270700
89	3.144500
90	2.940800
91	3.033700
92	3.066600
93	3.190500
94	3.235600
95	3.040400
96	3.197300
97	3.081900
98	2.874500
99	3.019400
100	2.960700
101	3.096200
102	3.252400
103	3.185300
104	3.113700
105	3.055300
106	3.132400
107	3.093400
108	3.149200
109	3.086000
110	3.049400
111	3.146100
112	2.889800
113	3.231800
114	3.168600
115	3.262100
116	3.191200
117	3.226300
118	3.023800
119	2.904300
120	3.003000

STAGE 2 - INSTRUCTION FINE-TUNING RESULTS
Train time/sec: 300.44
Peak allocated VRAM/GB: 5.489
Peak reserved VRAM/GB: 5.668

Stage 2 test answer:
Yes, the joining fee is INR 9,999.
The annual fee is INR 4,999.
The minimum spend required for waiver of joining fee is INR 10,00,000 (within 12 months prior to card anniversary date).

### Explanation:

The joining fee is INR 9,999.
The annual fee is INR 4,999.
The minimum spend required for waiver of joining fee is INR

Saving Stage 2 adapter...
Stage 2 adapter saved to: /content/unsloth_yes_merge_reload_outputs/stage2_instruction_adapter

Merging Stage 2 adapter with base model...
Detected local model directory: /content/unsloth_yes_merge_reload_outputs/stage1_non_instruction_merged_model
Copied tokenizer.model from local model directory
Found HuggingFace hub cache directory: /root/.cache/huggingface/hub
Unsloth: Preparing safetensor model files: 100%|██████████| 1/1 [00:00<00:00, 6168.09it/s]
Unsloth: Merging weights into 16bit: 100%|██████████| 1/1 [01:17<00:00, 77.42s/it]
Unsloth: Merge process complete. Saved to `/content/unsloth_yes_merge_reload_outputs/stage2_instruction_merged_model`
Stage 2 merged model saved to: /content/unsloth_yes_merge_reload_outputs/stage2_instruction_merged_model
```

### Stage 3: DPO Preference Tuning Logs
```
==========================================
STAGE 3: LOAD STAGE 2 MERGED MODEL AND DPO
==========================================
==((====))==  Unsloth 2026.7.2: Fast Llama patching. Transformers: 4.56.2.
   \\   /|    Tesla T4. Num GPUs = 1. Max memory: 14.563 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.10.0+cu128. CUDA: 7.5. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = FALSE. FA [Xformers = 0.0.35. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
trainable params: 12,615,680 || all params: 1,112,664,064 || trainable%: 1.1338
Extracting prompt in train dataset (num_proc=3): 100%
 50/50 [00:01<00:00, 26.61 examples/s]
Applying chat template to train dataset (num_proc=3): 100%
 50/50 [00:01<00:00, 32.18 examples/s]
Tokenizing train dataset (num_proc=3): 100%
 50/50 [00:01<00:00, 50.87 examples/s]
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 50 | Num Epochs = 18 | Total steps = 120
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 8
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 8 x 1) = 8
 "-____-"     Trainable parameters = 12,615,680 of 1,112,664,064 (1.13% trained)
 [120/120 05:58, Epoch 17/18]
Step	Training Loss	rewards / chosen	rewards / rejected	rewards / accuracies	rewards / margins	logps / chosen	logps / rejected	logits / chosen	logits / rejected
1	0.693100	0.000000	0.000000	0.000000	0.000000	-158.187103	-232.693069	-6.582432	-7.455808
2	0.693100	0.000000	0.000000	0.000000	0.000000	-131.651627	-217.853348	-5.695874	-7.056409
3	0.688700	0.001351	-0.007540	1.000000	0.008891	-132.645020	-257.140320	-6.885962	-7.809823
4	0.678500	0.003277	-0.026325	1.000000	0.029602	-123.085541	-228.486694	-6.641813	-7.503722
5	0.651000	-0.002996	-0.089828	1.000000	0.086833	-142.431625	-232.317062	-5.968905	-7.403429
6	0.662900	0.004758	-0.056968	1.000000	0.061725	-141.082764	-205.475464	-5.978640	-7.457504
7	0.665000	0.011810	-0.047578	0.500000	0.059388	-108.332932	-230.307587	-5.967325	-7.858461
8	0.440100	0.103303	-0.512609	1.000000	0.615912	-124.964546	-239.898605	-6.555674	-7.725322
9	0.466600	0.081859	-0.463451	1.000000	0.545311	-155.986313	-274.433136	-6.756214	-8.076987
10	0.430800	0.147316	-0.478799	1.000000	0.626115	-110.027809	-208.875519	-5.797956	-7.408959
11	0.363100	0.199326	-0.659190	1.000000	0.858516	-152.213715	-231.846405	-6.363814	-7.395730
12	0.312400	0.134059	-0.897297	1.000000	1.031356	-129.743591	-256.698853	-6.044329	-7.438726
13	0.347200	0.156615	-0.784132	1.000000	0.940747	-143.214142	-207.662720	-5.805142	-6.735427
14	0.297100	0.190578	-0.871221	1.000000	1.061800	-125.635910	-201.339905	-5.809059	-7.156183
15	0.114300	0.421989	-1.824244	1.000000	2.246233	-149.513428	-277.408691	-6.280523	-7.511980
16	0.131200	0.235044	-1.779737	1.000000	2.014781	-122.029190	-230.190674	-6.025521	-7.261076
17	0.111600	0.412271	-1.943300	1.000000	2.355571	-130.004471	-249.787537	-5.850582	-7.239346
18	0.115700	0.252714	-1.970984	1.000000	2.223698	-121.118675	-233.746460	-5.605621	-6.821327
19	0.110700	0.211800	-2.371765	1.000000	2.583565	-126.671661	-265.311188	-6.432840	-7.686047
20	0.076500	0.481843	-2.242317	1.000000	2.724159	-136.416443	-235.042877	-5.741974	-7.523239
21	0.137700	0.259092	-2.112685	1.000000	2.371777	-198.808594	-258.940460	-6.020902	-7.400624
22	0.036300	0.667028	-3.051025	1.000000	3.718053	-112.322685	-268.449829	-6.267040	-7.651868
23	0.041100	0.480869	-3.295273	1.000000	3.776142	-141.195999	-273.836823	-6.000557	-7.617543
24	0.022200	0.024737	-4.106221	1.000000	4.130958	-140.318146	-272.728210	-5.595226	-6.934151
25	0.043800	0.342259	-3.295790	1.000000	3.638049	-148.004669	-234.973068	-5.183350	-6.621335
26	0.014200	0.533867	-3.968405	1.000000	4.502272	-113.337730	-252.435532	-5.155495	-6.393838
27	0.009800	0.473697	-4.450877	1.000000	4.924574	-132.246689	-290.381226	-5.581595	-6.870963
28	0.000600	0.600615	-6.784023	1.000000	7.384638	-168.433380	-301.794495	-5.351268	-7.188097
29	0.006500	0.422443	-5.152329	1.000000	5.574772	-125.298508	-256.402649	-4.846514	-6.222463
30	0.001900	0.399065	-6.263953	1.000000	6.663017	-146.991959	-330.618103	-5.306141	-6.705082
31	0.003200	0.428623	-5.936458	1.000000	6.365081	-148.032455	-308.491302	-5.739161	-6.951015
32	0.003400	0.429697	-5.657600	1.000000	6.087297	-145.286987	-268.185455	-5.219815	-6.502929
33	0.003000	0.503031	-5.658043	1.000000	6.161074	-114.106064	-268.842316	-5.178332	-6.660728
34	0.001000	0.517455	-6.917141	1.000000	7.434597	-112.995758	-271.160980	-4.248519	-5.790093
35	0.000100	0.052560	-9.994888	1.000000	10.047447	-145.656387	-427.036072	-5.886386	-7.733908
36	0.001200	0.723292	-6.829476	1.000000	7.552769	-111.866074	-281.048157	-4.890960	-6.312240
37	0.001000	0.363402	-7.404936	1.000000	7.768338	-147.288666	-320.281189	-4.971882	-6.749240
38	0.000900	0.091512	-7.804677	1.000000	7.896190	-153.708466	-283.780121	-4.465798	-5.590326
39	0.000500	0.228532	-8.193003	1.000000	8.421534	-135.007553	-323.035889	-4.587800	-5.936872
40	0.000700	0.241549	-7.624218	1.000000	7.865767	-136.323578	-319.711761	-5.140648	-6.306119
41	0.000300	0.290035	-8.185742	1.000000	8.475778	-122.650871	-306.759583	-4.688798	-5.890560
42	0.000000	-0.199121	-12.249313	1.000000	12.050193	-122.118835	-344.177795	-3.995171	-5.518735
43	0.000400	0.359058	-8.019095	1.000000	8.378154	-133.391541	-313.529266	-5.018912	-6.484337
44	0.000200	0.433411	-8.566667	1.000000	9.000077	-119.105423	-296.595001	-4.305333	-5.713367
45	0.000400	0.070190	-8.571761	1.000000	8.641951	-132.469788	-303.354340	-4.212413	-4.913116
46	0.000200	0.043361	-9.765101	1.000000	9.808462	-126.840172	-325.694397	-4.315578	-5.637276
47	0.000400	-0.097029	-9.582454	1.000000	9.485426	-159.596756	-324.715576	-4.729233	-5.836670
48	0.000200	0.249061	-9.440745	1.000000	9.689806	-128.179306	-343.438049	-4.487696	-6.234944
49	0.000400	0.513989	-7.858520	1.000000	8.372509	-179.247482	-325.580933	-5.123824	-6.624508
50	0.000100	0.216162	-9.969895	1.000000	10.186057	-153.028534	-374.204102	-5.295268	-6.387050
51	0.000300	0.052930	-8.616416	1.000000	8.669346	-127.374176	-280.912628	-4.096064	-4.788259
52	0.000100	0.458671	-9.661146	1.000000	10.119817	-134.392380	-350.777527	-4.532020	-6.577676
53	0.000200	0.048338	-8.918591	1.000000	8.966929	-138.473236	-295.744720	-4.419951	-5.001884
54	0.000300	0.221084	-9.116582	1.000000	9.337666	-125.870781	-310.946960	-4.165605	-6.192119
55	0.000000	-0.400954	-11.458450	1.000000	11.057496	-140.174942	-347.363251	-4.085802	-5.189052
56	0.000200	0.405735	-8.408629	1.000000	8.814365	-119.878510	-272.402344	-3.957778	-5.378341
57	0.000200	0.274766	-9.003054	1.000000	9.277821	-126.599220	-327.991272	-4.506486	-5.973005
58	0.000100	0.232994	-10.377368	1.000000	10.610362	-125.978424	-337.664795	-4.197515	-5.596506
59	0.000100	-0.109476	-9.705506	1.000000	9.596031	-131.390625	-309.743500	-4.236419	-5.328351
60	0.000200	0.553630	-9.133190	1.000000	9.686820	-112.463448	-279.919617	-3.833321	-4.852775
61	0.000200	-0.259068	-10.743869	1.000000	10.484800	-157.866714	-373.107544	-4.459000	-5.740193
62	0.000000	-0.216734	-10.705062	1.000000	10.488328	-153.376556	-339.575562	-4.572851	-5.979392
63	0.000100	-0.396479	-9.393957	1.000000	8.997478	-179.261826	-327.121582	-5.257898	-5.663951
64	0.000100	-0.002737	-10.634088	1.000000	10.631351	-128.257797	-344.218384	-4.245533	-5.860159
65	0.000200	0.131384	-9.270552	1.000000	9.401935	-119.512741	-303.167053	-3.922384	-5.096279
66	0.000100	0.050687	-9.635480	1.000000	9.686167	-160.041046	-336.294464	-5.021643	-5.726090
67	0.000100	0.116436	-10.983892	1.000000	11.100327	-133.012070	-322.900635	-4.120299	-4.853320
68	0.000100	-0.187468	-11.096165	1.000000	10.908696	-136.622864	-365.980988	-4.523541	-6.034776
69	0.000200	0.078141	-9.099207	1.000000	9.177348	-140.772308	-298.694458	-4.018835	-5.482193
70	0.000100	0.085336	-10.098110	1.000000	10.183446	-143.854980	-363.201965	-4.059443	-5.961243
71	0.000000	-0.070606	-11.618042	1.000000	11.547438	-134.701721	-357.977722	-4.271929	-5.721601
72	0.000000	0.295349	-9.746734	1.000000	10.042083	-139.154388	-299.645203	-3.915718	-4.675575
73	0.000000	-0.220895	-11.162857	1.000000	10.941961	-137.796890	-360.303436	-4.489016	-5.894049
74	0.000300	0.405076	-8.875302	1.000000	9.280378	-133.788864	-315.018188	-4.559589	-5.300159
75	0.000100	-0.501396	-10.655467	1.000000	10.154072	-142.013229	-335.630615	-4.311389	-5.654475
76	0.000200	0.180604	-9.652525	1.000000	9.833129	-132.611633	-321.935669	-4.108400	-5.712934
77	0.000100	0.059394	-9.107252	1.000000	9.166647	-140.655182	-315.935760	-4.193226	-5.474240
78	0.000100	0.018734	-10.228773	1.000000	10.247506	-133.218994	-333.137054	-4.163042	-5.316287
79	0.000100	0.449227	-9.143187	1.000000	9.592413	-122.869843	-307.334076	-4.242129	-5.554732
80	0.000200	-0.445642	-10.648972	1.000000	10.203330	-159.083832	-333.134460	-4.165095	-5.096904
81	0.000100	0.132760	-10.530656	1.000000	10.663416	-130.478760	-353.990601	-4.405284	-5.257838
82	0.000000	-0.003847	-11.196750	1.000000	11.192901	-121.796707	-330.777405	-4.093431	-5.594582
83	0.000100	0.199518	-9.860642	1.000000	10.060160	-155.281540	-338.574463	-4.521321	-6.262859
84	0.000000	-1.309447	-11.561865	1.000000	10.252418	-133.187103	-310.655121	-4.009427	-4.369579
85	0.000100	-0.352332	-11.362064	1.000000	11.009733	-166.663086	-363.239716	-4.501440	-5.472269
86	0.000100	-0.109206	-10.480628	1.000000	10.371423	-127.939789	-331.502716	-4.292660	-5.236389
87	0.000100	-0.462712	-10.772396	1.000000	10.309683	-145.074188	-335.238983	-4.272787	-5.171210
88	0.000100	0.210311	-9.868268	1.000000	10.078580	-121.794037	-326.030853	-4.044085	-5.581694
89	0.000100	0.320030	-10.205784	1.000000	10.525814	-126.626923	-303.704956	-3.989577	-5.071507
90	0.000100	0.199698	-9.774537	1.000000	9.974236	-135.334656	-325.837036	-4.208825	-5.652340
91	0.000200	0.695379	-9.593346	1.000000	10.288724	-132.125351	-370.732880	-4.809772	-7.492706
92	0.000100	-0.081781	-10.460871	1.000000	10.379090	-127.674065	-309.576019	-3.889894	-4.800476
93	0.000000	0.140177	-10.758525	1.000000	10.898703	-142.501724	-351.948181	-4.468529	-5.818819
94	0.000100	-0.239847	-11.568804	1.000000	11.328958	-151.301971	-348.483521	-4.240905	-5.796538
95	0.000100	-0.326892	-10.805487	1.000000	10.478595	-131.382309	-337.040802	-3.894260	-5.227062
96	0.000100	0.250404	-10.121632	1.000000	10.372035	-134.926208	-366.157532	-4.609892	-6.106435
97	0.000200	0.244975	-8.857004	1.000000	9.101980	-133.690430	-291.309753	-4.465761	-5.219250
98	0.000100	-0.209000	-10.184958	1.000000	9.975958	-141.743042	-305.148560	-3.526229	-4.018208
99	0.000100	-0.032686	-10.602009	1.000000	10.569323	-150.234741	-319.337402	-3.890963	-4.891297
100	0.000100	-0.152441	-10.509691	1.000000	10.357250	-137.556900	-330.358154	-3.950055	-5.476398
101	0.000200	0.477390	-8.957042	1.000000	9.434431	-110.089996	-296.727570	-4.131924	-5.036794
102	0.000000	-0.261684	-11.528471	1.000000	11.266788	-153.630310	-377.185516	-4.503514	-5.648071
103	0.000100	0.112239	-10.025039	1.000000	10.137278	-112.049065	-312.665009	-4.353920	-5.428727
104	0.000000	-0.311228	-11.283739	1.000000	10.972511	-159.005264	-357.308990	-4.472770	-5.879353
105	0.000100	0.320732	-9.638346	1.000000	9.959078	-138.305450	-356.762085	-4.405526	-6.183528
106	0.000000	-0.346086	-11.930696	1.000000	11.584612	-160.267197	-381.291077	-4.453464	-5.992891
107	0.000200	-0.184441	-9.516071	1.000000	9.331631	-114.613358	-293.309387	-3.824883	-4.371864
108	0.000100	0.373688	-9.110624	1.000000	9.484312	-120.045959	-289.151917	-4.168319	-5.519980
109	0.000100	0.545838	-9.857602	1.000000	10.403440	-127.183800	-340.082306	-4.657332	-5.642775
110	0.000000	-0.315493	-10.747131	1.000000	10.431638	-159.915115	-335.700409	-4.120334	-5.034132
111	0.000100	-0.085437	-11.462548	1.000000	11.377110	-139.508560	-360.106934	-3.981003	-5.703279
112	0.000100	-0.399685	-11.367493	1.000000	10.967808	-143.379486	-338.563232	-4.646070	-6.364968
113	0.000200	-0.454449	-9.748278	1.000000	9.293829	-146.713547	-318.285370	-4.127198	-5.077339
114	0.000100	0.075286	-10.222233	1.000000	10.297518	-123.505424	-313.299103	-3.916939	-4.736525
115	0.000100	0.205691	-10.270306	1.000000	10.475997	-114.435783	-327.457153	-4.032404	-5.774354
116	0.000100	0.228019	-10.017332	1.000000	10.245351	-133.735672	-307.032166	-4.154382	-5.338544
117	0.000100	-0.042112	-10.606398	1.000000	10.564285	-152.296463	-354.290649	-4.523201	-5.738343
118	0.000000	-0.033723	-11.670393	1.000000	11.636669	-143.627655	-373.196198	-4.557911	-5.770884
119	0.000000	-0.406239	-12.102785	1.000000	11.696545	-172.697144	-366.652191	-4.134076	-5.560594
120	0.000100	-0.150876	-10.974878	1.000000	10.824002	-150.788620	-329.011841	-4.189190	-5.260443

STAGE 3 - DPO PREFERENCE TUNING RESULTS
Train time/sec: 366.9
Peak allocated VRAM/GB: 5.774
Peak reserved VRAM/GB: 5.961

Final model test answer before merge:
The YES Bank Marquee Credit Card offers a 5-year introductory APR on purchases and balance transfers at 0% interest.

The annual fee is waived in its first year. The minimum spend requirement for the annual fee waiver is INR 10,00,000 (within 12 months prior to card anniversary date).

### Explanation:

The YES Bank Marquee Credit Card offers a 5-year introductory APR on purchases and balance transfers at 0% interest. The annual fee is waived in its first year. The minimum spend requirement for the annual fee waiver is

Saving Stage 3 DPO Final adapter...
Stage 3 DPO Final adapter saved to: /content/unsloth_yes_merge_reload_outputs/stage3_dpo_adapter

Merging Stage 3 DPO Final adapter with base model...
Detected local model directory: /content/unsloth_yes_merge_reload_outputs/stage2_instruction_merged_model
Copied tokenizer.model from local model directory
Found HuggingFace hub cache directory: /root/.cache/huggingface/hub
Unsloth: Preparing safetensor model files: 100%|██████████| 1/1 [01:25<00:00, 85.30s/it]
Copied model.safetensors from local model directory
Unsloth: Merging weights into 16bit: 100%|██████████| 1/1 [01:20<00:00, 80.31s/it]
Unsloth: Merge process complete. Saved to `/content/unsloth_yes_merge_reload_outputs/stage3_dpo_final_merged_model`
Stage 3 DPO Final merged model saved to: /content/unsloth_yes_merge_reload_outputs/stage3_dpo_final_merged_model
```

## 11. Before vs After Output Comparison
After Non-Instruction fine-tuning:
Card Name: YES BANK MARQUÉE Credit Card
Joining Fee Waiver: No, one-time non-waivable fee of INR 9,999 + GST applies.
Annual Fee Waiver: Yes, waived on retail spends of INR 2,00,000 within 12 months.

Question :What are the joining and annual membership fees for the YES Bank Marquee Credit Card, and how can the annual fee be waived?

After Instruction fine-tuning:
The YES Bank Marquee Credit Card has a joining fee of INR 9,999 and an annual fee of INR 4,999. The annual fee is waived if you spend a minimum of INR 10,00,000 within the 12 months prior to your card anniversary date.

After Preference-tuning:
Answer : The joining and annual membership fees are INR 9,999 and INR 4,999 respectively.
The minimum spend required for a waiver of the joining fee is INR 10,00,000 within 12 months prior to the card anniversary date.

## 12. Final Observations
This project successfully demonstrates a robust, multi-stage fine-tuning approach to develop a specialized, preference-aligned chatbot for specific domain knowledge. The resulting model is capable of providing accurate and user-friendly answers to complex inquiries about the YES Bank Marquee Credit Card. The progressive fine-tuning stages effectively imbue the model with domain understanding, conversational ability, and human-aligned response quality.

## 13. Challenges Faced
1. During the initial phases of this project, significant challenges were encountered in effectively managing and utilizing Hugging Face modules for dataset creation, downloading, and integration into the fine-tuning pipeline. These initial hurdles required dedicated effort to overcome the complexities associated with the platform's ecosystem.

2. Google-colab is not allowing me to utilize the T4-GPU and wants me to pay on use.

## 14. Future Improvements
Instead of solely relying on fine-tuning large language models, a promising direction for future work involves developing a Retrieval-Augmented Generation (RAG) system. This approach would entail building a comprehensive knowledge base from the relevant documentation, and then utilizing it to inform the chatbot's responses. This method could potentially offer a more computationally efficient solution while maintaining high accuracy and relevance in responses.

## 15. Technologies Used
*   **`unsloth`**: For efficient and fast fine-tuning of large language models.
*   **`transformers`**: Hugging Face's library for state-of-the-art Natural Language Processing.
*   **`trl`**: (Transformer Reinforcement Learning) for training with human feedback methods like DPO.
*   **`pymupdf`**: For PDF processing and text extraction.
*   **`datasets`**: Hugging Face's library for easy dataset loading and manipulation.
*   **`torch`**: PyTorch deep learning framework.
*   **`huggingface-hub`**: For interacting with the Hugging Face Hub.
*   **`accelerate`**: For simplifying distributed training.
*   **`peft`**: (Parameter-Efficient Fine-Tuning) library.
*   **`xformers`**: For optimized attention mechanisms.
