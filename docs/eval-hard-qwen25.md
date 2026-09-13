# Hard eval appendix: Qwen2.5-3B and 14B on an RTX 3080 12 GB

Live independent-turn plate, greedy, `max_new_tokens = 256`, isolated workers.
Order: BF16 3B → BF16 14B → NF4 3B → NF4 14B. History protocol was not run.
Source run: `C:\dev\models\runs\hard-matrix-20260913-1430` (copied under
[`docs/runs/hard-qwen25/`](runs/hard-qwen25/)). This is not the smoke script
(Paris / Berlin / 323) and not a fixture.

![Hard eval plate: CUDA working set, nvidia-smi peak, TTFT, decode tok/s, accuracy, and the scored item table](img/hard-eval-qwen25.png)

*Figure. Hard eval on one RTX 3080 12 GB — Qwen2.5-3B-Instruct and
Qwen2.5-14B-Instruct, dense BF16 against the NF4 driver. Panels A–E are
session totals; panel F is the auto-scored item table. The same image sits
in the README. Full questions and replies are below, not in the figure.*

## Wall time per model

Prefill is time to the first generated token (prompt processing). Decode is
the rest of the reply. Wall = load + all prefills + all decodes in that
isolated process. It does not include waiting for the previous worker to exit.

| Run | Load | Prefill (12 items) | Decode (12 items) | Wall | Tokens out | Mean TTFT | Mean decode | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3B BF16 | 7.71 s | 0.67 s | 1.74 min (104.5 s) | **1.88 min (112.9 s)** | 2319 | 56 ms | 22.05 tok/s | 7/12 |
| 14B BF16 | 42.55 s | 34.38 s | 100.51 min (6030.4 s) | **101.79 min (6107.4 s)** | 2085 | 2865 ms | 0.66 tok/s | 9/12 |
| 3B NF4 | 9.97 s | 5.28 s | 2.18 min (130.7 s) | **2.43 min (146.0 s)** | 2210 | 440 ms | 16.90 tok/s | 8/12 |
| 14B NF4 | 15.79 s | 18.95 s | 5.19 min (311.6 s) | **5.77 min (346.4 s)** | 2066 | 1579 ms | 6.59 tok/s | 10/12 |

Sum of the four walls: **111.88 min (6712.6 s)**. The 14B BF16 wall is almost
entirely decode: 0.66 tok/s because the CUDA working set is 28,490 MiB on a
12,288 MiB card (Windows shared GPU memory). That is not a hang.

## What broke, where, and why

Scoring is fail-closed: GSM8K takes `#### N`, else a trailing “the answer is”,
else the last number; yes/no takes the last Yes/No. `notes = number miss` means
the extracted number is not the gold. No session crashed; `quality_all_ok` is
false on every run because at least one item missed.

| Item | Gold | 3B BF16 | 14B BF16 | 3B NF4 | 14B NF4 | Why |
|---|---:|---|---|---|---|---|
| `gsm8k-lamps` | 164 | miss `4` | ok `164` | ok `164` | ok `164` | 3B BF16 extracted 4 (a leftover from the arithmetic) instead of the three-day total 164. |
| `gsm8k-money` | 44 | ok `44` | ok `44` | miss `11` | ok `44` | 3B NF4 extracted 11 — likely one-fifth of the remainder treated as the leftover, not subtracted from it. |
| `gsm8k-tank` | 90 | ok `90` | ok `90` | ok `90` | miss `30` | 14B NF4 extracted 30 (the added litres) instead of 90 remaining after the drain. |
| `gsm8k-train` | 240 | miss `2` | miss `2` | miss `2` | miss `2` | All four extracted 2: they reported hours, not minutes (90/45 + 60/30 = 2 h = 240 min). Shared task miss, not a codec split. |
| `gsm8k-machines` | 108 | miss `18` | miss `6` | miss `4` | ok `108` | Rate-scaling miss. 3B BF16 → 18, 14B BF16 → 6, 3B NF4 → 4; only 14B NF4 kept 108. |
| `gsm8k-stickers` | 48 | miss `3` | miss `72` | miss `4` | ok `48` | 3B BF16 → 3, 14B BF16 → 72 (gave away 2/5 and stopped), 3B NF4 → 4; 14B NF4 kept 48. |
| `trap-sheep` | 9 | miss `8` | ok `9` | ok `9` | ok `9` | Only 3B BF16 hit the trap (8 instead of 9). The other three answered 9. |
| `logic-yesno` | no | ok `no` | ok `no` | ok `no` | ok `no` | All four: No. No miss. |
| `code-sum` | 15 | ok `15` | ok `15` | ok `15` | ok `15` | All four: 15. No miss. |
| `code-loop` | 33 | ok `33` | ok `33` | ok `33` | ok `33` | All four: 33. No miss. |
| `trap-batball` | 0.05 | ok `0.05` | ok `0.05` | ok `0.05` | ok `0.05` | All four: 0.05. No miss. |
| `prefill-warehouse` | 7429 | ok `7429` | ok `7429` | ok `7429` | ok `7429` | All four: 7429. Buried number survived the distractors. |

Shared misses (all four wrong) are **gsm8k-train** (hours vs minutes). That
item does not discriminate codecs. NF4 is not uniformly worse: on 14B it is
10/12 against BF16 9/12; on 3B it is 8/12 against BF16 7/12. Losses that are
NF4-only on this set: 3B `gsm8k-money` (44→11) and 14B `gsm8k-tank` (90→30).
3B BF16 uniquely failed `gsm8k-lamps` and `trap-sheep`.

Driver / harness: no OOM, no skip, no missing stop token on Qwen. 14B BF16
decode at 0.66 tok/s is the spill, not a bug. nvidia-smi peak 12,019 MiB is
the capped meter; CUDA working set 28,490 MiB is the honest footprint.

## Time per message

Columns: prefill = time to first token; decode = rest of the completion.

### 3B BF16

| # | Item | Prefill | Decode | tok/s | New tokens | Stop |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gsm8k-lamps` | 57.4 ms | 11.49 s | 22.20 | 256 | max_new_tokens |
| 2 | `gsm8k-money` | 53.1 ms | 11.59 s | 22.01 | 256 | max_new_tokens |
| 3 | `gsm8k-tank` | 56.1 ms | 11.60 s | 21.98 | 256 | max_new_tokens |
| 4 | `gsm8k-train` | 63.9 ms | 11.56 s | 22.05 | 256 | max_new_tokens |
| 5 | `gsm8k-machines` | 45.6 ms | 11.28 s | 22.61 | 256 | max_new_tokens |
| 6 | `gsm8k-stickers` | 51.5 ms | 11.47 s | 22.23 | 256 | max_new_tokens |
| 7 | `trap-sheep` | 61.6 ms | 6.47 s | 22.10 | 144 | eos |
| 8 | `logic-yesno` | 54.5 ms | 48.1 ms | 20.81 | 2 | eos |
| 9 | `code-sum` | 54.5 ms | 10.10 s | 21.78 | 221 | eos |
| 10 | `code-loop` | 51.9 ms | 10.86 s | 22.09 | 241 | eos |
| 11 | `trap-batball` | 57.1 ms | 7.78 s | 21.48 | 168 | eos |
| 12 | `prefill-warehouse` | 63.4 ms | 258.4 ms | 23.22 | 7 | eos |

### 14B BF16

| # | Item | Prefill | Decode | tok/s | New tokens | Stop |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gsm8k-lamps` | 2.45 s | 5.54 min | 0.66 | 219 | eos |
| 2 | `gsm8k-money` | 2.26 s | 55.00 min | 0.08 | 256 | max_new_tokens |
| 3 | `gsm8k-tank` | 1.74 s | 6.59 min | 0.61 | 243 | eos |
| 4 | `gsm8k-train` | 1.76 s | 4.40 min | 0.97 | 256 | max_new_tokens |
| 5 | `gsm8k-machines` | 1.81 s | 5.57 min | 0.76 | 256 | max_new_tokens |
| 6 | `gsm8k-stickers` | 1.75 s | 5.99 min | 0.71 | 256 | max_new_tokens |
| 7 | `trap-sheep` | 1.91 s | 2.55 min | 0.78 | 121 | eos |
| 8 | `logic-yesno` | 1.56 s | 1.04 s | 0.97 | 2 | eos |
| 9 | `code-sum` | 1.98 s | 1.67 min | 0.88 | 89 | eos |
| 10 | `code-loop` | 2.99 s | 4.01 min | 0.68 | 164 | eos |
| 11 | `trap-batball` | 1.76 s | 7.79 min | 0.39 | 182 | eos |
| 12 | `prefill-warehouse` | 12.42 s | 1.38 min | 0.48 | 41 | eos |

### 3B NF4

| # | Item | Prefill | Decode | tok/s | New tokens | Stop |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gsm8k-lamps` | 427.2 ms | 14.32 s | 15.44 | 222 | eos |
| 2 | `gsm8k-money` | 474.6 ms | 15.72 s | 16.22 | 256 | max_new_tokens |
| 3 | `gsm8k-tank` | 337.4 ms | 12.60 s | 16.74 | 212 | eos |
| 4 | `gsm8k-train` | 396.3 ms | 14.94 s | 17.07 | 256 | max_new_tokens |
| 5 | `gsm8k-machines` | 340.8 ms | 14.85 s | 17.17 | 256 | max_new_tokens |
| 6 | `gsm8k-stickers` | 409.1 ms | 14.86 s | 17.16 | 256 | max_new_tokens |
| 7 | `trap-sheep` | 274.9 ms | 3.95 s | 17.22 | 69 | eos |
| 8 | `logic-yesno` | 271.5 ms | 58.0 ms | 17.25 | 2 | eos |
| 9 | `code-sum` | 347.3 ms | 13.54 s | 17.06 | 232 | eos |
| 10 | `code-loop` | 348.2 ms | 13.66 s | 17.20 | 236 | eos |
| 11 | `trap-batball` | 345.3 ms | 11.42 s | 17.24 | 198 | eos |
| 12 | `prefill-warehouse` | 1.31 s | 821.0 ms | 17.05 | 15 | eos |

### 14B NF4

| # | Item | Prefill | Decode | tok/s | New tokens | Stop |
|---:|---|---:|---:|---:|---:|---|
| 1 | `gsm8k-lamps` | 1.49 s | 27.76 s | 6.48 | 181 | eos |
| 2 | `gsm8k-money` | 1.53 s | 39.44 s | 6.47 | 256 | max_new_tokens |
| 3 | `gsm8k-tank` | 1.25 s | 38.30 s | 6.66 | 256 | max_new_tokens |
| 4 | `gsm8k-train` | 1.26 s | 38.37 s | 6.65 | 256 | max_new_tokens |
| 5 | `gsm8k-machines` | 1.26 s | 39.01 s | 6.54 | 256 | max_new_tokens |
| 6 | `gsm8k-stickers` | 1.52 s | 37.72 s | 6.65 | 252 | eos |
| 7 | `trap-sheep` | 1.00 s | 9.89 s | 6.67 | 67 | eos |
| 8 | `logic-yesno` | 1.01 s | 155.7 ms | 6.42 | 2 | eos |
| 9 | `code-sum` | 1.29 s | 13.11 s | 6.64 | 88 | eos |
| 10 | `code-loop` | 1.26 s | 30.02 s | 6.66 | 201 | eos |
| 11 | `trap-batball` | 1.27 s | 30.19 s | 6.56 | 199 | eos |
| 12 | `prefill-warehouse` | 4.80 s | 7.68 s | 6.64 | 52 | eos |

## Full questions and replies

Gold is from [`gpu/lab/data/hard_items.json`](../gpu/lab/data/hard_items.json).
Replies are the raw `messages.csv` text, not the extracted number.

### 1. `gsm8k-lamps`

**Gold:** `164`

**Question**

```
A shop sold 35 lamps on Monday. Tuesday it sold 12 more than Monday. Wednesday it sold as many as Monday and Tuesday combined. How many lamps were sold in the three days? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `4` (miss); 
prefill 57.4 ms, decode 11.49 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Determine the number of lamps sold on Tuesday:**
   - The shop sold 35 lamps on Monday.
   - On Tuesday, they sold 12 more lamps than on Monday.
   - Therefore, the number of lamps sold on Tuesday is \( 35 + 12 = 47 \).

2. **Determine the number of lamps sold on Wednesday:**
   - On Wednesday, the shop sold as many lamps as the total sold on Monday and Tuesday combined.
   - The total number of lamps sold on Monday and Tuesday is \( 35 + 47 = 82 \).
   - Therefore, the number of lamps sold on Wednesday is 82.

3. **Calculate the total number of lamps sold over the three days:**
   - Total lamps sold = Lamps sold on Monday + Lamps sold on Tuesday + Lamps sold on Wednesday.
   - Substituting the values we have:
     \[
     35 + 47 + 82
     \]

4. **Perform the addition:**
   - First, add the lamps sold on Monday and Tuesday:
     \[
     35 + 4
```

**14B BF16** — extracted `164` (correct); 
prefill 2.45 s, decode 5.54 min, 219 tokens, stop `eos`.

```
To find out how many lamps were sold over the three days, we can follow these steps:

1. **Lamps Sold on Monday**: The shop sold 35 lamps.
   
2. **Lamps Sold on Tuesday**: It sold 12 more lamps than on Monday. Therefore, the number of lamps sold on Tuesday is:
   \[
   35 + 12 = 47
   \]

3. **Lamps Sold on Wednesday**: The shop sold as many lamps as the total from Monday and Tuesday combined. So, the number of lamps sold on Wednesday is:
   \[
   35 + 47 = 82
   \]

4. **Total Lamps Sold Over Three Days**: To get the total number of lamps sold over the three days, we add the number of lamps sold each day:
   \[
   35 + 47 + 82 = 164
   \]

Therefore, the total number of lamps sold in the three days is:
#### 164
```

**3B NF4** — extracted `164` (correct); 
prefill 427.2 ms, decode 14.32 s, 222 tokens, stop `eos`.

```
Let's break down the problem step by step:

1. **Monday**: The shop sold 35 lamps.
2. **Tuesday**: The shop sold 12 more lamps than on Monday. Therefore, the number of lamps sold on Tuesday is:
   \[
   35 + 12 = 47
   \]
3. **Wednesday**: The shop sold as many lamps as the total sold on Monday and Tuesday combined. Therefore, the number of lamps sold on Wednesday is:
   \[
   35 + 47 = 82
   \]

Now, to find the total number of lamps sold over the three days, we add the number of lamps sold on each day:
\[
35 + 47 + 82
\]

Let's add these numbers step by step:
\[
35 + 47 = 82
\]
\[
82 + 82 = 164
\]

Therefore, the total number of lamps sold in the three days is:
#### 164
```

**14B NF4** — extracted `164` (correct); 
prefill 1.49 s, decode 27.76 s, 181 tokens, stop `eos`.

```
Let's break down the sales for each day:

- On Monday, the shop sold 35 lamps.
- On Tuesday, the shop sold 12 more than it did on Monday, which means it sold \(35 + 12 = 47\) lamps.
- On Wednesday, the shop sold as many as Monday and Tuesday combined, which means it sold \(35 + 47 = 82\) lamps.

To find the total number of lamps sold over the three days, we add the sales from each day together:

- Monday: 35 lamps
- Tuesday: 47 lamps
- Wednesday: 82 lamps

Adding these together: \(35 + 47 + 82 = 164\).

Therefore, the total number of lamps sold in the three days is 164.

####164
```

### 2. `gsm8k-money`

**Gold:** `44`

**Question**

```
Maya has 84 dollars. She buys 3 notebooks at 7 dollars each and 2 pens at 4 dollars each. Then she spends one fifth of what remains on lunch. How many dollars does she have left? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `44` (correct); 
prefill 53.1 ms, decode 11.59 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Amount**: Maya starts with $84.

2. **Cost of Notebooks**: She buys 3 notebooks at $7 each.
   \[
   3 \times 7 = 21 \text{ dollars}
   \]
   After buying the notebooks, the remaining amount is:
   \[
   84 - 21 = 63 \text{ dollars}
   \]

3. **Cost of Pens**: She buys 2 pens at $4 each.
   \[
   2 \times 4 = 8 \text{ dollars}
   \]
   After buying the pens, the remaining amount is:
   \[
   63 - 8 = 55 \text{ dollars}
   \]

4. **Spending on Lunch**: She spends one-fifth of the remaining amount on lunch.
   \[
   \frac{1}{5} \times 55 = 11 \text{ dollars}
   \]
   After spending on lunch, the remaining amount is:
   \[
   55 - 11 = 44 \text{ dollars}
   \]

Therefore, the final amount of
```

**14B BF16** — extracted `44` (correct); 
prefill 2.26 s, decode 55.00 min, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Amount**: Maya starts with $84.
   
2. **Cost of Notebooks**: She buys 3 notebooks at $7 each. The total cost for the notebooks is \(3 \times 7 = 21\) dollars.
   
3. **Cost of Pens**: She buys 2 pens at $4 each. The total cost for the pens is \(2 \times 4 = 8\) dollars.
   
4. **Total Spent on Notebooks and Pens**: The total amount spent on notebooks and pens is \(21 + 8 = 29\) dollars.
   
5. **Remaining Amount After Purchases**: Subtracting the total spent from her initial amount gives \(84 - 29 = 55\) dollars remaining.
   
6. **Spending on Lunch**: Maya spends one fifth of the remaining amount on lunch. One fifth of $55 is \(\frac{55}{5} = 11\) dollars.
   
7. **Final Remaining Amount**: Subtracting the amount spent on lunch from the remaining amount gives \(55 - 11 = 44\) dollars.

Therefore, the final amount Maya has left is
```

**3B NF4** — extracted `11` (miss); 
prefill 474.6 ms, decode 15.72 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Amount**: Maya starts with $84.

2. **Cost of Notebooks**: Maya buys 3 notebooks at $7 each.
   \[
   \text{Total cost of notebooks} = 3 \times 7 = 21 \text{ dollars}
   \]
   After buying the notebooks, the remaining amount is:
   \[
   84 - 21 = 63 \text{ dollars}
   \]

3. **Cost of Pens**: Maya buys 2 pens at $4 each.
   \[
   \text{Total cost of pens} = 2 \times 4 = 8 \text{ dollars}
   \]
   After buying the pens, the remaining amount is:
   \[
   63 - 8 = 55 \text{ dollars}
   \]

4. **Spending on Lunch**: Maya spends one fifth of the remaining amount on lunch.
   \[
   \text{Amount spent on lunch} = \frac{1}{5} \times 55 = 11 \text{ dollars}
   \]
   After spending on lunch, the remaining amount is:
   \[
```

**14B NF4** — extracted `44` (correct); 
prefill 1.53 s, decode 39.44 s, 256 tokens, stop `max_new_tokens`.

```
First, let's calculate the total cost of the notebooks and pens that Maya buys.

The cost of 3 notebooks at 7 dollars each is:
\[ 3 \times 7 = 21 \, \text{dollars} \]

The cost of 2 pens at 4 dollars each is:
\[ 2 \times 4 = 8 \, \text{dollars} \]

Adding these costs together gives the total amount spent on notebooks and pens:
\[ 21 + 8 = 29 \, \text{dollars} \]

Maya starts with 84 dollars, so after buying the notebooks and pens, she has:
\[ 84 - 29 = 55 \, \text{dollars} \]

Next, she spends one fifth of what remains on lunch. One fifth of 55 dollars is:
\[ \frac{55}{5} = 11 \, \text{dollars} \]

Subtracting the cost of lunch from the remaining amount gives:
\[ 55 - 11 = 44 \, \text{dollars} \]

Therefore, the final amount Maya has left is:
\[ 44 \]
```

### 3. `gsm8k-tank`

**Gold:** `90`

**Question**

```
A tank holds 240 liters. It is 3/8 full. After 30 liters are added, one quarter of the new amount is drained. How many liters remain? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `90` (correct); 
prefill 56.1 ms, decode 11.60 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Volume Calculation:**
   The tank initially holds 240 liters and is \( \frac{3}{8} \) full.
   \[
   \text{Initial volume} = 240 \times \frac{3}{8} = 90 \text{ liters}
   \]

2. **Adding 30 Liters:**
   After adding 30 liters to the initial volume:
   \[
   \text{New volume} = 90 + 30 = 120 \text{ liters}
   \]

3. **Draining One Quarter of the New Amount:**
   One quarter of the new amount (120 liters) is drained:
   \[
   \text{Volume drained} = 120 \times \frac{1}{4} = 30 \text{ liters}
   \]
   Therefore, the remaining volume after draining:
   \[
   \text{Remaining volume} = 120 - 30 = 90 \text{ liters}
   \]

So, the number of liters that remain in the tank is 90. According to the
```

**14B BF16** — extracted `90` (correct); 
prefill 1.74 s, decode 6.59 min, 243 tokens, stop `eos`.

```
To solve this problem, we'll follow these steps:

1. **Determine how much water is currently in the tank:**
   The tank is 3/8 full of its total capacity of 240 liters.
   \[
   \text{Current amount} = \frac{3}{8} \times 240 = 90 \text{ liters}
   \]

2. **Add 30 liters to the current amount:**
   \[
   \text{New amount} = 90 + 30 = 120 \text{ liters}
   \]

3. **Drain one quarter of the new amount:**
   One quarter of 120 liters is:
   \[
   \frac{1}{4} \times 120 = 30 \text{ liters}
   \]
   So, after draining 30 liters, the remaining amount is:
   \[
   120 - 30 = 90 \text{ liters}
   \]

Therefore, the number of liters that remain in the tank is:
\[
\boxed{90}
\]
```

**3B NF4** — extracted `90` (correct); 
prefill 337.4 ms, decode 12.60 s, 212 tokens, stop `eos`.

```
Let's break down the problem step by forget to add the final integer at the end.

1. **Initial Condition**: The tank holds 240 liters and is initially 3/8 full.
   - Initial volume of water in the tank = \( \frac{3}{8} \times 240 = 90 \) liters.

2. **Adding 30 Liters**: After adding 30 liters to the tank.
   - New volume of water in the tank = \( 90 + 30 = 120 \) liters.

3. **Draining One Quarter**: One quarter of the new amount (120 liters) is drained.
   - Volume drained = \( \frac{1}{4} \times 120 = 30 \) liters.
   - Remaining volume after draining = \( 120 - 30 = 90 \) liters.

The final integer that remains after the process is 90 liters. ####90
```

**14B NF4** — extracted `30` (miss); 
prefill 1.25 s, decode 38.30 s, 256 tokens, stop `max_new_tokens`.

```
To solve the problem, let's break it down step by step.

1. **Determine the initial amount of water in the tank:**
   - The tank's total capacity is 240 liters.
   - It is 3/8 full.
   - Initial amount of water = \( \frac{3}{8} \times 240 \) liters.
   - Initial amount of water = \( \frac{3 \times 240}{8} = \frac{720}{8} = 90 \) liters.

2. **Add 30 liters to the initial amount:**
   - New amount of water = Initial amount + 30 liters.
   - New amount of water = 90 liters + 30 liters = 120 liters.

3. **Drain one quarter of the new amount:**
   - One quarter of the new amount = \( \frac{1}{4} \times 120 \) liters.
   - One quarter of the new amount = \( \frac{120}{4} = 30 \) liters.
   - Remaining amount after draining = New amount - One quarter of the new amount.
   - Remaining
```

### 4. `gsm8k-train`

**Gold:** `240`

**Question**

```
A train travels 90 km at 45 km/h, then 60 km at 30 km/h. How many minutes does the whole trip take? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `2` (miss); 
prefill 63.9 ms, decode 11.56 s, 256 tokens, stop `max_new_tokens`.

```
To calculate the total time taken for the entire trip, we need to find the time taken for each segment of the journey and then sum them up.

### Step 1: Calculate the time taken for the first part of the journey.
- Distance traveled: 90 km
- Speed: 45 km/h

The formula to calculate time is:
\[ \text{Time} = \frac{\text{Distance}}{\text{Speed}} \]

For the first part:
\[ \text{Time}_1 = \frac{90 \text{ km}}{45 \text{ km/h}} = 2 \text{ hours} \]

### Step 2: Calculate the time taken for the second part of the journey.
- Distance traveled: 60 km
- Speed: 30 km/h

For the second part:
\[ \text{Time}_2 = \frac{60 \text{ km}}{30 \text{ km/h}} = 2 \text{ hours} \]

### Step 3: Sum the times for both parts of the journey.
\[ \text{Total Time} = \text{Time}_1 + \text{Time}_2 = 2 \text{ hours}
```

**14B BF16** — extracted `2` (miss); 
prefill 1.76 s, decode 4.40 min, 256 tokens, stop `max_new_tokens`.

```
To calculate the total time taken for the entire trip, we need to find the time taken for each segment of the journey and then sum these times.

1. **First Segment:**
   - Distance = 90 km
   - Speed = 45 km/h

   Time is calculated using the formula:
   \[
   \text{Time} = \frac{\text{Distance}}{\text{Speed}}
   \]
   
   For the first segment:
   \[
   \text{Time}_1 = \frac{90 \text{ km}}{45 \text{ km/h}} = 2 \text{ hours}
   \]

2. **Second Segment:**
   - Distance = 60 km
   - Speed = 30 km/h

   Using the same formula:
   \[
   \text{Time}_2 = \frac{60 \text{ km}}{30 \text{ km/h}} = 2 \text{ hours}
   \]

3. **Total Time in Hours:**
   \[
   \text{Total Time} = \text{Time}_1 + \text{Time}_2 = 2 \text{ hours} + 2 \text
```

**3B NF4** — extracted `2` (miss); 
prefill 396.3 ms, decode 14.94 s, 256 tokens, stop `max_new_tokens`.

```
To calculate the total time taken for the entire trip, we need to calculate the time taken for each part of the journey separately and then add them together.

### Step 1: Calculate the time for the first part of the journey
- Distance: 90 km
- Speed: 45 km/h

The time taken for the first part can be calculated using the formula:
\[ \text{Time} = \frac{\text{Distance}}{\text{Speed}} \]

\[ \text{Time}_1 = \frac{90 \text{ km}}{45 \text{ km/h}} = 2 \text{ hours} \]

### Step 2: Calculate the time for the second part of the journey
- Distance: 60 km
- Speed: 30 km/h

\[ \text{Time}_2 = \frac{60 \text{ km}}{30 \text{ km/h}} = 2 \text{ hours} \]

### Step 3: Add the times for both parts of the journey
\[ \text{Total Time} = \text{Time}_1 + \text{Time}_2 = 2 \text{ hours} + 2 \text{
```

**14B NF4** — extracted `2` (miss); 
prefill 1.26 s, decode 38.37 s, 256 tokens, stop `max_new_tokens`.

```
To find the total time taken for the whole trip, we need to calculate the time taken for each segment of the trip separately and then add them together.

1. **First Segment:**
   - Distance: 90 km
   - Speed: 45 km/h

   Time taken for the first segment is given by the formula:
   \[
   \text{Time} = \frac{\text{Distance}}{\text{Speed}}
   \]
   \[
   \text{Time}_1 = \frac{90 \text{ km}}{45 \text{ km/h}} = 2 \text{ hours}
   \]

2. **Second Segment:**
   - Distance: 60 km
   - Speed: 30 km/h

   Time taken for the second segment is:
   \[
   \text{Time}_2 = \frac{60 \text{ km}}{30 \text{ km/h}} = 2 \text{ hours}
   \]

3. **Total Time:**
   Adding the time taken for both segments:
   \[
   \text{Total Time} = \text{Time}_1 + \text{Time}_2 = 2 \
```

### 5. `gsm8k-machines`

**Gold:** `108`

**Question**

```
8 machines make 96 parts in 4 hours. How many parts do 6 machines make in 6 hours at the same rate? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `18` (miss); 
prefill 45.6 ms, decode 11.28 s, 256 tokens, stop `max_new_tokens`.

```
To solve this problem, we'll first determine the rate at which the machines work and then use that rate to find out how many parts 6 machines can make in 6 hours.

### Step 1: Determine the rate of production for one machine

Given:
- 8 machines make 96 parts in 4 hours.

First, calculate the total number of parts made per hour by all 8 machines:
\[ \text{Total parts per hour} = \frac{96 \text{ parts}}{4 \text{ hours}} = 24 \text{ parts/hour} \]

Since this is the combined production of 8 machines, the production rate per machine per hour is:
\[ \text{Rate per machine per hour} = \frac{24 \text{ parts/hour}}{8 \text{ machines}} = 3 \text{ parts/machine/hour} \]

### Step 2: Calculate the production rate for 6 machines

Now, if we have 6 machines working at the same rate:
\[ \text{Production rate for 6 machines per hour} = 6 \text{ machines} \times 3 \text{ parts/machine/hour} = 18 \text
```

**14B BF16** — extracted `6` (miss); 
prefill 1.81 s, decode 5.57 min, 256 tokens, stop `max_new_tokens`.

```
To solve this problem, we first need to determine the rate at which the machines produce parts.

1. **Calculate the production rate per machine per hour:**
   - We know that 8 machines make 96 parts in 4 hours.
   - First, find out how many parts one machine makes in 4 hours:
     \[
     \text{Parts per machine in 4 hours} = \frac{96 \text{ parts}}{8 \text{ machines}} = 12 \text{ parts}
     \]
   - Now, calculate how many parts one machine makes in 1 hour:
     \[
     \text{Parts per machine per hour} = \frac{12 \text{ parts}}{4 \text{ hours}} = 3 \text{ parts per hour}
     \]

2. **Determine the production of 6 machines in 6 hours:**
   - If one machine produces 3 parts per hour, then 6 machines will produce:
     \[
     \text{Parts per hour for 6 machines} = 6 \times 3 = 18 \text{ parts per hour}
     \]
   - Over 6 hours, 6 machines will produce
```

**3B NF4** — extracted `4` (miss); 
prefill 340.8 ms, decode 14.85 s, 256 tokens, stop `max_new_tokens`.

```
To solve this problem, we need to determine the rate at which the machines work and then apply that rate to the new scenario.

### Step 1: Determine the Rate of One Machine

First, let's find out how many parts one machine can make in one hour.

- 8 machines make 96 parts in 4 hours.
- Therefore, in 4 hours, one machine makes \( \frac{96}{8} = 12 \) parts.

So, one machine makes 12 parts in 4 hours.

### Step 2: Determine the Rate of One Machine in One Hour

Since one machine makes 12 parts in 4 hours, the rate of one machine is:
\[ \text{Rate of one machine} = \frac{12 \text{ parts}}{4 \text{ hours}} = 3 \text{ parts per hour} \]

### Step 3: Determine the Rate of 6 Machines in One Hour

If one machine makes 3 parts per hour, then 6 machines will make:
\[ \text{Rate of 6 machines} = 6 \times 3 = 18 \text{ parts per hour} \]

### Step 4: Determine the Number
```

**14B NF4** — extracted `108` (correct); 
prefill 1.26 s, decode 39.01 s, 256 tokens, stop `max_new_tokens`.

```
To solve this problem, we first need to determine the rate at which the machines produce parts. We know that 8 machines make 96 parts in 4 hours. Let's break it down step by step.

1. **Determine the rate of production for 8 machines:**
   - 8 machines make 96 parts in 4 hours.
   - Therefore, the rate for 8 machines is 96 parts / 4 hours = 24 parts per hour.

2. **Determine the rate of production for 1 machine:**
   - If 8 machines produce 24 parts per hour, then 1 machine produces 24 parts per hour / 8 machines = 3 parts per hour.

3. **Determine the rate of production for 6 machines:**
   - If 1 machine produces 3 parts per hour, then 6 machines produce 3 parts per hour * 6 machines = 18 parts per hour.

4. **Calculate the total production for 6 machines in 6 hours:**
   - If 6 machines produce 18 parts per hour, then in 6 hours they produce 18 parts per hour * 6 hours = 108 parts
```

### 6. `gsm8k-stickers`

**Gold:** `48`

**Question**

```
Leah has 120 stickers. She gives 2/5 of them to her brother. Then she gives 1/3 of what she still has to her sister. How many stickers does Leah keep? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `3` (miss); 
prefill 51.5 ms, decode 11.47 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Stickers:**
   Leah starts with 120 stickers.

2. **Stickers Given to Her Brother:**
   Leah gives \(\frac{2}{5}\) of her stickers to her brother.
   
   To find out how many stickers this is:
   \[
   \frac{2}{5} \times 120 = \frac{2 \times 120}{5} = \frac{240}{5} = 48
   \]
   So, Leah gives 48 stickers to her brother.

3. **Remaining Stickers After Giving to Her Brother:**
   Subtract the stickers given to her brother from the initial total:
   \[
   120 - 48 = 72
   \]
   Leah now has 72 stickers left.

4. **Stickers Given to Her Sister:**
   Leah then gives \(\frac{1}{3}\) of the remaining stickers to her sister.
   
   To find out how many stickers this is:
   \[
   \frac{1}{3} \times 72 = \frac{72}{3} =
```

**14B BF16** — extracted `72` (miss); 
prefill 1.75 s, decode 5.99 min, 256 tokens, stop `max_new_tokens`.

```
To find out how many stickers Leah keeps, we can follow these steps:

1. **Calculate the number of stickers Leah gives to her brother:**
   - Leah starts with 120 stickers.
   - She gives \( \frac{2}{5} \) of her stickers to her brother.
   - The number of stickers given to her brother is \( \frac{2}{5} \times 120 = \frac{240}{5} = 48 \).

2. **Determine how many stickers Leah has left after giving some to her brother:**
   - Leah originally had 120 stickers.
   - After giving 48 stickers to her brother, she has \( 120 - 48 = 72 \) stickers left.

3. **Calculate the number of stickers Leah gives to her sister:**
   - Leah gives \( \frac{1}{3} \) of the remaining stickers to her sister.
   - The number of stickers given to her sister is \( \frac{1}{3} \times 72 = 24 \).

4. **Determine how many stickers Leah keeps:**
   - Leah had 72 stickers left after giving some
```

**3B NF4** — extracted `4` (miss); 
prefill 409.1 ms, decode 14.86 s, 256 tokens, stop `max_new_tokens`.

```
Let's break down the problem step by step.

1. **Initial Stickers:**
   Leah starts with 120 stickers.

2. **Stickers Given to Her Brother:**
   Leah gives \( \frac{2}{5} \) of her stickers to her brother.
   \[
   \text{Stickers given to her brother} = \frac{2}{5} \times 120 = \frac{240}{5} = 48
   \]
   After giving 48 stickers to her brother, the number of stickers Leah has left is:
   \[
   120 - 48 = 72
   \]

3. **Stickers Given to Her Sister:**
   Leah then gives \( \frac{1}{3} \) of the remaining stickers to her sister.
   \[
   \text{Stickers given to her sister} = \frac{1}{3} \times 72 = \frac{72}{3} = 24
   \]
   After giving 24 stickers to her sister, the number of stickers Leah has left is:
   \[
   72 - 24 = 4
```

**14B NF4** — extracted `48` (correct); 
prefill 1.52 s, decode 37.72 s, 252 tokens, stop `eos`.

```
To find out how many stickers Leah keeps, let's break down the problem step by step.

1. Leah starts with 120 stickers.
2. She gives 2/5 of her stickers to her brother. To find out how many stickers that is, we calculate:
   \[
   \frac{2}{5} \times 120 = 48
   \]
   So, Leah gives 48 stickers to her brother.
3. After giving 48 stickers to her brother, Leah has:
   \[
   120 - 48 = 72
   \]
   stickers left.
4. Leah then gives 1/3 of the remaining stickers to her sister. To find out how many stickers that is, we calculate:
   \[
   \frac{1}{3} \times 72 = 24
   \]
   So, Leah gives 24 stickers to her sister.
5. After giving 24 stickers to her sister, Leah has:
   \[
   72 - 24 = 48
   \]
   stickers left.

Therefore, Leah keeps 48 stickers.

#### 48
```

### 7. `trap-sheep`

**Gold:** `9`

**Question**

```
A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left? Show the steps, then put the final integer after ####.
```

**3B BF16** — extracted `8` (miss); 
prefill 61.6 ms, decode 6.47 s, 144 tokens, stop `eos`.

```
Let's break down the problem step by step:

1. The farmer starts with 17 sheep.
2. All but 9 sheep die. This means that out of the 17 sheep, all except 9 die.

To find out how many sheep are left, we subtract the number of sheep that died from the total number of sheep initially present:

\[ \text{Sheep left} = \text{Total sheep} - \text{Sheep that died} \]
\[ \text{Sheep left} = 17 - 9 \]
\[ \text{Sheep left} = 8 \]

So, the farmer has 8 sheep left. 

####8####
```

**14B BF16** — extracted `9` (correct); 
prefill 1.91 s, decode 2.55 min, 121 tokens, stop `eos`.

```
The problem states that all but 9 of the farmer's sheep die. This means that out of the original 17 sheep, only 9 survive.

To find out how many sheep the farmer has left:
- Start with the total number of sheep: 17.
- Subtract the number of sheep that died (which is 17 - 9 = 8).

So, the calculation is:
\[ 17 - (17 - 9) = 17 - 8 = 9 \]

Therefore, the farmer has 9 sheep left.

#### 9
```

**3B NF4** — extracted `9` (correct); 
prefill 274.9 ms, decode 3.95 s, 69 tokens, stop `eos`.

```
Let's break down the problem step by step:

1. The farmer starts with 17 sheep.
2. All but 9 die. This means that 17 - 9 = 8 sheep die.
3. Therefore, the number of sheep left is 9.

The final integer after #### is 9. ####9
```

**14B NF4** — extracted `9` (correct); 
prefill 1.00 s, decode 9.89 s, 67 tokens, stop `eos`.

```
The farmer initially has 17 sheep. The phrase "all but 9 die" means that out of the 17 sheep, 9 survive while the rest die. 

To find out how many sheep the farmer has left, we simply need to identify the number that survived, which is 9.

#### 9
```

### 8. `logic-yesno`

**Gold:** `no`

**Question**

```
Every alpha is a beta. No beta is a gamma. Can an alpha be a gamma? Reply with Yes or No only.
```

**3B BF16** — extracted `no` (correct); 
prefill 54.5 ms, decode 48.1 ms, 2 tokens, stop `eos`.

```
No
```

**14B BF16** — extracted `no` (correct); 
prefill 1.56 s, decode 1.04 s, 2 tokens, stop `eos`.

```
No
```

**3B NF4** — extracted `no` (correct); 
prefill 271.5 ms, decode 58.0 ms, 2 tokens, stop `eos`.

```
No
```

**14B NF4** — extracted `no` (correct); 
prefill 1.01 s, decode 155.7 ms, 2 tokens, stop `eos`.

```
No
```

### 9. `code-sum`

**Gold:** `15`

**Question**

```
What integer does this Python snippet print? Reason briefly, then put the final integer after ####.

s = 0
for i in range(1, 6):
    s += i
print(s)
```

**3B BF16** — extracted `15` (correct); 
prefill 54.5 ms, decode 10.10 s, 221 tokens, stop `eos`.

```
Let's break down the code step by step:

- We initialize `s` to 0.
- The `for` loop runs from 1 to 5 (inclusive), so it iterates over the numbers 1, 2, 3, 4, and 5.
- In each iteration, the current value of `i` is added to `s`.
- After the loop completes, the value of `s` will be the sum of the numbers from 1 to 5.

The sum of the first 5 natural numbers can be calculated using the formula for the sum of an arithmetic series: \( S = \frac{n(n + 1)}{2} \), where \( n \) is the number of terms. Here, \( n = 5 \).

So, the sum is:
\[ S = \frac{5(5 + 1)}{2} = \frac{5 \times 6}{2} = 15 \]

Therefore, the integer that the Python snippet prints is 15.

####15####
```

**14B BF16** — extracted `15` (correct); 
prefill 1.98 s, decode 1.67 min, 89 tokens, stop `eos`.

```
The Python snippet initializes `s` to 0 and then iterates over the range from 1 to 5 (inclusive), adding each number to `s`. The numbers added are 1, 2, 3, 4, and 5. Summing these gives \(1 + 2 + 3 + 4 + 5 = 15\). Therefore, the integer printed is:

#### 15
```

**3B NF4** — extracted `15` (correct); 
prefill 347.3 ms, decode 13.54 s, 232 tokens, stop `eos`.

```
Let's break down the Python snippet step by step to determine the final integer printed.

The snippet initializes a variable `s` to 0. Then, it enters a `for` loop that iterates from 1 to 5 (inclusive). In each iteration, the loop adds the current value of `i` to `s`.

Here's the step-by-step calculation:
- First iteration: `s += 1` → `s = 0 + 1 = 1`
- Second iteration: `s += 2` → `s = 1 + 2 = 3`
- Third iteration: `s += 3` → `s = 3 + 3 = 6`
- Fourth iteration: `s += 4` → `s = 6 + 4 = 10`
- Fifth iteration: `s += 5` → `s = 10 + 5 = 15`

After the loop completes, the final value of `s` is 15. Therefore, the Python snippet will print `15`.

The final integer is: ####15####
```

**14B NF4** — extracted `15` (correct); 
prefill 1.29 s, decode 13.11 s, 88 tokens, stop `eos`.

```
The Python snippet initializes a variable `s` to 0. It then iterates over a range from 1 to 5 (inclusive), adding each number in this range to `s`. The numbers added are 1, 2, 3, 4, and 5. The sum of these numbers is 15. Therefore, the final value of `s` is 15.

#### 15
```

### 10. `code-loop`

**Gold:** `33`

**Question**

```
What integer does this Python snippet print? Reason briefly, then put the final integer after ####.

x = 3
for _ in range(4):
    x = x * 2 - 1
print(x)
```

**3B BF16** — extracted `33` (correct); 
prefill 51.9 ms, decode 10.86 s, 241 tokens, stop `eos`.

```
Let's break down the code step by step to understand what it does:

1. We start with `x = 3`.
2. The `for` loop runs four times (`range(4)`).
3. In each iteration of the loop, `x` is updated according to the expression `x = x * 2 - 1`.

Let's go through each iteration:

- **First iteration**: `x = 3`
  - New value: `x = 3 * 2 - 1 = 5`

- **Second iteration**: `x = 5`
  - New value: `x = 5 * 2 - 1 = 9`

- **Third iteration**: `x = 9`
  - New value: `x = 9 * 2 - 1 = 17`

- **Fourth iteration**: `x = 17`
  - New value: `x = 17 * 2 - 1 = 33`

After the loop completes, the final value of `x` is `33`. 

Therefore, the integer that the Python snippet prints is `####33####`.
```

**14B BF16** — extracted `33` (correct); 
prefill 2.99 s, decode 4.01 min, 164 tokens, stop `eos`.

```
The initial value of `x` is set to 3. The loop runs 4 times, and in each iteration, the value of `x` is updated using the formula `x = x * 2 - 1`.

Let's break down the iterations:

- After the first iteration: `x = 3 * 2 - 1 = 5`
- After the second iteration: `x = 5 * 2 - 1 = 9`
- After the third iteration: `x = 9 * 2 - 1 = 17`
- After the fourth iteration: `x = 17 * 2 - 1 = 33`

Therefore, the final value of `x` printed by the code is 33.

#### 33
```

**3B NF4** — extracted `33` (correct); 
prefill 348.2 ms, decode 13.66 s, 236 tokens, stop `eos`.

```
Let's analyze the code step by step to determine the final value of `x`:

1. Initially, `x` is set to 3.
2. The `for` loop runs 4 times, with `_` (an unused variable) as the loop variable.
3. In each iteration of the loop, `x` is updated according to the expression `x * 2 - 1`.

Let's go through the iterations:

- **First iteration**: `x = 3 * 2 - 1 = 6 - 1 = 5`
- **Second iteration**: `x = 5 * 2 - 1 = 10 - 1 = 9`
- **Third iteration**: `x = 9 * 2 - 1 = 18 - 1 = 17`
- **Fourth iteration**: `x = 17 * 2 - 1 = 34 - 1 = 33`

After the loop completes, the final value of `x` is 33.

Therefore, the integer that the Python snippet prints is `####33`.

####33
```

**14B NF4** — extracted `33` (correct); 
prefill 1.26 s, decode 30.02 s, 201 tokens, stop `eos`.

```
The Python snippet starts with `x = 3`. It then enters a loop that runs 4 times (`for _ in range(4)`). In each iteration of the loop, the value of `x` is updated according to the expression `x = x * 2 - 1`.

Let's break down the iterations:

1. Initial value of `x` is 3.
2. After the first iteration: `x = 3 * 2 - 1 = 5`.
3. After the second iteration: `x = 5 * 2 - 1 = 9`.
4. After the third iteration: `x = 9 * 2 - 1 = 17`.
5. After the fourth iteration: `x = 17 * 2 - 1 = 33`.

Thus, after the loop completes, the value of `x` is 33. The final output of the program is:

#### 33
```

### 11. `trap-batball`

**Gold:** `0.05`

**Question**

```
A bat and a ball cost 1.10 dollars together. The bat costs 1.00 dollar more than the ball. How many dollars does the ball cost? Show the steps, then put the final number after ####.
```

**3B BF16** — extracted `0.05` (correct); 
prefill 57.1 ms, decode 7.78 s, 168 tokens, stop `eos`.

```
Let's denote the cost of the ball as \( x \) dollars.

According to the problem:
- The bat costs $1.00 more than the ball.
- Together, they cost $1.10.

So, we can write the following equations based on the information given:

1. \( x + (x + 1.00) = 1.10 \)

Simplifying this equation:
\[ 2x + 1.00 = 1.10 \]

Subtract 1.00 from both sides:
\[ 2x = 0.10 \]

Divide both sides by 2:
\[ x = 0.05 \]

Therefore, the ball costs \( 0.05 \) dollars.

####0.05####
```

**14B BF16** — extracted `0.05` (correct); 
prefill 1.76 s, decode 7.79 min, 182 tokens, stop `eos`.

```
Let's denote the cost of the ball as \( x \) dollars.

According to the problem, the bat costs $1.00 more than the ball. Therefore, the cost of the bat is \( x + 1.00 \) dollars.

The total cost of the bat and the ball is given as $1.10. So we can write the equation:

\[ x + (x + 1.00) = 1.10 \]

Simplifying this equation:

\[ 2x + 1.00 = 1.10 \]

Subtract 1.00 from both sides:

\[ 2x = 0.10 \]

Divide both sides by 2:

\[ x = 0.05 \]

So, the ball costs 0.05 dollars.

#### 0.05
```

**3B NF4** — extracted `0.05` (correct); 
prefill 345.3 ms, decode 11.42 s, 198 tokens, stop `eos`.

```
Let's denote the cost of the ball as \(x\) dollars. According to the problem, the bat costs \(x + 1.00\) dollars. The total cost of the bat and the ball together is \(1.10\) dollars.

The equation based on the given information is:
\[ x + (x + 1.00) = 1.10 \]

Simplifying the equation:
\[ 2x + 1.00 = 1.10 \]

Subtract \(1.00\) from both sides to isolate the term with \(x\):
\[ 2x = 0.10 \]

Now, divide both sides by \(2\) to solve for \(x\):
\[ x = \frac{0.10}{2} \]
\[ x = 0.05 \]

Therefore, the ball costs \(0.05\) dollars.

####0.05####
```

**14B NF4** — extracted `0.05` (correct); 
prefill 1.27 s, decode 30.19 s, 199 tokens, stop `eos`.

```
Let's denote the cost of the ball as \( x \) dollars. According to the problem, the bat costs 1.00 dollar more than the ball, so the cost of the bat is \( x + 1.00 \) dollars.

The total cost of the bat and the ball is 1.10 dollars, so we can write the equation:
\[ x + (x + 1.00) = 1.10 \]

Simplifying the left side of the equation:
\[ 2x + 1.00 = 1.10 \]

Subtract 1.00 from both sides to isolate the term with \( x \):
\[ 2x = 0.10 \]

Divide both sides by 2 to solve for \( x \):
\[ x = 0.05 \]

So, the ball costs 0.05 dollars.

#### 0.05
```

### 12. `prefill-warehouse`

**Gold:** `7429`

**Question**

```
You are given a logistics briefing. Read it carefully. Several numbers appear; only one of them is the warehouse access code.

In March 2021 the North dock of Meridian Freight processed 1800 pallets. Supervisor Elena Voss recorded a delay of 312 minutes when a scanner failed. The canteen bill that week was 55 dollars. None of those figures is the access code.

The night clerk kept a paper log. On Tuesday he wrote that bay 2 stayed closed and that 14 trucks arrived before 06:00. He also copied a sentence from a sealed envelope labelled dock procedure, revision 4. That sentence says the warehouse access code for gate C is 7429. The surrounding sentences are about pallet orientation, shrink-wrap tension, and a radio channel set to 19.

Elena's later notes mention last year's inventory variance of 0.4 percent, a visitor badge numbered 4501 that expired in April, a fire-panel test every 90 days, and 88 empty cages on a whiteboard. The night shift starts at 22:00. A spare key is tagged 2024.

Question: What is the warehouse access code for gate C? Put the final integer after ####.
```

**3B BF16** — extracted `7429` (correct); 
prefill 63.4 ms, decode 258.4 ms, 7 tokens, stop `eos`.

```
####7429####
```

**14B BF16** — extracted `7429` (correct); 
prefill 12.42 s, decode 1.38 min, 41 tokens, stop `eos`.

```
The warehouse access code for gate C is mentioned directly in the briefing: "the warehouse access code for gate C is 7429." Therefore, the answer is:

####7429
```

**3B NF4** — extracted `7429` (correct); 
prefill 1.31 s, decode 821.0 ms, 15 tokens, stop `eos`.

```
The warehouse access code for gate C is 7429.
```

**14B NF4** — extracted `7429` (correct); 
prefill 4.80 s, decode 7.68 s, 52 tokens, stop `eos`.

```
The warehouse access code for gate C is found in the sentence copied by the night clerk from a sealed envelope labelled dock procedure, revision 4. That sentence states the warehouse access code for gate C is 7429.

####7429
```

## Method reminders

- Two stacks: HuggingFace `generate` (BF16) vs `CompressedLinear` + `TokenLoop` (NF4).
- nvidia-smi is dedicated VRAM only and stops at 12,288 MiB.
- Display memory is inside those readings and is not subtracted.
- Notebook: [`notebooks/06_hard_eval.ipynb`](../notebooks/06_hard_eval.ipynb).
- Method: [`docs/eval.md`](eval.md).

