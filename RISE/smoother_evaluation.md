# Requirement: Benchmark Ours vs. Fully Differentiable Full-Backprop Attack

## 0. Objective

Implement a completely separate benchmark script to compare:

1. **Ours / Stop-Gradient Attack**
   - The existing attack implementation.
   - The explanation is generated normally.
   - Top-K masks are detached/frozen during attack optimization.
   - Existing Del/Ins curves and AUC values are already available and should be REUSED.
   - Do NOT recompute the existing attack's Del/Ins AUC unless necessary for validation.

2. **Full-Backprop Attack**
   - A new independent implementation.
   - Replace the hard/non-differentiable explanation → sorting → Top-K → mask pipeline with the differentiable/smoothed formulation provided by the IDExpO repository.
   - The full computation graph must remain differentiable from the faithfulness objective back to the adversarial perturbation.
   - This includes the differentiable explanation computation, differentiable sorting/ranking, and differentiable mask/reference-image construction used by IDExpO.
   - For gradient-based explainers such as Grad-CAM, support the required higher-order differentiation / second-order backward.
   - Do NOT modify the existing explainer implementation in the main project.

The purpose of this experiment is NOT simply to compare final attack performance.

The experiment must answer three questions:

A. Does full backpropagation through the explainer actually provide better attack performance?

B. If performance is comparable, how much additional computational cost does full backpropagation require?

C. Does the proposed stop-gradient design achieve a better performance-vs-resource trade-off?

---

# 1. Source of the Full-Backprop Formulation

There is an existing IDExpO repository located at:

D:\Adversarial-Robustness-Grad-Eclip\idexpo

The repository provides a differentiable formulation of insertion/deletion-related operations, including the smooth/differentiable treatment of operations such as sorting/ranking and reference-image construction.

IMPORTANT:

- Inspect the actual code in this repository.
- Reuse/adapt its mathematical formulation and implementation where appropriate.
- Do NOT blindly reimplement a different approximation.
- The exact differentiable formulation from the repository must be preserved as much as possible.
- Identify the exact functions/classes responsible for:
  - differentiable insertion,
  - differentiable deletion,
  - differentiable sorting/ranking,
  - reference-image construction,
  - smooth mask/weight generation,
  - any temperature/smoothing parameters.
- Document which IDExpO components are used.

The user will provide the repository if it is not currently accessible.

Do not assume that the current project's explainer implementation and IDExpO's implementation are interchangeable.

---

# 2. Critical Constraint: Do NOT Modify Existing Explainer Code

The existing explanation implementations in the main project must remain untouched.

This includes:

- Grad-CAM
- GradECLIP
- GAME
- MaskCLIP
- RISE
- any existing explanation utility
- any existing attack implementation

Do NOT:

- modify their source files,
- add second-order support directly into them,
- change their current APIs,
- change their current forward/backward behavior,
- replace their existing gradient implementation.

Instead, create a completely separate implementation for the full-backprop experiment.

Suggested structure:

    benchmark_fullbp/
        __init__.py
        fullbp_gradcam.py
        differentiable_faithfulness.py
        benchmark_fullbp.py
        metrics.py
        profiling.py
        config.py
        README.md

The exact directory structure can be adjusted to match the existing project, but the implementation must remain isolated.

---

# 3. Initial Explainer for This Experiment

For the first experiment, support ONLY:

    Grad-CAM

Do not implement all explainers initially.

The script must expose:

    --explainer gradcam

and reject unsupported explainers cleanly.

The reason for starting with Grad-CAM is that it provides a meaningful test of the computational overhead caused by differentiating through a gradient-based explanation.

---

# 4. Two Different Grad-CAM Paths

There must be two conceptually separate paths.

## 4.1 Existing/Ours Grad-CAM

Use the existing Grad-CAM implementation exactly as it is.

Do not modify it.

This is only used to identify the existing Ours results and/or run validation.

The existing method already uses:

    x_adv
       ↓
    Grad-CAM
       ↓
    saliency S
       ↓
    Top-K
       ↓
    mask M_t
       ↓
    detach / stop-gradient
       ↓
    masked image
       ↓
    predictor
       ↓
    faithfulness loss

---

## 4.2 Full-Backprop Grad-CAM

Create a new Grad-CAM implementation specifically for this experiment.

The new implementation must support higher-order differentiation.

Conceptually:

    x_adv
       ↓
    predictor forward
       ↓
    first-order gradient
       ↓
    Grad-CAM saliency
       ↓
    differentiable ranking/sorting
       ↓
    differentiable mask
       ↓
    differentiable Del/Ins objective
       ↓
    second-order backward
       ↓
    gradient w.r.t. x_adv / delta

The first gradient used to construct Grad-CAM must preserve its computation graph.

For PyTorch-style implementation, this typically means that the first-order gradient must be computed with graph construction enabled, e.g. conceptually:

    autograd.grad(..., create_graph=True)

Do NOT blindly copy this exact implementation if the existing model/Grad-CAM architecture requires another mechanism. Inspect the actual Grad-CAM implementation and adapt accordingly.

The essential requirement is:

    d(GradCAM(x)) / dx

must remain differentiable.

This means that when the faithfulness objective is backpropagated, the graph can reach the gradient operation used to produce the Grad-CAM saliency.

---

# 5. Mathematical Definition of the Two Backward Paths

For deletion, define:

    z_t = M_t(x_adv) ⊙ x_adv

where:

    x_adv = x + delta

and:

    M_t = TopK(G(x_adv))

For a fully differentiable formulation:

    dz_t / d(delta)
      =
        M_t
        +
        x_adv ⊙ dM_t/d(delta)

Therefore:

    dL/d(delta)
      =
        dL/dz_t
        [
            M_t
            +
            x_adv ⊙ dM_t/d(delta)
        ]

The Ours implementation intentionally removes the second term:

    dM_t/d(delta) = 0

because M_t is detached/fixed during the optimization step.

Therefore:

    dz_t/d(delta) = M_t

and the remaining direct gradient path is:

    L
      → predictor
      → M_t ⊙ x_adv
      → x_adv
      → delta

The Full-BP implementation must retain both terms:

    L
      → predictor
      → differentiable mask
      → differentiable saliency
      → Grad-CAM gradient operation
      → x_adv
      → delta

The benchmark must therefore NOT detach the differentiable mask or saliency in the Full-BP path.

---

# 6. Important: Hard Top-K Must NOT Be Used in the Full-BP Path

The Full-BP path cannot use the existing hard:

    TopK(...)

operation if that operation breaks the gradient path.

Instead, use the smooth/differentiable ranking/sorting/masking formulation from IDExpO.

The purpose is to construct:

    M_t^diff

such that:

    M_t^diff = differentiable_function(S)

and:

    dM_t^diff / dS

is available during backward.

Do NOT use:

    torch.topk(...)

followed by pretending that it is differentiable.

Do NOT use:

    detach()

on the differentiable mask.

Do NOT use:

    no_grad()

inside the Full-BP explanation-to-mask path.

---

# 7. Faithfulness Objective

The existing attack defines the prediction-preservation term:

    L_pred = D_KL(f(x) || f(x_adv))

and the causal deletion/insertion objectives.

Follow the exact objective used by the existing implementation.

In particular, preserve the current definitions:

    L_del_obj = L_del - L_pred

    L_ins_obj = -(L_ins + L_pred)

and:

    L_total
      =
        lambda_del * L_del_obj
        +
        lambda_ins * L_ins_obj

The Full-BP version must optimize the same conceptual objective.

The only intended difference is:

    Ours:
        hard mask + stop-gradient

    Full-BP:
        differentiable/smooth mask + full backward

Do NOT change:

- epsilon,
- alpha,
- PGD update rule,
- number of attack iterations,
- lambda_del,
- lambda_ins,
- target class,
- prediction-loss definition,

unless explicitly specified through command-line arguments.

---

# 8. Del/Ins Evaluation Must Remain Causal / Original

This is VERY IMPORTANT.

The differentiable Del/Ins formulation from IDExpO is used for:

    optimization

It must NOT automatically replace the final reported evaluation metric.

For Full-BP:

    differentiable Del/Ins
        ↓
    used during optimization

but after every attack step:

    x_adv
        ↓
    ordinary explanation/evaluation procedure
        ↓
    ordinary causal deletion curve
    ordinary causal insertion curve
        ↓
    numerical AUC

The reported Del-AUC and Ins-AUC must therefore be based on the same causal evaluation protocol used for Ours.

This prevents the Full-BP method from being evaluated using its own surrogate metric.

---

# 9. Per-Step Performance Measurement

For every PGD attack iteration k:

    k = 0, 1, ..., K

record:

    Del-AUC(k)
    Ins-AUC(k)

for both:

    Ours
    Full-BP

For Ours:

- Existing Del/Ins curve/AUC results are already available.
- Reuse those results.
- Do not rerun the entire existing attack merely to calculate the same AUC.

For Full-BP:

- The attack produces a new x_adv at every iteration.
- At each iteration, evaluate the adversarial example using the original causal Del/Ins procedure.
- Compute the complete Del curve.
- Compute the complete Ins curve.
- Compute AUC for each curve.

Output should contain:

    sample_id
    method
    attack_step
    del_auc
    ins_auc

---

# 10. Del/Ins Curve Evaluation

At each attack step, use the same number of causal evaluation steps T as the existing experiment.

For deletion:

    x_t_del = M_t ⊙ x_adv

For insertion:

    x_t_ins = (1 - M_t) ⊙ x_adv

where the evaluation mask is generated according to the same explanation/evaluation protocol used by the existing attack.

Do NOT use the smooth optimization mask as the final evaluation mask unless this is explicitly required by the original evaluation protocol.

The purpose is to ensure:

    Ours
    vs.
    Full-BP

are evaluated using the same metric.

---

# 11. AUC Calculation

Use exactly the same AUC convention as the existing project.

Do not silently change:

- normalization,
- integration range,
- x-axis,
- number of causal steps,
- interpolation method,
- direction of deletion,
- direction of insertion.

If the existing implementation already contains an AUC utility, reuse it for evaluation.

If it does not, reproduce the exact existing mathematical definition and document it.

The benchmark must report:

    Del-AUC
    Ins-AUC

with the same interpretation as the current paper.

---

# 12. Fair Experimental Conditions

Ours and Full-BP must use exactly the same:

    model
    dataset
    samples
    sample ordering
    random seed
    target class
    epsilon
    alpha
    PGD iterations
    causal evaluation steps
    Top-K ratio / K
    lambda_del
    lambda_ins

Only the gradient path should differ.

The comparison should therefore isolate:

    stop-gradient explanation optimization

vs.

    full differentiable explanation optimization

---

# 13. Number of Samples

The script must support selecting the number of samples.

Example:

    --num-samples 10

for debugging.

Then:

    --num-samples 100

for the actual benchmark.

If:

    --num-samples -1

is supported, use all available samples.

The exact default should be conservative, e.g.:

    --num-samples 10

to prevent accidentally launching a very expensive second-order experiment.

---

# 14. Command-Line Arguments

Implement arguments approximately as follows:

    --num-samples
    --seed
    --dataset
    --model
    --explainer
    --steps
    --alpha
    --epsilon
    --topk
    --causal-steps
    --lambda-del
    --lambda-ins
    --device
    --output-dir
    --temperature

The `--temperature` or equivalent smoothing parameter must be passed to the IDExpO differentiable formulation if required.

If IDExpO uses a different parameter name or multiple smoothing parameters, expose the relevant parameters clearly.

Example:

    python benchmark_fullbp.py \
        --num-samples 10 \
        --explainer gradcam \
        --steps 50 \
        --causal-steps 20 \
        --epsilon 8/255 \
        --alpha 1/255 \
        --lambda-del 1 \
        --lambda-ins 1 \
        --device cuda \
        --output-dir results/fullbp_gradcam

Adapt model/dataset arguments to the actual project API instead of inventing incompatible loaders.

---

# 15. Timing Measurement

The benchmark must measure computational time.

Record at minimum:

    explanation_time
    faithfulness_time
    backward_time
    update_time
    total_step_time

Also record:

    total_attack_time

and:

    average_step_time

IMPORTANT:

CUDA operations are asynchronous.

When using CUDA, synchronize before and after timing.

Conceptually:

    torch.cuda.synchronize()
    start = time.perf_counter()

    computation()

    torch.cuda.synchronize()
    end = time.perf_counter()

Do NOT report GPU timing using unsynchronized wall-clock measurements.

---

# 16. Separate Timing of First-Order and Second-Order Components

For Full-BP Grad-CAM, explicitly measure:

    gradcam_forward_time
    first_order_gradient_time
    differentiable_sort_mask_time
    faithfulness_forward_time
    second_order_backward_time
    optimizer_update_time

If the implementation makes it impossible to isolate one component precisely, document that limitation instead of producing a fake measurement.

The most important measurement is:

    second_order_backward_time

because this is the additional computational burden that the proposed stop-gradient design is intended to avoid.

---

# 17. GPU Memory Measurement

Measure peak GPU memory for every method.

Before an attack:

    torch.cuda.reset_peak_memory_stats()

After the relevant computation:

    torch.cuda.max_memory_allocated()

Record:

    peak_gpu_memory_mb

Optionally also record:

    peak_gpu_reserved_mb

Do not compare memory unless both methods are measured under the same CUDA device and runtime conditions.

The benchmark should report:

    Ours peak VRAM
    Full-BP peak VRAM
    memory overhead ratio

where:

    memory overhead ratio
      =
        Full-BP peak memory / Ours peak memory

---

# 18. Important Memory Requirement for Second-Order Gradients

The Full-BP implementation must correctly retain the computation graph required for higher-order differentiation.

For gradient-based Grad-CAM, the first-order gradient used to generate saliency may need graph construction enabled.

Conceptually:

    first_grad = grad(
        score,
        input,
        create_graph=True
    )

The exact implementation must follow the existing model/Grad-CAM architecture.

The benchmark must verify that:

    loss.backward()

actually produces a non-zero gradient with respect to:

    delta

through the explanation path.

Do not merely set:

    create_graph=True

and assume that the graph is correct.

---

# 19. Gradient-Path Verification

Implement an optional debug mode:

    --debug-gradient

It should verify:

1. Grad-CAM saliency requires gradient.
2. The first-order gradient has `requires_grad=True` when required.
3. The differentiable mask has `requires_grad=True`.
4. The final faithfulness loss has a valid gradient with respect to `delta`.
5. The gradient norm is non-zero.

Report:

    ||grad_delta||
    ||grad_through_explainer||

If possible, separately verify the contribution from the explanation path.

The Full-BP implementation must NOT silently fall back to a detached gradient.

---

# 20. Stop-Gradient Verification

For Ours, verify that:

    dM/d(delta) = 0

in the computational graph.

For Full-BP, verify that:

    dM_diff/d(delta)

is connected to the graph.

This is important because the benchmark is specifically testing:

    detached mask

vs.

    differentiable mask

---

# 21. Important Distinction: Explanation Forward vs. Explanation Backward

The benchmark must explicitly distinguish:

    explanation generation cost

from:

    differentiation through explanation

Ours still needs to generate the explanation.

Therefore, do NOT claim:

    "Ours does not require Grad-CAM computation."

That would be incorrect.

The intended comparison is:

    Ours:
        explanation forward / first-order explanation generation
        +
        ordinary faithfulness backward

vs.

    Full-BP:
        explanation forward / first-order explanation generation
        +
        retained graph
        +
        differentiable explanation path
        +
        higher-order backward

The experiment should quantify this additional cost.

---

# 22. Complexity Reporting

Do NOT claim that the two methods necessarily have different asymptotic Big-O complexity merely because one uses stop-gradient.

Instead report the computational decomposition.

For Ours:

    C_ours
      ≈
      C_explanation
      +
      C_faithfulness
      +
      C_ordinary_backward

For Full-BP:

    C_full
      ≈
      C_explanation
      +
      C_faithfulness
      +
      C_ordinary_backward
      +
      C_higher_order

where:

    C_higher_order

represents the additional cost caused by differentiating through the gradient-based explanation and differentiable ranking/masking.

For gradient-based explainers, this may involve second-order derivatives of the predictor.

The benchmark should empirically measure this cost rather than assuming a universal Big-O ratio.

---

# 23. Output Files

Create:

    results/
        fullbp_gradcam/
            per_step.csv
            per_sample.csv
            summary.json
            config.json
            curves.npz
            plots/

`per_step.csv`:

    sample_id
    method
    attack_step
    del_auc
    ins_auc
    explanation_time
    first_order_gradient_time
    differentiable_sort_time
    faithfulness_time
    backward_time
    second_order_backward_time
    update_time
    total_step_time
    peak_gpu_memory_mb

`per_sample.csv`:

    sample_id
    method
    final_del_auc
    final_ins_auc
    total_attack_time
    average_step_time
    peak_gpu_memory_mb

`summary.json` should contain:

    number_of_samples
    model
    dataset
    explainer
    steps
    epsilon
    alpha
    causal_steps
    lambda_del
    lambda_ins
    average_final_del_auc
    average_final_ins_auc
    average_total_time
    average_time_per_step
    average_peak_gpu_memory

Also report standard deviation across samples whenever meaningful.

---

# 24. Performance Curves

Generate at least:

## Figure 1: Del-AUC vs PGD Step

    x-axis:
        PGD iteration

    y-axis:
        Del-AUC

Curves:

    Ours
    Full-BP

Use the exact same metric direction/convention as the paper.

---

## Figure 2: Ins-AUC vs PGD Step

    x-axis:
        PGD iteration

    y-axis:
        Ins-AUC

Curves:

    Ours
    Full-BP

---

## Figure 3: Attack Effectiveness vs Wall-Clock Time

This is especially important.

Use:

    x-axis:
        wall-clock time

    y-axis:
        attack effectiveness

The exact effectiveness definition should be documented.

Possible definitions include normalized degradation from the clean/no-attack baseline.

The goal is to determine whether:

    Ours reaches the same attack effectiveness
    at a lower computational cost.

The desired evidence is:

    C_ours(E*)
        <
    C_fullBP(E*)

for the same effectiveness E*.

Do NOT fabricate such a result; calculate it from the actual benchmark.

---

## Figure 4: Peak GPU Memory vs Performance

Compare:

    Ours
    Full-BP

at equivalent attack performance.

This should demonstrate whether Full-BP requires substantially more memory due to retained/higher-order computation graphs.

---

# 25. Performance-at-Equal-Cost Analysis

Do not only compare final performance after K iterations.

Also compare:

    performance at equal number of steps

and:

    performance at equal wall-clock time

and, if possible:

    performance at equal GPU-memory budget.

For a target effectiveness E*:

    cost_ours(E*)
    cost_fullbp(E*)

and calculate:

    cost_ratio
      =
      cost_fullbp(E*) / cost_ours(E*)

If:

    cost_ratio > 1

then Full-BP requires more resource to achieve the same effectiveness.

Only report this if both curves actually reach E*.

If one method never reaches the selected target, report:

    not reached

rather than extrapolating.

---

# 26. Statistical Reporting

Across samples, report:

    mean
    standard deviation
    median

for:

    final Del-AUC
    final Ins-AUC
    total attack time
    average time/step
    peak GPU memory

For performance curves, preferably report mean ± standard deviation across samples.

---

# 27. Warm-Up and Benchmark Fairness

Before collecting timing measurements:

- perform a small CUDA warm-up,
- load the same model once,
- use the same device,
- use the same precision,
- use the same batch size,
- avoid unrelated background computations.

Do not include:

- model loading time,
- dataset loading time,
- plotting time,

inside the attack computation time.

Clearly define what is included in:

    attack_time

and:

    step_time

---

# 28. Caching

Caching is allowed only where it does not create an unfair advantage.

For example:

- model weights may be loaded once,
- dataset samples may be preloaded,
- static ground-truth information may be cached.

Do NOT cache:

- Full-BP saliency,
- differentiable masks,
- attack-dependent Del/Ins curves,

because these depend on the current `x_adv`.

Ours may reuse its already-computed Del/Ins results because those results already exist and the explicit experimental requirement is to reuse them.

---

# 29. Reproducibility

Save the complete configuration to:

    config.json

including:

    random seed
    model
    dataset
    sample count
    sample IDs
    epsilon
    alpha
    attack steps
    causal steps
    Top-K
    lambda values
    differentiable sorting temperature
    device
    dtype

The benchmark must print the configuration at startup.

---

# 30. Debug Mode

Implement:

    --debug

When enabled, run only 1–2 samples and 2–3 attack iterations.

Print:

    sample ID
    attack step
    loss_del
    loss_ins
    total loss
    grad norm
    peak GPU memory
    step time

For Full-BP also print:

    Grad-CAM first-order gradient norm
    final delta gradient norm
    whether differentiable mask requires grad
    whether saliency requires grad

This should make it easy to detect accidental `.detach()` or missing `create_graph=True`.

---

# 31. Sanity Checks

Before running the full benchmark, verify:

### Sanity check 1

With Full-BP disabled, the new benchmark must not alter the existing Ours implementation.

### Sanity check 2

Full-BP `delta.grad` must be non-zero.

### Sanity check 3

The differentiable mask must have a gradient path to `x_adv`.

### Sanity check 4

Removing `create_graph=True` in the gradient-based explanation path should break or eliminate the intended higher-order gradient. This can be tested only in debug mode.

### Sanity check 5

The Full-BP optimization loss should change across iterations.

### Sanity check 6

The reported Del/Ins evaluation must use the original causal evaluation procedure rather than the optimization surrogate.

---

# 32. No Silent Fallbacks

This is critical.

If:

- second-order gradient fails,
- differentiable sorting fails,
- the mask is detached,
- the explanation has no gradient path,
- CUDA memory is insufficient,

the program must raise a clear error.

Do NOT silently:

- detach tensors,
- switch to first-order gradients,
- use hard Top-K,
- use the existing non-differentiable explainer,
- skip the explanation path.

Otherwise the benchmark would incorrectly label a non-full-backprop method as Full-BP.

---

# 33. Expected Experimental Question

The final experiment should allow us to make a statement of the following form:

> We compare the proposed stop-gradient attack with a fully differentiable formulation that backpropagates through the explanation generation and differentiable ranking/masking operations. While both methods optimize the same faithfulness-oriented objective, the proposed method avoids the additional higher-order gradient computation required by full backpropagation.

Then empirically determine:

1. Whether Full-BP improves Del-AUC.
2. Whether Full-BP improves Ins-AUC.
3. Whether Full-BP improves attack effectiveness overall.
4. Whether Ours achieves comparable performance.
5. How much slower Full-BP is.
6. How much additional GPU memory Full-BP requires.
7. Whether Ours reaches a given effectiveness at lower wall-clock cost.

Do NOT assume the result beforehand.

---

# 34. Main Hypothesis

The experiment is designed to test the following hypothesis:

    Full-BP:
        potentially richer gradient information
        but substantially higher computational/memory cost.

    Ours:
        removes the explainer-gradient term
        but retains the direct faithfulness gradient through x_adv
        while repeatedly recomputing the explanation in the forward pass.

The empirical result should determine whether the removed explainer-gradient term provides meaningful additional attack performance.

---

# 35. Important Conceptual Point

The experiment must preserve this distinction:

    "No gradient through the explainer"

does NOT mean:

    "No explanation information."

Ours still computes:

    x_adv
      → Grad-CAM
      → saliency
      → Top-K
      → mask

during the forward process.

It only prevents:

    L
      → mask
      → saliency
      → explanation gradient
      → x_adv

during the backward process.

This distinction should be reflected in the implementation and documentation.

---

# 36. Final Deliverables

The implementation should provide:

1. A standalone Full-BP Grad-CAM implementation.
2. A standalone differentiable faithfulness implementation adapted from IDExpO.
3. A standalone benchmark script.
4. CLI support for selecting sample count.
5. Per-step Del/Ins AUC logging.
6. Wall-clock timing.
7. First-order and second-order backward timing where measurable.
8. Peak GPU memory measurement.
9. CSV/JSON outputs.
10. Performance-vs-step plots.
11. Performance-vs-time plots.
12. Memory-vs-performance plots.
13. Debug gradient verification.
14. Reproducible configuration.
15. README explaining how to run the benchmark.

---

# 37. Implementation Priority

Implement in this order:

### Phase 1
Inspect the existing project:

- identify model interface,
- identify current Grad-CAM,
- identify current attack,
- identify existing Del/Ins-AUC implementation,
- identify where Ours results are stored.

### Phase 2
Inspect IDExpO:

- identify differentiable sorting,
- identify differentiable reference image construction,
- identify differentiable insertion/deletion formulation,
- identify smoothing/temperature parameters.

### Phase 3
Implement isolated Full-BP Grad-CAM.

### Phase 4
Verify second-order gradient flow.

### Phase 5
Implement Full-BP attack.

### Phase 6
Implement causal evaluation and per-step AUC.

### Phase 7
Implement timing and memory profiling.

### Phase 8
Run a tiny debug experiment:

    2 samples
    3 PGD steps

### Phase 9
Run the requested benchmark:

    configurable sample count
    Grad-CAM only

### Phase 10
Generate final comparison tables and plots.

---

# 38. Do Not Do These Things

DO NOT:

- modify existing explainer code,
- modify existing attack code,
- replace existing Grad-CAM globally,
- use hard Top-K in Full-BP,
- detach the Full-BP mask,
- use `torch.no_grad()` in the Full-BP gradient path,
- accidentally use `create_graph=False` when higher-order differentiation is required,
- evaluate Full-BP using only its differentiable surrogate metric,
- compare different samples,
- compare different attack hyperparameters,
- claim asymptotically lower Big-O complexity without proof,
- claim lower resource usage without measuring it,
- claim better performance before running the experiment,
- silently fall back to first-order/non-differentiable computation.

---

# 39. Desired Final Comparison

The final result should be summarized in a table like:

| Method | Del-AUC ↓ | Ins-AUC ↓ | Time / Step ↓ | Total Time ↓ | Peak VRAM ↓ |
|--------|-----------|-----------|---------------|--------------|-------------|
| Ours (Stop-grad) | ... | ... | ... | ... | ... |
| Full-BP | ... | ... | ... | ... | ... |

Additionally:

| Target Effectiveness | Ours Cost | Full-BP Cost | Full-BP / Ours |
|----------------------|-----------|--------------|----------------|
| E1 | ... | ... | ... |
| E2 | ... | ... | ... |
| E3 | ... | ... | ... |

The most important result is not necessarily that Ours has better final performance.

The important question is whether:

    comparable effectiveness

can be obtained with:

    substantially lower time and/or memory

because the proposed method avoids the higher-order backward path through the explanation generator.