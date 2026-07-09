+++
title = "Distilling a trained RL actor into a much smaller one"
description = "A walkthrough of off-policy (behavioral cloning) and on-policy (DAgger) policy distillation, the three places a correct-looking implementation quietly gives you the wrong answer, and what the exercise actually measures"
date = 2026-07-06

[taxonomies]
tags = ["reinforcement-learning", "distillation", "imitation-learning", "machine-learning"]

[extra]
toc = true
comment = false
math = true
+++

This is a small tutorial on taking a large policy network that already works and pressing it into a much smaller one that works just as well. This came up in some of my ongoing research for analyzing communication in multi-agent RL. The pieces are generic: an off-policy stage (behavioral cloning on a fixed dataset) and an on-policy stage (DAgger), the loss that ties them together, and some plumbing to make a recurrent network work.

## The setup

I have a policy $\pi_{\text{teacher}}$ that is already trained and doing pretty good. In my case it is a transformer-based actor for a 20-agent [Pursuit](https://pettingzoo.farama.org/environments/sisl/pursuit/) task (twenty pursuers cornering eight evaders on a grid) from PettingZoo, trained with multi-agent PPO — [MAPPO](https://arxiv.org/abs/2103.01955), the multi-agent variant of [PPO](https://arxiv.org/abs/1707.06347). It has **~100k parameters** and captures 100% of the evaders. None of the PPO machinery matters for what follows, for the sake of this tutorial we can treat the teacher as a black box that maps a state to a distribution over actions.

The question we're attempting to answer for our case-study is simple:

> Can a network **much smaller** than the teacher **represent** the same policy?

This question is markedly different from the much harder question "can I train a small network from scratch to be this good". Just: does a small enough function class *contain* a good policy at all?

Distillation is an attempt to answer that. You freeze the teacher, use it as a supervisor, and fit a smaller student $\pi_\theta$ to it. If the small student reaches the teacher's performance, the small architecture can represent the policy.

A few terms to be familiar with:

* **Teacher / student**: the frozen good policy, and the smaller one being fit to it.
* **Off-policy distillation** (behavioral cloning): fit the student on a *fixed* dataset of states the teacher visited.
* **On-policy distillation** ([DAgger](https://arxiv.org/abs/1011.0686)): let the *student* choose which states to visit, and have the teacher label those.
* **Forward KL**: the loss, $\mathrm{KL}(\text{teacher} \Vert \text{student})$. It pulls the student's whole action distribution onto the teacher's, not just the top action.
* **Capture% / Done%**: task metrics — fraction of evaders caught, and fraction of episodes where all were caught.

## The words: imitation learning, behavioral cloning, distillation

These three terms are used interchangeably often and are worth clarifying because they will dictate the approach we are taking.

**Imitation learning** is the umbrella term that encompasses the idea of learning a policy from a supervisor instead of from reward. That's the only commitment and everything below is a choice made within it. (The other branch of imitation learning, inverse RL and adversarial methods like [GAIL](https://arxiv.org/abs/1606.03476) instead recovers a reward function and optimizes it. And those are separate beasts in themselves, and are useful when we don't have access to the internals of the teacher).

Within imitation learning there are two *independent* axes

**Axis 1 — what the target is.**

* A **hard action label**: the single action the expert took, matched with ordinary cross-entropy. This is what a demonstration dataset from a human gives us, we can see the action they took, not the distribution they drew it from. More relevant in setups when you only have access to the expert data but not the internals.
* The **full action distribution**: the expert's entire probability vector over actions, matched with KL. You only get this when the expert is a model we own and can read the logits out of.

**Axis 2 — where the training states come from.**

* **Off-policy**: a fixed dataset of states the *expert* visited. There is no interaction during training.
* **On-policy**: states the *learner* visits, which the expert then labels. Requires an expert you can query online

**Behavioral cloning** is the corner where both axes take their simplest value: hard labels, off-policy. A fixed dataset of (state, expert-action) pairs, supervised cross-entropy, done. It's the baseline, and it carries the covariate-shift weakness that motivates everything else.

**Classic DAgger** moves along axis 2 only: still hard action labels, but on-policy as the expert relabels the learner's own states with the action *it* would have taken.

**Policy distillation** moves along axis 1: the target is the expert's full distribution, matched with KL. The "expert" is a network we own (so we can read its distribution), and the usual goal is transfer or compression (a teacher into a smaller student) rather than learning from a demonstrator we couldn't otherwise optimize against. 

The two axes are independent, so there are four corners, not a single spectrum:

| | hard action label (cross-entropy) | full distribution (KL) |
| :--- | :--- | :--- |
| **off-policy** — expert's states | behavioral cloning | **offline distillation — Stage A here** |
| **on-policy** — learner's states | classic DAgger | **on-policy distillation — Stage B here** |


## The state distribution is the only difference 

Both stages minimize the same per-state loss — the forward KL between the two action distributions:

$$\mathrm{KL}\big(\pi_{\text{teacher}}(\cdot \mid s) \Vert \pi_\theta(\cdot \mid s)\big) = \sum_a \pi_{\text{teacher}}(a \mid s) \log \frac{\pi_{\text{teacher}}(a \mid s)}{\pi_\theta(a \mid s)}$$

The *only* difference between off-policy and on-policy is **which states $s$ you average this over**.

Off-policy (behavioral cloning) averages over states the teacher visits:

$$\min_\theta \mathbb E_{s \sim d_{\text{teacher}}}\big[\mathrm{KL}\big(\pi_{\text{teacher}}(\cdot \mid s) \Vert \pi_\theta(\cdot \mid s)\big)\big]$$

On-policy (DAgger) averages over states the *student* visits, and the teacher only supplies the labels:

$$\min_\theta \mathbb E_{s \sim d_\theta}\big[\mathrm{KL}\big(\pi_{\text{teacher}}(\cdot \mid s) \Vert \pi_\theta(\cdot \mid s)\big)\big]$$

That swap of $d_{\text{teacher}}$ for $d_\theta$ is the entire idea of DAgger. The reason it exists: a purely off-policy student is only ever trained on states the teacher reaches, but at deployment the student steers itself, makes a small error, and lands in a state slightly off the teacher's manifold (where it was never trained), so it makes more mistakes which in turn compunds over the episode. [Ross et al.](https://arxiv.org/abs/1011.0686) made this precise: plain behavioral cloning has worst-case regret growing like $O(T^2\varepsilon)$ over a horizon $T$, while iterating the on-policy relabel-and-retrain loop brings it to $O(T\varepsilon)$. The fix is to train on the student's own state distribution which you only get by letting the student drive.

So the recipe is: **do the cheap off-policy stage first, and only reach for the expensive on-policy stage if distribution shift is actually your problem.**

(We are saying the on-policy stage is expensive here because we need access to the expert network at real time, which is often much bigger than the student network)

## The target is a distribution, and you must evaluate by sampling

A peculiar thing that I noticed when I first did the off-policy stage was tht the teacher captured **100%** of evaders when its actions are *sampled* from $\pi_{\text{teacher}}(\cdot \mid s)$, and only about **35%** when its actions are the *argmax* of the same distribution. Same network, same weights, same states. Greedy play was somehow nearly three times worse than stochastic play.

The reason is specific to multi-agent coordination but I think the lesson generalizes. Twenty identical pursuers looking at a symmetric board compute nearly identical action distributions. If we take the argmax and they all commit to the *same* move , they clump, they mirror each other, the formation deadlocks and never closes the net. Sampling breaks the tie because with identical distributions, the randomness from the sampling of the distrubtions shatters the symmetry and the agents spread out to cover the evaders. **The competence lived in the entropy of the distribution, not in its mode.**

(In the MAPPO paper, an agent’s actions are not conditioned on other agents’ actions at the same time step. This trade‑off lets us either speed up training and inference by having all agents act simultaneously, or, as many other papers do, have agents decide causally based on others’ actions. This opens up a plethora of design decisions about how to sequence your agents taking actions. Since this is essentially auto‑regressive, we can explore other optimizations to speed it up like grouping, etc.)

Two consequences:

**1. Evaluate by sampling, not argmax.** The "obvious" way to evaluate a policy deterministically is to take the most likely action. Here that turns a 100% policy into a 35% one and you would conclude your teacher is broken before you even start distilling. Understand your setups well.

**2. Distill the full distribution, not the label.** The forward KL is, up to a constant, a cross-entropy against the teacher's *soft* distribution. Split the log to see it:

$$\mathrm{KL}(p \Vert q) = \sum_a p(a)\log\frac{p(a)}{q(a)} = \sum_a p(a)\log p(a) - \sum_a p(a)\log q(a) = H(p,q) - H(p)$$

The teacher's entropy is $H(p) = -\sum_a p(a)\log p(a)$, so the first sum is $-H(p)$. The cross-entropy of the student against the teacher is $H(p,q) = -\sum_a p(a)\log q(a)$, so the second sum is $-H(p,q)$. Entropy enters purely as bookkeeping: KL is the gap between those two.

$H(p)$ doesn't depend on the student, so minimizing forward KL is exactly minimizing the soft cross-entropy $-\sum_a p(a) \log q(a)$, the *entire* teacher distribution $p$ acts as the label. Contrasting this with the reflexive supervised-learning move: take $\arg\max_a p(a)$ as a hard one-hot label and minimize ordinary cross-entropy. That is $-\log q(\arg\max p)$. It throws away everything about `p` except its peak, which, as we just established, is precisely the part of the policy that *doesn't work*.

A small aside on the *direction* of the KL. Forward KL ($\mathrm{KL}(p \Vert q)$) with teacher $p$, student $q$ blows up wherever the teacher puts mass and the student doesn't, so to keep it finite the student must cover the teacher's *whole* distribution. This is usually referred to as the mass-covering, "mean-seeking" behavior, and it is exactly what we want when the competence is the spread. Reverse KL $\mathrm{KL}(q \Vert p)$ blows up the other way, wherever the student puts mass and the teacher doesn't so it rewards the student for concentrating on a single mode of the teacher and zeroing out the rest. This is the "mode-seeking" behaviour and here it is *precisely the failure mode*: a mode-collapsed student is the argmax-clumping student from the top of this section. 

Another aside on temperature, since the [distillation paper](https://arxiv.org/abs/1503.02531) makes a lot of it: you can soften both distributions by a temperature $T$ and scale the loss by $T^2$. For a 5-way action head at $T=1$ it barely moves the needle, so I left it at 1, but the knob is there in the code below.

## The one hook that makes all of this possible

Everything downstream needs one thing the standard RL forward pass doesn't expose: the full action logits. The RL path samples an action and returns it; distillation needs the whole distribution, for both collecting teacher targets and training the student. So I added a single method next to the existing forward pass, touching nothing on the RL path:

```python
def forward_logits(self, obs, rnn_states, masks, available_actions=None):
    """same data-flow as forward() but returns the full
    categorical log-probs instead of a sampled action. """
    obs   = check(obs).to(**self.tpdv)
    rnn_states = check(rnn_states).to(**self.tpdv)
    masks = check(masks).to(**self.tpdv)

    actor_features = self.base(obs)                 # encoder (here: a transformer)
    if self._use_recurrent_policy:
        actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)
    action_logits = self.act.action_out(actor_features, available_actions)
    return action_logits.logits, rnn_states         # log-softmax, (rows, n_actions)
```

* It serves both roles. The **teacher** uses it during data collection to emit targets; the **student** uses it during training as its trainable forward pass. Same code path, so teacher targets and student predictions are guaranteed comparable.
* The student is *just this actor*, no critic, no value function, no shared replay buffer, none of the RL scaffolding. So the student's width and depth are free parameters you can shrink at will without disturbing anything else.

## Step 1 — collect the teacher dataset (off-policy)

Roll the teacher through the environment and save complete episodes of `(obs, teacher_log_probs)`. **Roll it by sampling**, not greedily as you want to widen the support of the states the teacher actually visits in stochastic play, which is the regime where it works. Save the full log-softmax and not just the chosen action.

```python
teacher = R_Actor(args, obs_space, act_space, device=device)
teacher.load_state_dict(torch.load(f"{teacher_dir}/actor.pt", map_location=device))
teacher.eval()

obs   = envs.reset()                                    # (N_threads, n_agents, obs_dim)
rnn   = np.zeros((N, M, recN, H), dtype=np.float32)
masks = np.ones((N, M, 1), dtype=np.float32)
episodes, ep_obs, ep_logp = [], [[] for _ in range(N)], [[] for _ in range(N)]

while len(episodes) < num_episodes:
    with torch.no_grad():
        logits, rnn_new = teacher.forward_logits(
            np.concatenate(obs), np.concatenate(rnn), np.concatenate(masks))
        action = FixedCategorical(logits=logits).sample()      # SAMPLE, for coverage

    logp = logits.cpu().numpy().reshape(N, M, A)               # the distillation target
    for i in range(N):
        ep_obs[i].append(np.asarray(obs[i], np.float32))       # state the teacher acted on
        ep_logp[i].append(logp[i])                             # teacher's full distribution

    obs, _, dones, _ = envs.step([action.cpu().numpy().reshape(N, M)[i] for i in range(N)])
    done_env = np.all(dones, axis=-1)
    rnn = rnn_new.cpu().numpy().reshape(N, M, recN, H); rnn[done_env] = 0.0
    masks = np.ones((N, M, 1), np.float32); masks[done_env] = 0.0

    for i in np.flatnonzero(done_env):                          # close finished episodes
        episodes.append((np.stack(ep_obs[i]).astype(np.float32),   # (L, M, obs_dim)
                         np.stack(ep_logp[i]).astype(np.float16)))  # (L, M, A)
        ep_obs[i], ep_logp[i] = [], []

torch.save({"episodes": episodes}, out_path)
```

You collect this dataset **once** and reuse it for every student size. In my run it was 600 episodes, ~3M agent-steps. I chose to store the log-probs as `float16` as it halves the dataset size and the KL (in this case) is not too sensitive on the precision.

## Trap 2: a recurrent student is not an i.i.d. supervised problem

The student here is recurrent (a GRU rides on top of the encoder). The simplest way to do behavioral cloning — flatten every `(obs, target)` pair into one big table and shuffle it into minibatches — is *wrong* for a recurrent policy, and wrong in a way that trains fine and deploys badly.

The hidden state carries information across time. At deployment the student starts each episode with `h = 0` and rolls its hidden state forward step by step. If you train on shuffled individual timesteps, you have to invent a hidden state for each one (usually zero, or a stored stale value), and the student never learns to *use* the recurrence the way it will be used at test time. It fits the per-step targets and still can't reproduce the temporal behavior.

The fix for this is **whole-episode BPTT from a zero initial state**, where each stored episode is one training sequence, `h = 0` at `t = 0`, rolled through the whole episode exactly as at deployment. Episodes have different lengths, so we pad to the batch max length and mask the padding out of the loss.

```python
def build_batch(chunk, M, obs_dim, A, device):
    """Pad a list of variable-length episodes to (Lmax, B, M, ...) and flatten."""
    B, Lmax = len(chunk), max(o.shape[0] for o, _ in chunk)
    obs_b = torch.zeros(Lmax, B, M, obs_dim)
    tgt_b = torch.zeros(Lmax, B, M, A)                          # teacher log-probs
    mask  = torch.zeros(Lmax, B, M)                            # 1 on real steps, 0 on padding
    for b, (o, lp) in enumerate(chunk):
        L = o.shape[0]
        obs_b[:L, b] = torch.from_numpy(o)
        tgt_b[:L, b] = torch.from_numpy(lp.astype(np.float32))
        mask[:L, b]  = 1.0
    return (obs_b.reshape(Lmax*B*M, obs_dim).to(device),
            tgt_b.reshape(Lmax*B*M, A).to(device),
            mask.reshape(Lmax*B*M).to(device), B)

def kl_loss(student_logq, teacher_logp, mask, temp=1.0):
    """masked-mean forward KL(teacher ‖ student). Both args are log-softmax."""
    if temp != 1.0:
        teacher_logp = F.log_softmax(teacher_logp / temp, dim=-1)
        student_logq = F.log_softmax(student_logq / temp, dim=-1)
    p  = teacher_logp.exp()
    kl = (p * (teacher_logp - student_logq)).sum(-1)            # per agent-step
    return (kl * mask).sum() / mask.sum().clamp(min=1.0) * (temp**2)
```

The transformer encoder is non-recurrent over time, so padded timesteps don't interact with real ones and the masking is exact. Sorting episodes by length before batching keeps padding small.

## Step 2 — train the student

The most boring part of this whole study. Build a smaller actor, roll each episode from `h = 0`, forward KL, step. No RL, no critic, no reward.

```python
student = R_Actor(student_args, obs_space, act_space, device=device)   # smaller n_embd / n_block
opt = torch.optim.Adam(student.parameters(), lr=5e-4)

for epoch in range(distill_epochs):
    for chunk in batches_sorted_by_length(train_episodes, batch_episodes):
        obs_flat, tgt_flat, mask_flat, B = build_batch(chunk, M, obs_dim, A, device)
        rnn0  = torch.zeros(B*M, recN, student.hidden_size, device=device)   # h = 0
        Lmax  = obs_flat.shape[0] // (B*M)
        gmask = torch.ones(Lmax*B*M, 1, device=device)         # GRU never resets mid-episode

        log_s, _ = student.forward_logits(obs_flat, rnn0, gmask)   # whole-episode BPTT
        loss = kl_loss(log_s, tgt_flat, mask_flat, temp=1.0)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
```

We use 2 cheap validation signals here.  `val_KL` (held-out distillation KL) and `val_agree` (how often the student's argmax matches the teacher's on held-out states). But the metric that actually decides success is the **sampled task score** of the resulting actor. `val_agree` can drift down while sampled capture stays pinned at 100%, because the policy tolerates a lot of per-step disagreement as long as the *distribution* is close.

## The sweep

Fix the dataset, shrink the student, and read the sampled task score as a function of parameter count. The student's size is two knobs: the transformer's width (`n_embd`) and depth (`n_block`)

```bash
# one dataset, many students. n_head must divide n_embd.
for cfg in "64 2" "48 2" "32 2" "24 1" "16 2" "8 2" "6 1" "4 1" "3 1"; do
  set -- $cfg; N_EMBD=$1; N_BLOCK=$2
  python train_distill.py --n_embd $N_EMBD --hidden_size $N_EMBD --n_block $N_BLOCK \
      --n_head 4 --data_path s12_600ep.pt --distill_epochs 40 \
      --experiment_name "dst_w${N_EMBD}b${N_BLOCK}"
  # then eval each with SAMPLING (the SCoUT protocol)
done
```

The curve, on sampled 20-seed eval (teacher = 107,721 params, 100/100):

| student (n_embd / n_block) | params | Catch% | Done% |
| :--- | ---: | :---: | :---: |
| teacher | 107,721 | 100.0 | 100 |
| 48 / 2 | 48,105 | 100.0 | 100 |
| 32 / 2 | 22,921 | 100.0 | 100 |
| 16 / 2 | 6,953 | 100.0 | 100 |
| 8 / 2 | 2,425 | 100.0 | 100 |
| **6 / 1** | **1,377** | **100.0** | **100** |
| 4 / 1 | 889 | 98.1 | 85 |
| 3 / 1 | 681 | 61.9 | 5 |
| 2 / 1 | 497 | 4.4 | 0 |

The policy is astonishingly compressible. Capture stays at 100% down to roughly **1,377 parameters** (about 1/78 of the teacher) and it's robust (the ≥1,377 students hit 100/100 across multiple seeds). Around 900 params you're on a knife-edge (good runs depend on init luck), and below ~800 it falls off a cliff and by ~500 it's near-random. So there's a sharp **capacity wall at ~800 parameters**, and comfortably above it the policy is fully represented.

## The capacity wall

The cliff is tempting to attack with the on-policy stage. The tiny student (possibly) fails because it drifts into states the teacher's dataset never covered, and DAgger trains on the student's own states, therefore DAgger should rescue it (hopefully).

DAgger fixes one specific failure: **distribution shift**, that is when the student visits states it wasn't trained on. It does nothing for a different failure: **insufficient representational capacity**, where the function class is too small to fit the teacher's distribution *anywhere*. These look identical from the outside (student underperforms) and are diagnosable from the inside with one question:

> Does the training KL floor above zero on the student's *own* on-policy states?

If a 765-parameter student, trained directly on the states it itself visits, still can't drive its KL to the teacher below some floor, then the problem isn't which states it's seeing — it's that no setting of 765 parameters (with the given arch. constraints) fits the target.

We can run a quick control check. Warm-start a student at the cliff from its BC checkpoint (765 params, 63% capture), then 15 rounds of DAgger — student rolls out, teacher relabels, KL on the student's trajectories:

```python
@torch.no_grad()
def rollout_relabel(student, teacher, envs, n_ep):
    """Student acts (its own state distribution); teacher labels each visited state."""
    obs = envs.reset(); episodes = []
    while len(episodes) < n_ep:
        s_logits, _ = student.forward_logits(cat(obs), cat(s_rnn), cat(masks))
        t_logits, _ = teacher.forward_logits(cat(obs), cat(t_rnn), cat(masks))   # the labels
        act = FixedCategorical(logits=s_logits).sample()       # STUDENT drives — this is the point
        # store (obs, t_logits) as the training target; step the env with `act`; ...
    return episodes
```

The result: the rollout catch rate sat flat at ~58–62% every round and never climbed, the training KL floored at ~0.50 and stayed there, and the final sampled eval — **66.9 ± 15.3** capture — was statistically identical to where BC left it. No recovery. The KL floored because 765 parameters cannot represent the teacher's distribution, full stop; the student's state distribution was never the problem, so changing it changed nothing.

The clean statement the control buys you: **above ~900 params, off-policy BC already saturates (no shift to fix, DAgger adds nothing); at/below ~800, both BC and DAgger fail (a capacity wall, which on-policy data can't cross).** The ~800-parameter wall is real, not an artifact of doing the cheap thing.

The general rule, worth pinning up: **DAgger is the answer to "my student drifts off-distribution," never to "my student isn't big enough." Diagnose which one you have — does the on-policy KL floor? — before you build the on-policy loop.**

## The heuristic underneath all of this

Distillation gets sold as compression — "we shrank the model 44×." That framing invites the mistake this whole post is trying to prevent. The honest framing is that distillation is a **measurement instrument for representational capacity**, and it answers one question while staying silent on a second, more interesting one.

It answers: *can this small architecture represent a good policy?* Here, decisively yes — a ~1,400-parameter actor holds the full 20-agent solution.

It says nothing about: *can this small architecture be trained to that policy from scratch, by RL, under the real reward?* Those are different questions, and the gap between them has a name. The [lottery-ticket hypothesis](https://arxiv.org/abs/1803.03635) found that a pruned subnetwork which trivially *represents* a good function usually *cannot be retrained from scratch* to reach it without the original initialization — and separately, overparameterization is understood to help optimization even when the final function is simple. A tiny student scoring 100/100 is a statement about the target *existing* and being cheaply *deployable*. It is not a statement that you could have trained that tiny network to find the target on its own.

So the tiny-student result is a **representation and deployment** claim, not a training-method claim, and it is easy to over-sell into the second. In my case the legitimate reading is: the smaller architecture's low from-scratch RL scores are an *optimization / exploration* gap, not a *capacity* gap — the capacity is there, with two orders of magnitude to spare; RL just can't find it under sparse reward. Distillation didn't beat the from-scratch method; it *localized the from-scratch method's failure* to optimization rather than architecture. That is a far more useful thing to learn than a compression ratio, and it's the kind of read the code will never give you on its own.

## What I would keep in mind next time

* **Sampled vs greedy is the first thing to check, not an afterthought.** A policy whose competence lives in its entropy is destroyed by argmax. Evaluate it the way it's meant to run, and distill the whole distribution — the non-max probability mass is not noise, it's the policy.
* **Forward KL against soft targets, never cross-entropy against argmax labels.** They differ by exactly the entropy the good policy depends on.
* **A recurrent student is trained by whole-episode BPTT from `h = 0`, not by shuffling timesteps.** The shuffled version fits per-step targets and still can't reproduce the temporal behavior it will show at deployment.
* **Do the cheap off-policy stage first; only reach for on-policy DAgger if distribution shift is actually the failure.** Distinguish shift from capacity by asking whether the KL floors on the student's own states. DAgger crosses the first wall and never the second.
* **Read distillation as a capacity measurement, not a compression win.** It tells you the target exists and is deployable. It does not tell you a from-scratch method could have found it — capacity is not trainability.

## The whole pipeline in one place

Consolidated and stripped of argument-parsing and logging, so the data-flow is visible end to end. The three files in the repo (`collect_teacher_data.py`, `train_distill.py`, `dagger_distill.py`) are these three blocks with the boilerplate added back.

```python
import numpy as np, torch, torch.nn.functional as F
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
from onpolicy.algorithms.utils.distributions import FixedCategorical

# ---------------------------------------------------------------------------
# 0. The one hook. Same data-flow as the RL forward(), but returns the full
#    categorical log-probs. Add it to the actor; it touches nothing on the RL path.
# ---------------------------------------------------------------------------
def forward_logits(self, obs, rnn_states, masks, available_actions=None):
    obs, rnn_states, masks = (check(x).to(**self.tpdv) for x in (obs, rnn_states, masks))
    feats = self.base(obs)
    if self._use_recurrent_policy:
        feats, rnn_states = self.rnn(feats, rnn_states, masks)
    return self.act.action_out(feats, available_actions).logits, rnn_states   # log-softmax

# ---------------------------------------------------------------------------
# 1. COLLECT (off-policy dataset). Roll the teacher by SAMPLING; save complete
#    episodes of (obs, teacher_log_probs). Collect once, reuse for every student.
# ---------------------------------------------------------------------------
def collect(teacher, envs, num_episodes, N, M, recN, H, A):
    obs   = envs.reset()
    rnn   = np.zeros((N, M, recN, H), np.float32)
    masks = np.ones((N, M, 1), np.float32)
    eps, ep_o, ep_l = [], [[] for _ in range(N)], [[] for _ in range(N)]
    while len(eps) < num_episodes:
        with torch.no_grad():
            logits, rnn_new = teacher.forward_logits(np.concatenate(obs),
                                                     np.concatenate(rnn), np.concatenate(masks))
            action = FixedCategorical(logits=logits).sample()          # sample for coverage
        logp = logits.cpu().numpy().reshape(N, M, A)                   # full distribution = target
        for i in range(N):
            ep_o[i].append(np.asarray(obs[i], np.float32)); ep_l[i].append(logp[i])
        obs, _, dones, _ = envs.step([action.cpu().numpy().reshape(N, M)[i] for i in range(N)])
        d = np.all(dones, -1)
        rnn = rnn_new.cpu().numpy().reshape(N, M, recN, H); rnn[d] = 0.0
        masks = np.ones((N, M, 1), np.float32); masks[d] = 0.0
        for i in np.flatnonzero(d):
            eps.append((np.stack(ep_o[i]).astype(np.float32),
                        np.stack(ep_l[i]).astype(np.float16)))
            ep_o[i], ep_l[i] = [], []
    return eps

# ---------------------------------------------------------------------------
# 2. Whole-episode BPTT batching + forward KL (shared by off- and on-policy).
# ---------------------------------------------------------------------------
def build_batch(chunk, M, obs_dim, A, device):
    B, Lmax = len(chunk), max(o.shape[0] for o, _ in chunk)
    obs_b = torch.zeros(Lmax, B, M, obs_dim); tgt_b = torch.zeros(Lmax, B, M, A)
    mask  = torch.zeros(Lmax, B, M)
    for b, (o, lp) in enumerate(chunk):
        L = o.shape[0]
        obs_b[:L, b] = torch.from_numpy(o)
        tgt_b[:L, b] = torch.from_numpy(lp.astype(np.float32)); mask[:L, b] = 1.0
    return (obs_b.reshape(Lmax*B*M, obs_dim).to(device),
            tgt_b.reshape(Lmax*B*M, A).to(device),
            mask.reshape(Lmax*B*M).to(device), B)

def kl_loss(logq, logp, mask, temp=1.0):
    if temp != 1.0:
        logp, logq = F.log_softmax(logp/temp, -1), F.log_softmax(logq/temp, -1)
    kl = (logp.exp() * (logp - logq)).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1.0) * (temp**2)

def train_epoch(student, opt, episodes, batch_eps, M, obs_dim, A, recN, device):
    for chunk in batches_sorted_by_length(episodes, batch_eps):
        obs_f, tgt_f, mask_f, B = build_batch(chunk, M, obs_dim, A, device)
        rnn0 = torch.zeros(B*M, recN, student.hidden_size, device=device)      # h = 0
        gmask = torch.ones(obs_f.shape[0]//(B*M) * B*M, 1, device=device)
        log_s, _ = student.forward_logits(obs_f, rnn0, gmask)                  # BPTT over episode
        loss = kl_loss(log_s, tgt_f, mask_f, temp=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0); opt.step()

# ---------------------------------------------------------------------------
# 3a. OFF-POLICY (behavioral cloning): fit the student on the fixed dataset.
# ---------------------------------------------------------------------------
def distill_offpolicy(student_args, dataset, obs_space, act_space, device, epochs=40):
    student = R_Actor(student_args, obs_space, act_space, device=device)
    opt = torch.optim.Adam(student.parameters(), lr=5e-4)
    for _ in range(epochs):
        train_epoch(student, opt, dataset, 8, student_args.n_pursuers,
                    obs_space.shape[0], act_space.n, student_args.recurrent_N, device)
    return student                                    # deploy this; evaluate by SAMPLING

# ---------------------------------------------------------------------------
# 3b. ON-POLICY (DAgger): student drives, teacher labels, KL on student states.
#     Warm-start from a BC student. Only worth it if the failure is shift, not capacity.
# ---------------------------------------------------------------------------
def distill_onpolicy(student, teacher, envs, rounds=15, ep_per_round=48, cap=600):
    opt, buffer = torch.optim.Adam(student.parameters(), lr=5e-4), []
    for r in range(rounds):
        student.eval()
        fresh = rollout_relabel(student, teacher, envs, ep_per_round)   # student acts, teacher labels
        buffer = (buffer + fresh)[-cap:]
        student.train()
        for _ in range(8):
            train_epoch(student, opt, buffer, 8, ...)                   # same KL loop as 3a
        # DIAGNOSTIC: if train-KL floors above ~0 here, you are capacity-bound, not
        # shift-bound — stop; more on-policy rounds will not help.
    return student
```

The full sources, the per-seed run manifest, and the complete parameter-vs-capture curve live in the repo alongside this pipeline. But the three blocks above — the hook, the sampled soft-target collection, the whole-episode KL loop — are the entire method. Everything else is the boilerplate a coding agent is genuinely good at, and the three traps are the part it isn't.
