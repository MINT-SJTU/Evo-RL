# ARX5 Dual Infer RTC Design

## Goal

Create a new entrypoint:

- `src/lerobot/scripts/lerobot_arx5_dual_infer_rtc.py`

based on:

- `src/lerobot/scripts/lerobot_arx5_dual_infer.py`

with Real-Time Chunking (RTC) support for dual-arm ARX5 inference.

This version intentionally narrows scope:

- keep dual-arm policy inference
- keep camera pipeline
- keep keyboard stop / resume behavior
- keep action clipping
- remove ACP
- remove raw recorder
- remove VR handoff / teleop takeover
- remove "safe-mode press once to fetch one chunk" interaction

The target behavior is continuous runtime with RTC-style asynchronous chunk refresh.

## Why Single-Thread RTC Is Not Enough

If we stay single-threaded, the loop is still:

1. read observation
2. run policy inference
3. wait for inference to finish
4. execute actions
5. repeat

That is not real RTC execution. It is still synchronous chunking, even if `policy.predict_action_chunk(...)` receives RTC kwargs.

The core RTC idea is:

- while the robot is still executing leftover actions from the previous chunk
- start computing the next chunk early
- when the new chunk arrives, merge it with the execution already in progress

Without overlap between inference and execution, there is no latency hiding. In that case:

- `prev_chunk_left_over` exists
- `inference_delay` can be estimated
- RTC guidance can still run mathematically

but robot-side timing does not improve much, because inference is still blocking action rollout.

Conclusion:

- a "single-thread RTC" can be useful as a debugging baseline
- but it does not deliver the main benefit we want
- for actual deployment, the script should be asynchronous

## High-Level Design

Split runtime into two cooperating loops:

1. Main control loop
2. Inference worker

### Main Control Loop

Responsibilities:

- own robot arms
- own cameras
- own keyboard handling
- send one action every `args.duration`
- maintain executed-step counters
- maintain the currently active action queue
- apply action clipping before sending commands
- handle stop, home, and shutdown

This loop must never block on policy inference unless there is no action available at all.

### Inference Worker

Responsibilities:

- wait for inference requests
- preprocess observation
- run `policy.predict_action_chunk(...)`
- postprocess output
- measure inference latency
- return a chunk result plus metadata

This worker should not talk to hardware directly.

It only consumes snapshots:

- observation
- task string
- execution horizon
- RTC metadata

and returns tensors or parsed action dicts.

## Thread Model

Use one background thread:

- `threading.Thread(target=inference_worker_loop, daemon=True)`

Communication uses thread-safe queues:

- `queue.Queue(maxsize=1)` for requests
- `queue.Queue(maxsize=1)` for results

Recommended rule:

- at most one in-flight inference request
- at most one unread inference result

This keeps the system simple and prevents stale chunk accumulation.

## Core Runtime State

The RTC script should maintain a small explicit state object.

Suggested fields:

```python
@dataclass
class RTCExecutionState:
    action_queue: list[dict[str, float]]
    original_chunk_actions: list[dict[str, float]] | None
    prev_chunk_left_over: torch.Tensor | None
    chunk_index: int
    global_step: int
    request_id: int
    in_flight_request_id: int | None
    last_completed_request_id: int | None
    last_obs_time: float | None
    last_infer_latency_s: float | None
    last_infer_delay_steps: int
```

Meaning:

- `action_queue`: actions still available for execution
- `original_chunk_actions`: full untrimmed chunk returned by policy, before delay trimming
- `prev_chunk_left_over`: unexecuted tail from previous chunk, encoded as tensor for RTC guidance
- `chunk_index`: logical chunk counter
- `global_step`: total executed action steps
- `request_id`: monotonically increasing inference request id
- `in_flight_request_id`: request currently being processed by worker
- `last_completed_request_id`: latest result accepted by main loop
- `last_infer_latency_s`: measured worker latency
- `last_infer_delay_steps`: `round(latency / duration)` clipped into valid range

For V1, the value passed into RTC inference should be a fixed constant:

```python
rtc_inference_delay_steps = 6
```

This is separate from the executed-step difference used later for queue replacement.

## Data Flow

### Initial Warm Start

At startup there is no previous chunk, so:

- `prev_chunk_left_over = None`
- main loop captures one observation
- submit first inference request
- wait until first chunk arrives
- populate `action_queue`
- start stepping the robot

The first chunk is the only time we must block.

### Steady-State Loop

After warm start, the main loop repeats at control rate:

1. if enough actions remain, execute next action immediately
2. after each executed step, check whether we should launch the next inference request
3. if no inference is in flight and queue remaining length is below a threshold, capture a fresh observation and submit request
4. keep executing existing queued actions while worker is busy
5. when result arrives, compute real delay in steps and replace queue using RTC semantics

This is the key overlap:

- main thread is executing old actions
- worker thread is preparing new actions

## When To Trigger Next Inference

Do not wait until the queue is empty.

Recommended trigger:

- submit next inference when `len(action_queue) <= rtc_execution_horizon`

or more conservatively:

- submit when `len(action_queue) <= max(2, rtc_execution_horizon)`

The exact threshold can be a CLI argument later, but for first version we can use:

```python
prefetch_threshold = 10
```

Reason:

- if we wait too late, inference finishes after queue starvation
- if we trigger too early, the observation becomes too stale

For V1, use:

```python
execution_horizon = 30
prefetch_threshold = 10
```

This means:

- each policy call produces up to 30 executable actions
- when the remaining queue length drops to 10 or below, launch the next inference request
- this leaves about 10 control steps of buffer for async inference to complete

## RTC Request Contents

Each worker request should include:

```python
{
    "request_id": int,
    "observation": observation_snapshot,
    "task": args.task,
    "execution_horizon": execution_horizon,
    "inference_delay": rtc_inference_delay_steps,
    "prev_chunk_left_over": prev_chunk_left_over_tensor_or_none,
}
```

Important:

- `prev_chunk_left_over` must be computed at request submission time
- not at result consumption time

That is because RTC guidance needs the prefix that was still expected to execute when inference started.

## How To Compute `prev_chunk_left_over`

Use the same semantics as `lerobot.policies.rtc.action_queue.ActionQueue.get_left_over()`.

At the moment we submit a new inference request:

- let `action_queue` be the remaining executable actions from the current chunk
- convert that remaining tail into a tensor with shape `(1, T, action_dim)` or `(T, action_dim)`
- store it in the request as `prev_chunk_left_over`

This should represent the robot plan that is still being executed while the new chunk is being inferred.

## How To Compute `inference_delay`

For V1, do not estimate the RTC inference input delay from runtime latency.

Pass a fixed value into policy inference:

```python
inference_delay = 6
```

This fixed value is used only as the RTC guidance input to `policy.predict_action_chunk(...)`.

Still record worker latency for debugging:

```python
infer_delay_steps_by_time = int(round(infer_latency_s / args.duration))
```

Then clip:

```python
infer_delay_steps_by_time = max(0, min(infer_delay_steps_by_time, len(original_actions)))
```

This wall-clock-derived delay is only for logs and diagnostics in V1.

Also record:

- request submit time
- queue size at submit time
- executed global step at submit time

Then compare:

- wall-clock-derived delay
- actually consumed steps during worker execution

If they differ, log both. The consumed-step value is often more trustworthy than pure wall clock.

Recommended final value:

```python
real_delay = executed_steps_since_submit
```

with wall-clock delay used only as a warning/debug signal.

This matches the intent of `ActionQueue._check_delays(...)`.

Important distinction:

- `rtc_inference_delay_steps = 6` is the fixed value passed into `policy.predict_action_chunk(...)` for RTC guidance in V1
- `real_delay = executed_steps_since_submit` is the value used by the runtime to trim returned actions before replacing the queue

These two values do not need to be identical in V1.

## Result Merge Semantics

Worker returns:

```python
{
    "request_id": int,
    "infer_latency_s": float,
    "action_chunk_tensor": torch.Tensor,
    "actions": list[dict[str, float]],
}
```

When main loop accepts the result:

1. compute `real_delay`
2. discard the first `real_delay` actions from the new chunk
3. replace current queue with the delayed-trimmed new actions
4. compute the new leftover basis for the next request

This is RTC mode, so queue replacement is expected.

Do not append the new actions to the old queue.

Append semantics are for ordinary chunking, not RTC.

## Action Clipping

We are removing safe-mode keyboard chunk stepping, but keeping joint delta clipping.

That means clipping becomes unconditional when enabled by CLI.

Suggested CLI:

- `--clip-actions`
- `--max-joint-step 0.02`

Behavior:

- after worker result is converted into action dicts
- run `_clip_safe_actions(...)`
- save both unclipped and clipped forms if we want debug visibility

This is independent of RTC.

It should happen before queue replacement, so the executable queue always contains the clipped actions.

## Keyboard Behavior

Keep only simple continuous-control semantics:

- `[R]` enter running state
- `[Space]` stop and hold current pose
- `[H]` home
- `[Q]` quit
- `[O]/[P]` gripper open/close when stopped
- `[B]/[N]/[M]` teaching and recorded pose utilities

Remove:

- safe-mode "press `I` for next chunk"
- VR teleop entry and handoff logic

So for RTC script:

- `safe_mode` argument can be removed entirely
- `vr_*` arguments can be removed entirely in the RTC script

I recommend removing both from the RTC script for V1.

## Camera Handling

Only the main thread should read cameras.

Do not let the inference worker call `camera.async_read()`.

Instead:

1. main thread captures observation snapshot
2. snapshot is passed to worker

Reason:

- camera access is hardware-facing and should stay in one place
- this avoids multi-threaded camera race conditions

## Recommended Implementation Structure

### New Script

- `src/lerobot/scripts/lerobot_arx5_dual_infer_rtc.py`

### Reuse From Existing Script

Reuse directly where possible:

- `_visual_image_slot_names`
- `_required_image_slot_names`
- `_parse_camera_specs`
- `_make_camera_configs`
- `_build_dataset_features`
- `_read_dual_state`
- `_build_dual_observation`
- `_split_dual_action`
- `_clip_safe_actions`
- `_home_dual_arms_after_camera_failure`
- `_run_keyboard_command`
- trajectory recorders if still needed

### New RTC-Specific Helpers

Add helpers local to the new script:

- `_action_dicts_to_tensor(...)`
- `_tensor_to_action_dicts(...)`
- `_make_rtc_request(...)`
- `_start_inference_worker(...)`
- `_stop_inference_worker(...)`
- `_maybe_submit_rtc_request(...)`
- `_consume_ready_rtc_result(...)`
- `_reset_rtc_execution_state(...)`

## Suggested Worker API

```python
def inference_worker_loop(
    *,
    request_queue: Queue,
    result_queue: Queue,
    stop_event: threading.Event,
    policy,
    preprocessor,
    postprocessor,
    dataset_features,
    device,
    use_amp: bool,
    robot_type: str,
):
    ...
```

Important constraint:

- the policy object should only be touched by the worker thread once runtime begins

Main thread may still call:

- `policy.reset()`

but only while worker is idle or after worker is paused/stopped.

To keep this simple, use the rule:

- when stopping or homing, first ensure there is no active worker request being processed or discard pending results and reset worker generation id

## Stale Result Protection

This is important.

Suppose:

- request 8 is launched
- user presses Space
- runtime stops
- later request 8 returns

That result must not be applied to the queue.

Solution:

- maintain a scheduler generation id
- every request carries `(generation_id, request_id)`
- whenever runtime is reset, increment `generation_id`
- main loop only accepts results matching current generation

This is the cleanest way to survive stop/resume.

## Main Loop Sketch

```python
while running:
    key = keyboard.get_key()
    state, running = handle_keyboard(...)

    if state != LoopState.RUNNING:
        time.sleep(0.01)
        continue

    if action_queue:
        execute_one_action(action_queue.pop(0))
        global_step += 1
    else:
        if not has_inflight_request:
            observation = build_observation(...)
            submit_first_or_recovery_request(observation)
        result = wait_briefly_for_result()
        if result is None:
            hold_or_sleep_short()
            continue
        accept_result_and_fill_queue(result)
        continue

    maybe_submit_next_request_if_queue_low(...)
    maybe_accept_ready_result_nonblocking(...)
```

Key difference from current script:

- current script is chunk-blocking
- RTC script should become step-driven

The control primitive is no longer "execute chunk".

It becomes:

- "execute one action step"
- "service scheduler"

## Why Step-Driven Main Loop Is Better

Your current script naturally centers around:

- infer one chunk
- execute one chunk

RTC naturally centers around:

- keep action queue non-empty
- keep one-step control cadence stable

So the new RTC script should not try to keep `_execute_dual_chunk(...)` as the main primitive.

Instead, split it into:

- `_execute_dual_action_step(...)`
- scheduler logic around it

That will make the code simpler than trying to reuse the full chunk executor.

## Logging and Debugging

At minimum log:

- request id
- generation id
- queue size at request submit
- queue size at result apply
- inference latency seconds
- inferred delay steps from wall clock
- consumed steps during inference
- applied real delay
- final queue size after replacement

This will make RTC behavior debuggable on hardware.

## First-Version Simplifications

For V1, do not implement:

- ACP integration
- raw recorder integration
- safe-mode chunk stepping
- VR handoff / teleop takeover
- multi-worker parallel inference
- fancy latency prediction

For V1, do implement:

- one async inference worker
- generation id / stale result protection
- queue replacement RTC semantics
- fixed RTC inference delay input of 6 steps
- real delay computed from executed-step difference
- unconditional optional action clipping
- camera failure stop path

## CLI Proposal

Suggested new arguments for `lerobot_arx5_dual_infer_rtc.py`:

- `--execution-horizon` default `30`
- `--rtc-inference-delay-steps` default `6`
- `--rtc-enabled` default `True`
- `--rtc-max-guidance-weight`
- `--rtc-prefix-attention-schedule`
- `--rtc-prefetch-threshold` default `10`
- `--clip-actions`
- `--max-joint-step`

Suggested removed arguments in RTC script:

- `--acp-enable`
- `--acp-cfg-beta`
- `--safe-mode`
- `--raw-train-record-dir`
- `--vr-scale-factor`
- `--vr-no-camera`
- `--vr-no-camera-display`
- `--vr-control-hz`
- `--vr-visualize-placo`

## Recommended Build Order

1. Copy the current dual-arm script into `lerobot_arx5_dual_infer_rtc.py`.
2. Delete ACP and raw recorder code paths first.
3. Delete VR teleop / handoff code paths.
4. Delete safe-mode gating and the `I` key path.
5. Refactor chunk execution into one-step execution helper.
6. Add worker queues and inference thread.
7. Add scheduler state and request/result structs.
8. Implement warm start with `execution_horizon=30`.
9. Pass fixed `rtc_inference_delay_steps=6` into worker inference requests.
10. Implement steady-state queue prefetch at `prefetch_threshold=10`.
11. Use executed-step difference as the merge-time `real_delay`.
12. Add generation id reset on stop and resume.
13. Add logs and hardware validation.

## Recommendation

Use asynchronous RTC from the start.

Do not spend time building a "single-thread RTC" first unless you specifically want:

- an algorithmic correctness baseline
- or an easier unit-test target

For actual robot use, the async version matches the purpose of RTC much better and will likely end up simpler than trying to retrofit true RTC semantics into the current chunk-blocking loop.

## V1 Decisions

These choices are fixed for the first implementation:

- `execution_horizon = 30`
- `prefetch_threshold = 10`
- `rtc_inference_delay_steps = 6`
- `real_delay = global_step_now - submit_global_step`
- wall-clock latency is logged but not used as the primary merge signal
- no ACP
- no raw recorder
- no VR takeover
- no safe-mode "manual next chunk" interaction
