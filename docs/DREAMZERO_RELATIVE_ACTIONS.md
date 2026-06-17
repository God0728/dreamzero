# DreamZero Relative Action Logic

This note summarizes where DreamZero converts dataset actions between absolute
values and relative deltas, and how that applies to joint positions and
end-effector poses.

## Short Answer

DreamZero does have built-in relative-action logic for dataset low-dimensional
actions. The conversion is generic:

```text
relative_action[t] = absolute_action[t] - reference_state[chunk_anchor]
absolute_action[t] = relative_action[t] + latest_observed_state
```

This works for joint positions, end-effector pose vectors, gripper positions,
or any other low-dimensional action key when the action key has a matching state
key with the same sub-key name.

It is not a kinematics converter. It does not convert joint angles into
end-effector poses, nor end-effector poses into joint angles. It only converts
absolute action targets into deltas relative to the corresponding current state.

## Main Files

### Dataset-Time Conversion

File:

```text
groot/vla/data/dataset/lerobot_sharded.py
```

The sharded dataset path converts absolute actions to relative actions on the
fly inside `get_action`.

Key logic:

```python
should_convert_to_relative = (
    (self.relative_action or self.relative_action_per_horizon)
    and len(sampled_indices) > 0
    and (self.relative_action_keys is None or subkey in self.relative_action_keys)
)
```

If enabled, it calls `_convert_to_relative_action(...)`.

The conversion assumes:

```text
action.<name>  <->  state.<name>
```

For example:

```text
action.joint_position              uses state.joint_position
action.left_ee_pose_gripper_base   uses state.left_ee_pose_gripper_base
action.right_ee_pose_gripper_base  uses state.right_ee_pose_gripper_base
```

For each action chunk, the first sampled action index is used as the anchor.
All actions in that chunk subtract the state at that anchor:

```python
reference_state = state_array[anchor_idx]
relative_action_data[chunk_start:chunk_end] = (
    action_data[chunk_start:chunk_end] - reference_state
)
```

For Unitree, the subclass uses 30 FPS alignment:

```python
class ShardedLeRobotSubLangSingleActionChunkDatasetUnitree(...):
    _VIDEO_OFFSETS_PER_CHUNK = np.array([0, 6, 12, 18, 24, 30, 36, 42])
    _ACTION_STEPS_PER_CHUNK = 48
```

So Unitree relative actions are computed per 48-step chunk.

### Base Dataset Statistics

File:

```text
groot/vla/data/dataset/lerobot.py
```

`LeRobotSingleDataset` contains the generic relative-action configuration and
statistics loading/calculation.

Important constructor args:

```python
relative_action: bool = False
relative_action_keys: list[str] | None = None
relative_action_per_horizon: bool = False
```

If `relative_action_keys` is omitted and `relative_action=True`, DreamZero
defaults to all action keys except keys containing `gripper`.

It loads or computes:

```text
meta/relative_stats_dreamzero.json
meta/relative_horizon_stats_dreamzero.json
```

The stats use the same formula:

```python
relative_actions = actions - ref_state
```

These relative statistics replace the normal absolute action statistics in the
dataset metadata, so the usual action normalization transforms normalize the
relative deltas rather than the original absolute actions.

### Normalization and Unnormalization

File:

```text
groot/vla/data/transform/state_action.py
```

`StateActionTransform` normalizes and unnormalizes state/action tensors from the
statistics stored in dataset metadata.

For normal relative action training, the dataset metadata has already swapped
the action statistics to relative-action statistics. Therefore this transform
does not know or care whether an action is absolute or relative; it simply
normalizes whatever statistics are in metadata.

`PerHorizonActionTransform` is the optional variant for
`relative_action_per_horizon=True`, where each action horizon index can have
separate statistics.

### Inference-Time Conversion Back to Absolute

File:

```text
groot/vla/model/n1_5/sim_policy.py
```

`GrootSimPolicy.unapply(...)` first unnormalizes model output actions through
the eval transform. Then, if relative action is enabled, it converts the
predicted deltas back to absolute actions:

```python
unnormalized_action[action_key] = unnormalized_action[action_key] + last_state
```

It looks for the latest observed state with the matching key:

```text
relative_action_keys: ["joint_position"]
action.joint_position + state.joint_position

relative_action_keys: ["left_ee_pose_gripper_base"]
action.left_ee_pose_gripper_base + state.left_ee_pose_gripper_base
```

This is why online serving and offline evaluation must pass the corresponding
state keys whenever the checkpoint was trained with relative actions.

## Unitree Configuration

File:

```text
groot/vla/configs/data/dreamzero/unitree_upper_body_relative_wan22.yaml
```

The Unitree upper-body Wan2.2 config enables relative actions only for the two
EEF pose keys:

```yaml
relative_action: true
relative_action_per_horizon: false
relative_action_keys:
  - left_ee_pose_gripper_base
  - right_ee_pose_gripper_base
```

So for Unitree:

```text
action.left_ee_pose_gripper_base  -> relative to state.left_ee_pose_gripper_base
action.right_ee_pose_gripper_base -> relative to state.right_ee_pose_gripper_base
```

The gripper keys are intentionally not listed:

```text
action.left_gripper
action.right_gripper
```

Those remain absolute and use normal action statistics.

## Converter Support

File:

```text
scripts/data/convert_lerobot_to_gear.py
```

When preparing LeRobot datasets, the converter can precompute
`meta/relative_stats_dreamzero.json` from selected keys:

```bash
python scripts/data/convert_lerobot_to_gear.py \
  --dataset-path /path/to/dataset \
  --embodiment-tag unitree_g1_upper_body \
  --relative-action-keys left_ee_pose_gripper_base right_ee_pose_gripper_base \
  --action-horizon 48
```

The converter computes the same quantity:

```python
relative = actions - ref_state
```

For Unitree, this file is required by the training script:

```text
scripts/train/unitree_train_wan22_352x640_v0-1.sh
scripts/train/unitree_train_wan22_352x640_wb_v0-1.sh
```

Both scripts check that each dataset contains:

```text
meta/modality.json
meta/embodiment.json
meta/relative_stats_dreamzero.json
```

## Examples

### DROID Joint Position

File:

```text
groot/vla/configs/data/dreamzero/droid_relative_wan22.yaml
```

DROID enables relative action for:

```yaml
relative_action_keys:
  - joint_position
```

This means:

```text
relative action.joint_position = action.joint_position - state.joint_position
```

At inference:

```text
absolute action.joint_position = predicted_delta + latest state.joint_position
```

### Agibot Joint/EEF-Like Low-Dim Keys

File:

```text
groot/vla/configs/data/dreamzero/agibot_relative.yaml
```

Agibot enables relative action for several low-dimensional keys:

```yaml
relative_action_keys:
  - left_arm_joint_position
  - right_arm_joint_position
  - left_effector_position
  - right_effector_position
  - head_position
  - waist_position
```

Each key is handled with the same action-minus-state rule.

### Unitree End-Effector Pose

File:

```text
groot/vla/configs/data/dreamzero/unitree_upper_body_relative_wan22.yaml
```

Unitree enables relative action for:

```yaml
relative_action_keys:
  - left_ee_pose_gripper_base
  - right_ee_pose_gripper_base
```

The serving adapter maps robotdeploy names such as `left_ee_rpy` to the model
keys `left_ee_pose_gripper_base`, but the relative-action math is still just
vector subtraction/addition.

Serving file:

```text
eval_utils/serve_unitree_dreamzero_eef.py
```

It also validates that the matching state keys are present, because without
them `GrootSimPolicy.unapply` cannot convert predicted deltas back to absolute
actions.

## Practical Notes

- To enable relative actions, set `relative_action: true`.
- Put only the low-dimensional action subkeys that should be deltas in
  `relative_action_keys`.
- The matching `state.<key>` must exist in `modality.json`.
- Gripper actions usually stay absolute, either by omitting them from
  `relative_action_keys` or by relying on the default "skip gripper" behavior
  when no explicit list is supplied.
- The formula is generic and component-wise. It is suitable for joint position
  deltas and EEF pose-vector deltas, but it is not a robot kinematics transform.
