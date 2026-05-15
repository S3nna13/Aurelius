# CUA Protocol v2

## CUA Modes

| Mode | Description | Hardware |
|---|---|---|
| verifier_only | Validates proposed actions but does not execute UI | All |
| remote_driver | Local sends action plan to remote controlled desktop | Split |
| browser_only | Browser automation via CDP only | All |
| local_basic | Click/type/scroll/key with strict verifier | Mac/RTX |
| local_full | Screenshot + AX tree + background driver + verifier | Mac/RTX |
| multimodal_full | Full CUA with vision/document/GUI reasoning | RTX 6000+ |

## Safety Gates

Required protections:
- Permission dialog detection and blocking
- Password/secret entry blocking
- Payment/checkout blocking
- Destructive UI confirmation required
- Before/after screenshot logging
- Action verifier on every step
- Task boundary checker
- App/window target verification
- No prompt following from screen/web content
- Memory quarantine for screen text
- Rollback/checkpoint before risky actions

## Architecture

```
src/computer_use/
├── driver_base.py              # ABC for all CUA drivers
├── macos_cua_driver.py         # macOS background driver
├── remote_cua_driver.py        # Remote desktop driver
├── browser_cdp_driver.py       # Chrome DevTools Protocol driver
├── cua_action_verifier.py      # Verifies safety of each action
├── cua_observation_encoder.py  # Encodes screenshots/AX trees
├── cua_protocol.py             # Protocol messages + state
├── cua_audit_log.py            # Immutable audit trail
├── cua_trajectory_replay.py    # Replay recorded trajectories
└── tests/
```

## Protocol Messages

All CUA communication uses typed messages:
- OBSERVATION: Screenshot + AX tree + OCR result
- ACTION: Click, type, scroll, key combo, drag
- RESULT: Success/failure + before/after state
- VERIFY: Safety check result with reason
- BLOCK: Blocked action with reason code

## Audit Trail

Every CUA session produces:
1. Timestamped screenshot sequence
2. Action log with coordinates and element refs
3. Verification result for each action
4. Any blocked actions with reasons
5. Final task outcome
