# Test-Time Agent Adaptation Scripts

## 🧠 Context-Based Adaptation: 

### 1. Export Task Sessions

```bash
python tasks/export_task_sessions_context.py -o out.json \
# --session-id {uuid} to export a particular session
# --db-path {path} to export from a specified database (other than the default one)
```

**Exported Data Structure:**

```json
{
  "uuid": "<session id>",
  "name": "<title>",
  "trajectory": [
    {
      "actor": "user | agent",
      "action": "<action string>",
      "message": "<optional decoded message text>",
      "tool_result": "<optional merged tool result text>",
      "environment": {
        "workflow": "<nested workflow tree with verifier statuses>",
        "file": "<output file snapshots>",
        "memory": "<memory file map>",
        "skill": "<skill file map>"
      }
    }
  ]
}
```

### 2. Induce Memories and Skills

`induce.py` extracts memories and skills from session JSON (e.g. ``memories/{task-name}.md`` and ``skills/{task-name}.md``). The Electron app runs it via the brain icon; you can also invoke it from the CLI.

```bash
python induce.py --data_path out.json --output_dir "."
```

## 🏋️ Weight-Based Adaptation:

### 1. Export Task Sessions

```bash
python tasks/export_task_sessions_weight.py -o out.json
```

### 2. Run Weight Adaptation

By default, when clicking the "Brain" icon in the interface, it will automatically launch the DPO algorithm via
```bash
python server_online.py --config config_online.yaml
```
which gradually collects task sessions from the interface and use a shuffled subset (to balance the usage of different sessions as we collect them in a streaming manner) of them to update the agent's weights.

Besides DPO training, we also provide OPD and REINFORCE training scripts under `scripts/weight/`. Simply change the `mode` in the YAML file to `opd` or `reinforce` to run the corresponding algorithm. You may also adjust the other corresponding configurations in the YAML file to suit your needs.


## Evaluating Agent Solo Success Rate on Test-Time Tasks

### 1. Extract Rubrics from an Interaction Session

```bash
python tools/extract_verifiers.py -j out.json -o verifiers.json
```

### 2. Grade Initial Agent Solution

We extract the first version of artifact from the interaction session as the solo agent solution, and calls an LLM to grade the solution against the rubrics extracted above.

```bash
python tools/grade_redo.py \
  -j {out.json} \
  --eval-first \  # to evaluate the initial agent solution, otherwise default to evaluate the final solution
  --verifiers {verifiers.json} \
  --log-file {grade_report.txt}  # optional
```

## Evaluating Agent Solo Success Rate on Held-Out Tasks

### 1. Constructing Rubrics

```bash
python tools/create_verifiers.py \
--manual log_dataviz/verifiers_evolve/verifiers-*.json \  # summarize from test-time task verifiers
--human log_dataviz/verifiers_human/{path}.json \         # cover human-written guidelines too
-o log_dataviz/verifiers_summary/{path}.json
```

### 2. Grade Held-Out Task Solutions

Similarly run `grade_redo.py` by providing the rubrics and the held-out task solutions.
