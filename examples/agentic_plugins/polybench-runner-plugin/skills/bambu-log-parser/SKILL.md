---
name: bambu-log-parser
description: Parses a bambu-log file and writes a JSON file containing device, clock_period, memory_allocation_policy, and total_cycles. Use this skill when you need to extract synthesis and simulation parameters from a Bambu HLS log.
---

Parse a `bambu-log` file and produce a JSON summary.

## Fields to extract

| JSON key | Source in log |
|---|---|
| `device` | `--device=<value>` on the first line |
| `clock_period` | `--clock-period=<value>` on the first line (as a float) |
| `memory_allocation_policy` | `--memory-allocation-policy=<value>` on the first line |
| `total_cycles` | `Total cycles             : <N> cycles` line (integer) |

## Steps

1. **Read** the bambu-log file provided in the arguments.

2. **Extract** the four values using these patterns on the first line of the file (the `bambu` invocation line):
   - `--device=(\S+)`  → `device`
   - `--clock-period=([\d.]+)` → `clock_period` (parse as float)
   - `--memory-allocation-policy=(\S+)` → `memory_allocation_policy`

   And from anywhere in the file:
   - `Total cycles\s+:\s+(\d+) cycles` → `total_cycles` (parse as integer)

3. **Write** a JSON file named bambu_summary.json at the output path specified in the arguments. Format:

```json
{
  "device": "xcu280-2Lfsvh2892-VVD",
  "clock_period": 5.0,
  "memory_allocation_policy": "NO_BRAM",
  "total_cycles": 1146
}
```

4. **Report** the extracted values and the path of the written JSON file.

## Notes

- If a field is not found, use `null` for its value rather than omitting the key.
- Do not use any external tools or scripts — read the file with the Read tool and write with the Write tool.
