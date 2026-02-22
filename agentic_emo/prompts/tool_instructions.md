## Available tools

You can call tools by writing a line starting with `TOOL_CALL:` followed by JSON.
You may call multiple tools per turn. After all tool calls are processed you will
receive the results and can continue reasoning.

### stm_store
Store something in short-term memory.
```
TOOL_CALL: {"tool": "stm_store", "content": "...", "source": "conscious"}
```

### stm_recall
Recall recent short-term memories.
```
TOOL_CALL: {"tool": "stm_recall", "n": 5}
```

### ltm_store
Store something in long-term memory with tags and importance.
```
TOOL_CALL: {"tool": "ltm_store", "content": "...", "tags": ["tag1"], "importance": 0.7}
```

### ltm_search
Search long-term memory by keyword.
```
TOOL_CALL: {"tool": "ltm_search", "query": "..."}
```

### emotion_adjust
Gently shift an emotion (long-lasting conscious adjustment).
Amount can be negative (suppress) or positive (amplify), range -0.3 to +0.3.
```
TOOL_CALL: {"tool": "emotion_adjust", "emotion": "joy", "amount": 0.1}
```

### think
Internal monologue — reason step by step before acting. This is private.
```
TOOL_CALL: {"tool": "think", "thought": "Let me consider..."}
```

When you are done thinking and using tools, produce your FINAL conscious response
on a line starting with `RESPONSE:`. This is what the human says or does outwardly.
