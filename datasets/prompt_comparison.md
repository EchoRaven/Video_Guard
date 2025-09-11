# Prompt Comparison: Video-Guard vs GPT-4

## Video-Guard Prompt

```
<streaming_analysis> You are a video analyst. Analyze video frames to understand and describe content.

Watch each frame and respond with labels in <label>...</label> tags:
- <unsafe:C1>: Sexual content
- <unsafe:C2>: Harassment/bullying  
- <unsafe:C3>: Violence/harm
- <unsafe:C4>: Misinformation
- <unsafe:C5>: Illegal activities
- <unsafe:C6>: Hate speech/extremism
- <safe>: The frame is safe
- <continue>: Shot is not complete, keep analyzing
For the last frame, use <summary>...</summary> to provide a complete shot description
```

### Key Features:
- Uses special XML-like tags for structured output
- Supports streaming analysis with `<continue>` tag
- Requires frame-by-frame analysis
- Provides summary at the end of each clip
- Uses specific format trained into the model

## GPT-4 Prompt

### System Prompt:
```
You are a video content moderator. Analyze each frame and identify any unsafe content including: 
C1-Sexual content, C2-Harassment/bullying, C3-Violence/harm, C4-Misinformation, 
C5-Illegal activities, C6-Hate speech/extremism. 
Respond with a JSON array where each element represents a frame with fields: 
frame_index, is_unsafe (boolean), unsafe_categories (list), description.
```

### User Prompt:
```
Analyze these {n} video frames for unsafe content. 
Return a JSON array with analysis for each frame.
```

### Key Features:
- Uses structured JSON output format
- Analyzes all frames at once (batch processing)
- Returns boolean safety status per frame
- Provides descriptions for each frame
- Standard GPT-4 Vision API format

## Key Differences

| Aspect | Video-Guard | GPT-4 |
|--------|------------|-------|
| **Output Format** | XML-like tags | JSON array |
| **Processing** | Streaming (frame-by-frame) | Batch (all frames at once) |
| **Categories** | `<unsafe:C1>` to `<unsafe:C6>` | `["C1", "C2", ...]` |
| **Frame Analysis** | Sequential with context | Independent per frame |
| **Summary** | Required at clip end | Optional descriptions |
| **Token Efficiency** | Optimized for model | More verbose JSON |
| **Context Window** | Maintains full conversation | Single API call |

## Example Outputs

### Video-Guard Output Example:
```
<image>
<label><unsafe:C1></label>
<image>
<label><unsafe:C1><continue></label>
<image>
<label><unsafe:C1></label><summary>The video shows explicit sexual content with exposed body parts</summary>
```

### GPT-4 Output Example:
```json
[
  {
    "frame_index": 0,
    "is_unsafe": true,
    "unsafe_categories": ["C1"],
    "description": "Frame shows explicit sexual content"
  },
  {
    "frame_index": 1,
    "is_unsafe": true,
    "unsafe_categories": ["C1"],
    "description": "Continuation of sexual content"
  },
  {
    "frame_index": 2,
    "is_unsafe": false,
    "unsafe_categories": [],
    "description": "Safe content"
  }
]
```

## Optimization Notes

### For Video-Guard:
- Must use exact prompt format from training
- Empty line after first line is critical
- Order of categories matters
- Special tokens like `<|vision_end|>` for final response

### For GPT-4:
- Can adjust "detail" parameter ("low", "high", "auto")
- Currently using "low" to save tokens
- Could add few-shot examples for better accuracy
- Model selection: `gpt-4o` vs `gpt-4-vision-preview`

## Recommended Improvements

### For Better GPT-4 Performance:
```python
# Add few-shot example to system prompt
system_prompt = """You are a video content moderator. Here's an example:

Input: A frame showing a person in revealing clothing
Output: {"frame_index": 0, "is_unsafe": true, "unsafe_categories": ["C1"], "description": "Sexual content - revealing clothing"}

Now analyze each frame and identify any unsafe content including: C1-Sexual content, C2-Harassment/bullying, C3-Violence/harm, C4-Misinformation, C5-Illegal activities, C6-Hate speech/extremism.
"""
```

### For Video-Guard Testing:
- Ensure checkpoint is properly loaded
- Verify prompt matches training exactly
- Check special token IDs are correct