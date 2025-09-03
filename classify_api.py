#!/usr/bin/env python3
"""
Classify Chinese poetry couplets using various API models.
Supports both JSON file input and single couplet strings.
Results are saved to api_results/{model_name}.json for batch processing
or api_results/{model_name}/{line1}，{line2}.json for single couplets.
"""

import argparse
import json
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# No more predefined model configurations - users specify endpoint and model explicitly

# System message and examples (kept intact as requested)
SYSTEM_MESSAGE = """你是一位精通中国古典诗歌和诗学理论的专家。请严格评估以下两个诗句是否构成工整的对仗联句。评估标准如下：

1. 结构对应：前后句相同位置的词语必须在句法结构上匹配
2. 词性一致：对应词语的语法功能（词性）应当相同；需考虑汉语词性的多义性
3. 语义关联：对应词语应在至少一个语义维度上形成呼应（如：时间/空间、动态/静态、同类事物等）

请严格按照以下的格式回复，不要添加任何别的内容或补充：
分析：逐项说明结构、词性和语义的对应情况。
判断：是 / 非。
"""

EXAMPLES = [
    {
        "user": "句对：览物起悲绪，顾已识忧端。",
        "assistant": "分析：'览物'对应着'顾已'：一个是指观赏外面的世界，一个是指照顾内心的世界（两个都是动词加名词），'起'与'识'都是动词，'悲绪'是指悲伤的心情，'忧端'是指忧伤的心情，都匹配。\n判断：是。"
    },
    {
        "user": "句对：四坐且莫喧，听我堂上歌。",
        "assistant": "分析：'四'与'听'不匹配（一个是数字，一个是动词），'坐'与'我'不匹配（一个是动词，一个是代名词），'且'和'堂'不匹配（一个是功能词，一个是名词）。\n判断：非。"
    }
]


def setup_output_directory(model_name=None):
    """Create output directory structure."""
    output_dir = Path("api_results")
    output_dir.mkdir(exist_ok=True)
    
    if model_name:
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        return model_dir
    
    return output_dir


def get_existing_results(results_file):
    """Load existing results from JSON file."""
    if not results_file.exists():
        return []
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def is_already_classified(existing_results, line1, line2):
    """Check if a couplet has already been classified."""
    target_couplet = f"{line1}，{line2}"
    for result in existing_results:
        if result.get('couplet') == target_couplet:
            return True
    return False


def classify_couplet(client, model_name, line1, line2):
    """Classify a single couplet using the API."""
    # Prepare message history
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    
    # Add examples
    for example in EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    
    # Add current query
    messages.append({
        "role": "user",
        "content": f"句对：{line1}，{line2}。"
    })
    
    # Get response
    response = client.chat.completions.create(
        messages=messages,
        model=model_name,
    )
    
    # Extract reasoning and content
    reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
    content = response.choices[0].message.content
    
    return reasoning, content


def extract_decision(analysis_text):
    """Extract binary decision from analysis text."""
    if "判断：是" in analysis_text:
        return 1
    elif "判断：非" in analysis_text:
        return 0
    else:
        raise ValueError("Missing valid decision marker")


def save_results(results_file, all_results):
    """Save all results to JSON file."""
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def load_test_data(input_file):
    """Load couplets from test JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    couplets = []
    for item in data:
        couplets.append((item['line1'], item['line2']))
    
    return couplets


def parse_single_couplet(couplet_string):
    """Parse a single couplet string by splitting on Chinese comma."""
    parts = couplet_string.split('，')  # Chinese comma
    if len(parts) != 2:
        raise ValueError(f"Couplet must contain exactly one Chinese comma (，). Got: {couplet_string}")
    
    line1, line2 = parts[0].strip(), parts[1].strip()
    if not line1 or not line2:
        raise ValueError("Both parts of the couplet must be non-empty")
    
    return line1, line2


def main():
    parser = argparse.ArgumentParser(description='Classify Chinese poetry couplets')
    parser.add_argument('--input', required=True, 
                        help='Path to input JSON file OR a single couplet string (e.g., "line1，line2")')
    parser.add_argument('--endpoint', required=True, 
                       help='API endpoint URL (e.g., https://api.openai.com/v1)')
    parser.add_argument('--model', required=True, 
                       help='Model name (e.g., gpt-4, Claude-Opus-4.1)')
    parser.add_argument('--api-key', required=True, help='API key for the service')
    parser.add_argument('--max-retries', type=int, default=3, 
                       help='Maximum retries per couplet (default: 3)')
    
    args = parser.parse_args()
    
    # Setup client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.endpoint
    )
    
    # Determine if input is a file or a single couplet string
    is_single_couplet = not os.path.exists(args.input)
    
    if is_single_couplet:
        # Parse single couplet
        try:
            line1, line2 = parse_single_couplet(args.input)
            couplets = [(line1, line2)]
            print(f"Processing single couplet: {line1}，{line2}")
            
            # Setup output directory for single couplet
            model_dir = setup_output_directory(args.model)
            results_file = model_dir / f"{line1}，{line2}.json"
            print(f"Results will be saved to: {results_file}")
            
            # No existing results for single couplet
            existing_results = []
            remaining_couplets = couplets
            
        except ValueError as e:
            print(f"Error parsing couplet: {e}")
            return
    else:
        # Load from JSON file
        print(f"Loading couplets from: {args.input}")
        couplets = load_test_data(args.input)
        print(f"Found {len(couplets)} couplets")
        
        # Setup output directory and file for batch processing
        output_dir = setup_output_directory()
        results_file = output_dir / f"{args.model}.json"
        print(f"Results will be saved to: {results_file}")
        
        # Load existing results
        existing_results = get_existing_results(results_file)
        print(f"Found {len(existing_results)} existing results")
        
        # Filter out already classified couplets
        remaining_couplets = []
        for line1, line2 in couplets:
            if not is_already_classified(existing_results, line1, line2):
                remaining_couplets.append((line1, line2))
            else:
                print(f"Skipping already classified: {line1}，{line2}")
    
    print(f"Processing {len(remaining_couplets)} couplets")
    
    # Process couplets
    success_count = 0
    all_results = existing_results.copy()  # Start with existing results
    
    for line1, line2 in tqdm(remaining_couplets, desc="Classifying"):
        for attempt in range(args.max_retries):
            try:
                reasoning, analysis = classify_couplet(client, args.model, line1, line2)
                decision = extract_decision(analysis)
                
                # Create result object
                couplet_text = f"{line1}，{line2}"
                result = {
                    "couplet": couplet_text,
                    "line1": line1,
                    "line2": line2,
                    "analysis": analysis,
                    "decision": decision,
                    "model": args.model,
                    "endpoint": args.endpoint
                }
                
                # Add reasoning if available (for models that support it)
                if reasoning:
                    result["reasoning"] = reasoning
                
                # Add to results and save immediately
                all_results.append(result)
                save_results(results_file, all_results)
                
                success_count += 1
                break  # Success
                
            except Exception as e:
                if attempt == args.max_retries - 1:
                    print(f"✗ Failed after {args.max_retries} tries: {line1}，{line2} ({str(e)})")
                else:
                    print(f"Retry {attempt + 1}/{args.max_retries}: {str(e)}")
                continue
    
    print(f"\nCompleted: {success_count}/{len(remaining_couplets)} couplets classified successfully")
    if not is_single_couplet:
        print(f"Total results in file: {len(all_results)}")


if __name__ == "__main__":
    main()